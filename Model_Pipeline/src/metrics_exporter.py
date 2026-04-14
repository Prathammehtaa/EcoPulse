"""
EcoPulse — Prometheus Metrics Exporter
=======================================
Exposes model performance metrics as a Prometheus endpoint or pushes
them to a Pushgateway for CI consumption.

Usage:
    python metrics_exporter.py --serve   # HTTP server on port 8000
    python metrics_exporter.py --push    # Push to Pushgateway once, then exit

Environment variables:
    PUSHGATEWAY_URL         Pushgateway address (default: http://localhost:9091)
    REPORTS_DIR             Path to reports root (default: Model_Pipeline/reports)
    VALIDATION_REPORT_PATH  Override validation_report.json location
    METRICS_PORT            HTTP server port (default: 8000)
    PUSH_INTERVAL           Seconds between metric refreshes in serve mode (default: 30)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    push_to_gateway,
    start_http_server,
)

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ecopulse.metrics")

# ── Configurable paths via environment variables ─────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SRC_DIR)
_DEFAULT_REPORTS_DIR = os.path.join(_PIPELINE_DIR, "reports")

REPORTS_DIR = os.environ.get("REPORTS_DIR", _DEFAULT_REPORTS_DIR)
VALIDATION_REPORT_PATH = os.environ.get(
    "VALIDATION_REPORT_PATH",
    os.path.join(REPORTS_DIR, "validation", "validation_report.json"),
)
BIAS_DIR = os.path.join(REPORTS_DIR, "bias")
COMPARISON_CSV_PATH = os.path.join(REPORTS_DIR, "full_comparison.csv")
DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_report.json")

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))
PUSH_INTERVAL = int(os.environ.get("PUSH_INTERVAL", "30"))

HORIZONS = ["1h", "6h", "12h", "24h"]

# ── Prometheus registry and gauges ───────────────────────────────
registry = CollectorRegistry()

ecopulse_mae = Gauge(
    "ecopulse_mae",
    "Mean Absolute Error per horizon and model type",
    ["horizon", "model_type"],
    registry=registry,
)
ecopulse_rmse = Gauge(
    "ecopulse_rmse",
    "Root Mean Square Error per horizon and model type",
    ["horizon", "model_type"],
    registry=registry,
)
ecopulse_r2 = Gauge(
    "ecopulse_r2",
    "R-squared per horizon and model type",
    ["horizon", "model_type"],
    registry=registry,
)
ecopulse_mae_pct_change = Gauge(
    "ecopulse_mae_pct_change",
    "MAE percentage change from previous run",
    ["horizon"],
    registry=registry,
)
ecopulse_bias_critical_count = Gauge(
    "ecopulse_bias_critical_count",
    "Number of critical bias failures across all horizons",
    registry=registry,
)
ecopulse_training_timestamp = Gauge(
    "ecopulse_training_timestamp",
    "Unix timestamp of last training run",
    registry=registry,
)
ecopulse_drift_detected = Gauge(
    "ecopulse_drift_detected",
    "Whether data drift was detected (1 = yes, 0 = no)",
    registry=registry,
)


# ── Report readers ───────────────────────────────────────────────

def _read_json(path: str) -> dict | None:
    """Read a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.warning("Report not found: %s — skipping", path)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s — skipping", path, exc)
        return None


def _load_validation_report() -> dict | None:
    """Read the validation_report.json fresh from disk."""
    return _read_json(VALIDATION_REPORT_PATH)


def _load_drift_report() -> dict | None:
    """Read the drift_report.json fresh from disk."""
    return _read_json(DRIFT_REPORT_PATH)


def _find_latest_bias_csvs() -> dict[str, str]:
    """
    Scan the bias directory and return the most recently modified
    disparity_report CSV per horizon. Never hardcodes filenames since
    they contain timestamps.

    Returns:
        dict mapping horizon label (e.g. "1h") to the file path
    """
    if not os.path.isdir(BIAS_DIR):
        logger.warning("Bias directory not found: %s — skipping bias metrics", BIAS_DIR)
        return {}

    latest: dict[str, tuple[float, str]] = {}

    for entry in os.scandir(BIAS_DIR):
        if not entry.name.startswith("disparity_report_") or not entry.name.endswith(".csv"):
            continue

        # Extract horizon from filename like disparity_report_xgboost_1h_20260414_033429.csv
        parts = entry.name.replace(".csv", "").split("_")
        horizon_label = None
        for part in parts:
            if part in ("1h", "6h", "12h", "24h"):
                horizon_label = part
                break

        if horizon_label is None:
            continue

        mtime = entry.stat().st_mtime
        if horizon_label not in latest or mtime > latest[horizon_label][0]:
            latest[horizon_label] = (mtime, entry.path)

    return {h: path for h, (_, path) in latest.items()}


BIAS_CRITICAL_THRESHOLD = float(os.environ.get("BIAS_CRITICAL_THRESHOLD", "80"))


def _count_bias_critical(bias_csvs: dict[str, str]) -> int:
    """
    Count bias failures where abs(pct_deviation) exceeds the critical
    threshold (default 80%).  Only these are truly critical — lower
    deviations are flagged but not critical.
    """
    total = 0
    for horizon, path in bias_csvs.items():
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get("pct_deviation", "").strip()
                    try:
                        pct_val = abs(float(raw.replace("%", "").replace("+", "")))
                    except (ValueError, AttributeError):
                        continue
                    if pct_val > BIAS_CRITICAL_THRESHOLD:
                        total += 1
        except (OSError, csv.Error) as exc:
            logger.warning("Failed to read bias CSV %s: %s — skipping", path, exc)
    return total


def _load_baseline_mae() -> dict[str, float]:
    """
    Read full_comparison.csv and return baseline MAE per horizon.
    Scans for rows where model_type == 'baseline'.

    Returns:
        dict mapping horizon label (e.g. "1h") to baseline MAE float
    """
    baselines: dict[str, float] = {}
    try:
        with open(COMPARISON_CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_type", "").strip() != "baseline":
                    continue
                horizon_raw = row.get("horizon", "").strip()
                mae_str = row.get("mae", "").strip()
                if not horizon_raw or not mae_str:
                    continue
                try:
                    baselines[f"{horizon_raw}h"] = float(mae_str)
                except ValueError:
                    continue
    except FileNotFoundError:
        logger.warning("Comparison CSV not found: %s — MAE pct change skipped", COMPARISON_CSV_PATH)
    except (OSError, csv.Error) as exc:
        logger.warning("Failed to read comparison CSV %s: %s — skipping", COMPARISON_CSV_PATH, exc)
    return baselines


# ── Metric collection ────────────────────────────────────────────

def collect_metrics() -> None:
    """
    Read all report files fresh from disk and update Prometheus gauges.
    Called on every scrape cycle in serve mode, or once in push mode.
    """
    # ── Validation report (MAE / RMSE / R²) ──────────────────
    report = _load_validation_report()
    if report:
        for horizon in HORIZONS:
            h_data = report.get(horizon)
            if not h_data:
                continue
            test_metrics = h_data.get("metrics", {}).get("test", {})
            if not test_metrics:
                continue

            model_type = "xgboost_tuned"
            if "mae" in test_metrics:
                ecopulse_mae.labels(horizon=horizon, model_type=model_type).set(
                    test_metrics["mae"]
                )
            if "rmse" in test_metrics:
                ecopulse_rmse.labels(horizon=horizon, model_type=model_type).set(
                    test_metrics["rmse"]
                )
            if "r2" in test_metrics:
                ecopulse_r2.labels(horizon=horizon, model_type=model_type).set(
                    test_metrics["r2"]
                )

        # Training timestamp — use validation report file mtime
        try:
            mtime = os.path.getmtime(VALIDATION_REPORT_PATH)
            ecopulse_training_timestamp.set(mtime)
        except OSError:
            pass
    else:
        logger.warning("No validation report available — MAE/RMSE/R² metrics skipped")

    # ── MAE % change (current vs baseline from full_comparison.csv) ─
    baseline_mae = _load_baseline_mae()
    if report and baseline_mae:
        for horizon in HORIZONS:
            h_data = report.get(horizon)
            if not h_data:
                continue
            current_mae = h_data.get("metrics", {}).get("test", {}).get("mae")
            bl = baseline_mae.get(horizon)
            if current_mae is not None and bl is not None and bl > 0:
                pct_change = (current_mae - bl) / bl
                ecopulse_mae_pct_change.labels(horizon=horizon).set(pct_change)
    elif not baseline_mae:
        logger.info("No baseline MAE available — MAE pct change metrics skipped")

    # ── Drift report ─────────────────────────────────────────
    drift_report = _load_drift_report()
    if drift_report:
        drift_flag = 1.0 if drift_report.get("drift_detected", False) else 0.0
        ecopulse_drift_detected.set(drift_flag)
    else:
        logger.info("No drift report available — drift metric skipped")

    # ── Bias critical count ──────────────────────────────────
    bias_csvs = _find_latest_bias_csvs()
    if bias_csvs:
        critical_count = _count_bias_critical(bias_csvs)
        ecopulse_bias_critical_count.set(critical_count)
        logger.info("Bias critical count: %d (from %d horizon files)", critical_count, len(bias_csvs))
    else:
        logger.info("No bias CSVs found — bias metric skipped")


# ── Modes ────────────────────────────────────────────────────────

def serve() -> None:
    """Start an HTTP server that exposes metrics on METRICS_PORT."""
    logger.info("Starting metrics server on port %d", METRICS_PORT)
    start_http_server(METRICS_PORT, registry=registry)
    logger.info("Metrics server running at http://0.0.0.0:%d/metrics", METRICS_PORT)

    while True:
        collect_metrics()
        time.sleep(PUSH_INTERVAL)


def push() -> None:
    """Push metrics to Pushgateway once, then exit."""
    collect_metrics()
    logger.info("Pushing metrics to Pushgateway at %s", PUSHGATEWAY_URL)
    try:
        push_to_gateway(PUSHGATEWAY_URL, job="ecopulse-metrics", registry=registry)
        logger.info("Metrics pushed successfully")
    except Exception as exc:
        logger.error("Failed to push metrics to Pushgateway: %s", exc)
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EcoPulse Prometheus Metrics Exporter",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--serve",
        action="store_true",
        help="Start HTTP server exposing metrics continuously",
    )
    group.add_argument(
        "--push",
        action="store_true",
        help="Push metrics to Pushgateway once and exit",
    )
    args = parser.parse_args()

    if args.serve:
        serve()
    elif args.push:
        push()


if __name__ == "__main__":
    main()
