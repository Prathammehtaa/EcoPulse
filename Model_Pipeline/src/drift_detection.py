"""
EcoPulse — TFDV Data Drift Detection
======================================
Generates TFDV statistics on an incoming data file, compares them against
saved training data statistics, and flags drift when > 30 % of features
show distribution anomalies.  Results are logged to MLflow under the
'ecopulse-monitoring' experiment.

On the first run (or when --train-stats is absent) the script loads the
canonical training split, generates baseline statistics, and saves them
so subsequent runs load instantly.

Requires: tensorflow-data-validation  (listed in requirements.txt)

Output:
    Model_Pipeline/reports/drift_report.json
    Model_Pipeline/reports/train_stats.pb   (generated on first run)

CLI:
    python drift_detection.py --data path/to/data.parquet
    python drift_detection.py --data path/to/data.parquet \\
                              --train-stats custom/train_stats.pb
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import mlflow

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlflow_config import setup_mlflow, _git_commit
from utils import REPORTS_DIR, TRAIN_PATH

# ============================================================
# TFDV — optional import guard (mirrors tfdv_bias_analysis.py)
# ============================================================
try:
    import tensorflow_data_validation as tfdv
    TFDV_AVAILABLE = True
except ImportError:
    tfdv            = None          # type: ignore[assignment]
    TFDV_AVAILABLE  = False

# ============================================================
# CONFIG
# ============================================================
MONITORING_EXPERIMENT = "ecopulse-monitoring"

DRIFT_THRESHOLD = 0.30   # > 30 % of features anomalous → drifted

DEFAULT_TRAIN_STATS_PATH = os.path.join(REPORTS_DIR, "train_stats.pb")
DRIFT_REPORT_PATH        = os.path.join(REPORTS_DIR, "drift_report.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecopulse-model.drift")


# ============================================================
# TFDV helpers
# ============================================================

def _require_tfdv():
    if not TFDV_AVAILABLE:
        raise ImportError(
            "tensorflow-data-validation is required but not installed.\n"
            "Install with: pip install tensorflow-data-validation"
        )


def generate_stats(df: pd.DataFrame):
    """Generate TFDV statistics from a DataFrame."""
    _require_tfdv()
    logger.info(
        "Generating TFDV statistics for %d rows × %d cols …", *df.shape
    )
    stats = tfdv.generate_statistics_from_dataframe(df)
    logger.info("Statistics generated.")
    return stats


def save_stats(stats, path: str) -> None:
    """Persist statistics to disk (TFDV text-protobuf format)."""
    _require_tfdv()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tfdv.write_stats_text(stats, path)
    logger.info("Statistics saved → %s", path)


def load_stats(path: str):
    """Load statistics from disk."""
    _require_tfdv()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats file not found: {path}")
    stats = tfdv.load_stats_text(path)
    logger.info("Loaded statistics from %s", path)
    return stats


def load_or_generate_train_stats(stats_path: str):
    """
    Return training baseline statistics:
      • If stats_path exists → load and return immediately.
      • Otherwise → load the canonical training parquet (TRAIN_PATH),
        generate statistics, save to stats_path, and return.
    """
    if os.path.exists(stats_path):
        logger.info(
            "Found existing train stats at %s — loading.", stats_path
        )
        return load_stats(stats_path)

    logger.info(
        "No train stats found at %s — generating from training data…",
        stats_path,
    )
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_PATH}. "
            "Run the data pipeline before drift detection."
        )
    train_df = pd.read_parquet(TRAIN_PATH)
    logger.info("Loaded training data: %d rows × %d cols", *train_df.shape)
    stats = generate_stats(train_df)
    save_stats(stats, stats_path)
    return stats


# ============================================================
# Drift detection logic
# ============================================================

def _count_features(stats) -> int:
    """Total number of features in a TFDV statistics object."""
    try:
        return len(stats.datasets[0].features)
    except (IndexError, AttributeError):
        return 0


def detect_drift(train_stats, new_stats) -> dict:
    """
    Infer a schema from *train_stats*, validate *new_stats* against it,
    and compute the drift fraction.

    Returns:
        {
          "n_total_features":     int,
          "n_anomalous_features": int,
          "drift_fraction":       float,   # 0.0 – 1.0
          "drifted":              bool,
          "anomalies":            {feature_name: short_description, …}
        }
    """
    _require_tfdv()

    schema    = tfdv.infer_schema(train_stats)
    anomalies = tfdv.validate_statistics(statistics=new_stats, schema=schema)

    n_total    = _count_features(train_stats)
    anomaly_map = {
        feat: info.short_description
        for feat, info in anomalies.anomaly_info.items()
    }
    n_anomalous  = len(anomaly_map)
    drift_frac   = round(n_anomalous / n_total, 4) if n_total > 0 else 0.0
    drifted      = drift_frac > DRIFT_THRESHOLD

    logger.info(
        "Drift check: %d / %d features anomalous (%.1f%%) — %s",
        n_anomalous, n_total, drift_frac * 100,
        "DRIFTED" if drifted else "OK",
    )
    for feat, desc in anomaly_map.items():
        logger.warning("  Anomalous feature: %s — %s", feat, desc)

    return {
        "n_total_features":     n_total,
        "n_anomalous_features": n_anomalous,
        "drift_fraction":       drift_frac,
        "drifted":              drifted,
        "anomalies":            anomaly_map,
    }


def build_per_feature_report(drift_result: dict, new_stats) -> dict:
    """
    Build a per-feature status dict.

    Every feature in *new_stats* receives either::
        {"status": "ok"}
    or::
        {"status": "drifted", "description": "<short_description>"}
    """
    _require_tfdv()
    anomalies   = drift_result["anomalies"]
    per_feature = {}
    try:
        for feature in new_stats.datasets[0].features:
            name = feature.name
            if name in anomalies:
                per_feature[name] = {
                    "status":      "drifted",
                    "description": anomalies[name],
                }
            else:
                per_feature[name] = {"status": "ok"}
    except (IndexError, AttributeError) as exc:
        logger.warning("Could not extract per-feature stats: %s", exc)
    return per_feature


# ============================================================
# Full pipeline
# ============================================================

def run_drift_detection(
    data_path: str,
    train_stats_path: str = DEFAULT_TRAIN_STATS_PATH,
) -> dict:
    """
    End-to-end drift detection:

    1. Load (or generate + cache) training baseline statistics.
    2. Load incoming data, generate statistics.
    3. Compare → classify drift.
    4. Build per-feature report.
    5. Return the complete report dict (not yet persisted — call
       save_drift_report separately).
    """
    if not TFDV_AVAILABLE:
        return {
            "error":      "tensorflow-data-validation is not installed.",
            "drifted":    False,
            "checked_at": datetime.utcnow().isoformat(),
        }

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    logger.info("Loading incoming data from %s …", data_path)
    new_df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows × %d cols", *new_df.shape)

    train_stats  = load_or_generate_train_stats(train_stats_path)
    new_stats    = generate_stats(new_df)
    drift_result = detect_drift(train_stats, new_stats)
    per_feature  = build_per_feature_report(drift_result, new_stats)

    return {
        "checked_at":           datetime.utcnow().isoformat(),
        "data_file":            str(data_path),
        "train_stats_path":     train_stats_path,
        "drift_threshold_pct":  DRIFT_THRESHOLD * 100,
        "drifted":              drift_result["drifted"],
        "n_total_features":     drift_result["n_total_features"],
        "n_anomalous_features": drift_result["n_anomalous_features"],
        "drift_fraction":       drift_result["drift_fraction"],
        "per_feature":          per_feature,
    }


# ============================================================
# Report persistence
# ============================================================

def save_drift_report(report: dict) -> str:
    """Write the drift report to JSON.  Returns the file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Drift report saved → %s", DRIFT_REPORT_PATH)
    return DRIFT_REPORT_PATH


# ============================================================
# MLflow logging
# ============================================================

def log_drift_to_mlflow(report: dict) -> Optional[str]:
    """
    Log drift results as a new run under 'ecopulse-monitoring'.
    The JSON report must already be saved to disk before calling this.
    Returns the run_id, or None on failure.
    """
    try:
        setup_mlflow(MONITORING_EXPERIMENT)
        run_name = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "project":    "ecopulse",
                "run_type":   "drift_detection",
                "git_commit": _git_commit(),
                "drifted":    str(report.get("drifted", False)),
            })
            mlflow.log_params({
                "data_file":           os.path.basename(
                    str(report.get("data_file", "unknown"))
                ),
                "drift_threshold_pct": DRIFT_THRESHOLD * 100,
            })
            mlflow.log_metrics({
                "n_total_features":     float(report.get("n_total_features", 0)),
                "n_anomalous_features": float(report.get("n_anomalous_features", 0)),
                "drift_fraction":       float(report.get("drift_fraction", 0.0)),
            })
            if os.path.exists(DRIFT_REPORT_PATH):
                mlflow.log_artifact(DRIFT_REPORT_PATH, artifact_path="drift")

            run_id = mlflow.active_run().info.run_id
            logger.info("Drift run logged to MLflow (run_id=%s).", run_id)
            return run_id

    except Exception as exc:
        logger.error("Failed to log drift run to MLflow: %s", exc)
        return None


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "EcoPulse TFDV drift detection — compares incoming data "
            "against the training baseline statistics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to the incoming data parquet file to check for drift.",
    )
    parser.add_argument(
        "--train-stats",
        default=DEFAULT_TRAIN_STATS_PATH,
        metavar="PATH",
        help=(
            f"Path to saved training statistics (default: {DEFAULT_TRAIN_STATS_PATH}). "
            "Auto-generated from training data if absent."
        ),
    )
    return parser


def main():
    args = _build_parser().parse_args()

    print("=" * 60)
    print("EcoPulse — TFDV Drift Detection")
    print("=" * 60)

    if not TFDV_AVAILABLE:
        print("\n  ERROR: tensorflow-data-validation is not installed.")
        print("  Install with: pip install tensorflow-data-validation")
        sys.exit(1)

    report = run_drift_detection(
        data_path=args.data,
        train_stats_path=args.train_stats,
    )

    # Persist before logging (MLflow artifact upload reads from disk)
    save_drift_report(report)
    log_drift_to_mlflow(report)

    # ── Summary printout ───────────────────────────────────────
    drifted      = report.get("drifted", False)
    n_total      = report.get("n_total_features", 0)
    n_anomalous  = report.get("n_anomalous_features", 0)
    drift_frac   = report.get("drift_fraction", 0.0)
    threshold    = report.get("drift_threshold_pct", DRIFT_THRESHOLD * 100)

    print()
    print(f"  Overall drift status : {'DRIFTED' if drifted else 'OK'}")
    print(f"  Checked at           : {report['checked_at']}")
    print(f"  Drift threshold      : > {threshold:.0f}% of features")
    print(f"  Features checked     : {n_total}")
    print(f"  Anomalous features   : {n_anomalous}  ({drift_frac * 100:.1f}%)")

    per_feature = report.get("per_feature", {})
    drifted_feats = [
        (feat, info["description"])
        for feat, info in per_feature.items()
        if info.get("status") == "drifted"
    ]
    if drifted_feats:
        print(f"\n  Drifted features ({len(drifted_feats)}):")
        for feat, desc in drifted_feats:
            print(f"    - {feat}: {desc}")

    print(f"\n  Report → {DRIFT_REPORT_PATH}\n")

    if drifted:
        sys.exit(1)


if __name__ == "__main__":
    main()
