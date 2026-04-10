"""
EcoPulse — Model Performance Monitor
=====================================
Queries MLflow for the most recent training run metrics per horizon,
compares current MAE against the training reference baseline, flags
performance decay, and logs results to MLflow under 'ecopulse-monitoring'.

Decay thresholds (relative MAE increase from reference):
    WARNING  — > 20 %
    CRITICAL — > 35 %

When only one training run exists for a horizon, the naive baseline MAE
stored in mlflow_config._BASELINE_MAE is used as the reference instead.

Output:
    Model_Pipeline/reports/monitoring_report.json

CLI:
    python monitor.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlflow_config import (
    setup_mlflow,
    TRAINING_EXPERIMENT_NAME,
    _BASELINE_MAE,
    _git_commit,
)
from utils import HORIZONS, REPORTS_DIR

# ============================================================
# CONFIG
# ============================================================
MONITORING_EXPERIMENT = "ecopulse-monitoring"

DECAY_WARNING  = 0.20   # > 20 % MAE increase → WARNING
DECAY_CRITICAL = 0.35   # > 35 % MAE increase → CRITICAL

REPORT_PATH = os.path.join(REPORTS_DIR, "monitoring_report.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecopulse-model.monitor")


# ============================================================
# MLflow helpers
# ============================================================

def _client_and_exp(experiment_name: str):
    """Return (MlflowClient, experiment_entity | None)."""
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    return client, exp


def get_recent_runs(horizon: int, n: int = 2):
    """
    Return up to *n* most recent FINISHED training runs for *horizon*,
    sorted newest-first.  Returns [] if the experiment doesn't exist.
    """
    client, exp = _client_and_exp(TRAINING_EXPERIMENT_NAME)
    if exp is None:
        return []
    return client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"tags.horizon_h = '{horizon}' "
            f"AND attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=n,
    )


# ============================================================
# Decay classification
# ============================================================

def classify_decay(current_mae: float, reference_mae: float):
    """
    Compare *current_mae* against *reference_mae*.

    Returns (status, pct_change) where:
        pct_change > 0  → degradation
        status          → 'healthy' | 'warning' | 'critical' | 'unknown'
    """
    if not reference_mae:
        return "unknown", 0.0
    pct = (current_mae - reference_mae) / reference_mae
    if pct > DECAY_CRITICAL:
        status = "critical"
    elif pct > DECAY_WARNING:
        status = "warning"
    else:
        status = "healthy"
    return status, round(pct * 100, 2)


# ============================================================
# Core monitoring check
# ============================================================

def run_monitoring_check() -> dict:
    """
    For each horizon:
      1. Fetch the most recent finished training run → current_mae.
      2. If a second run exists, use its MAE as the reference (run-over-run
         decay).  Otherwise fall back to the naive baseline from
         _BASELINE_MAE.
      3. Classify the horizon status and accumulate the overall status.

    Returns the full monitoring report dict.
    """
    logger.info("Starting EcoPulse performance monitoring check…")

    # Ensure the tracking URI is set before any client calls.
    setup_mlflow(TRAINING_EXPERIMENT_NAME)

    horizons_report = {}
    overall_status  = "healthy"

    for horizon in HORIZONS:
        runs = get_recent_runs(horizon, n=2)

        if not runs:
            logger.warning("No finished training runs found for horizon %sh.", horizon)
            horizons_report[f"{horizon}h"] = {
                "status":  "missing",
                "message": "No finished training run found for this horizon.",
            }
            continue

        latest_run = runs[0]
        current_mae = latest_run.data.metrics.get("test_mae")
        if current_mae is None:
            logger.warning(
                "Horizon %sh run %s has no test_mae metric — skipping.",
                horizon, latest_run.info.run_id[:8],
            )
            horizons_report[f"{horizon}h"] = {
                "status":  "missing",
                "message": "Latest run has no test_mae metric.",
            }
            continue

        # Choose reference: previous run MAE, else naive baseline
        if len(runs) > 1:
            prev_mae         = runs[1].data.metrics.get("test_mae")
            comparison_label = "vs_previous_run"
            comparison_mae   = round(prev_mae, 4) if prev_mae else None
        else:
            comparison_label = "vs_naive_baseline"
            comparison_mae   = _BASELINE_MAE.get(horizon)

        if comparison_mae is None:
            logger.warning(
                "Horizon %sh: no reference MAE available — marking as healthy.",
                horizon,
            )
            status, pct_change = "healthy", 0.0
        else:
            status, pct_change = classify_decay(current_mae, comparison_mae)

        horizons_report[f"{horizon}h"] = {
            "status":            status,
            "current_mae":       round(current_mae, 4),
            "comparison_label":  comparison_label,
            "comparison_mae":    comparison_mae,
            "pct_change":        pct_change,
            "model_type":        latest_run.data.tags.get("model_type", "unknown"),
            "reference_run_id":  latest_run.info.run_id,
            "naive_baseline_mae": _BASELINE_MAE.get(horizon),
        }

        # Roll up overall status (critical > warning > healthy)
        if status == "critical":
            overall_status = "critical"
        elif status == "warning" and overall_status == "healthy":
            overall_status = "warning"

        logger.info(
            "Horizon %sh → %s  current=%.4f  %s=%.4f  Δ%+.1f%%",
            horizon, status.upper(), current_mae,
            comparison_label, comparison_mae or 0.0, pct_change,
        )

    return {
        "status":     overall_status,
        "checked_at": datetime.utcnow().isoformat(),
        "thresholds": {
            "warning_pct":  DECAY_WARNING  * 100,
            "critical_pct": DECAY_CRITICAL * 100,
        },
        "horizons": horizons_report,
    }


# ============================================================
# Report persistence
# ============================================================

def save_report(report: dict) -> str:
    """Write the monitoring report to JSON.  Returns the file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Monitoring report saved → %s", REPORT_PATH)
    return REPORT_PATH


# ============================================================
# MLflow logging
# ============================================================

def log_monitoring_run(report: dict) -> Optional[str]:
    """
    Log the monitoring report as a new run under 'ecopulse-monitoring'.
    The JSON report must already exist on disk (call save_report first).
    Returns the run_id, or None on failure.
    """
    try:
        setup_mlflow(MONITORING_EXPERIMENT)
        run_name = f"monitor_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "project":        "ecopulse",
                "run_type":       "monitoring",
                "git_commit":     _git_commit(),
                "overall_status": report["status"],
            })
            mlflow.log_params({
                "warning_threshold_pct":  report["thresholds"]["warning_pct"],
                "critical_threshold_pct": report["thresholds"]["critical_pct"],
            })

            for horizon_key, h in report["horizons"].items():
                if "current_mae" not in h:
                    continue
                mlflow.log_metric(f"{horizon_key}.current_mae", h["current_mae"])
                mlflow.log_metric(f"{horizon_key}.pct_change",  h["pct_change"])
                mlflow.log_param( f"{horizon_key}.status",      h["status"])

            if os.path.exists(REPORT_PATH):
                mlflow.log_artifact(REPORT_PATH, artifact_path="monitoring")

            run_id = mlflow.active_run().info.run_id
            logger.info("Monitoring run logged to MLflow (run_id=%s).", run_id)
            return run_id

    except Exception as exc:
        logger.error("Failed to log monitoring run to MLflow: %s", exc)
        return None


# ============================================================
# CLI
# ============================================================

def main():
    print("=" * 60)
    print("EcoPulse — Model Performance Monitor")
    print("=" * 60)

    report = run_monitoring_check()

    # Save first — log_monitoring_run uploads the file as an artifact.
    save_report(report)
    log_monitoring_run(report)

    # ── Summary table ──────────────────────────────────────────
    w  = report["thresholds"]["warning_pct"]
    cr = report["thresholds"]["critical_pct"]
    print(f"\n  Overall status : {report['status'].upper()}")
    print(f"  Checked at     : {report['checked_at']}")
    print(f"  Thresholds     : WARNING >{w:.0f}%   CRITICAL >{cr:.0f}%")
    print()
    print(f"  {'Horizon':<8} {'Status':<10} {'Current MAE':>12} "
          f"{'Reference MAE':>14} {'Δ %':>8}  Comparison")
    print(f"  {'-'*70}")
    for key, h in report["horizons"].items():
        if "current_mae" not in h:
            print(f"  {key:<8} {'MISSING':<10}")
            continue
        icon = ("✗ " if h["status"] == "critical"
                else ("⚠ " if h["status"] == "warning" else "✓ "))
        print(
            f"  {icon}{key:<6} {h['status'].upper():<10} "
            f"{h['current_mae']:>12.4f} {h['comparison_mae']:>14.4f} "
            f"{h['pct_change']:>+8.1f}%  ({h['comparison_label']})"
        )

    print(f"\n  Report → {REPORT_PATH}\n")

    if report["status"] == "critical":
        sys.exit(1)


if __name__ == "__main__":
    main()
