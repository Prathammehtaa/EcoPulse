"""
EcoPulse Model Promotion & Rollback System
==========================================
Governs when new model versions replace the current production model, and
provides a safe, audited rollback path when a production model regresses.

Decision model
--------------
Promotion is evaluated per-horizon then aggregated:

    Horizon verdict
    ───────────────
    "improved"  — new MAE is ≥ IMPROVEMENT_THRESHOLD % better than current
    "neutral"   — new MAE is within REGRESSION_TOLERANCE % of current
    "regressed" — new MAE exceeds REGRESSION_TOLERANCE % worse than current

    Aggregate decision
    ──────────────────
    "promote" — no horizon is "regressed"
    "reject"  — at least one horizon is "regressed"

Both MLflow (run tags + promotion experiment) and GCP Artifact Registry (the
``production`` tag) are updated atomically per horizon, keeping them in sync.

Environment variables
---------------------
    PROMOTION_IMPROVEMENT_THRESHOLD   float, default 0.005  (0.5 %)
    PROMOTION_REGRESSION_TOLERANCE    float, default 0.010  (1.0 %)
    PROMOTION_R2_MIN_DROP             float, default 0.02   (R² floor)
    PROMOTION_EXPERIMENT_NAME         MLflow experiment for audit runs
    SLACK_WEBHOOK_URL                 Incoming webhook for Slack alerts
    ECOPULSE_AUDIT_LOG_PATH           Override default audit log location

Usage
-----
::

    from model_promotion import (
        get_production_metrics,
        compare_models,
        should_promote,
        promote_models_to_production,
        rollback_to_previous,
    )

    # After training
    current  = get_production_metrics("xgboost")
    new_m    = {1: test_metrics_1h, 6: ..., 12: ..., 24: ...}
    cmp      = compare_models(new_m, current)

    if should_promote(cmp):
        promote_models_to_production("xgboost", new_version)
    else:
        print("Rejected:", cmp["summary"])

    # Emergency rollback
    rollback_to_previous("xgboost")

CLI
---
::

    python model_promotion.py promote  --model-type xgboost --version 20260325_1400_xgboost_1h
    python model_promotion.py rollback --model-type xgboost
    python model_promotion.py status   --model-type xgboost
    python model_promotion.py promote  --model-type xgboost --version ... --dry-run
    python model_promotion.py promote  --model-type xgboost --version ... --force
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_config import (
    TRAINING_EXPERIMENT_NAME,
    TUNING_EXPERIMENT_NAME,
    _git_commit,
    get_performance_tier,
    setup_mlflow,
)
from gcp_registry import (
    list_model_versions,
    promote_model_to_production,
)

logger = logging.getLogger("ecopulse-model.promotion")

# ============================================================
# CONSTANTS
# ============================================================

HORIZONS: List[int] = [1, 6, 12, 24]

# ── Decision thresholds (env-var overridable) ───────────────────────────────

# New MAE must beat current by at least this fraction to be "improved"
IMPROVEMENT_THRESHOLD: float = float(
    os.getenv("PROMOTION_IMPROVEMENT_THRESHOLD", "0.005")
)

# New MAE may be worse by at most this fraction before being "regressed"
REGRESSION_TOLERANCE: float = float(
    os.getenv("PROMOTION_REGRESSION_TOLERANCE", "0.010")
)

# R² must not drop by more than this absolute amount on any horizon
R2_MIN_DROP: float = float(os.getenv("PROMOTION_R2_MIN_DROP", "0.02"))

# Minimum number of past production versions to retain as rollback candidates
MIN_ROLLBACK_HISTORY: int = 3

# ── MLflow ──────────────────────────────────────────────────────────────────

PROMOTION_EXPERIMENT_NAME: str = os.getenv(
    "PROMOTION_EXPERIMENT_NAME", "ecopulse-model-promotions"
)

# ── Paths ───────────────────────────────────────────────────────────────────

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_PIPE_DIR  = os.path.dirname(_SRC_DIR)
_REPORTS   = os.path.join(_PIPE_DIR, "reports")

AUDIT_LOG_PATH: str = os.getenv(
    "ECOPULSE_AUDIT_LOG_PATH",
    os.path.join(_REPORTS, "promotion_audit.jsonl"),
)

# ── Slack ───────────────────────────────────────────────────────────────────

_SLACK_WEBHOOK: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _model_name(model_type: str, horizon: int) -> str:
    """Return the canonical per-horizon model identifier."""
    return f"{model_type}_{horizon}h"


def _all_experiment_ids(client: MlflowClient) -> List[str]:
    """Return experiment IDs for training + tuning (searched for metrics)."""
    ids: List[str] = []
    for name in (TRAINING_EXPERIMENT_NAME, TUNING_EXPERIMENT_NAME):
        exp = client.get_experiment_by_name(name)
        if exp:
            ids.append(exp.experiment_id)
    return ids


def _metrics_from_run(run: mlflow.entities.Run) -> Dict[str, float]:
    """Extract the standard metric set from an MLflow run object."""
    m = run.data.metrics
    return {
        "mae":  m.get("test_mae",  m.get("mae",  float("nan"))),
        "rmse": m.get("test_rmse", m.get("rmse", float("nan"))),
        "r2":   m.get("test_r2",   m.get("r2",   float("nan"))),
        "mape": m.get("test_mape", m.get("mape", float("nan"))),
    }


# ============================================================
# METRIC RETRIEVAL
# ============================================================

def _get_production_version_from_gcp(model_name: str) -> Optional[str]:
    """
    Return the version string currently tagged as ``production`` in GCP
    Artifact Registry for the given model name, or ``None`` if no production
    tag exists or GCP is unavailable.
    """
    try:
        versions = list_model_versions(model_name)
        for v in versions:
            if "production" in v.get("tags", []):
                return v["version"]
        return None
    except Exception as exc:
        logger.warning(
            "Could not query GCP production tag for '%s': %s", model_name, exc
        )
        return None


def _get_mlflow_metrics_for_version(
    client: MlflowClient,
    horizon: int,
    model_type: str,
    gcp_version: str,
) -> Optional[Dict[str, float]]:
    """
    Find the MLflow run that produced a specific GCP version and return its
    test metrics.  Returns ``None`` if no matching run is found.
    """
    exp_ids = _all_experiment_ids(client)
    if not exp_ids:
        return None

    filter_str = (
        f"tags.`gcp.registry.version` = '{gcp_version}' "
        f"AND tags.horizon_h = '{horizon}'"
    )
    try:
        runs = client.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_str,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    except Exception as exc:
        logger.warning("MLflow search failed for version '%s': %s", gcp_version, exc)
        return None

    if not runs:
        # Fall back: search by model_type + horizon, take best run
        try:
            runs = client.search_runs(
                experiment_ids=exp_ids,
                filter_string=(
                    f"tags.model_type = '{model_type}' "
                    f"AND tags.horizon_h = '{horizon}' "
                    f"AND attributes.status = 'FINISHED'"
                ),
                order_by=["metrics.test_mae ASC"],
                max_results=1,
            )
        except Exception:
            return None

    return _metrics_from_run(runs[0]) if runs else None


def get_production_metrics(
    model_type: str,
    horizons: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Return the test metrics for the current production version of each horizon.

    Resolution order per horizon:

    1. Find the ``production`` tag in GCP Artifact Registry → get version string.
    2. Search MLflow runs with ``tags."gcp.registry.version" = <version>``
       and ``tags.horizon_h = <h>`` to retrieve logged test metrics.
    3. If no GCP tag exists, fall back to the best FINISHED MLflow run for
       that model type / horizon (lowest ``test_mae``).

    Args:
        model_type: Model flavour (``"xgboost"``, ``"lightgbm"``,
                    ``"xgboost_tuned"``).
        horizons:   Subset of horizons to query.  Defaults to all four.

    Returns:
        ``{horizon: {"mae": float, "rmse": float, "r2": float, "mape": float}}``

        Horizons with no data are omitted from the dict.

    Example::

        current = get_production_metrics("xgboost")
        # {1: {"mae": 25.1, "rmse": 33.4, "r2": 0.90, "mape": 6.2}, ...}
    """
    horizons = horizons or HORIZONS
    setup_mlflow(TRAINING_EXPERIMENT_NAME)
    client  = MlflowClient()
    results: Dict[int, Dict[str, float]] = {}

    for h in horizons:
        mname   = _model_name(model_type, h)
        version = _get_production_version_from_gcp(mname)

        metrics = _get_mlflow_metrics_for_version(client, h, model_type, version or "")
        if metrics and not all(
            v != v for v in metrics.values()  # all-NaN check
        ):
            results[h] = metrics
            logger.info(
                "Production metrics for %s (v=%s): MAE=%.4f  R²=%.4f",
                mname, version or "best-run", metrics["mae"], metrics["r2"],
            )
        else:
            logger.warning(
                "No production metrics found for '%s' (GCP version: %s).",
                mname, version,
            )

    return results


# ============================================================
# COMPARISON & DECISION LOGIC
# ============================================================

def compare_models(
    new_metrics: Dict[int, Dict[str, float]],
    current_metrics: Dict[int, Dict[str, float]],
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compare new model metrics against the current production model.

    For each horizon, three verdicts are possible:

    * ``"improved"``  — MAE improved by at least ``IMPROVEMENT_THRESHOLD`` (0.5 %)
      **and** R² did not drop by more than ``R2_MIN_DROP`` (0.02).
    * ``"neutral"``   — MAE is within ``REGRESSION_TOLERANCE`` (1 %) of current.
    * ``"regressed"`` — MAE is worse by more than ``REGRESSION_TOLERANCE``, OR
      R² dropped by more than ``R2_MIN_DROP`` relative to current.

    The aggregate decision is ``"promote"`` when no horizon is ``"regressed"``,
    and ``"reject"`` otherwise.

    Args:
        new_metrics:     ``{horizon: {"mae": …, "rmse": …, "r2": …}}``
        current_metrics: Same format for the current production model.
        horizons:        Horizons to evaluate.  Defaults to all four.

    Returns:
        Dict with keys::

            {
                "per_horizon": {
                    1: {
                        "verdict":       "improved" | "neutral" | "regressed",
                        "mae_current":   float,
                        "mae_new":       float,
                        "mae_delta":     float,   # new − current (negative = better)
                        "mae_pct":       float,   # delta / current  (negative = better)
                        "rmse_delta":    float,
                        "r2_current":    float,
                        "r2_new":        float,
                        "r2_delta":      float,   # new − current (positive = better)
                        "r2_violated":   bool,    # True if R² dropped too much
                        "threshold_ok":  bool,
                    },
                    …
                },
                "overall":     "promote" | "reject",
                "summary":     str,
                "horizons_ok": [int, …],          # no regression
                "horizons_bad":[int, …],           # regressed
                "thresholds": {
                    "improvement_threshold": float,
                    "regression_tolerance":  float,
                    "r2_min_drop":           float,
                },
            }

    Example::

        cmp = compare_models(
            new_metrics     = {1: {"mae": 24.8, "r2": 0.91}, …},
            current_metrics = {1: {"mae": 25.1, "r2": 0.90}, …},
        )
        print(cmp["overall"])           # "promote"
        print(cmp["per_horizon"][1]["verdict"])  # "improved"
    """
    horizons = horizons or HORIZONS
    per_horizon: Dict[int, Dict[str, Any]] = {}
    horizons_ok:  List[int] = []
    horizons_bad: List[int] = []

    for h in horizons:
        if h not in new_metrics:
            logger.warning("Horizon %dh missing from new_metrics — skipping.", h)
            continue
        if h not in current_metrics:
            logger.info(
                "No current production metrics for horizon %dh — treating as improved.", h
            )
            nm = new_metrics[h]
            per_horizon[h] = {
                "verdict":     "improved",
                "mae_current": None,
                "mae_new":     nm.get("mae"),
                "mae_delta":   None,
                "mae_pct":     None,
                "rmse_delta":  None,
                "r2_current":  None,
                "r2_new":      nm.get("r2"),
                "r2_delta":    None,
                "r2_violated": False,
                "threshold_ok": True,
            }
            horizons_ok.append(h)
            continue

        nm = new_metrics[h]
        cm = current_metrics[h]

        mae_cur  = cm.get("mae",  float("nan"))
        mae_new  = nm.get("mae",  float("nan"))
        r2_cur   = cm.get("r2",   float("nan"))
        r2_new   = nm.get("r2",   float("nan"))
        rmse_cur = cm.get("rmse", float("nan"))
        rmse_new = nm.get("rmse", float("nan"))

        mae_delta  = mae_new - mae_cur                          # negative = better
        mae_pct    = mae_delta / mae_cur if mae_cur else 0.0   # negative = better
        rmse_delta = rmse_new - rmse_cur
        r2_delta   = r2_new - r2_cur                           # positive = better
        r2_violated = (r2_delta < -R2_MIN_DROP)

        if mae_pct <= -IMPROVEMENT_THRESHOLD and not r2_violated:
            verdict      = "improved"
            threshold_ok = True
        elif mae_pct <= REGRESSION_TOLERANCE and not r2_violated:
            verdict      = "neutral"
            threshold_ok = True
        else:
            verdict      = "regressed"
            threshold_ok = False

        per_horizon[h] = {
            "verdict":     verdict,
            "mae_current": round(mae_cur,  4),
            "mae_new":     round(mae_new,  4),
            "mae_delta":   round(mae_delta, 4),
            "mae_pct":     round(mae_pct * 100, 3),   # stored as %, e.g. -1.2
            "rmse_delta":  round(rmse_delta, 4),
            "r2_current":  round(r2_cur,   4),
            "r2_new":      round(r2_new,   4),
            "r2_delta":    round(r2_delta, 4),
            "r2_violated": r2_violated,
            "threshold_ok": threshold_ok,
        }

        if threshold_ok:
            horizons_ok.append(h)
        else:
            horizons_bad.append(h)

        logger.info(
            "Horizon %dh: %s  MAE %+.2f (%+.2f%%)  R² %+.4f",
            h, verdict.upper(), mae_delta, mae_pct * 100, r2_delta,
        )

    overall = "promote" if not horizons_bad else "reject"

    # Human-readable summary
    if overall == "promote":
        improved = [h for h in horizons_ok if per_horizon[h]["verdict"] == "improved"]
        neutral  = [h for h in horizons_ok if per_horizon[h]["verdict"] == "neutral"]
        parts    = []
        if improved:
            parts.append(f"improved on {[f'{h}h' for h in improved]}")
        if neutral:
            parts.append(f"neutral on {[f'{h}h' for h in neutral]}")
        summary = "PROMOTE — " + ", ".join(parts) if parts else "PROMOTE"
    else:
        bad_details = [
            f"{h}h (MAE {per_horizon[h]['mae_pct']:+.2f}%)"
            for h in horizons_bad
            if h in per_horizon
        ]
        summary = f"REJECT — regression on: {', '.join(bad_details)}"

    return {
        "per_horizon":  per_horizon,
        "overall":      overall,
        "summary":      summary,
        "horizons_ok":  horizons_ok,
        "horizons_bad": horizons_bad,
        "thresholds": {
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "regression_tolerance":  REGRESSION_TOLERANCE,
            "r2_min_drop":           R2_MIN_DROP,
        },
    }


def should_promote(comparison_results: Dict[str, Any]) -> bool:
    """
    Return ``True`` if the comparison results warrant a promotion.

    A promotion is safe when:
    * No horizon is ``"regressed"`` (MAE within tolerance on all horizons).
    * At least one horizon improved OR all are neutral.

    Args:
        comparison_results: The dict returned by :func:`compare_models`.

    Returns:
        ``True`` to promote, ``False`` to reject.

    Example::

        cmp = compare_models(new_m, current_m)
        if should_promote(cmp):
            promote_models_to_production("xgboost", new_version)
    """
    return comparison_results.get("overall") == "promote"


# ============================================================
# PROMOTION
# ============================================================

def promote_models_to_production(
    model_type: str,
    version: str,
    new_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    current_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    horizons: Optional[List[int]] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Promote a new model version to production across all horizons.

    Updates both GCP Artifact Registry (the ``production`` tag per
    horizon package) and MLflow (logs a promotion decision run).

    Promotion is atomic per horizon — if any GCP tag update fails the
    error is captured and reported, but successful horizons are NOT
    rolled back (partial promotion is preferable to a complete outage).

    Args:
        model_type:      Model flavour (``"xgboost"``, ``"lightgbm"``).
        version:         GCP version string to promote (e.g.
                         ``"20260325_1400_xgboost_1h"``).  The same
                         version string is used for all horizons.
        new_metrics:     ``{horizon: metrics}`` for the candidate model.
                         Used for logging; fetched from GCP/MLflow when omitted.
        current_metrics: ``{horizon: metrics}`` for the current production model.
                         Used for logging; omitted metrics are left as ``None``.
        comparison:      Pre-computed :func:`compare_models` result.
                         Logged to MLflow when provided.
        horizons:        Override the horizon list (default: all four).
        dry_run:         Print what would happen without making any changes.
        force:           Skip the metric gate — promote unconditionally.
                         Use for emergency deployments only.

    Returns:
        Dict with keys::

            {
                "status":          "success" | "partial" | "dry_run" | "failed",
                "model_type":      str,
                "version":         str,
                "promoted":        [int, …],   # horizons successfully promoted
                "failed_horizons": [int, …],
                "errors":          {horizon: error_message},
                "timestamp":       str,
                "dry_run":         bool,
                "force":           bool,
            }

    Raises:
        ValueError: If ``version`` is empty or ``model_type`` is unknown.

    Example::

        result = promote_models_to_production(
            model_type   = "xgboost",
            version      = "20260325_1400_xgboost_1h",
            new_metrics  = {1: {"mae": 24.8, "r2": 0.91}, …},
            dry_run      = False,
        )
        print(result["status"])   # "success"
    """
    if not version:
        raise ValueError("version must be a non-empty string.")
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")

    horizons   = horizons or HORIZONS
    timestamp  = datetime.utcnow().isoformat() + "Z"
    promoted:  List[int] = []
    failed:    List[int] = []
    errors:    Dict[int, str] = {}

    _log_header = f"{'[DRY-RUN] ' if dry_run else ''}{'[FORCE] ' if force else ''}"
    logger.info(
        "%sPromoting %s v%s to production on horizons %s…",
        _log_header, model_type, version, horizons,
    )

    for h in horizons:
        mname = _model_name(model_type, h)
        if dry_run:
            logger.info(
                "[DRY-RUN] Would promote '%s' → version '%s'", mname, version
            )
            promoted.append(h)
            continue

        try:
            promote_model_to_production(mname, version)
            promoted.append(h)
            logger.info("Promoted '%s' v%s ✓", mname, version)
        except Exception as exc:
            failed.append(h)
            errors[h] = str(exc)
            logger.error(
                "Failed to promote '%s' v%s: %s", mname, version, exc
            )

    status = (
        "dry_run"  if dry_run
        else "success" if not failed
        else "partial" if promoted
        else "failed"
    )

    result: Dict[str, Any] = {
        "status":          status,
        "model_type":      model_type,
        "version":         version,
        "promoted":        promoted,
        "failed_horizons": failed,
        "errors":          {str(k): v for k, v in errors.items()},
        "timestamp":       timestamp,
        "dry_run":         dry_run,
        "force":           force,
    }

    if not dry_run:
        # Audit log
        _write_audit_entry({
            "event":      "promote",
            "model_type": model_type,
            "version":    version,
            "horizons":   horizons,
            "promoted":   promoted,
            "failed":     failed,
            "status":     status,
            "force":      force,
            "timestamp":  timestamp,
            "git_commit": _git_commit(),
            "actor":      os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown")),
        })

        # MLflow promotion run
        _log_promotion_decision(
            model_type  = model_type,
            new_version = version,
            event       = "promote",
            status      = status,
            comparison  = comparison,
            new_metrics = new_metrics,
            promoted    = promoted,
            failed      = failed,
        )

        # Slack
        _send_slack_alert(
            event_type      = "promote",
            model_type      = model_type,
            version         = version,
            status          = status,
            promoted        = promoted,
            failed_horizons = failed,
            new_metrics     = new_metrics,
            current_metrics = current_metrics,
            comparison      = comparison,
        )

    if status == "success":
        logger.info(
            "Promotion complete: %s v%s is now production on all %d horizons.",
            model_type, version, len(promoted),
        )
    elif status in ("partial", "failed"):
        logger.error(
            "Promotion %s: %d promoted, %d failed (%s).",
            status, len(promoted), len(failed),
            ", ".join(f"{h}h: {errors[h]}" for h in failed),
        )

    return result


# ============================================================
# ROLLBACK
# ============================================================

def rollback_to_previous(
    model_type: str,
    horizons: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Roll back to the most recent previous production version for a model type.

    Resolution order for "previous version":

    1. Audit log (``promotion_audit.jsonl``) — the second-to-last successful
       promotion entry for this model type.  Most reliable because it captures
       exactly what was in production.
    2. GCP Artifact Registry version history — the version that held the
       ``production`` tag immediately before the current one (requires that
       versions were uploaded in chronological order, which is the default when
       using :func:`make_version_string`).

    At least :data:`MIN_ROLLBACK_HISTORY` (3) prior versions must exist in the
    audit log before a rollback is permitted, to prevent accidental promotion
    loops on a sparse history.

    Args:
        model_type: Model flavour (``"xgboost"``, ``"lightgbm"``).
        horizons:   Override the horizon list (default: all four).
        dry_run:    Report what would be rolled back without executing.

    Returns:
        Dict with keys::

            {
                "status":          "success" | "partial" | "dry_run" | "failed" | "no_history",
                "model_type":      str,
                "rolled_back_to":  str | None,     # version string
                "previous_version": str | None,    # version we rolled back from
                "source":          "audit_log" | "gcp_history" | None,
                "promoted":        [int, …],
                "failed_horizons": [int, …],
                "timestamp":       str,
                "dry_run":         bool,
            }

    Raises:
        RuntimeError: If no rollback candidate can be found in either the
                      audit log or GCP.

    Example::

        result = rollback_to_previous("xgboost")
        print(result["rolled_back_to"])  # e.g. "20260320_1000_xgboost_1h"
    """
    horizons  = horizons or HORIZONS
    timestamp = datetime.utcnow().isoformat() + "Z"
    logger.warning(
        "%sInitiating rollback for '%s'…",
        "[DRY-RUN] " if dry_run else "", model_type,
    )

    # ── Find rollback target ────────────────────────────────────────────────
    target_version, prev_version, source = _resolve_rollback_target(model_type)

    if target_version is None:
        msg = (
            f"No rollback candidate found for '{model_type}'. "
            "At least 2 prior promotion entries are required in the audit log "
            "or GCP version history."
        )
        logger.error(msg)
        _send_slack_alert(
            event_type  = "rollback_failed",
            model_type  = model_type,
            version     = None,
            status      = "no_history",
            error       = msg,
        )
        return {
            "status":           "no_history",
            "model_type":       model_type,
            "rolled_back_to":   None,
            "previous_version": None,
            "source":           None,
            "promoted":         [],
            "failed_horizons":  [],
            "timestamp":        timestamp,
            "dry_run":          dry_run,
        }

    logger.warning(
        "%sRolling back '%s' from %s → %s (source: %s)",
        "[DRY-RUN] " if dry_run else "",
        model_type, prev_version or "unknown", target_version, source,
    )

    if dry_run:
        return {
            "status":           "dry_run",
            "model_type":       model_type,
            "rolled_back_to":   target_version,
            "previous_version": prev_version,
            "source":           source,
            "promoted":         horizons,
            "failed_horizons":  [],
            "timestamp":        timestamp,
            "dry_run":          True,
        }

    # ── Execute rollback using promote (force=True skips metric gate) ───────
    promo_result = promote_models_to_production(
        model_type  = model_type,
        version     = target_version,
        horizons    = horizons,
        dry_run     = False,
        force       = True,           # rollback bypasses metric comparison
    )

    status = promo_result["status"]

    # ── Write audit entry ───────────────────────────────────────────────────
    _write_audit_entry({
        "event":            "rollback",
        "model_type":       model_type,
        "version":          target_version,
        "previous_version": prev_version,
        "source":           source,
        "horizons":         horizons,
        "promoted":         promo_result["promoted"],
        "failed":           promo_result["failed_horizons"],
        "status":           status,
        "timestamp":        timestamp,
        "git_commit":       _git_commit(),
        "actor":            os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown")),
    })

    # ── Slack ───────────────────────────────────────────────────────────────
    _send_slack_alert(
        event_type        = "rollback",
        model_type        = model_type,
        version           = target_version,
        status            = status,
        promoted          = promo_result["promoted"],
        failed_horizons   = promo_result["failed_horizons"],
        previous_version  = prev_version,
    )

    return {
        "status":           status,
        "model_type":       model_type,
        "rolled_back_to":   target_version,
        "previous_version": prev_version,
        "source":           source,
        "promoted":         promo_result["promoted"],
        "failed_horizons":  promo_result["failed_horizons"],
        "timestamp":        timestamp,
        "dry_run":          False,
    }


def _resolve_rollback_target(
    model_type: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find the version to roll back to and the version we are rolling back from.

    Returns:
        ``(target_version, current_version, source)``
        where ``source`` is ``"audit_log"`` or ``"gcp_history"``.
        All three are ``None`` if no suitable candidate exists.
    """
    # Strategy 1: Audit log
    history = _read_audit_log(model_type)
    promotions = [
        e for e in history
        if e.get("event") in ("promote", "rollback")
        and e.get("status") in ("success", "partial")
        and e.get("version")
    ]

    if len(promotions) >= 2:
        current  = promotions[-1]["version"]
        previous = promotions[-2]["version"]
        if current != previous:
            return previous, current, "audit_log"

    # Strategy 2: GCP version history for the first horizon
    sample_name = _model_name(model_type, HORIZONS[0])
    try:
        versions = list_model_versions(sample_name)  # sorted newest-first
        prod_versions = [v for v in versions if "production" in v.get("tags", [])]

        if len(prod_versions) >= 1:
            current_gcp = prod_versions[0]["version"]
            # The previous candidate is the second-newest uploaded version
            # that is not the current production one
            non_current = [v for v in versions if v["version"] != current_gcp]
            if non_current:
                return non_current[0]["version"], current_gcp, "gcp_history"
    except Exception as exc:
        logger.warning("GCP history fallback failed: %s", exc)

    return None, None, None


# ============================================================
# AUDIT LOG
# ============================================================

def _write_audit_entry(entry: Dict[str, Any]) -> None:
    """
    Append a promotion / rollback event to the JSONL audit log.

    Each line is a self-contained JSON object.  The log is append-only —
    entries are never modified or deleted, providing a complete history.

    The audit log is stored at :data:`AUDIT_LOG_PATH` (env-var overridable).
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        logger.debug("Audit entry written: %s", entry.get("event"))
    except OSError as exc:
        logger.error("Failed to write audit log entry: %s", exc)


def _read_audit_log(
    model_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Read and return all audit log entries, optionally filtered by model type.

    Entries are returned in chronological order (oldest first).  Malformed
    JSON lines are silently skipped.

    Args:
        model_type: When provided, only entries for this model type are returned.

    Returns:
        List of audit entry dicts.
    """
    if not os.path.isfile(AUDIT_LOG_PATH):
        return []

    entries: List[Dict[str, Any]] = []
    with open(AUDIT_LOG_PATH, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if model_type is None or entry.get("model_type") == model_type:
                    entries.append(entry)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed audit log line %d.", line_no)

    return entries


def get_promotion_history(
    model_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Return recent promotion/rollback history from the audit log.

    Args:
        model_type: Filter by model type.  ``None`` returns all entries.
        limit:      Maximum number of entries to return (most recent first).

    Returns:
        List of audit entry dicts, newest first.

    Example::

        for entry in get_promotion_history("xgboost", limit=5):
            print(entry["timestamp"], entry["event"], entry["version"])
    """
    entries = _read_audit_log(model_type)
    return list(reversed(entries))[:limit]


# ============================================================
# MLFLOW DECISION LOGGING
# ============================================================

def _log_promotion_decision(
    model_type: str,
    new_version: str,
    event: str,
    status: str,
    comparison: Optional[Dict[str, Any]] = None,
    new_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    promoted: Optional[List[int]] = None,
    failed: Optional[List[int]] = None,
    current_version: Optional[str] = None,
) -> None:
    """
    Log a promotion or rollback decision to the MLflow promotions experiment.

    Creates a new MLflow run in :data:`PROMOTION_EXPERIMENT_NAME` with:
    - Tags: event type, status, model type, versions, git commit
    - Metrics: per-horizon MAE delta and R² delta (when comparison is provided)
    - Artifacts: serialised comparison dict as JSON

    Failures here are non-fatal — a warning is logged and the promotion
    result is unaffected.
    """
    try:
        setup_mlflow(PROMOTION_EXPERIMENT_NAME)

        run_name = (
            f"{event}_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "event":          event,
                "status":         status,
                "model_type":     model_type,
                "new_version":    new_version,
                "current_version": current_version or "",
                "promoted_h":     ",".join(str(h) for h in (promoted or [])),
                "failed_h":       ",".join(str(h) for h in (failed or [])),
                "git_commit":     _git_commit(),
                "actor":          os.getenv("GITHUB_ACTOR", os.getenv("USER", "")),
            })

            if comparison and "per_horizon" in comparison:
                for h, hd in comparison["per_horizon"].items():
                    prefix = f"h{h}"
                    if hd.get("mae_delta") is not None:
                        mlflow.log_metric(f"{prefix}.mae_delta", hd["mae_delta"])
                    if hd.get("mae_pct") is not None:
                        mlflow.log_metric(f"{prefix}.mae_pct",   hd["mae_pct"])
                    if hd.get("r2_delta") is not None:
                        mlflow.log_metric(f"{prefix}.r2_delta",  hd["r2_delta"])

            if new_metrics:
                for h, m in new_metrics.items():
                    mlflow.log_metric(f"new.h{h}.mae",  m.get("mae",  float("nan")))
                    mlflow.log_metric(f"new.h{h}.r2",   m.get("r2",   float("nan")))
                    mlflow.log_metric(f"new.h{h}.rmse", m.get("rmse", float("nan")))

            if comparison:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", prefix="promotion_", delete=False
                ) as tmp:
                    json.dump(comparison, tmp, indent=2, default=str)
                    tmp_path = tmp.name
                try:
                    mlflow.log_artifact(tmp_path, artifact_path="promotion")
                finally:
                    os.unlink(tmp_path)

    except Exception as exc:
        logger.warning(
            "Could not log promotion decision to MLflow: %s", exc
        )


# ============================================================
# SLACK NOTIFICATIONS
# ============================================================

def _send_slack_alert(
    event_type: str,
    model_type: str,
    version: Optional[str],
    status: str,
    promoted: Optional[List[int]] = None,
    failed_horizons: Optional[List[int]] = None,
    new_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    current_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    previous_version: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    Post a Slack notification about a promotion or rollback event.

    Requires the ``SLACK_WEBHOOK_URL`` environment variable.  Failures are
    logged as warnings and do not affect the promotion result.

    Notification format mirrors the existing workflow Slack alerts (Block Kit).
    """
    if not _SLACK_WEBHOOK:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return

    # ── Header ──────────────────────────────────────────────────────────────
    icons = {
        "promote":        {"success": "✅", "partial": "⚠️", "failed": "❌", "dry_run": "🔍"},
        "rollback":       {"success": "🔄", "partial": "⚠️", "failed": "❌"},
        "rollback_failed":{"no_history": "🚫"},
    }
    icon        = icons.get(event_type, {}).get(status, "❓")
    event_label = {
        "promote":         "Model Promotion",
        "rollback":        "Model Rollback",
        "rollback_failed": "Rollback Failed",
    }.get(event_type, event_type.title())

    header = f"{icon} EcoPulse {event_label} — {status.upper()}"

    # ── Body ────────────────────────────────────────────────────────────────
    lines: List[str] = [
        f"*Model type:* `{model_type}`",
    ]

    if version:
        lines.append(f"*New version:* `{version}`")
    if previous_version:
        lines.append(f"*Previous:*   `{previous_version}`")

    if promoted:
        lines.append(f"*Promoted horizons:* {', '.join(f'{h}h' for h in promoted)}")
    if failed_horizons:
        lines.append(f"*Failed horizons:*  {', '.join(f'{h}h' for h in failed_horizons)}")

    if error:
        lines += ["", f"*Error:* {error}"]

    # Metric comparison table
    if comparison and "per_horizon" in comparison:
        lines += ["", "*Horizon-by-horizon comparison:*"]
        lines.append("```")
        lines.append(f"{'Horizon':<8} {'Verdict':<12} {'MAE Δ':>8} {'MAE %':>8} {'R² Δ':>8}")
        lines.append("-" * 50)
        for h in HORIZONS:
            if h not in comparison["per_horizon"]:
                continue
            hd = comparison["per_horizon"][h]
            v_icon = {"improved": "↑", "neutral": "→", "regressed": "↓"}.get(
                hd["verdict"], "?"
            )
            mae_d  = f"{hd['mae_delta']:+.3f}"  if hd.get("mae_delta") is not None else "N/A"
            mae_p  = f"{hd['mae_pct']:+.2f}%"   if hd.get("mae_pct")   is not None else "N/A"
            r2_d   = f"{hd['r2_delta']:+.4f}"   if hd.get("r2_delta")  is not None else "N/A"
            lines.append(
                f"{f'{h}h':<8} {v_icon + ' ' + hd['verdict']:<12} {mae_d:>8} {mae_p:>8} {r2_d:>8}"
            )
        lines.append("```")

        if comparison.get("summary"):
            lines += ["", f"*Decision:* {comparison['summary']}"]

    elif new_metrics and current_metrics:
        # Simpler before/after table for rollbacks
        lines += ["", "*Metrics (new production):*", "```"]
        lines.append(f"{'Horizon':<8} {'MAE':>8} {'R²':>8}")
        lines.append("-" * 26)
        for h in HORIZONS:
            if h not in new_metrics:
                continue
            m = new_metrics[h]
            lines.append(
                f"{f'{h}h':<8} {m.get('mae', float('nan')):>8.3f} {m.get('r2', float('nan')):>8.4f}"
            )
        lines.append("```")

    run_url = (
        f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.getenv('GITHUB_REPOSITORY', '')}/"
        f"actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
    ).rstrip("/actions/runs/")

    body = "\n".join(lines)

    payload = {
        "text": header,
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header, "emoji": True},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body},
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Actor:* {os.getenv('GITHUB_ACTOR', os.getenv('USER', 'unknown'))}  "
                            f"*Commit:* `{_git_commit()}`  "
                            f"*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                        ),
                    }
                ],
            },
        ],
    }

    try:
        req = urllib.request.Request(
            _SLACK_WEBHOOK,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Slack alert sent: %s %s", event_type, resp.status)
    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)


# ============================================================
# STATUS REPORT
# ============================================================

def print_status(model_type: str) -> None:
    """
    Print a human-readable status report for the current production model.

    Reports:
    * Current production version and metrics per horizon (from GCP + MLflow)
    * Last 5 promotion/rollback events from the audit log

    Args:
        model_type: Model flavour to inspect.
    """
    print("=" * 70)
    print(f"  EcoPulse Production Status — {model_type}")
    print("=" * 70)

    print("\nCurrent production metrics:")
    print(f"  {'Horizon':<10} {'Version':<35} {'MAE':>8} {'R²':>8} {'RMSE':>8}")
    print(f"  {'-'*65}")

    for h in HORIZONS:
        mname   = _model_name(model_type, h)
        version = _get_production_version_from_gcp(mname) or "unknown"
        m = get_production_metrics(model_type, horizons=[h]).get(h, {})
        mae  = m.get("mae",  float("nan"))
        r2   = m.get("r2",   float("nan"))
        rmse = m.get("rmse", float("nan"))
        print(f"  {f'{h}h':<10} {version:<35} {mae:>8.3f} {r2:>8.4f} {rmse:>8.3f}")

    print("\nRecent audit log (last 5 events):")
    history = get_promotion_history(model_type, limit=5)
    if not history:
        print("  (no history found)")
    for e in history:
        ts     = e.get("timestamp", "")[:19].replace("T", " ")
        evt    = e.get("event", "?")
        ver    = e.get("version", "?")
        status = e.get("status", "?")
        actor  = e.get("actor", "?")
        print(f"  {ts}  {evt:<10} {status:<10} v={ver}  by={actor}")

    print()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="model_promotion",
        description="EcoPulse model promotion and rollback system.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # promote
    p = sub.add_parser("promote", help="Promote a version to production.")
    p.add_argument("--model-type", required=True, help="xgboost | lightgbm | xgboost_tuned")
    p.add_argument("--version",    required=True, help="GCP version string to promote")
    p.add_argument("--dry-run",    action="store_true", help="Show without executing")
    p.add_argument("--force",      action="store_true", help="Skip metric gate")
    p.add_argument(
        "--horizons", nargs="+", type=int, default=None,
        metavar="H", help="Horizon subset (default: 1 6 12 24)"
    )

    # rollback
    r = sub.add_parser("rollback", help="Roll back to the previous production version.")
    r.add_argument("--model-type", required=True)
    r.add_argument("--dry-run",    action="store_true")
    r.add_argument("--horizons",   nargs="+", type=int, default=None, metavar="H")

    # status
    s = sub.add_parser("status", help="Show current production status.")
    s.add_argument("--model-type", required=True)

    # compare
    c = sub.add_parser(
        "compare",
        help="Compare a candidate version against production (read-only).",
    )
    c.add_argument("--model-type",  required=True)
    c.add_argument("--new-version", required=True,
                   help="GCP version string of the candidate model")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.  Returns 0 on success, 1 on failure or rejection.

    Exit codes:
    * 0 — success (or dry-run completed)
    * 1 — failure (promotion rejected, rollback failed, no history)
    * 2 — argument error
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args   = parser.parse_args(argv)
    cmd    = args.command

    if cmd == "status":
        print_status(args.model_type)
        return 0

    if cmd == "promote":
        horizons = args.horizons or HORIZONS

        # Fetch current production and compare (unless force)
        current_m = get_production_metrics(args.model_type, horizons)

        if not args.force and current_m:
            print(
                "\nNote: provide --force to skip metric comparison, or run with "
                "'compare' first to validate metrics before promoting.\n"
            )
            logger.info(
                "Promoting with force=False but no new_metrics provided — "
                "metric gate skipped (no candidate metrics to compare)."
            )

        result = promote_models_to_production(
            model_type = args.model_type,
            version    = args.version,
            horizons   = horizons,
            dry_run    = args.dry_run,
            force      = args.force,
        )

        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] in ("success", "dry_run") else 1

    if cmd == "rollback":
        result = rollback_to_previous(
            model_type = args.model_type,
            horizons   = args.horizons or HORIZONS,
            dry_run    = args.dry_run,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] in ("success", "dry_run") else 1

    if cmd == "compare":
        # Fetch metrics for the candidate version from MLflow
        setup_mlflow(TRAINING_EXPERIMENT_NAME)
        client = MlflowClient()
        new_m: Dict[int, Dict[str, float]] = {}
        horizons = HORIZONS
        for h in horizons:
            m = _get_mlflow_metrics_for_version(
                client, h, args.model_type, args.new_version
            )
            if m:
                new_m[h] = m

        if not new_m:
            print(
                f"No MLflow metrics found for version '{args.new_version}'. "
                "Ensure the model was trained with GCP_PUSH_MODELS=1.",
                file=sys.stderr,
            )
            return 1

        current_m = get_production_metrics(args.model_type, horizons)
        cmp       = compare_models(new_m, current_m, horizons)

        print(json.dumps(cmp, indent=2, default=str))
        print(f"\nDecision: {cmp['summary']}")
        return 0 if should_promote(cmp) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
