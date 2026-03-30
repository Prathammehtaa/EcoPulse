"""
EcoPulse MLflow Configuration
==============================
Centralised MLflow utilities: tracking setup, tagging helpers,
artifact logging, model registry, and comparison dashboard queries.

Every other script imports from here — no more inline setup duplication.

Environment variables:
    MLFLOW_TRACKING_URI         Override the default SQLite backend
    MLFLOW_EXPERIMENT_NAME      Override the main experiment name
    MLFLOW_TUNING_EXP_NAME      Override the tuning experiment name
    MLFLOW_REGISTER_MODELS      Set to "1" to register models to the registry
"""

import os
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# ============================================================
# PATHS — resolved once at import time
# ============================================================
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SRC_DIR)
_MLRUNS_DIR   = os.path.join(_PIPELINE_DIR, "mlruns")
_DEFAULT_DB   = os.path.join(_MLRUNS_DIR, "mlflow.db")

# ============================================================
# EXPERIMENT NAMES  (env-var overridable)
# ============================================================
TRAINING_EXPERIMENT_NAME   = os.getenv("MLFLOW_EXPERIMENT_NAME",    "ecopulse-carbon-forecasting")
TUNING_EXPERIMENT_NAME     = os.getenv("MLFLOW_TUNING_EXP_NAME",    "ecopulse-hyperparameter-tuning")
COMPARISON_EXPERIMENT_NAME = os.getenv("MLFLOW_COMPARISON_EXP_NAME","ecopulse-model-comparison")

# ============================================================
# MODEL REGISTRY
# ============================================================
REGISTRY_BASE_NAME = "EcoPulse-CarbonForecaster"

# Baseline MAE values (naive benchmark) — used for performance tier
_BASELINE_MAE = {1: 57.48, 6: 71.33, 12: 76.36, 24: 68.79}


# ============================================================
# SETUP
# ============================================================

def setup_mlflow(experiment_name: str = None) -> None:
    """
    Configure the MLflow tracking URI and activate an experiment.

    Reads MLFLOW_TRACKING_URI from the environment; falls back to a
    local SQLite database at Model_Pipeline/mlruns/mlflow.db.

    Args:
        experiment_name: Experiment to activate.  Defaults to
                         TRAINING_EXPERIMENT_NAME (env-var overridable).
    """
    os.makedirs(_MLRUNS_DIR, exist_ok=True)
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{_DEFAULT_DB}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name or TRAINING_EXPERIMENT_NAME)


# ============================================================
# TAGGING HELPERS
# ============================================================

def _git_commit() -> str:
    """Return the current short git SHA, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_SRC_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_run_tags(model_type: str, horizon: int, **extra) -> dict:
    """
    Build the standard tag dict for a training / tuning run.

    Standard tags:
        model_type    — e.g. "xgboost", "lightgbm", "xgboost_tuned"
        horizon_h     — forecast horizon in hours (string)
        project       — always "ecopulse"
        git_commit    — short SHA from the working tree
        run_timestamp — ISO-ish datetime string

    Extra keyword arguments are merged in (values are cast to str).

    Usage::
        mlflow.set_tags(build_run_tags("xgboost", 6))
    """
    tags = {
        "model_type":    model_type,
        "horizon_h":     str(horizon),
        "project":       "ecopulse",
        "git_commit":    _git_commit(),
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tags.update({k: str(v) for k, v in extra.items()})
    return tags


def get_performance_tier(test_mae: float, horizon: int) -> str:
    """
    Classify model quality relative to the naive baseline MAE.

    Tiers:
        "excellent" — ≥60 % improvement over baseline
        "good"      — ≥45 % improvement
        "fair"      — ≥25 % improvement
        "poor"      — below 25 % improvement
    """
    baseline = _BASELINE_MAE.get(horizon, float("inf"))
    if baseline == 0:
        return "unknown"
    pct = (baseline - test_mae) / baseline
    if   pct >= 0.60: return "excellent"
    elif pct >= 0.45: return "good"
    elif pct >= 0.25: return "fair"
    else:             return "poor"


# ============================================================
# DATASET INFO
# ============================================================

def log_dataset_info(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    datetime_col: str = "datetime",
    zone_col:     str = "zone",
) -> None:
    """
    Log dataset statistics as MLflow params inside the active run.

    Logs row counts, column count, date ranges (if a datetime column
    exists), and zone list (if a zone column exists).
    """
    mlflow.log_params({
        "dataset.train_rows":  len(train_df),
        "dataset.val_rows":    len(val_df),
        "dataset.test_rows":   len(test_df),
        "dataset.total_rows":  len(train_df) + len(val_df) + len(test_df),
        "dataset.n_columns":   len(train_df.columns),
    })

    if datetime_col in train_df.columns:
        mlflow.log_params({
            "dataset.train_start": str(pd.to_datetime(train_df[datetime_col]).min().date()),
            "dataset.train_end":   str(pd.to_datetime(train_df[datetime_col]).max().date()),
            "dataset.val_start":   str(pd.to_datetime(val_df[datetime_col]).min().date()),
            "dataset.val_end":     str(pd.to_datetime(val_df[datetime_col]).max().date()),
            "dataset.test_start":  str(pd.to_datetime(test_df[datetime_col]).min().date()),
            "dataset.test_end":    str(pd.to_datetime(test_df[datetime_col]).max().date()),
        })

    if zone_col in train_df.columns:
        zones = sorted(train_df[zone_col].dropna().unique())
        mlflow.log_params({
            "dataset.n_zones": len(zones),
            "dataset.zones":   ",".join(str(z) for z in zones),
        })


# ============================================================
# ARTIFACT LOGGING — PLOTS
# ============================================================

def log_residual_plot(
    y_true,
    y_pred,
    split:      str,
    horizon:    int,
    model_type: str,
    save_dir:   str,
) -> str:
    """
    Generate a 2-panel residual diagnostic plot and log it to MLflow.

    Panel 1  Predicted vs Actual scatter with the ideal line
    Panel 2  Residual histogram with zero-error and mean-error lines

    The PNG is written to ``save_dir`` so it persists locally, then
    logged as an MLflow artifact under ``plots/residuals/``.

    Returns the local file path.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_type.upper()} {horizon}h — Residual Diagnostics ({split})",
        fontsize=13, fontweight="bold",
    )

    # --- Predicted vs Actual ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="#2ecc71", rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual (gCO₂/kWh)", fontsize=11)
    ax.set_ylabel("Predicted (gCO₂/kWh)", fontsize=11)
    ax.set_title(f"Predicted vs Actual  (MAE = {mae:.2f})", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # --- Residual histogram ---
    ax = axes[1]
    ax.hist(residuals, bins=60, color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red",    linestyle="--", lw=1.5, label="Zero error")
    ax.axvline(residuals.mean(), color="orange", linestyle="-", lw=1.2,
               label=f"Mean = {residuals.mean():.1f}")
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Error Distribution  (σ = {residuals.std():.2f})", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_type}_residuals_{horizon}h_{split}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(path, artifact_path="plots/residuals")
    return path


def log_feature_importance_plot(
    importance_df: pd.DataFrame,
    horizon:    int,
    model_type: str,
    save_dir:   str,
    top_n:      int = 20,
) -> str:
    """
    Generate a horizontal bar chart of the top-N feature importances
    and log it to MLflow under ``plots/feature_importance/``.

    Returns the local file path.
    """
    top = importance_df.head(top_n).copy().sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, int(top_n * 0.38))))

    cmap   = plt.cm.YlOrRd                                      # warm colour ramp
    colors = cmap(np.linspace(0.3, 0.9, len(top)))
    bars   = ax.barh(top["feature"], top["importance"], color=colors, alpha=0.9)

    # Inline value labels
    max_val = top["importance"].max()
    for bar, val in zip(bars, top["importance"]):
        ax.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8,
        )

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(
        f"{model_type.upper()} {horizon}h — Top {top_n} Feature Importances",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_type}_feat_importance_{horizon}h.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(path, artifact_path="plots/feature_importance")
    return path


# ============================================================
# MODEL REGISTRY  (opt-in via MLFLOW_REGISTER_MODELS=1)
# ============================================================

def register_model(
    run_id:        str,
    artifact_path: str,
    horizon:       int,
    model_type:    str,
    stage:         str = "Staging",
) -> None:
    """
    Register a logged model version in the MLflow Model Registry.

    No-op unless the environment variable ``MLFLOW_REGISTER_MODELS``
    is set to "1".

    Args:
        run_id:        MLflow run ID that contains the artifact.
        artifact_path: Artifact sub-path used in log_model (e.g. "xgboost_1h").
        horizon:       Forecast horizon in hours.
        model_type:    "xgboost" | "lightgbm" | "xgboost_tuned".
        stage:         Registry stage: "Staging", "Production", or None to skip.
    """
    if os.getenv("MLFLOW_REGISTER_MODELS", "0") != "1":
        return

    client = MlflowClient()
    name   = f"{REGISTRY_BASE_NAME}-{model_type}-{horizon}h"
    uri    = f"runs:/{run_id}/{artifact_path}"

    # Create registered model if it does not already exist
    try:
        client.create_registered_model(
            name,
            description=f"EcoPulse carbon intensity forecaster — {model_type} {horizon}h",
            tags={"project": "ecopulse", "model_type": model_type},
        )
    except mlflow.exceptions.MlflowException:
        pass  # already registered

    mv = client.create_model_version(
        name=name,
        source=uri,
        run_id=run_id,
        tags={"horizon_h": str(horizon), "model_type": model_type},
        description=f"Trained {model_type} for {horizon}h carbon-intensity forecast",
    )

    if stage:
        # Poll until version is READY before transitioning
        import time
        for _ in range(15):
            mv = client.get_model_version(name, mv.version)
            if mv.status == "READY":
                break
            time.sleep(1)
        client.transition_model_version_stage(
            name=name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=False,
        )


# ============================================================
# COMPARISON DASHBOARD UTILITIES
# ============================================================

def get_best_run_per_horizon(
    experiment_name: str = None,
    model_type:      str = None,
    metric:          str = "test_mae",
) -> dict:
    """
    Query MLflow for the best run (lowest metric) per horizon.

    Args:
        experiment_name: Defaults to TRAINING_EXPERIMENT_NAME.
        model_type:      Filter by "model_type" tag (e.g. "xgboost").
                         Pass None to search across all model types.
        metric:          Metric to rank by (lower = better).

    Returns:
        dict mapping horizon (int) → mlflow.entities.Run
    """
    setup_mlflow(experiment_name)
    client   = MlflowClient()
    exp_name = experiment_name or TRAINING_EXPERIMENT_NAME
    exp      = client.get_experiment_by_name(exp_name)
    if exp is None:
        return {}

    best = {}
    for horizon in [1, 6, 12, 24]:
        filters = [f"tags.horizon_h = '{horizon}'"]
        if model_type:
            filters.append(f"tags.model_type = '{model_type}'")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=" AND ".join(filters),
            order_by=[f"metrics.{metric} ASC"],
            max_results=1,
        )
        if runs:
            best[horizon] = runs[0]

    return best


def build_mlflow_comparison_df(experiment_name: str = None) -> pd.DataFrame:
    """
    Build a comparison DataFrame from all FINISHED runs in an experiment.

    Returns columns::
        run_id, run_name, model_type, horizon_h, perf_tier,
        test_mae, test_rmse, test_r2, test_mape, val_mae,
        best_iteration, run_timestamp, git_commit

    Useful as a programmatic alternative to reading CSV files.
    """
    setup_mlflow(experiment_name)
    client   = MlflowClient()
    exp_name = experiment_name or TRAINING_EXPERIMENT_NAME
    exp      = client.get_experiment_by_name(exp_name)
    if exp is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=500,
    )

    rows = []
    for run in runs:
        m = run.data.metrics
        t = run.data.tags
        rows.append({
            "run_id":        run.info.run_id,
            "run_name":      run.info.run_name,
            "model_type":    t.get("model_type", "unknown"),
            "horizon_h":     t.get("horizon_h", ""),
            "perf_tier":     t.get("perf_tier", ""),
            "test_mae":      m.get("test_mae"),
            "test_rmse":     m.get("test_rmse"),
            "test_r2":       m.get("test_r2"),
            "test_mape":     m.get("test_mape"),
            "val_mae":       m.get("val_mae"),
            "best_iteration":m.get("best_iteration"),
            "run_timestamp": t.get("run_timestamp", ""),
            "git_commit":    t.get("git_commit", ""),
        })

    return pd.DataFrame(rows)


def log_comparison_artifacts(comparison_df: pd.DataFrame, reports_dir: str) -> None:
    """
    Open a dedicated MLflow run in COMPARISON_EXPERIMENT_NAME and log:
      - The full comparison DataFrame as a CSV artifact
      - Any chart PNGs that exist in reports_dir
      - Summary metrics (best MAE per model/horizon combination)

    Safe to call outside an existing run context — creates its own run.
    """
    setup_mlflow(COMPARISON_EXPERIMENT_NAME)

    chart_files = [
        "model_comparison_mae.png",
        "model_comparison_r2.png",
        "model_improvement.png",
        "full_comparison.csv",
    ]

    with mlflow.start_run(run_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tags({
            "project":    "ecopulse",
            "run_type":   "model_comparison",
            "git_commit": _git_commit(),
        })

        # Log comparison CSV
        csv_path = os.path.join(reports_dir, "full_comparison.csv")
        if os.path.exists(csv_path):
            mlflow.log_artifact(csv_path, artifact_path="comparison")

        # Log charts
        for fname in ["model_comparison_mae.png", "model_comparison_r2.png", "model_improvement.png"]:
            fpath = os.path.join(reports_dir, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath, artifact_path="comparison/charts")

        # Log scalar summary metrics
        if not comparison_df.empty:
            for _, row in comparison_df.iterrows():
                mt      = str(row.get("model_type", "model")).replace(" ", "_")
                horizon = row.get("horizon", row.get("horizon_h", "?"))
                prefix  = f"{mt}_{horizon}h"
                if pd.notna(row.get("mae")):
                    mlflow.log_metric(f"{prefix}.test_mae",  float(row["mae"]))
                if pd.notna(row.get("r2")):
                    mlflow.log_metric(f"{prefix}.test_r2",   float(row["r2"]))
                if pd.notna(row.get("rmse")):
                    mlflow.log_metric(f"{prefix}.test_rmse", float(row["rmse"]))
