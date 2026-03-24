"""
EcoPulse Bias Detection
=======================
Model Development Phase | Bias Detection & Mitigation
----------------------------------------------------------------
Evaluates trained XGBoost/LightGBM models for performance disparities
across data slices: zones, seasons, carbon intensity buckets, and
weather conditions.

Verified against actual parquet column names from test_split.parquet.

Depends on:
  - utils.py              : shared constants, loaders, metrics
  - train_xgboost.py      : produces models/xgboost_{horizon}h.joblib
  - train_lightgbm.py     : produces models/lightgbm_{horizon}h.joblib
  - test_split.parquet    : Data_Pipeline/data/processed/test_split.parquet

Outputs:
  - reports/bias/slice_metrics_{model}_{horizon}h_{ts}.csv
  - reports/bias/disparity_report_{model}_{horizon}h_{ts}.csv
  - MLflow metrics logged under experiment: ecopulse-carbon-forecasting

Usage:
  python bias_detection.py --model xgboost
  python bias_detection.py --model lightgbm
  python bias_detection.py --model both        (default)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_split, prepare_Xy, compute_metrics,
    ensure_dirs,
    TARGET_COL, ZONE_COL, FORECAST_TARGETS, HORIZONS,
    MODELS_DIR, REPORTS_DIR,
    logger
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SRC_DIR          = Path(__file__).resolve().parent
MODEL_PIPELINE   = SRC_DIR.parent
BIAS_REPORTS_DIR = MODEL_PIPELINE / "reports" / "bias"
BIAS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Verified column names from test_split.parquet
# ──────────────────────────────────────────────
# Weather columns — actual names in the parquet (NOT temperature_2m etc.)
WEATHER_COL_TEMP  = "temperature_2m_c"
WEATHER_COL_WIND  = "wind_speed_100m_ms"
WEATHER_COL_CLOUD = "cloud_cover_pct"
WEATHER_COLS      = {WEATHER_COL_TEMP, WEATHER_COL_WIND, WEATHER_COL_CLOUD}

# Season column — present in parquet as "season"
SEASON_COL = "season"

# Carbon bucket config — over TARGET_COL = "carbon_intensity_gco2_per_kwh"
CARBON_BUCKET_BINS   = [0, 100, 200, 350, 500, float("inf")]
CARBON_BUCKET_LABELS = ["Very Low (0-100)", "Low (100-200)", "Medium (200-350)",
                         "High (350-500)", "Very High (500+)"]

# Weather condition slices using verified column names
WEATHER_CONDITIONS = {
    "Cold (< 5C)":       lambda df: df[WEATHER_COL_TEMP] < 5,
    "Mild (5-20C)":      lambda df: (df[WEATHER_COL_TEMP] >= 5) & (df[WEATHER_COL_TEMP] < 20),
    "Hot (>= 20C)":      lambda df: df[WEATHER_COL_TEMP] >= 20,
    "Windy (>= 10 m/s)": lambda df: df[WEATHER_COL_WIND] >= 10,
    "Calm (< 10 m/s)":   lambda df: df[WEATHER_COL_WIND] < 10,
    "Overcast (>= 75%)": lambda df: df[WEATHER_COL_CLOUD] >= 75,
    "Clear (< 25%)":     lambda df: df[WEATHER_COL_CLOUD] < 25,
}

# Flag slices deviating more than 20% from overall baseline MAE
DISPARITY_THRESHOLD = 0.20


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model(model_name: str, horizon: int):
    """
    Load a trained model saved by train_xgboost.py or train_lightgbm.py.
    Naming convention: models/xgboost_{horizon}h.joblib
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{horizon}h.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_{model_name}.py first (Person 1)."
        )
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_path}")
    return model


# ──────────────────────────────────────────────
# Slice Helpers
# ──────────────────────────────────────────────

def add_carbon_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add carbon_bucket column based on carbon_intensity_gco2_per_kwh."""
    df = df.copy()
    df["carbon_bucket"] = pd.cut(
        df[TARGET_COL],
        bins=CARBON_BUCKET_BINS,
        labels=CARBON_BUCKET_LABELS,
        right=False,
    ).astype(str)
    return df


def get_reference_feature_cols(df: pd.DataFrame, horizon: int) -> list:
    """
    Get the full feature column list from the complete test set.
    All slices must align to this — same as what the model trained on.
    """
    X, _, feature_cols = prepare_Xy(df, horizon)
    return list(feature_cols)


# ──────────────────────────────────────────────
# Per-Slice Evaluation
# ──────────────────────────────────────────────

def evaluate_on_slice(
    df_slice: pd.DataFrame,
    model,
    horizon: int,
    model_name: str,
    slice_type: str,
    slice_value: str,
    reference_feature_cols: list,
) -> dict:
    """
    Prepare features for a single data slice and compute metrics.
    Fills any missing zone dummy columns with 0 to match training space.
    """
    if len(df_slice) < 10:
        logger.warning(f"  [SKIP] {slice_type}={slice_value} — only {len(df_slice)} samples")
        return None

    try:
        X_slice, y_slice, _ = prepare_Xy(df_slice, horizon)

        # Fill missing zone dummies (slices won't have all zones)
        for col in reference_feature_cols:
            if col not in X_slice.columns:
                X_slice[col] = 0
        X_slice = X_slice[reference_feature_cols]

        y_pred  = model.predict(X_slice.values)
        metrics = compute_metrics(y_slice.values, y_pred)

    except Exception as e:
        logger.error(f"  [ERROR] {slice_type}={slice_value} horizon={horizon}h: {e}")
        return None

    result = {
        "model":       model_name,
        "horizon":     horizon,
        "slice_type":  slice_type,
        "slice_value": str(slice_value),
        "n_samples":   len(df_slice),
        **metrics,
    }

    print(
        f"  {slice_type:<20} {str(slice_value):<28} "
        f"n={len(df_slice):>6,}  "
        f"MAE={metrics['mae']:>7.3f}  "
        f"RMSE={metrics['rmse']:>7.3f}  "
        f"R2={metrics['r2']:>7.4f}"
    )
    return result


def run_slice_evaluation(
    test_df: pd.DataFrame,
    model,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Run evaluation across all slice types for one model + horizon.
    Order: overall -> zone -> season -> carbon_bucket -> weather_condition
    """
    results        = []
    reference_cols = get_reference_feature_cols(test_df, horizon)

    def _eval(df_slice, stype, sval):
        r = evaluate_on_slice(df_slice, model, horizon, model_name,
                              stype, sval, reference_cols)
        if r:
            results.append(r)

    # ── 1. Overall baseline ──────────────────────────────────
    print(f"\n  [Overall Baseline]")
    _eval(test_df, "overall", "all")

    # ── 2. Zone slices ───────────────────────────────────────
    # ZONE_COL = "zone" — confirmed present in parquet
    if ZONE_COL in test_df.columns:
        print(f"\n  [Zone Slices]")
        for zone, grp in test_df.groupby(ZONE_COL):
            _eval(grp, "zone", zone)

    # ── 3. Season slices ─────────────────────────────────────
    # "season" — confirmed present in parquet (e.g. "Fall")
    if SEASON_COL in test_df.columns:
        print(f"\n  [Season Slices]")
        for season, grp in test_df.groupby(SEASON_COL):
            _eval(grp, "season", season)

    # ── 4. Carbon intensity bucket slices ────────────────────
    if "carbon_bucket" in test_df.columns:
        print(f"\n  [Carbon Intensity Bucket Slices]")
        for bucket, grp in test_df.groupby("carbon_bucket"):
            _eval(grp, "carbon_bucket", bucket)

    # ── 5. Weather condition slices ──────────────────────────
    # Verified column names: temperature_2m_c, wind_speed_100m_ms, cloud_cover_pct
    if WEATHER_COLS.issubset(test_df.columns):
        print(f"\n  [Weather Condition Slices]")
        for condition_name, condition_fn in WEATHER_CONDITIONS.items():
            mask = condition_fn(test_df)
            _eval(test_df[mask], "weather_condition", condition_name)
    else:
        missing_wx = WEATHER_COLS - set(test_df.columns)
        print(f"\n  [Weather Condition Slices] SKIPPED — missing: {missing_wx}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# Disparity Detection
# ──────────────────────────────────────────────

def detect_disparities(
    results_df: pd.DataFrame,
    threshold: float = DISPARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Compare each slice's MAE to the overall baseline MAE.
    Flags slices where relative deviation exceeds threshold (default 20%).
    Positive relative_diff = worse than baseline.
    Negative relative_diff = better than baseline.
    """
    overall_rows = results_df[results_df["slice_type"] == "overall"]
    if overall_rows.empty:
        logger.warning("No overall baseline row — skipping disparity detection.")
        return pd.DataFrame()

    baseline_mae = float(overall_rows["mae"].values[0])
    flagged_rows = []

    for _, row in results_df.iterrows():
        if row["slice_type"] == "overall":
            continue

        rel_diff   = (row["mae"] - baseline_mae) / baseline_mae if baseline_mae != 0 else 0
        is_flagged = abs(rel_diff) > threshold

        flagged_rows.append({
            "model":         row["model"],
            "horizon":       row["horizon"],
            "slice_type":    row["slice_type"],
            "slice_value":   row["slice_value"],
            "n_samples":     row["n_samples"],
            "slice_mae":     row["mae"],
            "baseline_mae":  round(baseline_mae, 4),
            "relative_diff": round(rel_diff, 4),
            "pct_deviation": f"{rel_diff*100:+.1f}%",
            "flagged":       is_flagged,
        })

    disparity_df = pd.DataFrame(flagged_rows)
    n_flagged    = int(disparity_df["flagged"].sum()) if not disparity_df.empty else 0

    print(f"\n  [Disparity Detection]")
    print(f"  Baseline MAE: {baseline_mae:.3f}  |  "
          f"Threshold: {threshold*100:.0f}%  |  "
          f"Flagged: {n_flagged} / {len(disparity_df)} slices")

    if n_flagged > 0:
        print(f"\n  {'Slice Type':<20} {'Slice Value':<28} {'MAE':>7}  {'Deviation':>10}")
        print(f"  {'-'*68}")
        for _, r in disparity_df[disparity_df["flagged"]].iterrows():
            print(f"  {r['slice_type']:<20} {r['slice_value']:<28} "
                  f"{r['slice_mae']:>7.3f}  {r['pct_deviation']:>10}")

    return disparity_df


# ──────────────────────────────────────────────
# MLflow Logging
# ──────────────────────────────────────────────

def log_to_mlflow(
    results_df: pd.DataFrame,
    disparity_df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> None:
    """
    Log per-slice metrics into the same MLflow experiment used by
    train_xgboost.py and train_lightgbm.py (ecopulse-carbon-forecasting).
    """
    mlflow_db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mlruns", "mlflow.db"
    )
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ecopulse-carbon-forecasting")

    run_name = (f"bias_{model_name}_{horizon}h_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task",    "bias_detection")
        mlflow.set_tag("model",   model_name)
        mlflow.set_tag("horizon", f"{horizon}h")
        mlflow.log_param("disparity_threshold", DISPARITY_THRESHOLD)

        for _, row in results_df.iterrows():
            key = (f"{row['slice_type']}__{str(row['slice_value'])}"
       .replace(" ", "_").replace("/", "_").replace("(", "")
       .replace(")", "").replace("+", "plus").replace("%", "pct")
       .replace("<", "lt").replace(">", "gt").replace("=", "eq")
       .replace("-", "").replace(".", ""))
            mlflow.log_metric(f"{key}__mae",  row["mae"])
            mlflow.log_metric(f"{key}__rmse", row["rmse"])
            mlflow.log_metric(f"{key}__r2",   row["r2"])

        if not disparity_df.empty:
            mlflow.log_metric("n_slices_total",   len(disparity_df))
            mlflow.log_metric("n_slices_flagged", int(disparity_df["flagged"].sum()))

    logger.info(f"[MLflow] Logged: {run_name}")


# ──────────────────────────────────────────────
# Report Saving
# ──────────────────────────────────────────────

def save_bias_reports(
    results_df: pd.DataFrame,
    disparity_df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> None:
    """Save CSVs to reports/bias/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_path   = BIAS_REPORTS_DIR / f"slice_metrics_{model_name}_{horizon}h_{ts}.csv"
    disparity_path = BIAS_REPORTS_DIR / f"disparity_report_{model_name}_{horizon}h_{ts}.csv"

    results_df.to_csv(metrics_path,    index=False)
    disparity_df.to_csv(disparity_path, index=False)

    print(f"\n  Slice metrics    -> {metrics_path}")
    print(f"  Disparity report -> {disparity_path}")


# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────

def run_bias_detection(model_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  EcoPulse Bias Detection  |  Model: {model_name.upper()}")
    print(f"{'='*70}")

    # Load test split and add carbon bucket column
    test_df = load_split("test")
    test_df = add_carbon_bucket(test_df)

    ensure_dirs()

    for horizon in HORIZONS:
        print(f"\n{'─'*70}")
        print(f"  Horizon: {horizon}h ahead")
        print(f"{'─'*70}")

        try:
            model = load_model(model_name, horizon)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        results_df = run_slice_evaluation(test_df, model, model_name, horizon)

        if results_df.empty:
            logger.warning(f"No results for {model_name} {horizon}h — skipping.")
            continue

        disparity_df = detect_disparities(results_df)

        try:
            log_to_mlflow(results_df, disparity_df, model_name, horizon)
        except Exception as e:
            logger.warning(f"[MLflow] Logging skipped: {e}")

        save_bias_reports(results_df, disparity_df, model_name, horizon)

    print(f"\n  Bias detection complete for: {model_name}")
    print(f"  Reports saved to: reports/bias/")


def main():
    parser = argparse.ArgumentParser(description="EcoPulse Bias Detection")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "both"],
        default="both",
        help="Which model to evaluate (default: both)",
    )
    args = parser.parse_args()

    if args.model == "both":
        run_bias_detection("xgboost")
        run_bias_detection("lightgbm")
    else:
        run_bias_detection(args.model)


if __name__ == "__main__":
    main()
