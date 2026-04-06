"""
EcoPulse Bias Mitigation
========================
Model Development Phase | Bias Detection & Mitigation
----------------------------------------------------------------
Implements mitigation strategies targeting disparities found in bias_detection.py:

  1. Zone-Aware Sample Weighting
     - US-NW-PACW is consistently 27-34% worse than baseline
     - Upweights PACW samples during training so the model pays more
       attention to that zone

  2. Carbon Bucket Oversampling (SMOTE-style)
     - Very Low (0-100) bucket has only 35 samples — model performs
       terribly on it (R2 negative in several horizons)
     - Oversamples this bucket in the training data to improve coverage

  3. Re-evaluation after mitigation
     - Retrains XGBoost and LightGBM with mitigated training data
     - Re-runs slice evaluation to measure improvement
     - Saves comparison report: before vs after mitigation

Depends on:
  - utils.py           : shared constants, loaders, metrics
  - bias_detection.py  : slice evaluation logic (imported directly)
  - train_split.parquet, val_split.parquet, test_split.parquet

Outputs:
  - models/xgboost_{horizon}h_mitigated.joblib
  - models/lightgbm_{horizon}h_mitigated.joblib
  - reports/bias/mitigation_comparison_{model}_{horizon}h_{ts}.csv
  - MLflow runs logged under experiment: ecopulse-carbon-forecasting

Usage:
  python mitigation.py --model xgboost
  python mitigation.py --model lightgbm
  python mitigation.py --model both        (default)
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
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_split, prepare_Xy, align_columns, compute_metrics,
    ensure_dirs, save_model,
    TARGET_COL, ZONE_COL, FORECAST_TARGETS, HORIZONS,
    MODELS_DIR, REPORTS_DIR,
    logger
)

# Import slice evaluation from bias_detection to avoid duplication
from bias_detection import (
    add_carbon_bucket,
    run_slice_evaluation,
    detect_disparities,
    save_bias_reports,
    BIAS_REPORTS_DIR,
    DISPARITY_THRESHOLD,
)
from mlflow_config import setup_mlflow

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Mitigation Config
# ──────────────────────────────────────────────

# Zone weight multiplier for underperforming zones
# PACW is ~27-34% worse — we upweight it so model trains harder on it
PACW_ZONE         = "US-NW-PACW"
PACW_WEIGHT       = 3.0   # PACW samples count 3x during training

# Carbon bucket oversampling
# Very Low (0-100) has only 35 samples — oversample to at least this many
VERY_LOW_LABEL        = "Very Low (0-100)"
VERY_LOW_TARGET_COUNT = 300  # oversample from 35 to 300

# Model hyperparameters — same as Person 1's training scripts
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ──────────────────────────────────────────────
# Strategy 1 — Zone-Aware Sample Weighting
# ──────────────────────────────────────────────

def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Assign higher sample weights to underperforming zones.
    PACW gets PACW_WEIGHT (3x), all other zones get weight 1.0.

    These weights are passed to model.fit() so the model penalises
    errors on PACW samples more during training.
    """
    weights = np.ones(len(train_df))
    if ZONE_COL in train_df.columns:
        pacw_mask = train_df[ZONE_COL].values == PACW_ZONE
        weights[pacw_mask] = PACW_WEIGHT
        n_pacw = pacw_mask.sum()
        logger.info(f"  Sample weights: {n_pacw:,} PACW samples upweighted to {PACW_WEIGHT}x")
    return weights


# ──────────────────────────────────────────────
# Strategy 2 — Carbon Bucket Oversampling
# ──────────────────────────────────────────────

def oversample_very_low_bucket(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample the Very Low (0-100) carbon intensity bucket by random
    duplication with small Gaussian noise on numeric columns.

    WHY DUPLICATION WITH NOISE instead of pure duplication:
    Pure duplication would give the model identical rows which doesn't
    help generalisation. Adding small noise (std=0.01 * column std)
    creates slightly varied copies that teach the model the general
    pattern without overfitting to exact duplicates.

    WHY NOT SMOTE:
    SMOTE requires imbalanced-learn and works best for classification.
    For regression with only 35 samples, noisy duplication is simpler
    and more interpretable.
    """
    # Add carbon bucket column if not present
    if "carbon_bucket" not in train_df.columns:
        train_df = add_carbon_bucket(train_df)

    very_low_mask = train_df["carbon_bucket"] == VERY_LOW_LABEL
    very_low_df   = train_df[very_low_mask]
    current_count = len(very_low_df)

    if current_count >= VERY_LOW_TARGET_COUNT:
        logger.info(f"  Very Low bucket already has {current_count} samples — no oversampling needed")
        return train_df

    n_to_generate = VERY_LOW_TARGET_COUNT - current_count
    logger.info(f"  Oversampling Very Low bucket: {current_count} -> {VERY_LOW_TARGET_COUNT} "
                f"(generating {n_to_generate} noisy copies)")

    # Identify numeric columns to add noise to
    # Exclude target columns, datetime, zone, categorical columns
    exclude = {TARGET_COL, "datetime", "zone", "season", "carbon_bucket",
               "aws_region", "gcp_region", "azure_region"}
    numeric_cols = [
        c for c in very_low_df.columns
        if c not in exclude
        and very_low_df[c].dtype in ["float64", "float32", "int64", "int32"]
    ]

    synthetic_rows = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_to_generate):
        # Sample a random row from the Very Low bucket
        base_row = very_low_df.sample(n=1, random_state=rng.integers(0, 10000)).copy()

        # Add small Gaussian noise to numeric columns
        for col in numeric_cols:
            col_std = very_low_df[col].std()
            if col_std > 0:
                noise = rng.normal(0, 0.01 * col_std)
                base_row[col] = base_row[col] + noise

        synthetic_rows.append(base_row)

    synthetic_df  = pd.concat(synthetic_rows, ignore_index=True)
    augmented_df  = pd.concat([train_df, synthetic_df], ignore_index=True)

    logger.info(f"  Training set size: {len(train_df):,} -> {len(augmented_df):,} rows")
    return augmented_df


# ──────────────────────────────────────────────
# Mitigated Model Training
# ──────────────────────────────────────────────

def train_mitigated_xgboost(
    X_train, y_train, X_val, y_val,
    sample_weights: np.ndarray,
    horizon: int,
) -> xgb.XGBRegressor:
    """Train XGBoost with sample weights for zone bias mitigation."""
    params = dict(XGBOOST_PARAMS)
    model  = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"  XGBoost mitigated {horizon}h — best iteration: {model.best_iteration}")
    return model


def train_mitigated_lightgbm(
    X_train, y_train, X_val, y_val,
    sample_weights: np.ndarray,
    horizon: int,
) -> lgb.LGBMRegressor:
    """Train LightGBM with sample weights for zone bias mitigation."""
    params = dict(LIGHTGBM_PARAMS)
    model  = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=-1),   # silent
        ],
    )
    logger.info(f"  LightGBM mitigated {horizon}h — best iteration: {model.best_iteration_}")
    return model


# ──────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────

def build_comparison_report(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Merge before/after slice metrics into a single comparison DataFrame.
    Shows MAE before, MAE after, and absolute + relative improvement.
    """
    before = before_df[["slice_type", "slice_value", "mae", "r2"]].copy()
    after  = after_df[["slice_type", "slice_value", "mae", "r2"]].copy()

    before = before.rename(columns={"mae": "mae_before", "r2": "r2_before"})
    after  = after.rename(columns={"mae": "mae_after",  "r2": "r2_after"})

    comparison = before.merge(after, on=["slice_type", "slice_value"], how="outer")
    comparison["mae_improvement"]     = comparison["mae_before"] - comparison["mae_after"]
    comparison["mae_improvement_pct"] = (
        (comparison["mae_before"] - comparison["mae_after"]) / comparison["mae_before"] * 100
    ).round(2)
    comparison["model"]   = model_name
    comparison["horizon"] = horizon

    return comparison


def print_comparison(comparison_df: pd.DataFrame) -> None:
    """Print a readable before/after comparison table."""
    print(f"\n  {'Slice Type':<20} {'Slice Value':<28} "
          f"{'MAE Before':>11} {'MAE After':>10} {'Improvement':>12}")
    print(f"  {'-'*83}")
    for _, row in comparison_df.iterrows():
        imp = row["mae_improvement_pct"]
        arrow = "↓" if imp > 0 else "↑" if imp < 0 else "="
        print(
            f"  {row['slice_type']:<20} {str(row['slice_value']):<28} "
            f"{row['mae_before']:>11.3f} {row['mae_after']:>10.3f} "
            f"{f'{imp:+.1f}% {arrow}':>12}"
        )


def save_comparison_report(
    comparison_df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> None:
    """Save comparison CSV to reports/bias/."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BIAS_REPORTS_DIR / f"mitigation_comparison_{model_name}_{horizon}h_{ts}.csv"
    comparison_df.to_csv(path, index=False)
    print(f"\n  Mitigation comparison -> {path}")


# ──────────────────────────────────────────────
# MLflow Logging
# ──────────────────────────────────────────────

def log_mitigation_to_mlflow(
    comparison_df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> None:
    """Log mitigation improvement metrics to MLflow."""
    setup_mlflow()

    run_name = f"mitigation_{model_name}_{horizon}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task",    "bias_mitigation")
        mlflow.set_tag("model",   model_name)
        mlflow.set_tag("horizon", f"{horizon}h")
        mlflow.log_param("pacw_weight",            PACW_WEIGHT)
        mlflow.log_param("very_low_target_count",  VERY_LOW_TARGET_COUNT)

        for _, row in comparison_df.iterrows():
            key = (f"{row['slice_type']}__{str(row['slice_value'])}"
                   .replace(" ", "_").replace("/", "_").replace("(", "")
                   .replace(")", "").replace("+", "plus").replace("%", "pct")
                   .replace("<", "lt").replace(">", "gt").replace("=", "eq")
                   .replace("-", "").replace(".", ""))
            mlflow.log_metric(f"{key}__mae_before", row["mae_before"])
            mlflow.log_metric(f"{key}__mae_after",  row["mae_after"])
            mlflow.log_metric(f"{key}__improvement_pct", row["mae_improvement_pct"])

        # Overall improvement on flagged slices
        flagged_types = ["zone", "carbon_bucket"]
        flagged = comparison_df[comparison_df["slice_type"].isin(flagged_types)]
        if not flagged.empty:
            avg_imp = flagged["mae_improvement_pct"].mean()
            mlflow.log_metric("avg_improvement_pct_flagged_slices", round(avg_imp, 4))

    logger.info(f"[MLflow] Logged mitigation run: {run_name}")


# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────

def run_mitigation(model_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  EcoPulse Bias Mitigation  |  Model: {model_name.upper()}")
    print(f"{'='*70}")

    # Load all splits
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    # Apply mitigation strategies to training data
    print(f"\n  [Strategy 1] Zone-Aware Sample Weighting (PACW = {PACW_WEIGHT}x)")
    print(f"  [Strategy 2] Carbon Bucket Oversampling (Very Low: 35 -> {VERY_LOW_TARGET_COUNT})")

    # Oversample Very Low carbon bucket in training data
    train_mitigated = oversample_very_low_bucket(train_df)
    train_mitigated = add_carbon_bucket(train_mitigated)

    # Prepare test set for evaluation (add carbon bucket)
    test_df_eval = add_carbon_bucket(test_df)

    ensure_dirs()

    for horizon in HORIZONS:
        print(f"\n{'─'*70}")
        print(f"  Horizon: {horizon}h ahead")
        print(f"{'─'*70}")

        # ── Prepare features ──────────────────────────────────────────
        X_train_orig, y_train_orig, _ = prepare_Xy(train_df, horizon)
        X_train_mit,  y_train_mit,  _ = prepare_Xy(train_mitigated, horizon)
        X_val,        y_val,        _ = prepare_Xy(val_df, horizon)
        X_test,       y_test,       _ = prepare_Xy(test_df, horizon)

        # Align columns across all splits
        X_train_mit, X_val, X_test, feature_cols = align_columns(
            X_train_mit, X_val, X_test
        )
        # Also align original train to same feature space
        for col in feature_cols:
            if col not in X_train_orig.columns:
                X_train_orig[col] = 0
        X_train_orig = X_train_orig[feature_cols]

        # ── Compute sample weights on mitigated training data ────────
        # Weights apply to original rows; synthetic rows from Very Low
        # bucket get weight 1.0 by default (they're already oversampled)
        n_original  = len(train_df)
        n_synthetic = len(train_mitigated) - n_original
        weights_original  = compute_sample_weights(train_df)
        weights_synthetic = np.ones(n_synthetic)
        sample_weights    = np.concatenate([weights_original, weights_synthetic])

        # ── Load original model for "before" evaluation ──────────────
        original_model_path = os.path.join(MODELS_DIR, f"{model_name}_{horizon}h.joblib")
        if not os.path.exists(original_model_path):
            logger.error(f"Original model not found: {original_model_path} — skipping {horizon}h")
            continue
        original_model = joblib.load(original_model_path)
        logger.info(f"  Loaded original model: {original_model_path}")

        # ── Before: slice evaluation with original model ─────────────
        print(f"\n  [Before Mitigation]")
        before_df = run_slice_evaluation(test_df_eval, original_model, model_name, horizon)

        # ── Train mitigated model ────────────────────────────────────
        print(f"\n  [Training Mitigated Model]")
        if model_name == "xgboost":
            mitigated_model = train_mitigated_xgboost(
                X_train_mit, y_train_mit,
                X_val, y_val,
                sample_weights, horizon,
            )
        else:
            mitigated_model = train_mitigated_lightgbm(
                X_train_mit, y_train_mit,
                X_val, y_val,
                sample_weights, horizon,
            )

        # Save mitigated model
        mitigated_name = f"{model_name}_{horizon}h_mitigated"
        save_model(mitigated_model, mitigated_name)

        # ── After: slice evaluation with mitigated model ─────────────
        print(f"\n  [After Mitigation]")
        after_df = run_slice_evaluation(
            test_df_eval, mitigated_model, f"{model_name}_mitigated", horizon
        )

        # ── Disparity check on mitigated model ───────────────────────
        print(f"\n  [Disparity Check — Mitigated Model]")
        detect_disparities(after_df, threshold=DISPARITY_THRESHOLD)

        # ── Build and print comparison ────────────────────────────────
        comparison_df = build_comparison_report(before_df, after_df, model_name, horizon)
        print(f"\n  [Before vs After Comparison]")
        print_comparison(comparison_df)

        # ── Log to MLflow ─────────────────────────────────────────────
        try:
            log_mitigation_to_mlflow(comparison_df, model_name, horizon)
        except Exception as e:
            logger.warning(f"[MLflow] Logging skipped: {e}")

        # ── Save reports ──────────────────────────────────────────────
        save_comparison_report(comparison_df, model_name, horizon)
        save_bias_reports(after_df, detect_disparities(after_df), f"{model_name}_mitigated", horizon)

    print(f"\n  Mitigation complete for: {model_name}")
    print(f"  Mitigated models saved to: models/")
    print(f"  Reports saved to: reports/bias/")


def main():
    parser = argparse.ArgumentParser(description="EcoPulse Bias Mitigation")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "both"],
        default="both",
        help="Which model to mitigate (default: both)",
    )
    args = parser.parse_args()

    if args.model == "both":
        run_mitigation("xgboost")
        run_mitigation("lightgbm")
    else:
        run_mitigation(args.model)


if __name__ == "__main__":
    main()
