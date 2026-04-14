"""
EcoPulse LightGBM Model Training
=================================
Trains LightGBM regressors for each forecast horizon (1h, 12h, 24h).
Same structure as XGBoost for fair comparison.

Author: Aaditya Krishna (ML Modelling Lead)
Run: cd Model_Pipeline/src && python train_lightgbm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import joblib
from datetime import datetime

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, print_metrics_table,
    save_results, save_model, ensure_dirs,
    FORECAST_TARGETS, HORIZONS, MODELS_DIR, REPORTS_DIR,
    logger, get_timestamp
)
from mlflow_config import (
    setup_mlflow, build_run_tags, get_performance_tier,
    log_dataset_info, log_feature_importance_plot,
    register_model, TRAINING_EXPERIMENT_NAME,
)

# ============================================================
# LIGHTGBM HYPERPARAMETERS
# ============================================================
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


def train_single_horizon(train_df, val_df, test_df, horizon, params=LIGHTGBM_PARAMS):
    """Train LightGBM for one forecast horizon."""
    target_col = FORECAST_TARGETS[horizon]
    logger.info(f"Training LightGBM for {horizon}h horizon...")

    # Prepare features
    X_train, y_train, _ = prepare_Xy(train_df, horizon)
    X_val, y_val, _ = prepare_Xy(val_df, horizon)
    X_test, y_test, _ = prepare_Xy(test_df, horizon)

    # Align columns
    X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

    logger.info(f"  Features: {len(feature_cols)}, "
                f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50)
        ]
    )

    best_iteration = model.best_iteration_ if hasattr(model, "best_iteration_") else params["n_estimators"]
    logger.info(f"  Best iteration: {best_iteration}")

    # Evaluate
    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        preds = model.predict(X)
        metrics = compute_metrics(y, preds)
        results[split_name] = metrics

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Print results
    print(f"\n  LightGBM {horizon}h Results:")
    for split_name in ["train", "val", "test"]:
        print_metrics(results[split_name], f"    {split_name}")

    print(f"\n  Top 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<50} {row['importance']}")
    
    from mlflow.models.signature import infer_signature
    X_sample = X_train.iloc[:50]
    signature = infer_signature(X_sample, model.predict(X_sample))
    
    return {
        "model": model,
        "results": results,
        "importance": importance_df,
        "feature_cols": feature_cols,
        "best_iteration": best_iteration,
        "horizon": horizon,
        "signature": signature,
        "X_sample": X_sample,
        "preds_by_split": {
            "val": (y_val, model.predict(X_val)),
            "test": (y_test, model.predict(X_test)),
    }


def train_all_horizons():
    """Train LightGBM for all horizons with MLflow tracking."""
    print("=" * 80)
    print("ECOPULSE LIGHTGBM TRAINING")
    print("=" * 80)

    # Load data
    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    # Setup MLflow
    mlflow_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(mlflow_dir, 'mlflow.db')}")
    mlflow.set_experiment("ecopulse-carbon-forecasting")

    ensure_dirs()
    all_results = []
    timestamp = get_timestamp()

    for horizon in HORIZONS:
        print(f"\n{'='*80}")
        print(f"  TRAINING: {horizon}h HORIZON")
        print(f"{'='*80}")

        with mlflow.start_run(run_name=f"lightgbm_{horizon}h_{timestamp}"):
            mlflow.log_params(LIGHTGBM_PARAMS)
            mlflow.log_param("forecast_horizon", horizon)
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("train_rows", len(train))
            mlflow.log_param("val_rows", len(val))
            mlflow.log_param("test_rows", len(test))

            # Train
            result = train_single_horizon(train, val, test, horizon)

            # Log metrics
            for split_name, metrics in result["results"].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", value)

            mlflow.log_metric("best_iteration", result["best_iteration"])

            # Log model
            mlflow.lightgbm.log_model(
                result["model"],
                artifact_path=f"lightgbm_{horizon}h",
            )

            # Save locally
            save_model(result["model"], f"lightgbm_{horizon}h")

            # Save feature importance
            importance_path = os.path.join(REPORTS_DIR, f"lgb_importance_{horizon}h.csv")
            result["importance"].to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            # --- Performance tier tag ---
            tier = get_performance_tier(result["results"]["test"]["mae"], horizon)
            mlflow.set_tag("perf_tier", tier)

            # --- Log model with schema signature ---
            mlflow.lightgbm.log_model(
                result["model"],
                artifact_path=f"lightgbm_{horizon}h",
                signature=result["signature"],
                input_example=result["X_sample"],
            )

            # --- Save model locally ---
            save_model(result["model"], f"lightgbm_{horizon}h")

            # --- Feature importance CSV + plot ---
            importance_path = os.path.join(REPORTS_DIR, f"lgb_importance_{horizon}h.csv")
            result["importance"].to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, artifact_path="feature_importance")
            log_feature_importance_plot(
                result["importance"], horizon, "lightgbm", REPORTS_DIR
            )

            # --- Residual plots for val and test splits ---
            for split_name in ["val", "test"]:
                y_true, y_pred = result["preds_by_split"][split_name]
                log_residual_plot(y_true, y_pred, split_name, horizon, "lightgbm", REPORTS_DIR)

            # --- Opt-in model registry ---
            run_id = mlflow.active_run().info.run_id
            register_model(run_id, f"lightgbm_{horizon}h", horizon, "lightgbm")

            # --- GCP Artifact Registry push (opt-in via GCP_PUSH_MODELS=1) ---
            if _GCP_PUSH:
                push_after_mlflow_log(
                    model_path=os.path.join(MODELS_DIR, f"lightgbm_{horizon}h.joblib"),
                    model_name=f"lightgbm_{horizon}h",
                    version=os.getenv("RETRAIN_VERSION") or make_version_string("lightgbm", horizon),
                    mlflow_run_id=run_id,
                    horizon=horizon,
                    model_type="lightgbm",
                    metrics=result["results"]["test"],
                    performance_tier=tier,
                    auto_promote=True,
                )

            # --- Collect test results for summary ---
            test_metrics          = result["results"]["test"].copy()
            test_metrics["model"] = f"LightGBM ({horizon}h)"
            test_metrics["horizon"] = horizon
            all_results.append(test_metrics)

    # Summary
    print(f"\n{'='*80}")
    print("LIGHTGBM TRAINING SUMMARY (TEST SET)")
    print(f"{'='*80}")
    print_metrics_table(all_results)

    # Save
    save_results(all_results, "lightgbm_results.csv")

    # Compare with baselines and XGBoost
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINES & XGBOOST")
    print(f"{'='*80}")

    baseline_targets = {1: 57.48, 6: 71.33, 12: 76.36, 24: 68.79}
    xgboost_targets = {1: 25.14, 6: 34.34, 12: 39.97, 24: 43.01}

    for r in all_results:
        h = r["horizon"]
        bl_mae = baseline_targets[h]
        xgb_mae = xgboost_targets[h]
        bl_imp = ((bl_mae - r["mae"]) / bl_mae) * 100
        vs_xgb = "BEATS" if r["mae"] < xgb_mae else "LOSES TO"
        print(f"  {h}h: LightGBM MAE={r['mae']:.2f} | "
              f"vs Baseline: {bl_imp:+.1f}% | "
              f"{vs_xgb} XGBoost ({xgb_mae:.2f})")

    print(f"\n✅ LightGBM training complete!")
    print(f"   Models saved to: models/")
    print(f"   Results saved to: reports/lightgbm_results.csv")

    return all_results


if __name__ == "__main__":
    results = train_all_horizons()
