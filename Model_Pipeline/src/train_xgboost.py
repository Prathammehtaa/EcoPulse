"""
EcoPulse XGBoost Model Training
================================
Trains XGBoost regressors for each forecast horizon (1h, 12h, 24h).
Uses validation set for early stopping, evaluates on held-out test set.
Tracks all experiments in MLflow.
Run: cd Model_Pipeline/src && python train_xgboost.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
from datetime import datetime
from mlflow.models.signature import infer_signature

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, print_metrics_table,
    save_results, save_model, ensure_dirs,
    FORECAST_TARGETS, HORIZONS, MODELS_DIR, REPORTS_DIR,
    logger, get_timestamp
)
from mlflow_config import (
    setup_mlflow, build_run_tags, get_performance_tier,
<<<<<<< HEAD
    log_dataset_info, log_feature_importance_plot, log_residual_plot,
    register_model, TRAINING_EXPERIMENT_NAME,
)

# GCP Artifact Registry push — opt-in via GCP_PUSH_MODELS=1
_GCP_PUSH = os.getenv("GCP_PUSH_MODELS", "0") == "1"
if _GCP_PUSH:
    try:
        from gcp_registry import push_after_mlflow_log, make_version_string
    except ImportError:
        _GCP_PUSH = False
=======
    log_dataset_info, log_feature_importance_plot,log_residual_plot,
    register_model, TRAINING_EXPERIMENT_NAME,
)
>>>>>>> ef132e8b79b948f2ac4bcefede05d444b54a34b5

# ============================================================
# XGBOOST HYPERPARAMETERS
# ============================================================
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


def train_single_horizon(train_df, val_df, test_df, horizon, params=XGBOOST_PARAMS):
    """
    Train XGBoost for one forecast horizon.

    1. Prepare features (one-hot encode zone, drop targets/datetime)
    2. Compute fairness-aware sample weights
    3. Train XGBoost with early stopping on validation set
    4. Evaluate on train, val, and test
    5. Extract feature importances
    6. Return everything for logging
    """
    target_col = FORECAST_TARGETS[horizon]
    logger.info(f"Training XGBoost for {horizon}h horizon...")

    # Prepare features
    X_train, y_train, _ = prepare_Xy(train_df, horizon)
    X_val, y_val, _ = prepare_Xy(val_df, horizon)
    X_test, y_test, _ = prepare_Xy(test_df, horizon)

    # Align columns across splits
    X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

    # Fairness-aware sample weights
    ci = train_df["carbon_intensity_gco2_per_kwh"].loc[y_train.index]
    sample_weights = np.ones(len(y_train))
    sample_weights[ci < 100] = 3.0       # 3x weight for Very Low
    sample_weights[(ci >= 100) & (ci < 200)] = 1.5  # 1.5x for Low
    logger.info(f"  Sample weights: Very Low(3x)={int((ci < 100).sum())}, "
                f"Low(1.5x)={int(((ci >= 100) & (ci < 200)).sum())}, "
                f"Normal(1x)={int((ci >= 200).sum())}")

    logger.info(f"  Features: {len(feature_cols)}, "
                f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]
    logger.info(f"  Best iteration: {best_iteration}")

    # Evaluate on all splits
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
    print(f"\n  XGBoost {horizon}h Results:")
    for split_name in ["train", "val", "test"]:
        print_metrics(results[split_name], f"    {split_name}")

    print(f"\n  Top 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<50} {row['importance']:.4f}")
    
    from mlflow.models.signature import infer_signature
    X_sample = X_train.iloc[:50]
    signature = infer_signature(X_sample, model.predict(X_sample))

    # MLflow signature
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
        },
    }


def train_all_horizons():
    """Train XGBoost for all horizons with MLflow tracking."""
    print("=" * 80)
    print("ECOPULSE XGBOOST TRAINING")
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

        with mlflow.start_run(run_name=f"xgboost_{horizon}h_{timestamp}"):
            # Log parameters
            mlflow.log_params(XGBOOST_PARAMS)
            mlflow.log_param("forecast_horizon", horizon)
            mlflow.log_param("model_type", "xgboost")
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

<<<<<<< HEAD
            # --- Performance tier tag ---
            tier = get_performance_tier(result["results"]["test"]["mae"], horizon)
            mlflow.set_tag("perf_tier", tier)
=======
            # Log model to MLflow
            mlflow.xgboost.log_model(
                result["model"],
                artifact_path=f"xgboost_{horizon}h",
            )

            # Save model locally
            save_model(result["model"], f"xgboost_{horizon}h")

            # Save feature importance
            importance_path = os.path.join(REPORTS_DIR, f"xgb_importance_{horizon}h.csv")
            result["importance"].to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)


            # --- Performance tier tag ---
            tier = get_performance_tier(result["results"]["test"]["mae"], horizon)
            mlflow.set_tag("perf_tier", tier)

            # --- Performance tier tag ---
            tier = get_performance_tier(result["results"]["test"]["mae"], horizon)
            mlflow.set_tag("perf_tier", tier)

>>>>>>> ef132e8b79b948f2ac4bcefede05d444b54a34b5

            # --- Log model with schema signature ---
            mlflow.xgboost.log_model(
                result["model"],
                artifact_path=f"xgboost_{horizon}h",
                signature=result["signature"],
                input_example=result["X_sample"],
            )

            # --- Save model locally ---
            save_model(result["model"], f"xgboost_{horizon}h")

            # --- Feature importance CSV + plot ---
            importance_path = os.path.join(REPORTS_DIR, f"xgb_importance_{horizon}h.csv")
            result["importance"].to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path, artifact_path="feature_importance")
            log_feature_importance_plot(
                result["importance"], horizon, "xgboost", REPORTS_DIR
            )

            # --- Residual plots for val and test splits ---
            for split_name in ["val", "test"]:
                y_true, y_pred = result["preds_by_split"][split_name]
                log_residual_plot(y_true, y_pred, split_name, horizon, "xgboost", REPORTS_DIR)

            # --- Opt-in model registry ---
            run_id = mlflow.active_run().info.run_id
            register_model(run_id, f"xgboost_{horizon}h", horizon, "xgboost")

            # --- GCP Artifact Registry push (opt-in) ---
            if _GCP_PUSH:
                push_after_mlflow_log(
                    model_path=os.path.join(MODELS_DIR, f"xgboost_{horizon}h.joblib"),
                    model_name=f"xgboost_{horizon}h",
                    version=os.getenv("RETRAIN_VERSION") or make_version_string("xgboost", horizon),
                    mlflow_run_id=run_id,
                    horizon=horizon,
                    model_type="xgboost",
                    metrics=result["results"]["test"],
                    performance_tier=tier,
                    auto_promote=True,
                )

<<<<<<< HEAD
=======

                # --- Save model locally ---
                save_model(result["model"], f"xgboost_{horizon}h")

                # --- Feature importance CSV + plot ---
                importance_path = os.path.join(REPORTS_DIR, f"xgb_importance_{horizon}h.csv")
                result["importance"].to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, artifact_path="feature_importance")
                log_feature_importance_plot(
                    result["importance"], horizon, "xgboost", REPORTS_DIR
                )

                # --- Residual plots for val and test splits ---
                for split_name in ["val", "test"]:
                    y_true, y_pred = result["preds_by_split"][split_name]
                    log_residual_plot(y_true, y_pred, split_name, horizon, "xgboost", REPORTS_DIR)

                # --- Opt-in model registry ---
                run_id = mlflow.active_run().info.run_id
                register_model(run_id, f"xgboost_{horizon}h", horizon, "xgboost")

                # --- GCP Artifact Registry push (opt-in via GCP_PUSH_MODELS=1) ---
                if _GCP_PUSH:
                    push_after_mlflow_log(
                        model_path=os.path.join(MODELS_DIR, f"xgboost_{horizon}h.joblib"),
                        model_name=f"xgboost_{horizon}h",
                        version=os.getenv("RETRAIN_VERSION") or make_version_string("xgboost", horizon),
                        mlflow_run_id=run_id,
                        horizon=horizon,
                        model_type="xgboost",
                        metrics=result["results"]["test"],
                        performance_tier=tier,
                        auto_promote=True,
                    )

                # --- Collect test results for summary ---
                test_metrics          = result["results"]["test"].copy()
                test_metrics["model"] = f"XGBoost ({horizon}h)"
                test_metrics["horizon"] = horizon
                all_results.append(test_metrics)

>>>>>>> ef132e8b79b948f2ac4bcefede05d444b54a34b5
            # --- Collect test results for summary ---
            test_metrics = result["results"]["test"].copy()
            test_metrics["model"] = f"XGBoost ({horizon}h)"
            test_metrics["horizon"] = horizon
            all_results.append(test_metrics)
<<<<<<< HEAD
=======

>>>>>>> ef132e8b79b948f2ac4bcefede05d444b54a34b5

    # Summary
    print(f"\n{'='*80}")
    print("XGBOOST TRAINING SUMMARY (TEST SET)")
    print(f"{'='*80}")
    print_metrics_table(all_results)

    # Save summary
    save_results(all_results, "xgboost_results.csv")

    # Compare with baselines
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*80}")

    baseline_targets = {1: 57.48, 6: 71.33, 12: 76.36, 24: 68.79}
    for r in all_results:
        h = r["horizon"]
        baseline_mae = baseline_targets[h]
        improvement = ((baseline_mae - r["mae"]) / baseline_mae) * 100
        status = "BEATS" if r["mae"] < baseline_mae else "LOSES TO"
        print(f"  {h}h: XGBoost MAE={r['mae']:.2f} {status} baseline MAE={baseline_mae:.2f} "
              f"({improvement:+.1f}% improvement)")

    print(f"\n✅ XGBoost training complete!")
    print(f"   Models saved to: models/")
    print(f"   Results saved to: reports/xgboost_results.csv")
    print(f"   MLflow runs logged to: mlruns/")

    return all_results


if __name__ == "__main__":
    results = train_all_horizons()
