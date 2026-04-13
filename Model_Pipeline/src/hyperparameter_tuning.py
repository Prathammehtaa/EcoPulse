"""
EcoPulse Hyperparameter Tuning with Optuna
===========================================
Systematically searches for the best XGBoost hyperparameters
for each forecast horizon using Bayesian optimization.

HOW IT WORKS:
Optuna tries different combinations of hyperparameters (max_depth,
learning_rate, etc.) and uses the validation MAE to decide which
combinations to try next. Unlike grid search (which tries everything),
Optuna is smart — it focuses on promising regions of the search space.

Author: Aaditya Krishna (ML Modelling Lead)
Run: cd Model_Pipeline/src && python hyperparameter_tuning.py
"""

import sys
import os

# Reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
np.random.seed(42)

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib
from datetime import datetime
import mlflow
import mlflow.xgboost
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
    log_dataset_info, log_feature_importance_plot,
    register_model, TUNING_EXPERIMENT_NAME,
)

# GCP Artifact Registry push — opt-in via GCP_PUSH_MODELS=1
_GCP_PUSH = os.getenv("GCP_PUSH_MODELS", "0") == "1"
if _GCP_PUSH:
    try:
        from gcp_registry import push_after_mlflow_log, make_version_string
    except ImportError:
        logger.warning("gcp_registry imports failed — GCP push disabled for this run")
        _GCP_PUSH = False

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Number of trials per horizon
N_TRIALS = 20


def compute_sample_weights(train_df, y_train):
    """
    Fairness-aware sample weights: prioritize rare carbon intensity slices.
    Very Low (< 100 gCO2/kWh) = 3x weight
    Low (100-200) = 1.5x weight
    Normal (>= 200) = 1x weight
    """
    ci = train_df["carbon_intensity_gco2_per_kwh"].loc[y_train.index]
    weights = np.ones(len(y_train))
    weights[ci < 100] = 3.0
    weights[(ci >= 100) & (ci < 200)] = 1.5
    logger.info(f"  Sample weights: Very Low(3x)={int((ci < 100).sum())}, "
                f"Low(1.5x)={int(((ci >= 100) & (ci < 200)).sum())}, "
                f"Normal(1x)={int((ci >= 200).sum())}")
    return weights


def create_objective(X_train, y_train, X_val, y_val, sample_weights=None):
    """
    Create an Optuna objective function for a specific horizon.
    
    WHAT OPTUNA DOES:
    Each "trial" picks a set of hyperparameters from the defined ranges.
    It trains an XGBoost model with those params, evaluates on validation,
    and returns the MAE. Optuna uses TPE (Tree-structured Parzen Estimator)
    to intelligently pick the next set of params based on what worked before.
    
    SEARCH SPACE:
    - n_estimators: 200-1500 (more trees since current models used all 500)
    - max_depth: 3-10 (shallow to deep trees)
    - learning_rate: 0.01-0.3 (small careful steps to large aggressive steps)
    - subsample: 0.5-1.0 (how much data each tree sees)
    - colsample_bytree: 0.5-1.0 (how many features each tree sees)
    - min_child_weight: 1-20 (minimum samples per leaf)
    - reg_alpha: 0.0-2.0 (L1 regularization)
    - reg_lambda: 0.0-5.0 (L2 regularization)
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
            "seed": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        mae = np.mean(np.abs(y_val - preds))
        mlflow.log_metric("trial_mae", mae, step=trial.number)
        return mae

    return objective


def tune_horizon(train_df, val_df, test_df, horizon, n_trials=N_TRIALS):
    """Run Optuna tuning for a single horizon."""
    logger.info(f"Tuning XGBoost for {horizon}h horizon ({n_trials} trials)...")

    # Prepare data
    X_train, y_train, _ = prepare_Xy(train_df, horizon)
    X_val, y_val, _ = prepare_Xy(val_df, horizon)
    X_test, y_test, _ = prepare_Xy(test_df, horizon)
    X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

    # Compute fairness-aware sample weights
    sample_weights = compute_sample_weights(train_df, y_train)

    # Create and run study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"xgboost_{horizon}h"
    )

    objective = create_objective(X_train, y_train, X_val, y_val, sample_weights)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Best params
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["seed"] = 42
    best_params["n_jobs"] = -1
    best_params["early_stopping_rounds"] = 30

    print(f"\n  Best params for {horizon}h:")
    for k, v in best_params.items():
        if k not in ["random_state", "seed", "n_jobs", "early_stopping_rounds"]:
            print(f"    {k}: {v}")
    print(f"  Best validation MAE: {study.best_value:.4f}")

    # Retrain with best params on full training data (with weights)
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on all splits
    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        preds = final_model.predict(X)
        metrics = compute_metrics(y, preds)
        results[split_name] = metrics

    print(f"\n  Tuned XGBoost {horizon}h Results:")
    for split_name in ["train", "val", "test"]:
        print_metrics(results[split_name], f"    {split_name}")

    # Feature importance for the retrained model
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Small sample for MLflow signature and input_example
    X_sample  = X_train.iloc[:50]
    signature = infer_signature(X_sample, final_model.predict(X_sample))

    return {
        "model":        final_model,
        "results":      results,
        "best_params":  best_params,
        "best_val_mae": study.best_value,
        "study":        study,
        "feature_cols": feature_cols,
        "horizon":      horizon,
        "importance":   importance_df,
        "signature":    signature,
        "X_sample":     X_sample,
    }


def run_tuning():
    """Run hyperparameter tuning for all horizons."""
    print("=" * 80)
    print("ECOPULSE HYPERPARAMETER TUNING (OPTUNA)")
    print(f"Trials per horizon: {N_TRIALS}")
    print(f"Sample weights: Very Low=3x, Low=1.5x, Normal=1x")
    print("=" * 80)

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    # MLflow setup for the tuning experiment (separate from training runs)
    setup_mlflow(TUNING_EXPERIMENT_NAME)

    ensure_dirs()
    all_results = []
    all_params  = {}
    timestamp   = get_timestamp()

    # Base XGBoost results for comparison (on 240K dataset)
    original_mae = {1: 27.15, 6: 36.02, 12: 40.89, 24: 42.60}

    # Parent run groups all horizon tuning into one session
    with mlflow.start_run(run_name=f"optuna_all_{timestamp}"):
        mlflow.set_tags({
            "model_type":    "xgboost_tuned",
            "tuner":         "optuna",
            "n_trials":      str(N_TRIALS),
            "horizons":      ",".join(str(h) for h in HORIZONS),
            "project":       "ecopulse",
            "sample_weights": "very_low=3x,low=1.5x",
        })
        mlflow.log_param("n_trials", N_TRIALS)

        for horizon in HORIZONS:
            print(f"\n{'='*80}")
            print(f"  TUNING: {horizon}h HORIZON")
            print(f"{'='*80}")

            with mlflow.start_run(run_name=f"optuna_{horizon}h_{timestamp}", nested=True):
                # Tags
                mlflow.set_tags(build_run_tags(
                    "xgboost_tuned", horizon,
                    tuner="optuna",
                    n_trials=str(N_TRIALS),
                ))

                # Dataset provenance
                log_dataset_info(train, val, test)
                mlflow.log_param("n_trials",        N_TRIALS)
                mlflow.log_param("forecast_horizon", horizon)

                # Run tuning
                result = tune_horizon(train, val, test, horizon)

                # Best hyperparameters (exclude non-loggable fixed params)
                loggable_params = {
                    k: v for k, v in result["best_params"].items()
                    if k not in ["random_state", "seed", "n_jobs", "early_stopping_rounds"]
                }
                mlflow.log_params(loggable_params)
                mlflow.log_metric("optuna_best_val_mae", result["best_val_mae"])
                mlflow.log_metric("n_features", len(result["feature_cols"]))
                mlflow.log_metric("best_trial_number", result["study"].best_trial.number)
                mlflow.log_metric("best_trial_duration_s", result["study"].best_trial.duration.total_seconds())

                # Split metrics
                for split_name, metrics in result["results"].items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{split_name}_{metric_name}", value)

                # Performance tier
                tier = get_performance_tier(result["results"]["test"]["mae"], horizon)
                mlflow.set_tag("perf_tier", tier)

                # Feature importance plot
                log_feature_importance_plot(
                    result["importance"], horizon, "xgboost_tuned", REPORTS_DIR
                )

                # Log model artifact with signature
                mlflow.xgboost.log_model(
                    result["model"],
                    artifact_path=f"xgboost_tuned_{horizon}h",
                    signature=result["signature"],
                    input_example=result["X_sample"],
                )

                # Opt-in registry
                run_id = mlflow.active_run().info.run_id
                register_model(run_id, f"xgboost_tuned_{horizon}h", horizon, "xgboost_tuned")

                # Save tuned model locally
                save_model(result["model"], f"xgboost_tuned_{horizon}h")

                # --- GCP Artifact Registry push (opt-in via GCP_PUSH_MODELS=1) ---
                if _GCP_PUSH:
                    push_after_mlflow_log(
                        model_path=os.path.join(MODELS_DIR, f"xgboost_tuned_{horizon}h.joblib"),
                        model_name=f"xgboost_tuned_{horizon}h",
                        version=make_version_string("xgboost_tuned", horizon),
                        mlflow_run_id=run_id,
                        horizon=horizon,
                        model_type="xgboost_tuned",
                        metrics=result["results"]["test"],
                        performance_tier=tier,
                        auto_promote=True,
                    )

            # Collect results
            test_metrics = result["results"]["test"].copy()
            test_metrics["model"] = f"XGBoost Tuned ({horizon}h)"
            test_metrics["horizon"] = horizon
            all_results.append(test_metrics)
            all_params[horizon] = result["best_params"]

        # Save best params and log as MLflow artifact (still inside parent run)
        params_df = pd.DataFrame([
            {"horizon": h, **{k: v for k, v in p.items()
             if k not in ["random_state", "seed", "n_jobs", "early_stopping_rounds"]}}
            for h, p in all_params.items()
        ])
        params_path = os.path.join(REPORTS_DIR, "best_hyperparameters.csv")
        params_df.to_csv(params_path, index=False)
        mlflow.log_artifact(params_path)

    # Summary
    print(f"\n{'='*80}")
    print("TUNING SUMMARY (TEST SET)")
    print(f"{'='*80}")
    print_metrics_table(all_results)

    # Comparison with original
    print(f"\n{'='*80}")
    print("TUNED vs BASE XGBOOST (with sample weights)")
    print(f"{'='*80}")
    print(f"  {'Horizon':<10} {'Base MAE':>15} {'Tuned MAE':>15} {'Improvement':>15}")
    print(f"  {'-'*55}")
    for r in all_results:
        h = r["horizon"]
        orig = original_mae.get(h, 0)
        tuned = r["mae"]
        imp = ((orig - tuned) / orig) * 100 if orig > 0 else 0
        status = "IMPROVED" if tuned < orig else "NO GAIN"
        print(f"  {f'{h}h':<10} {orig:>15.2f} {tuned:>15.2f} {f'{imp:+.1f}% {status}':>15}")

    # Save results
    save_results(all_results, "tuned_xgboost_results.csv")

    print(f"\n✅ Hyperparameter tuning complete!")
    print(f"   Tuned models saved to: models/xgboost_tuned_*.joblib")
    print(f"   Results saved to: reports/tuned_xgboost_results.csv")
    print(f"   Best params saved to: reports/best_hyperparameters.csv")

    return all_results


if __name__ == "__main__":
    results = run_tuning()