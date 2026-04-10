"""
EcoPulse — LightGBM Hyperparameter Tuning (Optuna)
Uses the same prepare_Xy from utils.py as XGBoost tuning
to ensure consistent feature sets across all models.
"""
import os
import sys

# Reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd
import optuna
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, print_metrics_table,
    save_results, save_model, ensure_dirs,
    HORIZONS, MODELS_DIR, REPORTS_DIR,
    logger, get_timestamp
)

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIGURATION ─────────────────────────────────────
N_TRIALS = 20
SEED = 42


def create_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    return objective


def tune_horizon(train, val, test, horizon):
    # Use the SAME prepare_Xy as XGBoost and bias_detection
    X_train, y_train, _ = prepare_Xy(train, horizon)
    X_val, y_val, _ = prepare_Xy(val, horizon)
    X_test, y_test, _ = prepare_Xy(test, horizon)
    X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

    print("\n  Train: {}, Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))
    print("  Features: {}".format(len(feature_cols)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        create_objective(X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n  Best trial: MAE = {:.2f}".format(study.best_trial.value))

    # Retrain best model
    best_params = study.best_trial.params
    best_params["random_state"] = SEED
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1
    best_params["deterministic"] = True
    best_params["force_row_wise"] = True

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate on all splits
    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        preds = final_model.predict(X)
        metrics = compute_metrics(y, preds)
        results[split_name] = metrics

    print("\n  Tuned LightGBM {}h Results:".format(horizon))
    for split_name in ["train", "val", "test"]:
        print_metrics(results[split_name], "    {}".format(split_name))

    return {
        "model": final_model,
        "results": results,
        "best_params": best_params,
        "feature_cols": feature_cols,
        "horizon": horizon,
    }


def run_tuning():
    print("=" * 80)
    print("ECOPULSE LIGHTGBM HYPERPARAMETER TUNING (OPTUNA)")
    print("Trials per horizon: {}".format(N_TRIALS))
    print("=" * 80)

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    ensure_dirs()
    all_results = []

    # Original LightGBM MAE for comparison
    try:
        orig = pd.read_csv(os.path.join(REPORTS_DIR, "lightgbm_results.csv"))
        original_mae = dict(zip(orig["horizon"], orig["mae"]))
    except Exception:
        original_mae = {1: 28.54, 6: 37.88, 12: 42.14, 24: 42.94}

    for horizon in HORIZONS:
        print("\n{}".format("=" * 80))
        print("  TUNING: {}h HORIZON".format(horizon))
        print("=" * 80)

        result = tune_horizon(train, val, test, horizon)

        # Save tuned model
        save_model(result["model"], "lightgbm_tuned_{}h".format(horizon))

        row = result["results"]["test"].copy()
        row["model"] = "LightGBM Tuned ({}h)".format(horizon)
        row["horizon"] = horizon
        all_results.append(row)

    # Summary
    print("\n{}".format("=" * 80))
    print("TUNING SUMMARY (TEST SET)")
    print("=" * 80)
    print_metrics_table(all_results)

    print("\n{}".format("=" * 80))
    print("TUNED vs ORIGINAL LIGHTGBM")
    print("=" * 80)
    print("  {:<10} {:>15} {:>15} {:>15}".format("Horizon", "Original MAE", "Tuned MAE", "Improvement"))
    print("  {}".format("-" * 55))
    for r in all_results:
        h = r["horizon"]
        orig = original_mae.get(h, 0)
        tuned = r["mae"]
        imp = ((orig - tuned) / orig) * 100 if orig > 0 else 0
        status = "IMPROVED" if tuned < orig else "NO GAIN"
        print("  {:<10} {:>15.2f} {:>15.2f} {:>13.1f}% {}".format(
            "{}h".format(h), orig, tuned, imp, status))

    # Save results
    save_results(all_results, "tuned_lightgbm_results.csv")

    print("\nLightGBM tuning complete!")
    print("   Models: models/lightgbm_tuned_*.joblib")

    return all_results


if __name__ == "__main__":
    run_tuning()