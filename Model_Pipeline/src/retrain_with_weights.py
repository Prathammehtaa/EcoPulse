"""
Retrain tuned XGBoost models with sample weights
to reduce Very Low bucket bias without re-running Optuna.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, save_model, ensure_dirs,
    MODELS_DIR, REPORTS_DIR, logger
)

HORIZONS = [1, 6, 12, 24]
SEED = 42

def compute_weights(train_df, y_train):
    """Give higher weight to Very Low carbon rows."""
    ci = train_df["carbon_intensity_gco2_per_kwh"].loc[y_train.index]
    weights = np.ones(len(y_train))
    weights[ci < 100] = 3.0       # 3x for Very Low
    weights[(ci >= 100) & (ci < 200)] = 1.5  # 1.5x for Low
    
    vl_count = (ci < 100).sum()
    print("  Sample weights: Very Low (3x) = {:,}, Low (1.5x) = {:,}, Normal (1x) = {:,}".format(
        vl_count, ((ci >= 100) & (ci < 200)).sum(), (ci >= 200).sum()))
    return weights

def run():
    print("=" * 80)
    print("RETRAIN XGBOOST WITH SAMPLE WEIGHTS")
    print("Using existing best params from Optuna tuning")
    print("=" * 80)

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    ensure_dirs()

    # Load existing best hyperparameters
    params_path = os.path.join(REPORTS_DIR, "best_hyperparameters.csv")
    if os.path.exists(params_path):
        params_df = pd.read_csv(params_path)
        print("  Loaded best params from: {}".format(params_path))
    else:
        print("  No best_hyperparameters.csv found — using defaults")
        params_df = None

    for horizon in HORIZONS:
        print("\n{}".format("=" * 80))
        print("  RETRAINING: {}h HORIZON (with sample weights)".format(horizon))
        print("=" * 80)

        X_train, y_train, _ = prepare_Xy(train, horizon)
        X_val, y_val, _ = prepare_Xy(val, horizon)
        X_test, y_test, _ = prepare_Xy(test, horizon)
        X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

        print("  Train: {}, Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))

        # Get best params for this horizon
        if params_df is not None and horizon in params_df["horizon"].values:
            row = params_df[params_df["horizon"] == horizon].iloc[0]
            best_params = {k: v for k, v in row.to_dict().items() 
                          if k != "horizon" and pd.notna(v)}
            # Fix types
            for k in ["n_estimators", "max_depth", "min_child_weight"]:
                if k in best_params:
                    best_params[k] = int(best_params[k])
        else:
            best_params = {}

        best_params["random_state"] = SEED
        best_params["seed"] = SEED
        best_params["n_jobs"] = -1
        best_params["early_stopping_rounds"] = 30

        print("  Params: {}".format({k: v for k, v in best_params.items() 
              if k not in ["random_state", "seed", "n_jobs", "early_stopping_rounds"]}))

        # Compute sample weights
        weights = compute_weights(train, y_train)

        # Train with weights
        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        for split_name, X, y in [("train", X_train, y_train),
                                  ("val", X_val, y_val),
                                  ("test", X_test, y_test)]:
            preds = model.predict(X)
            metrics = compute_metrics(y, preds)
            print_metrics(metrics, "    {}".format(split_name))

        # Check Very Low specifically
        ci_test = test["carbon_intensity_gco2_per_kwh"].loc[y_test.index]
        vl_mask = ci_test < 100
        if vl_mask.sum() > 0:
            vl_preds = model.predict(X_test[vl_mask])
            vl_actual = y_test[vl_mask]
            vl_mae = np.mean(np.abs(vl_actual - vl_preds))
            overall_mae = np.mean(np.abs(y_test - model.predict(X_test)))
            deviation = (vl_mae - overall_mae) / overall_mae * 100
            print("\n    Very Low bucket: n={}, MAE={:.2f}, Overall MAE={:.2f}, Deviation={:+.1f}%".format(
                vl_mask.sum(), vl_mae, overall_mae, deviation))

        # Save — overwrite the tuned model
        save_model(model, "xgboost_tuned_{}h".format(horizon))
        print("  Saved: xgboost_tuned_{}h.joblib".format(horizon))

    print("\n" + "=" * 80)
    print("DONE — All XGBoost models retrained with sample weights")
    print("=" * 80)

if __name__ == "__main__":
    run()