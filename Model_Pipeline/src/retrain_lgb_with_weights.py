import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, save_model, ensure_dirs,
    MODELS_DIR, REPORTS_DIR, logger
)

HORIZONS = [1, 6, 12, 24]
SEED = 42

def compute_weights(train_df, y_train):
    ci = train_df["carbon_intensity_gco2_per_kwh"].loc[y_train.index]
    weights = np.ones(len(y_train))
    weights[ci < 100] = 3.0
    weights[(ci >= 100) & (ci < 200)] = 1.5
    vl = (ci < 100).sum()
    print("  Weights: Very Low (3x)={:,}, Low (1.5x)={:,}, Normal={:,}".format(
        vl, ((ci >= 100) & (ci < 200)).sum(), (ci >= 200).sum()))
    return weights

def run():
    print("=" * 70)
    print("RETRAIN LIGHTGBM WITH SAMPLE WEIGHTS")
    print("Using best params from Optuna tuning")
    print("=" * 70)

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    ensure_dirs()

    # Load best params from tuned LightGBM results
    params_path = os.path.join(REPORTS_DIR, "tuned_lightgbm_results.csv")

    for horizon in HORIZONS:
        print("\n{}".format("=" * 70))
        print("  RETRAINING: {}h HORIZON (with sample weights)".format(horizon))
        print("=" * 70)

        X_train, y_train, _ = prepare_Xy(train, horizon)
        X_val, y_val, _ = prepare_Xy(val, horizon)
        X_test, y_test, _ = prepare_Xy(test, horizon)
        X_train, X_val, X_test, feature_cols = align_columns(X_train, X_val, X_test)

        print("  Train: {}, Val: {}, Test: {}".format(X_train.shape, X_val.shape, X_test.shape))

        # Load existing tuned model to get its params
        tuned_path = os.path.join(MODELS_DIR, "lightgbm_tuned_{}h.joblib".format(horizon))
        if os.path.exists(tuned_path):
            tuned_model = joblib.load(tuned_path)
            best_params = tuned_model.get_params()
            print("  Loaded params from tuned model")
        else:
            best_params = {
                "n_estimators": 500, "max_depth": 8,
                "learning_rate": 0.05, "num_leaves": 63,
                "random_state": SEED, "n_jobs": -1, "verbose": -1,
            }
            print("  Using default params (no tuned model found)")

        best_params["random_state"] = SEED
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        best_params["deterministic"] = True
        best_params["force_row_wise"] = True

        weights = compute_weights(train, y_train)

        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, sample_weight=weights)

        for split_name, X, y in [("train", X_train, y_train),
                                  ("val", X_val, y_val),
                                  ("test", X_test, y_test)]:
            preds = model.predict(X)
            metrics = compute_metrics(y, preds)
            print_metrics(metrics, "    {}".format(split_name))

        ci_test = test["carbon_intensity_gco2_per_kwh"].loc[y_test.index]
        vl_mask = ci_test < 100
        if vl_mask.sum() > 0:
            vl_preds = model.predict(X_test[vl_mask])
            vl_actual = y_test[vl_mask]
            vl_mae = np.mean(np.abs(vl_actual - vl_preds))
            overall_mae = np.mean(np.abs(y_test - model.predict(X_test)))
            deviation = (vl_mae - overall_mae) / overall_mae * 100
            print("\n    Very Low: n={}, MAE={:.2f}, Overall={:.2f}, Deviation={:+.1f}%".format(
                vl_mask.sum(), vl_mae, overall_mae, deviation))

        save_model(model, "lightgbm_tuned_{}h".format(horizon))
        print("  Saved: lightgbm_tuned_{}h.joblib".format(horizon))

    print("\n" + "=" * 70)
    print("DONE — All LightGBM models retrained with sample weights")
    print("=" * 70)

if __name__ == "__main__":
    run()