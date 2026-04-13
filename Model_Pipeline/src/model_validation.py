"""
EcoPulse — Model Validation & Sensitivity Analysis (Person 4)
=============================================================
Validates tuned XGBoost models across all 4 forecast horizons.
Generates SHAP explanations, LIME local explanations,
hyperparameter sensitivity analysis, and full validation reports.

Run:
    cd ~/EcoPulse/Model_Pipeline/src
    python3 model_validation.py

Outputs (all saved to Model_Pipeline/reports/validation/):
    metrics_val.json / metrics_test.json       — MAE, RMSE, R2, MAPE
    confusion_matrix_<h>h.png                  — Bucket-level heatmap
    shap_summary_<h>h.png                      — SHAP bar plot
    shap_beeswarm_<h>h.png                     — SHAP beeswarm plot
    lime_sample_<h>h_<i>.html                  — LIME local explanations
    sensitivity_<h>h.png / sensitivity_<h>h.json — Hyperparam sensitivity
    validation_report.json                     — Master summary report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from utils import (
    load_all_splits, prepare_Xy, align_columns,
    compute_metrics, print_metrics, print_metrics_table,
    ensure_dirs, MODELS_DIR, REPORTS_DIR, HORIZONS,
    FORECAST_TARGETS, logger
)

# ============================================================
# CONFIG
# ============================================================
VALIDATION_DIR = os.path.join(REPORTS_DIR, "validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Best hyperparameters from Person 2 (best_hyperparameters.csv)
BEST_PARAMS = {
    1:  {"n_estimators": 1436, "max_depth": 10, "learning_rate": 0.0242,
         "subsample": 0.6867, "colsample_bytree": 0.9573,
         "min_child_weight": 18, "reg_alpha": 1.09, "reg_lambda": 2.02},
    6:  {"n_estimators": 1427, "max_depth": 8,  "learning_rate": 0.0434,
         "subsample": 0.7574, "colsample_bytree": 0.6609,
         "min_child_weight": 10, "reg_alpha": 1.52, "reg_lambda": 3.24},
    12: {"n_estimators": 1400, "max_depth": 8,  "learning_rate": 0.0488,
         "subsample": 0.7064, "colsample_bytree": 0.6895,
         "min_child_weight": 14, "reg_alpha": 0.318, "reg_lambda": 2.65},
    24: {"n_estimators": 1248, "max_depth": 10, "learning_rate": 0.0179,
         "subsample": 0.5176, "colsample_bytree": 0.6068,
         "min_child_weight": 4,  "reg_alpha": 0.094, "reg_lambda": 3.82},
}

# Baseline MAE from full_comparison.csv
BASELINE_MAE = {1: 57.48, 6: 71.33, 12: 76.36, 24: 68.79}

# Carbon intensity bucket edges (gCO2/kWh)
BUCKET_EDGES  = [0, 100, 200, 350, 500, float("inf")]
BUCKET_LABELS = ["Very Low\n(<100)", "Low\n(100-200)",
                 "Medium\n(200-350)", "High\n(350-500)", "Very High\n(500+)"]

LIME_SAMPLES = 3


# ============================================================
# HELPERS
# ============================================================
def load_model(horizon):
    path = os.path.join(MODELS_DIR, f"xgboost_tuned_{horizon}h.joblib")
    if not os.path.exists(path):
        path = os.path.join(MODELS_DIR, f"xgboost_{horizon}h.joblib")
        logger.warning(f"Tuned model not found, using base model: {path}")
    model = joblib.load(path)
    logger.info(f"Loaded model: {path}")
    return model


def bucketize(values):
    return np.digitize(values, BUCKET_EDGES[1:-1])


def save_json(data, filename):
    path = os.path.join(VALIDATION_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved -> {path}")


# ============================================================
# 1. VALIDATION METRICS
# ============================================================
def run_metrics(model, X_val, y_val, X_test, y_test, horizon):
    logger.info(f"[{horizon}h] Computing metrics...")

    m_val  = compute_metrics(y_val,  model.predict(X_val))
    m_test = compute_metrics(y_test, model.predict(X_test))

    m_test["baseline_mae"]        = BASELINE_MAE[horizon]
    m_test["mae_improvement_pct"] = round(
        ((BASELINE_MAE[horizon] - m_test["mae"]) / BASELINE_MAE[horizon]) * 100, 2
    )

    print(f"\n  [{horizon}h] Metrics:")
    print_metrics(m_val,  f"    val ")
    print_metrics(m_test, f"    test")
    print(f"  [{horizon}h] Improvement over baseline: {m_test['mae_improvement_pct']:+.1f}%")

    return {"val": m_val, "test": m_test}


# ============================================================
# 2. CONFUSION MATRIX (bucket-level)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, horizon):
    logger.info(f"[{horizon}h] Plotting confusion matrix...")

    tb = bucketize(np.array(y_true))
    pb = bucketize(np.array(y_pred))

    n  = len(BUCKET_LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(tb, pb):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=BUCKET_LABELS, yticklabels=BUCKET_LABELS, ax=ax)
    ax.set_ylabel("True Bucket")
    ax.set_xlabel("Predicted Bucket")
    ax.set_title(f"Bucket Confusion Matrix - XGBoost {horizon}h")
    plt.tight_layout()
    path = os.path.join(VALIDATION_DIR, f"confusion_matrix_{horizon}h.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved -> {path}")


# ============================================================
# 3. SHAP FEATURE IMPORTANCE
# ============================================================
def run_shap(model, X_val, horizon):
    logger.info(f"[{horizon}h] Running SHAP analysis...")

    sample      = X_val.sample(min(500, len(X_val)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP Feature Importance - XGBoost {horizon}h")
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, f"shap_summary_{horizon}h.png"), dpi=150)
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    plt.title(f"SHAP Beeswarm - XGBoost {horizon}h")
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, f"shap_beeswarm_{horizon}h.png"), dpi=150)
    plt.close()

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:10]
    return [{"feature": sample.columns[i], "mean_shap": round(float(mean_shap[i]), 4)}
            for i in top_idx]


# ============================================================
# 4. LIME LOCAL EXPLANATIONS
# ============================================================
def run_lime(model, X_train, X_val, horizon):
    logger.info(f"[{horizon}h] Running LIME ({LIME_SAMPLES} samples)...")

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression",
        random_state=42,
    )

    for i in range(LIME_SAMPLES):
        exp = explainer.explain_instance(
            X_val.iloc[i].values,
            model.predict,
            num_features=15,
        )
        path = os.path.join(VALIDATION_DIR, f"lime_sample_{horizon}h_{i}.html")
        exp.save_to_file(path)
        logger.info(f"Saved -> {path}")


# ============================================================
# 5. HYPERPARAMETER SENSITIVITY ANALYSIS
# ============================================================
def run_sensitivity(X_train, y_train, horizon):
    logger.info(f"[{horizon}h] Running sensitivity analysis...")

    bp = BEST_PARAMS[horizon]

    space = {
        "n_estimators": [
            max(100, bp["n_estimators"] - 600),
            max(100, bp["n_estimators"] - 300),
            bp["n_estimators"],
            bp["n_estimators"] + 200,
            bp["n_estimators"] + 400,
        ],
        "max_depth": sorted(set([
            max(2, bp["max_depth"] - 3),
            max(2, bp["max_depth"] - 1),
            bp["max_depth"],
            bp["max_depth"] + 1,
            bp["max_depth"] + 2,
        ])),
        "learning_rate": [
            round(bp["learning_rate"] * 0.25, 4),
            round(bp["learning_rate"] * 0.5,  4),
            round(bp["learning_rate"],         4),
            round(bp["learning_rate"] * 2.0,   4),
            round(bp["learning_rate"] * 4.0,   4),
        ],
    }

    results = {}
    for param, values in space.items():
        scores = []
        for v in values:
            p = {**bp, "random_state": 42, "n_jobs": 1,
                 "n_estimators": min(bp["n_estimators"], 300)}
            v = min(v, 500) if param == "n_estimators" else v
            p[param] = v
            cv = cross_val_score(xgb.XGBRegressor(**p), X_train, y_train,
                                 cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
            scores.append(round(float(-cv.mean()), 4))
        results[param] = {"values": values, "mae_scores": scores}
        logger.info(f"  {param}: done")

    save_json(results, f"sensitivity_{horizon}h.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (param, data) in zip(axes, results.items()):
        ax.plot(data["values"], data["mae_scores"], marker="o", linewidth=2)
        ax.axvline(x=bp[param], color="red", linestyle="--",
                   alpha=0.7, label=f"Best: {bp[param]}")
        ax.set_title(f"{param}\n(horizon={horizon}h)")
        ax.set_xlabel(param)
        ax.set_ylabel("CV MAE (3-fold)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Hyperparameter Sensitivity - XGBoost {horizon}h", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, f"sensitivity_{horizon}h.png"), dpi=150)
    plt.close()

    return results


# ============================================================
# MASTER RUNNER
# ============================================================
def validate_all_horizons():
    print("=" * 80)
    print("ECOPULSE - MODEL VALIDATION & SENSITIVITY ANALYSIS")
    print("=" * 80)

    splits = load_all_splits()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    master_report = {}
    all_test_results = []

    for horizon in HORIZONS:
        print(f"\n{'='*80}")
        print(f"  VALIDATING: {horizon}h HORIZON")
        print(f"{'='*80}")

        X_train, y_train, _ = prepare_Xy(train_df, horizon)
        X_val,   y_val,   _ = prepare_Xy(val_df,   horizon)
        X_test,  y_test,  _ = prepare_Xy(test_df,  horizon)
        X_train, X_val, X_test, _ = align_columns(X_train, X_val, X_test)

        logger.info(f"Features: {X_train.shape[1]}, "
                    f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        model       = load_model(horizon)
        metrics     = run_metrics(model, X_val, y_val, X_test, y_test, horizon)
        plot_confusion_matrix(y_test, model.predict(X_test), horizon)
        top_feats   = run_shap(model, X_val, horizon)
        run_lime(model, X_train, X_val, horizon)
        sensitivity = run_sensitivity(X_train, y_train, horizon)

        master_report[f"{horizon}h"] = {
            "metrics":           metrics,
            "top_shap_features": top_feats,
        }

        t = metrics["test"]
        all_test_results.append({
            "model": f"XGBoost Tuned ({horizon}h)",
            "mae": t["mae"], "rmse": t["rmse"],
            "r2":  t["r2"],  "mape": t["mape"],
        })

    save_json(master_report, "validation_report.json")

    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY (TEST SET)")
    print(f"{'='*80}")
    print_metrics_table(all_test_results)

    print(f"\n{'='*80}")
    print("IMPROVEMENT OVER BASELINES")
    print(f"{'='*80}")
    for horizon in HORIZONS:
        m = master_report[f"{horizon}h"]["metrics"]["test"]
        print(f"  {horizon}h: MAE={m['mae']:.2f}  "
              f"Baseline={m['baseline_mae']:.2f}  "
              f"Improvement={m['mae_improvement_pct']:+.1f}%")

    print(f"\n✅ Validation complete!")
    print(f"   Reports saved to: {VALIDATION_DIR}")
    print(f"\n   Output files:")
    for f in sorted(os.listdir(VALIDATION_DIR)):
        print(f"     {f}")

    return master_report


if __name__ == "__main__":
    validate_all_horizons()