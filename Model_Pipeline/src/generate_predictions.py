"""
EcoPulse — Generate & Save Prediction Outputs
===============================================
Loads trained XGBoost models, runs predictions on test set,
and saves:
  1. test_predictions.csv     — full table of actual vs predicted per timestamp
  2. forecast_curves.png      — time-series plots of actual vs predicted (per zone, per horizon)
  3. scatter_actual_vs_pred.png — scatter plot showing prediction accuracy
  4. sample_24h_forecast.csv  — example 24-hour forecast window (demo-ready)

Run from: Model_Pipeline/src/
Usage: python generate_predictions.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# ============================================================
# CONFIG
# ============================================================
HORIZONS = [1, 6, 12, 24]
TARGET_COL = "carbon_intensity_gco2_per_kwh"
ZONE_COL = "zone"
DATETIME_COL = "datetime"

# Paths (adjust if running from different directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Model_Pipeline/
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data_Pipeline", "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Columns to drop before prediction (must match training)
DROP_COLS = ["datetime", "zone", "aws_region", "gcp_region", "azure_region"]
TARGET_COLS = [f"carbon_intensity_target_{h}h" for h in HORIZONS]


def load_test_data():
    """Load test split parquet."""
    # Try multiple possible paths
    for path in [
        os.path.join(DATA_DIR, "splits", "test.parquet"),
        os.path.join(DATA_DIR, "processed", "test_split.parquet"),
    ]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"Loaded test data: {df.shape} from {path}")
            return df
    raise FileNotFoundError("Cannot find test parquet file")


def prepare_features(df):
    """
    Prepare features exactly as done during training.
    Must match train_xgboost.py's prepare_features() logic.
    """
    X = df.copy()

    # One-hot encode zone
    if ZONE_COL in X.columns:
        zone_dummies = pd.get_dummies(X[ZONE_COL], prefix="zone")
        X = pd.concat([X, zone_dummies], axis=1)

    # Drop non-feature columns
    cols_to_drop = [c for c in DROP_COLS + TARGET_COLS + [TARGET_COL]
                    if c in X.columns]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    # Drop any remaining string columns
    string_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = X.drop(columns=string_cols, errors="ignore")

    return X


def load_model(horizon, model_type="xgboost_tuned"):
    """
    Load trained model. Try tuned first, fall back to base.
    """
    for prefix in [model_type, "xgboost", "lightgbm"]:
        path = os.path.join(MODELS_DIR, f"{prefix}_{horizon}h.joblib")
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"  Loaded: {prefix}_{horizon}h.joblib")
            return model, prefix
    raise FileNotFoundError(f"No model found for {horizon}h horizon")


def generate_all_predictions():
    """Generate predictions for all horizons on test set."""
    test_df = load_test_data()
    X_test = prepare_features(test_df)

    # Align columns with what the model expects
    all_predictions = []

    for horizon in HORIZONS:
        model, model_name = load_model(horizon)

        # Get model's expected features and align
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            # Add missing columns as 0, drop extra columns
            for col in expected_cols:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_aligned = X_test[expected_cols]
        else:
            X_aligned = X_test

        # Predict
        preds = model.predict(X_aligned)

        # Get actual values
        target_col = f"carbon_intensity_target_{horizon}h"
        actuals = test_df[target_col].values if target_col in test_df.columns else None

        # Build output dataframe
        pred_df = pd.DataFrame({
            "datetime": test_df[DATETIME_COL].values,
            "zone": test_df[ZONE_COL].values,
            "horizon_hours": horizon,
            "model": model_name,
            "actual_carbon_intensity": actuals,
            "predicted_carbon_intensity": np.round(preds, 2),
        })

        if actuals is not None:
            pred_df["error"] = np.round(pred_df["predicted_carbon_intensity"]
                                        - pred_df["actual_carbon_intensity"], 2)
            pred_df["abs_error"] = np.abs(pred_df["error"])

        all_predictions.append(pred_df)

    combined = pd.concat(all_predictions, ignore_index=True)
    return combined, test_df


def save_predictions_csv(predictions_df):
    """Save full predictions table."""
    path = os.path.join(REPORTS_DIR, "test_predictions.csv")
    predictions_df.to_csv(path, index=False)
    print(f"\n✅ Saved: {path}")
    print(f"   Rows: {len(predictions_df)}")
    print(f"   Columns: {list(predictions_df.columns)}")


def plot_forecast_curves(predictions_df):
    """
    Plot actual vs predicted carbon intensity over time.
    One subplot per horizon, colored by zone.
    THIS IS THE MAIN VISUAL PROOF.
    """
    zones = predictions_df["zone"].unique()
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 4 * len(HORIZONS)),
                              sharex=True)
    if len(HORIZONS) == 1:
        axes = [axes]

    colors = {"US-MIDA-PJM": "#e74c3c", "US-NW-PACW": "#3498db"}

    for i, horizon in enumerate(HORIZONS):
        ax = axes[i]
        h_data = predictions_df[predictions_df["horizon_hours"] == horizon].copy()
        h_data["datetime"] = pd.to_datetime(h_data["datetime"])
        h_data = h_data.sort_values("datetime")

        for zone in zones:
            z_data = h_data[h_data["zone"] == zone]
            color = colors.get(zone, "#2ecc71")

            # Actual (solid line)
            ax.plot(z_data["datetime"], z_data["actual_carbon_intensity"],
                    color=color, alpha=0.7, linewidth=1.0,
                    label=f"{zone} — Actual")

            # Predicted (dashed line)
            ax.plot(z_data["datetime"], z_data["predicted_carbon_intensity"],
                    color=color, alpha=0.9, linewidth=1.2, linestyle="--",
                    label=f"{zone} — Predicted")

        # Calculate MAE for title
        mae = h_data["abs_error"].mean()
        r2 = 1 - (np.sum((h_data["actual_carbon_intensity"] - h_data["predicted_carbon_intensity"])**2)
                   / np.sum((h_data["actual_carbon_intensity"] - h_data["actual_carbon_intensity"].mean())**2))

        ax.set_title(f"{horizon}h Forecast — MAE: {mae:.2f} gCO₂/kWh | R²: {r2:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Carbon Intensity\n(gCO₂/kWh)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[-1].set_xlabel("Date (Test Period: Oct–Dec 2025)", fontsize=11)
    plt.suptitle("EcoPulse — Actual vs Predicted Carbon Intensity (Test Set)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "forecast_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_scatter(predictions_df):
    """Scatter plot: Actual vs Predicted for each horizon."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 5))
    if len(HORIZONS) == 1:
        axes = [axes]

    for i, horizon in enumerate(HORIZONS):
        ax = axes[i]
        h_data = predictions_df[predictions_df["horizon_hours"] == horizon]

        actual = h_data["actual_carbon_intensity"]
        predicted = h_data["predicted_carbon_intensity"]

        ax.scatter(actual, predicted, alpha=0.3, s=8, color="#3498db")

        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5,
                label="Perfect prediction")

        mae = h_data["abs_error"].mean()
        ax.set_title(f"{horizon}h Horizon\nMAE = {mae:.2f}", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Actual (gCO₂/kWh)")
        ax.set_ylabel("Predicted (gCO₂/kWh)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("EcoPulse — Prediction Accuracy (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "scatter_actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def generate_sample_forecast(predictions_df):
    """
    Generate a sample 24-hour forecast window — the demo artifact.
    Shows: "If you asked EcoPulse at time T, here's what it would tell you."
    """
    # Pick a random day from the test period for PJM zone
    h1_data = predictions_df[
        (predictions_df["horizon_hours"] == 1)
        & (predictions_df["zone"] == "US-MIDA-PJM")
    ].copy()
    h1_data["datetime"] = pd.to_datetime(h1_data["datetime"])
    h1_data = h1_data.sort_values("datetime")

    # Pick a 24-hour window from the middle of the test period
    mid_idx = len(h1_data) // 2
    sample = h1_data.iloc[mid_idx : mid_idx + 24].copy()

    # Add green window identification
    threshold = sample["predicted_carbon_intensity"].quantile(0.25)
    sample["is_green_window"] = sample["predicted_carbon_intensity"] <= threshold
    sample["recommendation"] = sample["is_green_window"].map(
        {True: "✅ SCHEDULE NOW — Low carbon", False: "⏳ DEFER — High carbon"}
    )

    # Save
    path = os.path.join(REPORTS_DIR, "sample_24h_forecast.csv")
    sample[["datetime", "zone", "predicted_carbon_intensity",
            "actual_carbon_intensity", "is_green_window", "recommendation"]
    ].to_csv(path, index=False)
    print(f"✅ Saved: {path}")

    # Print it nicely
    print(f"\n{'='*90}")
    print("SAMPLE 24-HOUR FORECAST — US-MIDA-PJM (Demo-Ready)")
    print(f"{'='*90}")
    print(f"{'Hour':<22} {'Predicted':>12} {'Actual':>12} {'Green?':>8} {'Action'}")
    print("-" * 90)
    for _, row in sample.iterrows():
        icon = "🟢" if row["is_green_window"] else "🔴"
        print(f"{str(row['datetime']):<22} "
              f"{row['predicted_carbon_intensity']:>10.1f}  "
              f"{row['actual_carbon_intensity']:>10.1f}  "
              f"{icon:>6}   "
              f"{row['recommendation']}")

    # Green window summary
    green_hours = sample["is_green_window"].sum()
    avg_green = sample.loc[sample["is_green_window"],
                           "predicted_carbon_intensity"].mean()
    avg_red = sample.loc[~sample["is_green_window"],
                         "predicted_carbon_intensity"].mean()

    print(f"\n📊 Green windows found: {green_hours}/24 hours")
    if green_hours > 0 and not np.isnan(avg_red):
        savings_pct = ((avg_red - avg_green) / avg_red) * 100
        print(f"📊 Avg intensity in green windows: {avg_green:.1f} gCO₂/kWh")
        print(f"📊 Avg intensity in red windows:   {avg_red:.1f} gCO₂/kWh")
        print(f"📊 Potential CO₂ reduction by scheduling in green: {savings_pct:.1f}%")

    return sample


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ECOPULSE — GENERATING PREDICTION OUTPUTS")
    print("=" * 80)

    # 1. Generate predictions
    predictions, test_df = generate_all_predictions()

    # 2. Save full CSV
    save_predictions_csv(predictions)

    # 3. Plot forecast curves (actual vs predicted over time)
    print("\nGenerating forecast curve plots...")
    plot_forecast_curves(predictions)

    # 4. Plot scatter (actual vs predicted)
    print("Generating scatter plots...")
    plot_scatter(predictions)

    # 5. Generate sample 24h forecast (demo artifact)
    print("\nGenerating sample 24-hour forecast...")
    sample = generate_sample_forecast(predictions)

    print(f"\n{'='*80}")
    print("✅ ALL PREDICTION OUTPUTS GENERATED")
    print(f"{'='*80}")
    print(f"  📄 reports/test_predictions.csv         — Full predictions table")
    print(f"  📈 reports/forecast_curves.png           — Actual vs Predicted time series")
    print(f"  📊 reports/scatter_actual_vs_predicted.png — Scatter accuracy plots")
    print(f"  🟢 reports/sample_24h_forecast.csv       — Demo-ready 24h forecast")
    print(f"\nThese files are your VISUAL PROOF that the model works.")