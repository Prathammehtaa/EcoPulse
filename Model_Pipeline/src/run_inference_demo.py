"""
EcoPulse — Inference Pipeline End-to-End Demo
===============================================
Demonstrates the FULL inference flow:
    1. Load trained models
    2. Load recent data (using test set as a stand-in for real-time)
    3. Build features
    4. Predict carbon intensity for all 4 horizons
    5. Find green windows
    6. Schedule a workload
    7. Print the recommendation

This is what happens behind the FastAPI endpoint.

Run from: Model_Pipeline/src/
Usage: python run_inference_demo.py
"""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging

from inference.predict import CarbonPredictor
from inference.feature_builder import FeatureBuilder
from inference.green_window import GreenWindowDetector, WorkloadScheduler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("ECOPULSE — INFERENCE PIPELINE DEMO")
    print("=" * 80)

    # ──────────────────────────────────────────────────────
    # STEP 1: Load the trained models
    # ──────────────────────────────────────────────────────
    print("\n📦 Step 1: Loading trained models...")
    predictor = CarbonPredictor()
    
    model_info = predictor.get_model_info()
    for h, info in model_info.items():
        print(f"   {h}: {info['model_type']} ({info['n_features']} features)")

    # ──────────────────────────────────────────────────────
    # STEP 2: Load recent data
    # In production, this would come from:
    #   - Electricity Maps API (real-time grid data)
    #   - Open-Meteo API (real-time weather data)
    #   - A cache of the last 8 days of hourly data
    #
    # For this demo, we use the TEST SET as a stand-in.
    # We take the last 200 rows (enough history for lags)
    # and pretend the LAST row is "right now."
    # ──────────────────────────────────────────────────────
    print("\n📊 Step 2: Loading recent data (test set as demo stand-in)...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_paths = [
        os.path.join(base_dir, "Data_Pipeline", "data", "processed",
                     "test_split.parquet"),
        os.path.join(base_dir, "Data_Pipeline", "data", "splits",
                     "test.parquet"),
    ]

    test_df = None
    for path in test_paths:
        if os.path.exists(path):
            test_df = pd.read_parquet(path)
            break

    if test_df is None:
        print("   ❌ No test data found. Cannot run demo.")
        return

    # Use one zone for demo, take last 200 rows (enough for lag_168h)
    demo_zone = "US-MIDA-PJM"
    zone_data = test_df[test_df["zone"] == demo_zone].copy()
    zone_data = zone_data.sort_values("datetime").tail(200)
    print(f"   Zone: {demo_zone}")
    print(f"   Data range: {zone_data['datetime'].min()} to {zone_data['datetime'].max()}")
    print(f"   Rows: {len(zone_data)}")

    # ──────────────────────────────────────────────────────
    # STEP 3: Predict carbon intensity
    # The test data already has features built, so we can
    # predict directly. In production, we'd call
    # FeatureBuilder.build_features() first.
    # ──────────────────────────────────────────────────────
    print("\n🔮 Step 3: Predicting carbon intensity...")

    # Predict the LAST 24 rows (simulating a 24-hour forecast)
    last_24 = zone_data.tail(24).copy()

    # Get 1h-ahead predictions for each of those 24 hours
    predictions_1h = predictor.predict(last_24, horizon=1)
    last_24["predicted_carbon_intensity"] = np.round(predictions_1h, 2)

    print(f"\n   24-Hour Forecast for {demo_zone}:")
    print(f"   {'Hour':<22} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print(f"   {'-'*55}")

    target_col = "carbon_intensity_target_1h"
    for _, row in last_24.iterrows():
        actual = row[target_col] if target_col in last_24.columns else "N/A"
        pred = row["predicted_carbon_intensity"]
        error = f"{pred - actual:.1f}" if isinstance(actual, (int, float)) else "—"
        actual_str = f"{actual:.1f}" if isinstance(actual, (int, float)) else actual
        print(f"   {str(row['datetime']):<22} {actual_str:>10} {pred:>10.1f} {error:>10}")

    # ──────────────────────────────────────────────────────
    # STEP 4: Find green windows
    # ──────────────────────────────────────────────────────
    print(f"\n🟢 Step 4: Identifying green windows...")

    detector = GreenWindowDetector(method="percentile", percentile=25)
    result = detector.find_green_windows(last_24)

    print(f"\n   Threshold: {result['threshold_gco2_kwh']} gCO₂/kWh")
    print(f"   Green hours: {result['green_hours']}/{result['total_hours']} "
          f"({result['green_pct']}%)")
    print(f"   Avg green intensity: {result['avg_green_intensity']} gCO₂/kWh")
    print(f"   Avg red intensity:   {result['avg_red_intensity']} gCO₂/kWh")
    print(f"   CO₂ savings potential: {result['co2_savings_pct']}%")

    if result["windows"]:
        print(f"\n   Green Windows Found:")
        for i, w in enumerate(result["windows"], 1):
            print(f"   Window {i}: {w['start_time']} → {w['end_time']} "
                  f"({w['duration_hours']}h, avg {w['avg_intensity']} gCO₂/kWh)")

    # ──────────────────────────────────────────────────────
    # STEP 5: Schedule a workload
    # ──────────────────────────────────────────────────────
    print(f"\n📋 Step 5: Scheduling a workload...")

    scheduler = WorkloadScheduler()
    schedule = scheduler.find_optimal_schedule(
        forecast=result["hourly"],
        runtime_hours=3,
        flexibility_hours=12,
        energy_kwh=150,
    )

    print(f"\n   Workload: 3-hour ML training job, 150 kWh, flexible 12 hours")
    print(f"   {'─'*60}")
    print(f"   If run NOW:")
    print(f"     Avg intensity: {schedule['immediate_intensity_gco2_kwh']} gCO₂/kWh")
    print(f"     CO₂ emitted:   {schedule['immediate_co2_kg']} kg")
    print(f"   ")
    print(f"   RECOMMENDED:")
    print(f"     Start at:      {schedule['recommended_start']}")
    print(f"     Wait:          {schedule['hours_to_wait']} hours")
    print(f"     Avg intensity: {schedule['expected_intensity_gco2_kwh']} gCO₂/kWh")
    print(f"     CO₂ emitted:   {schedule['optimal_co2_kg']} kg")
    print(f"     CO₂ SAVED:     {schedule['co2_saved_kg']} kg "
          f"({schedule['co2_savings_pct']}% reduction)")
    print(f"   ")
    print(f"   💡 {schedule['recommendation']}")

    # ──────────────────────────────────────────────────────
    # STEP 6: Multi-horizon forecast
    # ──────────────────────────────────────────────────────
    print(f"\n🎯 Step 6: Multi-horizon forecast (latest data point)...")

    latest_row = zone_data.tail(1)
    all_horizons = predictor.predict_all_horizons(latest_row)

    current_time = latest_row["datetime"].values[0]
    current_intensity = latest_row["carbon_intensity_gco2_per_kwh"].values[0]

    print(f"\n   Current time: {current_time}")
    print(f"   Current carbon intensity: {current_intensity:.1f} gCO₂/kWh")
    print(f"   ")
    print(f"   {'Horizon':<12} {'Predicted':>12} {'Direction':>12}")
    print(f"   {'─'*40}")

    for horizon, preds in all_horizons.items():
        pred = preds[0]
        direction = "📈 Rising" if pred > current_intensity else "📉 Falling"
        print(f"   {horizon}h ahead{'':<5} {pred:>10.1f}   {direction}")

    # ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("✅ INFERENCE PIPELINE DEMO COMPLETE")
    print(f"{'='*80}")
    print(f"\nThis is exactly what the FastAPI endpoint will do:")
    print(f"  POST /recommend → Step 5 output (scheduling recommendation)")
    print(f"  GET  /forecast  → Step 3 output (24h forecast)")
    print(f"  GET  /health    → Step 1 output (model info)")


if __name__ == "__main__":
    main()