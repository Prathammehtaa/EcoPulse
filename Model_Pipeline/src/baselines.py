"""
EcoPulse Baseline Models
========================
Establishes naive benchmarks that XGBoost/LightGBM must beat.
If a fancy model can't beat these, the features aren't useful.

Three baselines:
1. Naive Persistence: "future = current value"
2. 24h-Ago Persistence: "future = same hour yesterday"  
3. Historical Hourly Mean: "future = avg for this (zone, hour)"

Run: cd Model_Pipeline && python src/01_baselines.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils import (
    load_all_splits, compute_metrics, print_metrics_table,
    save_results, ensure_dirs,
    TARGET_COL, DATETIME_COL, ZONE_COL, FORECAST_TARGETS, HORIZONS,
    logger
)


def naive_persistence(test_df: pd.DataFrame) -> list:
    """
    Baseline 1: Predict future carbon intensity = current carbon intensity.
    
    WHY THIS WORKS AS A BASELINE:
    Carbon intensity changes slowly hour-to-hour, so "no change" is 
    surprisingly accurate for short horizons (1h) but terrible for 
    long horizons (24h). Our ML model MUST beat this.
    """
    results = []
    for h in HORIZONS:
        target_col = FORECAST_TARGETS[h]
        y_true = test_df[target_col].values
        y_pred = test_df[TARGET_COL].values  # current value as prediction
        
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = f"Naive Persistence ({h}h)"
        metrics["horizon"] = h
        metrics["baseline_type"] = "naive_persistence"
        results.append(metrics)
    
    return results


def lag24h_persistence(test_df: pd.DataFrame) -> list:
    """
    Baseline 2: Predict future = carbon intensity 24 hours ago.
    
    WHY THIS WORKS:
    Grid carbon intensity has strong daily seasonality — the energy 
    mix at 3pm today is similar to 3pm yesterday. This captures 
    that pattern without any ML.
    """
    # Find the lag_24h column
    lag_24h_col = [c for c in test_df.columns 
                   if "carbon_intensity" in c and "lag_24h" in c][0]
    
    results = []
    for h in HORIZONS:
        target_col = FORECAST_TARGETS[h]
        y_true = test_df[target_col].values
        y_pred = test_df[lag_24h_col].values
        
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = f"24h-Ago Persistence ({h}h)"
        metrics["horizon"] = h
        metrics["baseline_type"] = "lag24h_persistence"
        results.append(metrics)
    
    return results


def historical_hourly_mean(train_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> list:
    """
    Baseline 3: Predict future = historical average for this (zone, hour).
    
    WHY THIS WORKS:
    Each zone has a characteristic daily carbon curve — PJM peaks 
    in the evening, PACW is lower overall due to hydro. Computing 
    the mean per (zone, hour_of_day) captures this pattern.
    
    IMPORTANT: Means are computed from TRAINING data only to avoid leakage.
    """
    # Compute historical means from training data
    zone_hour_means = (
        train_df.groupby([ZONE_COL, "hour_of_day"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "hist_mean"})
    )
    
    # Merge onto test data
    test_with_mean = test_df.merge(
        zone_hour_means, on=[ZONE_COL, "hour_of_day"], how="left"
    )
    
    results = []
    for h in HORIZONS:
        target_col = FORECAST_TARGETS[h]
        y_true = test_with_mean[target_col].values
        y_pred = test_with_mean["hist_mean"].values
        
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = f"Historical Hourly Mean ({h}h)"
        metrics["horizon"] = h
        metrics["baseline_type"] = "historical_mean"
        results.append(metrics)
    
    return results


def run_all_baselines():
    """Run all baselines and output comparison table."""
    print("=" * 80)
    print("ECOPULSE BASELINE MODELS")
    print("=" * 80)
    
    # Load data
    splits = load_all_splits()
    train, test = splits["train"], splits["test"]
    
    # Run baselines
    print("\nRunning Naive Persistence...")
    r1 = naive_persistence(test)
    
    print("Running 24h-Ago Persistence...")
    r2 = lag24h_persistence(test)
    
    print("Running Historical Hourly Mean...")
    r3 = historical_hourly_mean(train, test)
    
    # Combine results
    all_results = r1 + r2 + r3
    
    # Print grouped by horizon
    for h in HORIZONS:
        print(f"\n{'='*80}")
        print(f"  HORIZON: {h}h AHEAD")
        print(f"{'='*80}")
        horizon_results = [r for r in all_results if r["horizon"] == h]
        print_metrics_table(horizon_results)
        
        # Identify best baseline for this horizon
        best = min(horizon_results, key=lambda x: x["mae"])
        print(f"\n  → Best baseline: {best['model']} (MAE={best['mae']:.2f})")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("FULL BASELINE SUMMARY (TEST SET)")
    print(f"{'='*80}")
    print_metrics_table(all_results)
    
    # Save
    ensure_dirs()
    save_results(all_results, "baseline_results.csv")
    
    # Print what the ML models need to beat
    print(f"\n{'='*80}")
    print("TARGETS FOR ML MODELS (must beat best baseline per horizon)")
    print(f"{'='*80}")
    for h in HORIZONS:
        horizon_results = [r for r in all_results if r["horizon"] == h]
        best = min(horizon_results, key=lambda x: x["mae"])
        print(f"  {h}h horizon: beat MAE < {best['mae']:.2f} "
              f"(currently: {best['model']})")
    
    print(f"\n✅ Baseline evaluation complete!")
    print(f"   Results saved to: reports/baseline_results.csv")
    
    return all_results


if __name__ == "__main__":
    results = run_all_baselines()