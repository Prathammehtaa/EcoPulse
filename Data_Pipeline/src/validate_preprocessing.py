"""
EcoPulse - Preprocessing Validation Script
Checks that all outputs are correct and ready for modeling.
"""

import pandas as pd
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")

print("=" * 60)
print("ECOPULSE PREPROCESSING VALIDATION")
print("=" * 60)

# ============================================================
# 1. GRID DATA
# ============================================================
print("\n--- GRID DATA ---")
grid = pd.read_parquet("data/processed/grid_data_processed.parquet")

check("Grid file loads", True)
check("Grid has 2 zones", len(grid["zone"].unique()) == 2,
      f"Got {grid['zone'].unique().tolist()}")
check("Grid zones correct",
      set(grid["zone"].unique()) == {"US-MIDA-PJM", "US-NW-PACW"})
check("Grid no nulls", grid.isnull().sum().sum() == 0,
      f"Found {grid.isnull().sum().sum()} nulls")
check("Grid date starts 2024-01-01",
      str(grid["datetime"].min()).startswith("2024-01-01"))
check("Grid date ends 2025-12-31",
      str(grid["datetime"].max()).startswith("2025-12-31"))
check("Grid rows ~35000", 34000 < len(grid) < 36000,
      f"Got {len(grid)}")
check("Grid has cloud mapping columns",
      all(c in grid.columns for c in ["aws_region", "gcp_region", "azure_region"]))
check("Carbon intensity range valid",
      grid["carbon_intensity_gco2_per_kwh"].between(0, 1500).all())
check("Carbon free energy 0-100%",
      grid["carbon_free_energy_pct"].between(0, 100).all())

# Check equal rows per zone
zone_counts = grid.groupby("zone").size()
check("Equal rows per zone",
      zone_counts.nunique() == 1,
      f"Counts: {zone_counts.to_dict()}")

# Check hourly frequency (no gaps)
for zone in grid["zone"].unique():
    z = grid[grid["zone"] == zone].sort_values("datetime")
    diffs = z["datetime"].diff().dropna()
    check(f"Hourly frequency ({zone})",
          (diffs == pd.Timedelta(hours=1)).all(),
          f"Found non-hourly gaps")

# ============================================================
# 2. WEATHER DATA
# ============================================================
print("\n--- WEATHER DATA ---")
weather = pd.read_parquet("data/processed/weather_data_processed.parquet")

check("Weather file loads", True)
check("Weather has 2 zones", len(weather["zone"].unique()) == 2)
check("Weather no nulls", weather.isnull().sum().sum() == 0)
check("Weather same rows as grid", len(weather) == len(grid))
check("Weather has 7 features",
      len([c for c in weather.columns if c not in ("datetime", "zone")]) == 7)
check("Temperature range valid",
      weather["temperature_2m_c"].between(-50, 60).all())
check("Cloud cover 0-100%",
      weather["cloud_cover_pct"].between(0, 100).all())
check("Solar radiation >= 0",
      (weather["shortwave_radiation_wm2"] >= 0).all())

# ============================================================
# 3. MERGED DATA
# ============================================================
print("\n--- MERGED DATA ---")
merged = pd.read_parquet("data/processed/merged_dataset.parquet")

check("Merged file loads", True)
check("Merged no nulls", merged.isnull().sum().sum() == 0)
check("Merged has grid + weather columns",
      len(merged.columns) == len(grid.columns) + len(weather.columns) - 2,
      f"Expected {len(grid.columns) + len(weather.columns) - 2}, got {len(merged.columns)}")
check("No row loss in merge", len(merged) == len(grid),
      f"Grid={len(grid)}, Merged={len(merged)}")

# ============================================================
# 4. FEATURE TABLE
# ============================================================
print("\n--- FEATURE TABLE ---")
feat = pd.read_parquet("data/features/feature_table.parquet")

check("Feature table loads", True)
check("Feature table no nulls", feat.isnull().sum().sum() == 0,
      f"Found {feat.isnull().sum().sum()} nulls")
check("Feature table ~40 columns", 35 < len(feat.columns) < 50,
      f"Got {len(feat.columns)}")
check("Same rows as grid", len(feat) == len(grid))

# Check temporal features
check("Has hour_of_day", "hour_of_day" in feat.columns)
check("hour_of_day range 0-23",
      feat["hour_of_day"].between(0, 23).all())
check("Has day_of_week", "day_of_week" in feat.columns)
check("Has is_weekend", "is_weekend" in feat.columns)
check("Has season", "season" in feat.columns)
check("Has cyclical encoding", "hour_sin" in feat.columns and "hour_cos" in feat.columns)

# Check lag features
check("Has lag_1h", "lag_1h" in feat.columns)
check("Has lag_24h", "lag_24h" in feat.columns)
check("Has lag_168h (weekly)", "lag_168h" in feat.columns)

# Check rolling features
check("Has rolling_mean_4h", "rolling_mean_4h" in feat.columns)
check("Has rolling_mean_24h", "rolling_mean_24h" in feat.columns)
check("Has rolling_std_24h", "rolling_std_24h" in feat.columns)

# Check interaction features
check("Has solar_potential", "solar_potential" in feat.columns)
check("Has temp_demand_proxy", "temp_demand_proxy" in feat.columns)
check("Has carbon_change_1h", "carbon_change_1h" in feat.columns)

# Check target column exists
check("Target column exists",
      "carbon_intensity_gco2_per_kwh" in feat.columns)

# ============================================================
# 5. DATA QUALITY STATS
# ============================================================
print("\n--- DATA QUALITY SUMMARY ---")
target = feat["carbon_intensity_gco2_per_kwh"]
print(f"  Target (carbon intensity):")
print(f"    Mean:   {target.mean():.1f} gCO2/kWh")
print(f"    Std:    {target.std():.1f}")
print(f"    Min:    {target.min():.1f}")
print(f"    Max:    {target.max():.1f}")
print(f"    Median: {target.median():.1f}")

print(f"\n  By zone:")
for zone in feat["zone"].unique():
    z = feat[feat["zone"] == zone]
    t = z["carbon_intensity_gco2_per_kwh"]
    print(f"    {zone}: mean={t.mean():.1f}, std={t.std():.1f}, range=[{t.min():.1f}, {t.max():.1f}]")

# ============================================================
# FINAL RESULT
# ============================================================
print(f"\n{'=' * 60}")
print(f"RESULT: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
if FAIL == 0:
    print("ALL CHECKS PASSED - Preprocessing is correct!")
else:
    print(f"WARNING: {FAIL} check(s) failed - review above")
print(f"{'=' * 60}")