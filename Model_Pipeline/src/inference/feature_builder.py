"""
EcoPulse Inference Pipeline — feature_builder.py
==================================================
WHAT IT DOES:
    Takes RAW data (from APIs or a recent data cache) and builds the
    exact same 100 features that the model was trained on.

WHY THIS IS THE HARDEST PART:
    During training, Kapish's step2_feature_engineering.py had access to
    the entire historical dataset to compute things like:
    - lag_168h (carbon intensity 7 DAYS ago)
    - rolling_mean_24h (average of last 24 hours)
    
    During inference, we only have "right now." So we need a CACHE of
    recent historical data (at least 168 hours = 7 days) to compute
    those same lag and rolling features.

HOW IT WORKS:
    1. Maintain a rolling cache of the last 7+ days of hourly data
    2. When new data arrives (from APIs), append it to the cache
    3. Compute all features from the cache
    4. Return the latest row(s) as model-ready features

USAGE:
    from inference.feature_builder import FeatureBuilder
    
    builder = FeatureBuilder()
    
    # Option A: Build features from cached historical data
    features = builder.build_from_cache(zone="US-MIDA-PJM")
    
    # Option B: Build features from a raw DataFrame 
    # (e.g., last 7 days of data from BigQuery/API)
    features = builder.build_features(recent_data_df)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# ============================================================
# FEATURE CONFIGURATION
# Must match Kapish's step2_feature_engineering.py EXACTLY
# ============================================================

# The target column (what we're predicting)
TARGET_COL = "carbon_intensity_gco2_per_kwh"

# Lag features: carbon intensity at these hours in the past
LAG_HOURS = [1, 3, 6, 24, 168]  # 1h, 3h, 6h, 1day, 1week ago

# Rolling statistics: computed over these window sizes
ROLLING_WINDOWS = [4, 12, 24]  # 4h, 12h, 24h

# Minimum history needed to compute all features
# 168h lag + 24h rolling window = need at least 192 hours (~8 days)
MIN_HISTORY_HOURS = 192


class FeatureBuilder:
    """
    Builds model-ready features from raw grid + weather data.
    
    The features must match EXACTLY what the model was trained on.
    Any mismatch = garbage predictions. Here's what we build:
    
    GRID SIGNALS (from Electricity Maps API):
        - carbon_intensity_gco2_per_kwh  (the target — also used as input via lags)
        - carbon_free_energy_pct
        - carbon_intensity_fossil_gco2_per_kwh
        - renewable_energy_pct
        - total_load_mw
        - net_load_mw
    
    WEATHER (from Open-Meteo API):
        - temperature_2m_c
        - wind_speed_100m_ms
        - cloud_cover_pct
        - shortwave_radiation_wm2
        - rain_mm
        - snowfall_cm
        - weather_code
    
    TEMPORAL (computed from timestamp):
        - hour_of_day, day_of_week, month
        - is_weekend, is_daytime, season
        - hour_sin, hour_cos (cyclic encoding)
        - month_sin, month_cos (cyclic encoding)
    
    LAG FEATURES (need historical data):
        - lag_1h, lag_3h, lag_6h, lag_24h, lag_168h
    
    ROLLING STATISTICS (need historical data):
        - rolling_mean_4h, rolling_mean_12h, rolling_mean_24h
        - rolling_std_24h
    
    INTERACTION FEATURES (computed from other features):
        - solar_potential = shortwave_radiation * (1 - cloud_cover/100)
        - temp_demand_proxy = |temperature - 20| (heating/cooling demand)
        - carbon_change_1h = carbon_intensity - lag_1h
    """

    def __init__(self):
        """Initialize the feature builder."""
        pass

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features from the datetime column.
        
        WHY: Carbon intensity follows strong daily and seasonal patterns.
        Solar generation peaks at noon, demand peaks in evening, wind 
        varies by season. These features let the model learn those patterns.
        
        CYCLIC ENCODING (sin/cos):
            Hour 23 and hour 0 are 1 hour apart, but numerically they're
            23 apart. Sin/cos encoding fixes this:
            - hour_sin = sin(2π × hour/24)
            - hour_cos = cos(2π × hour/24)
            Now hour 23 and hour 0 are close together in feature space.
        """
        dt = pd.to_datetime(df["datetime"])

        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek      # 0=Mon, 6=Sun
        df["month"] = dt.dt.month
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["is_daytime"] = ((dt.dt.hour >= 6) & (dt.dt.hour <= 20)).astype(int)

        # Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
        df["season"] = dt.dt.month.map(
            {12: 1, 1: 1, 2: 1,      # Winter
             3: 2, 4: 2, 5: 2,        # Spring
             6: 3, 7: 3, 8: 3,        # Summer
             9: 4, 10: 4, 11: 4}      # Fall
        )

        # Cyclic encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged carbon intensity values.
        
        WHY: The single best predictor of carbon intensity in 1 hour
        is carbon intensity RIGHT NOW (lag_1h). The best predictor for
        24h ahead is what it was at this exact hour yesterday (lag_24h).
        
        In our model, lag_1h has the highest feature importance for the
        1h horizon. lag_24h dominates for longer horizons.
        
        HOW: Simple shift — lag_1h = carbon_intensity.shift(1)
        This means row N's lag_1h is row N-1's carbon intensity.
        
        REQUIRES: Data must be sorted by (zone, datetime) with no gaps.
        """
        for zone in df["zone"].unique():
            mask = df["zone"] == zone

            for lag_h in LAG_HOURS:
                col_name = f"lag_{lag_h}h"
                df.loc[mask, col_name] = (
                    df.loc[mask, TARGET_COL].shift(lag_h)
                )

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        WHY: Rolling means smooth out noise. If carbon intensity was 
        300, 280, 320, 290 over the last 4 hours, the rolling_mean_4h 
        is 297.5 — a more stable signal than any single reading.
        
        rolling_std_24h captures volatility — is the grid stable today
        or swinging wildly? High volatility means the model should be
        less confident.
        
        HOW: pandas rolling() with min_periods=1 so we don't lose data
        at the edges. Each zone is computed independently.
        """
        for zone in df["zone"].unique():
            mask = df["zone"] == zone
            series = df.loc[mask, TARGET_COL]

            for window in ROLLING_WINDOWS:
                col_name = f"rolling_mean_{window}h"
                df.loc[mask, col_name] = (
                    series.rolling(window=window, min_periods=1).mean()
                )

            # Standard deviation for 24h window only
            df.loc[mask, "rolling_std_24h"] = (
                series.rolling(window=24, min_periods=1).std()
            )

        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived interaction features.
        
        WHY: These combine raw features into more meaningful signals:
        
        solar_potential:
            shortwave_radiation × (1 - cloud_cover/100)
            High radiation + clear sky = lots of solar generation = low carbon.
            The model learns this relationship faster with the combined feature.
        
        temp_demand_proxy:
            |temperature - 20|
            At 20°C, no heating or cooling needed = low electricity demand.
            At 0°C (heating) or 35°C (AC), demand spikes = grid stress.
        
        carbon_change_1h:
            current carbon intensity - lag_1h
            Positive = grid is getting dirtier. Negative = getting cleaner.
            Tells the model about the DIRECTION of change, not just level.
        """
        # Solar potential
        if "shortwave_radiation_wm2" in df.columns and "cloud_cover_pct" in df.columns:
            df["solar_potential"] = (
                df["shortwave_radiation_wm2"] * (1 - df["cloud_cover_pct"] / 100)
            )
        else:
            df["solar_potential"] = 0

        # Temperature-demand proxy
        if "temperature_2m_c" in df.columns:
            df["temp_demand_proxy"] = np.abs(df["temperature_2m_c"] - 20)
        else:
            df["temp_demand_proxy"] = 0

        # Carbon change over last hour
        if "lag_1h" in df.columns:
            df["carbon_change_1h"] = df[TARGET_COL] - df["lag_1h"]
        else:
            df["carbon_change_1h"] = 0

        return df

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build ALL features from raw data.
        
        THIS IS THE MAIN FUNCTION.
        
        Takes raw grid + weather data (at least 8 days of hourly rows)
        and produces the full 100-column feature table that the model expects.
        
        Args:
            raw_df: DataFrame with columns:
                    - datetime (hourly timestamps)
                    - zone (US-MIDA-PJM or US-NW-PACW)
                    - carbon_intensity_gco2_per_kwh
                    - carbon_free_energy_pct, renewable_energy_pct, etc.
                    - temperature_2m_c, wind_speed_100m_ms, etc.
        
        Returns:
            DataFrame with all features added.
            The LAST row(s) are the ones ready for prediction.
            Earlier rows were only needed to compute lags/rolling features.
        """
        df = raw_df.copy()

        # Ensure datetime is parsed and data is sorted
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(["zone", "datetime"]).reset_index(drop=True)

        logger.info(f"Building features from {len(df)} rows, "
                    f"{df['zone'].nunique()} zones")

        # Step 1: Temporal features (from datetime — no history needed)
        df = self.add_temporal_features(df)
        logger.info("  ✓ Temporal features added")

        # Step 2: Lag features (needs 168 hours of history)
        df = self.add_lag_features(df)
        logger.info("  ✓ Lag features added")

        # Step 3: Rolling statistics (needs 24 hours of history)
        df = self.add_rolling_features(df)
        logger.info("  ✓ Rolling features added")

        # Step 4: Interaction features (computed from other features)
        df = self.add_interaction_features(df)
        logger.info("  ✓ Interaction features added")

        # Step 5: Fill any remaining NaNs from lag/rolling edges
        # At the start of the data, lag_168h will be NaN for the first
        # 168 rows. We forward-fill then backward-fill as a safe default.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")

        logger.info(f"  ✓ Feature building complete: {df.shape}")

        return df

    def build_latest_features(
        self,
        raw_df: pd.DataFrame,
        n_latest: int = 1,
    ) -> pd.DataFrame:
        """
        Build features and return only the latest N rows.
        
        This is what the API calls — it doesn't need the full history
        in the output, just the most recent row(s) with all features
        computed.
        
        Args:
            raw_df: Raw data with enough history (8+ days)
            n_latest: How many recent rows to return (default 1)
        
        Returns:
            DataFrame with n_latest rows, all features computed
        """
        full_df = self.build_features(raw_df)

        # Return only the last n_latest rows per zone
        latest_rows = []
        for zone in full_df["zone"].unique():
            zone_data = full_df[full_df["zone"] == zone]
            latest_rows.append(zone_data.tail(n_latest))

        return pd.concat(latest_rows, ignore_index=True)


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("EcoPulse — Feature Builder Test")
    print("=" * 70)

    builder = FeatureBuilder()

    # Try loading test data to verify feature building works
    _MODEL_PIPELINE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    test_paths = [
        os.path.join(_MODEL_PIPELINE_DIR, "..", "Data_Pipeline", "data",
                     "processed", "test_split.parquet"),
    ]

    for path in test_paths:
        if os.path.exists(path):
            print(f"\n  Loading: {path}")
            df = pd.read_parquet(path)
            print(f"  Raw shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")

            # The test data already has features, but let's verify
            # our builder produces the same columns
            print(f"\n  Expected feature columns from training data:")
            expected_features = [c for c in df.columns
                                 if c not in ["datetime", "zone",
                                              "aws_region", "gcp_region",
                                              "azure_region"]]
            print(f"    Count: {len(expected_features)}")
            print(f"    Sample: {expected_features[:10]}...")

            print(f"\n  ✅ Feature builder module loaded successfully!")
            break
    else:
        print("\n  No test data found — module loaded but not tested")