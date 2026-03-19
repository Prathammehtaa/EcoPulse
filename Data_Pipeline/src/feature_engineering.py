"""
Step 2: Feature Engineering
===========================
Adds temporal, lag, rolling, and interaction features to raw_combined.parquet
Output: feature_table.parquet (~40+ columns)
"""
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURE_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "pipeline_config" / "preprocessing_config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_raw_data():
    """Load deduplicated dataset."""
    input_path = os.path.join(PROCESSED_DIR, 'raw_combined.parquet')
    print(f'Loading: {input_path}')
    df = pd.read_parquet(input_path)
    print(f'  Loaded {len(df)} rows')
    return df


def add_temporal_features(df):
    """Add temporal features."""
    print('\n--- ADDING TEMPORAL FEATURES ---')

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Basic temporal
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['year'] = df['datetime'].dt.year

    # Binary flags
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Sat=5, Sun=6
    df['is_daytime'] = ((df['hour_of_day'] >= 6) & (
        df['hour_of_day'] < 18)).astype(int)

    # Season
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['season'] = df['month'].map(season_map)

    print(f'  Added 13 temporal features')
    return df


def add_cyclical_encodings(df):
    """Add cyclical encodings for circular features."""
    print('\n--- ADDING CYCLICAL ENCODINGS ---')

    # Hour cyclical (0-23 hours)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # Month cyclical (1-12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day-of-week cyclical (0-6 days)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print(f'  Added 6 cyclical encodings')
    return df


def add_lag_features(df, lags=[1, 3, 6, 24, 168]):
    """Add lagged features (per zone)."""
    print('\n--- ADDING LAG FEATURES ---')

    lag_cols = ['carbon_intensity_gco2_per_kwh', 'total_load_mw',
                'temperature_2m_c', 'wind_speed_100m_ms']

    for zone in df['zone'].unique():
        zone_mask = df['zone'] == zone
        zone_data = df.loc[zone_mask].copy()

        for col in lag_cols:
            for lag in lags:
                feature_name = f'{col}_lag_{lag}h'
                df.loc[zone_mask, feature_name] = zone_data[col].shift(
                    lag).values

    print(
        f'  Added {len(lag_cols)} × {len(lags)} = {len(lag_cols) * len(lags)} lag features')
    return df


def add_rolling_features(df, windows=[4, 12, 24]):
    """Add rolling statistics (per zone)."""
    print('\n--- ADDING ROLLING FEATURES ---')

    rolling_cols = ['carbon_intensity_gco2_per_kwh',
                    'total_load_mw', 'temperature_2m_c']

    for zone in df['zone'].unique():
        zone_mask = df['zone'] == zone
        zone_data = df.loc[zone_mask].copy()

        for col in rolling_cols:
            for window in windows:
                # Mean
                feature_name = f'{col}_mean_{window}h'
                df.loc[zone_mask, feature_name] = zone_data[col].rolling(
                    window=window, min_periods=1).mean().values

                # Std dev
                feature_name = f'{col}_std_{window}h'
                df.loc[zone_mask, feature_name] = zone_data[col].rolling(
                    window=window, min_periods=1).std().values

                # Min/Max
                feature_name = f'{col}_min_{window}h'
                df.loc[zone_mask, feature_name] = zone_data[col].rolling(
                    window=window, min_periods=1).min().values

                feature_name = f'{col}_max_{window}h'
                df.loc[zone_mask, feature_name] = zone_data[col].rolling(
                    window=window, min_periods=1).max().values

    print(f'  Added {len(rolling_cols)} × {len(windows)} × 4 stats = {len(rolling_cols) * len(windows) * 4} rolling features')
    return df


def add_interaction_features(df):
    """Add domain-specific interaction features."""
    print('\n--- ADDING INTERACTION FEATURES ---')

    # Solar potential proxy (daytime + clear weather)
    df['solar_potential'] = df['is_daytime'] * \
        (100 - df['cloud_cover_pct']) / 100

    # Temperature-demand proxy
    df['temp_demand_proxy'] = df['total_load_mw'] * \
        np.abs(df['temperature_2m_c'] - 15) / 100

    # Carbon change rate (1h)
    df['carbon_change_1h'] = df.groupby(
        'zone')['carbon_intensity_gco2_per_kwh'].diff().fillna(0)

    # Clean energy availability (renewable + carbon-free)
    df['clean_energy_score'] = (
        df['carbon_free_energy_pct'] + df['renewable_energy_pct']) / 2

    # Load volatility (proxy)
    df['load_variability'] = df.groupby('zone')['total_load_mw'].transform(
        lambda x: x.rolling(24).std()).fillna(0)

    print(f'  Added 5 interaction features')
    return df


def handle_missing_values(df):
    """Handle NaN values from lags and rolling windows."""
    print('\n--- HANDLING MISSING VALUES ---')

    # Forward fill for lag features within each zone
    for zone in df['zone'].unique():
        zone_mask = df['zone'] == zone
        df.loc[zone_mask] = df.loc[zone_mask].fillna(
            method='bfill').fillna(method='ffill')

    # Fill any remaining with 0
    df = df.fillna(0)

    nulls_remaining = df.isnull().sum().sum()
    print(f'  Null values after handling: {nulls_remaining}')

    return df


def save_output(config, df):
    print('\n--- SAVING OUTPUT ---')

    output_path = os.path.join(FEATURE_DIR, 'feature_table.parquet')
    df.to_parquet(output_path, index=False)

    proc_dir = f"gs://{config['gcs']['bucket']}/{config['gcs']['paths']['features']}"
    feature_path = f"{proc_dir}/{config['output']['files']['feature_table']}"
    df.to_parquet(feature_path, index=False)

    print(f'  Saved: {output_path}')
    print(f'  Shape: {df.shape} rows × columns')
    print(f'  Columns: {df.columns.tolist()}')

    return output_path


def main():
    print('='*60)
    print('STEP 2: FEATURE ENGINEERING')
    print('='*60)

    # Load
    df = load_raw_data()
    print(f'Initial columns: {len(df.columns)}')
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )
    # Add features
    df = add_temporal_features(df)
    df = add_cyclical_encodings(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)

    # Clean
    df = handle_missing_values(df)
    print(f'\nTotal columns after feature engineering: {len(df.columns)}')

    # Save
    output_path = save_output(df)

    print('\n' + '='*60)
    print('✅ STEP 2 COMPLETE')
    print('='*60)
    print(f'Output: {output_path}')
    print(f'Ready for: Target engineering + temporal split')


if __name__ == '__main__':
    main()
