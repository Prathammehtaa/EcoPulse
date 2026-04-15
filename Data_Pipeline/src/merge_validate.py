"""
Step 1: Data Reconstruction & Validation
=========================================
Removes duplicates from merged dataset and validates:
- Temporal coverage (no gaps)
- Zone coverage
- Value ranges
Output: raw_combined.parquet
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ECO_PULSE_ROOT = os.path.dirname(PROJECT_ROOT)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def load_raw_data():
    """Load original merged dataset."""
    input_path = os.path.join(PROCESSED_DIR, 'merged_dataset.parquet')
    print(f'Loading raw data: {input_path}')
    df = pd.read_parquet(input_path)
    print(f'  Loaded {len(df)} rows')
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    print('\n--- REMOVING DUPLICATES ---')
    initial_rows = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    duplicates_removed = initial_rows - len(df)
    print(f'  Rows before: {initial_rows}')
    print(f'  Rows after: {len(df)}')
    print(
        f'  Duplicates removed: {duplicates_removed} ({(duplicates_removed/initial_rows)*100:.2f}%)')

    return df


def validate_temporal_coverage(df):
    """Validate no gaps in temporal data."""
    print('\n--- VALIDATING TEMPORAL COVERAGE ---')

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    for zone in df['zone'].unique():
        zone_data = df[df['zone'] == zone].sort_values('datetime')

        print(f'\n  Zone: {zone}')
        print(f'    Start: {zone_data["datetime"].min()}')
        print(f'    End: {zone_data["datetime"].max()}')
        print(f'    Rows: {len(zone_data)}')

        # Check for hourly frequency
        diffs = zone_data['datetime'].diff().dropna().unique()
        expected_diff = pd.Timedelta(hours=1)

        if all(diff == expected_diff for diff in diffs):
            print(f'    ✅ Hourly frequency verified')
        else:
            print(f'    ⚠️  Non-hourly gaps detected: {diffs}')

        # Expected rows: 3 years × 365 days × 24 hours ≈ 26,280
        expected_rows = 3 * 365 * 24
        print(
            f'    Expected rows (~3 years): {expected_rows}, Actual: {len(zone_data)}')


def validate_zones(df):
    """Validate zone coverage."""
    print('\n--- VALIDATING ZONE COVERAGE ---')

    zones = df['zone'].unique()
    print(f'  Zones found: {list(zones)}')
    print(f'  Expected: 2 zones (US-NE-ISNE, US-NW-PACW, US-MIDA-PJM, etc.)')

    for zone in zones:
        count = len(df[df['zone'] == zone])
        pct = (count / len(df)) * 100
        print(f'    {zone}: {count} rows ({pct:.1f}%)')


def validate_value_ranges(df):
    """Validate numeric column ranges."""
    print('\n--- VALIDATING VALUE RANGES ---')

    # Carbon intensity
    ci_col = 'carbon_intensity_gco2_per_kwh'
    print(f'\n  {ci_col}:')
    print(f'    Min: {df[ci_col].min():.2f}, Max: {df[ci_col].max():.2f}')
    print(f'    Mean: {df[ci_col].mean():.2f}, Std: {df[ci_col].std():.2f}')
    print(f'    Nulls: {df[ci_col].isnull().sum()}')

    # Temperature
    temp_col = 'temperature_2m_c'
    if temp_col in df.columns:
        print(f'\n  {temp_col}:')
        print(
            f'    Min: {df[temp_col].min():.2f}, Max: {df[temp_col].max():.2f}')
        print(
            f'    Mean: {df[temp_col].mean():.2f}, Std: {df[temp_col].std():.2f}')
        print(f'    Nulls: {df[temp_col].isnull().sum()}')

    # Wind speed
    wind_col = 'wind_speed_100m_ms'
    if wind_col in df.columns:
        print(f'\n  {wind_col}:')
        print(
            f'    Min: {df[wind_col].min():.2f}, Max: {df[wind_col].max():.2f}')
        print(
            f'    Mean: {df[wind_col].mean():.2f}, Std: {df[wind_col].std():.2f}')
        print(f'    Nulls: {df[wind_col].isnull().sum()}')

    # Load
    load_col = 'total_load_mw'
    if load_col in df.columns:
        print(f'\n  {load_col}:')
        print(
            f'    Min: {df[load_col].min():.2f}, Max: {df[load_col].max():.2f}')
        print(
            f'    Mean: {df[load_col].mean():.2f}, Std: {df[load_col].std():.2f}')
        print(f'    Nulls: {df[load_col].isnull().sum()}')


def validate_nulls(df):
    """Check for missing values."""
    print('\n--- CHECKING FOR NULL VALUES ---')

    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f'  ⚠️  Null values found:')
        for col, count in nulls[nulls > 0].items():
            pct = (count / len(df)) * 100
            print(f'    {col}: {count} ({pct:.2f}%)')
    else:
        print(f'  ✅ No null values found')


def save_output(df):
    """Save cleaned dataset."""
    print('\n--- SAVING OUTPUT ---')

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, 'raw_combined.parquet')

    df.to_parquet(output_path, index=False)
    print(f'  Saved: {output_path}')
    print(f'  Size: {len(df)} rows × {len(df.columns)} columns')

    return output_path


def main():
    print('='*60)
    print('STEP 1: DATA RECONSTRUCTION & VALIDATION')
    print('='*60)

    # Load
    df = load_raw_data()

    # Remove duplicates
    df = remove_duplicates(df)

    # Validate
    validate_temporal_coverage(df)
    validate_zones(df)
    validate_value_ranges(df)
    validate_nulls(df)

    # Save
    output_path = save_output(df)

    print('\n' + '='*60)
    print('✅ STEP 1 COMPLETE')
    print('='*60)
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
