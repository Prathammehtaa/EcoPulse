"""
Unit tests for preprocessing functions
Tests grid, weather, and merge/feature engineering logic
"""
import sys
import os
from pathlib import Path

# Add src directory to path FIRST
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Grid preprocessing imports
from grid_preprocessing import (
    remove_duplicates,
    fill_timeline_gaps,
    handle_missing_values,
    validate_and_clip,
    filter_training_window
)

# Weather preprocessing imports
from weather_preprocessing import (
    remove_duplicates as weather_remove_duplicates,
    fill_timeline_gaps as weather_fill_timeline_gaps,
    handle_missing_values as weather_handle_missing_values,
    validate_and_clip as weather_validate_and_clip
)

# Merge
from merge_and_features import merge_datasets

# Feature engineering — functions live in feature_engineering.py
from feature_engineering import (
    add_temporal_features,
    add_cyclical_encodings,
    add_lag_features,
    add_rolling_features,
    add_interaction_features,
    handle_missing_values as handle_feature_nulls,
)


@pytest.fixture
def sample_grid_data():
    """Create sample grid data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
    return pd.DataFrame({
        'datetime': dates,
        'zone': ['US-MIDA-PJM'] * 100,
        'carbon_intensity_gco2_per_kwh': np.random.randint(200, 600, 100).astype(float),
        'renewable_energy_pct': np.random.randint(20, 80, 100).astype(float),
        'total_load_mw': np.random.randint(5000, 10000, 100).astype(float),
        'temperature_2m_c': np.random.uniform(10, 30, 100),
        'wind_speed_100m_ms': np.random.uniform(5, 20, 100),
        'cloud_cover_pct': np.random.randint(0, 100, 100).astype(float),
        'carbon_free_energy_pct': np.random.randint(20, 80, 100).astype(float),
    })


@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
    return pd.DataFrame({
        'datetime': dates,
        'zone': ['US-MIDA-PJM'] * 100,
        'temperature_2m_c': np.random.uniform(10, 30, 100),
        'wind_speed_100m_ms': np.random.uniform(5, 20, 100),
        'cloud_cover_pct': np.random.randint(0, 100, 100).astype(float),
        'shortwave_radiation_wm2': np.random.uniform(100, 800, 100)
    })


@pytest.fixture
def data_with_duplicates():
    """Create data with duplicate rows"""
    dates = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
    df = pd.DataFrame({
        'datetime': list(dates) + [dates[0], dates[1]],  # Add 2 duplicates
        'zone': ['US-MIDA-PJM'] * 12,
        'value': range(12)
    })
    return df


@pytest.fixture
def data_with_gaps():
    """Create data with timeline gaps"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
    dates_with_gaps = [dates[i] for i in range(len(dates)) if i % 10 != 5]
    return pd.DataFrame({
        'datetime': dates_with_gaps,
        'zone': ['US-MIDA-PJM'] * len(dates_with_gaps),
        'value': range(len(dates_with_gaps))
    })


@pytest.fixture
def data_with_nulls():
    """Create data with missing values"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
    values = np.random.randint(200, 600, 100).astype(float)
    values[10:15] = np.nan
    return pd.DataFrame({
        'datetime': dates,
        'zone': ['US-MIDA-PJM'] * 100,
        'value': values
    })


class TestRemoveDuplicates:
    """Test duplicate removal"""

    def test_removes_duplicates(self, data_with_duplicates):
        result = remove_duplicates(data_with_duplicates)
        assert len(result) == 10
        assert not result.duplicated(subset=['datetime', 'zone']).any()

    def test_no_duplicates_unchanged(self, sample_grid_data):
        result = remove_duplicates(sample_grid_data)
        assert len(result) == len(sample_grid_data)


class TestFillTimelineGaps:
    """Test timeline gap filling"""

    def test_fills_gaps(self, data_with_gaps):
        zone = 'US-MIDA-PJM'
        result = fill_timeline_gaps(data_with_gaps, zone)
        time_diffs = result['datetime'].diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all()

    def test_preserves_existing_data(self, data_with_gaps):
        zone = 'US-MIDA-PJM'
        original_values = data_with_gaps[['datetime', 'value']].copy()
        result = fill_timeline_gaps(data_with_gaps, zone)
        merged = pd.merge(
            original_values,
            result[['datetime', 'value']],
            on='datetime',
            suffixes=('_orig', '_result')
        )
        assert (merged['value_orig'] == merged['value_result']).all()


class TestHandleMissingValues:
    """Test missing value handling"""

    def test_fills_missing_values(self, data_with_nulls):
        result = handle_missing_values(data_with_nulls, ['value'])
        assert result['value'].isnull().sum() == 0

    def test_ffill_bfill_logic(self):
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'value': [1.0, np.nan, np.nan, 4.0, 5.0]
        })
        result = handle_missing_values(df, ['value'])
        expected = [1.0, 1.0, 1.0, 4.0, 5.0]
        assert result['value'].tolist() == expected


class TestValidateAndClip:
    """Test value validation and clipping"""

    def test_clips_outliers(self):
        df = pd.DataFrame({'value': [50, 100, 150, 200, 250]})
        value_ranges = {'value': {'min': 75, 'max': 225}}
        result = validate_and_clip(df, value_ranges)
        assert result['value'].min() == 75
        assert result['value'].max() == 225

    def test_preserves_valid_values(self):
        df = pd.DataFrame({'value': [100, 150, 200]})
        value_ranges = {'value': {'min': 50, 'max': 250}}
        result = validate_and_clip(df, value_ranges)
        assert result['value'].tolist() == [100, 150, 200]


class TestFilterTrainingWindow:
    """Test training window filtering"""

    def test_filters_by_date_range(self):
        dates = pd.date_range('2024-01-01', periods=365, freq='D', tz='UTC')
        df = pd.DataFrame({'datetime': dates, 'value': range(365)})
        result = filter_training_window(df, '2024-02-01', '2024-02-29')
        assert result['datetime'].min() >= pd.Timestamp('2024-02-01', tz='UTC')
        assert result['datetime'].max() <= pd.Timestamp('2024-02-29 23:59:59', tz='UTC')
        assert len(result) == 29


class TestMergeDatasets:
    """Test dataset merging"""

    def test_merge_grid_weather(self, sample_grid_data, sample_weather_data):
        # sample_grid_data already has weather cols; use minimal frames
        grid = sample_grid_data[['datetime', 'zone', 'carbon_intensity_gco2_per_kwh']].copy()
        weather = sample_weather_data.copy()
        result = merge_datasets(grid, weather)
        assert len(result) == 100
        assert 'carbon_intensity_gco2_per_kwh' in result.columns
        assert 'temperature_2m_c' in result.columns

    def test_merge_preserves_zones(self, sample_grid_data, sample_weather_data):
        grid = sample_grid_data[['datetime', 'zone', 'carbon_intensity_gco2_per_kwh']].copy()
        result = merge_datasets(grid, sample_weather_data)
        assert 'zone' in result.columns
        assert len(result['zone'].unique()) >= 1


class TestAddTemporalFeatures:
    """Test temporal feature creation"""

    def test_adds_hour_features(self, sample_grid_data):
        result = add_temporal_features(sample_grid_data)
        assert 'hour_of_day' in result.columns
        assert result['hour_of_day'].between(0, 23).all()

    def test_adds_cyclical_features(self, sample_grid_data):
        result = add_temporal_features(sample_grid_data)
        result = add_cyclical_encodings(result)
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        assert result['hour_sin'].between(-1, 1).all()
        assert result['hour_cos'].between(-1, 1).all()

    def test_adds_weekend_flag(self, sample_grid_data):
        result = add_temporal_features(sample_grid_data)
        assert 'is_weekend' in result.columns
        assert result['is_weekend'].isin([0, 1]).all()


class TestAddLagFeatures:
    """Test lag feature creation"""

    def test_adds_lag_features(self, sample_grid_data):
        """add_lag_features(df, lags) uses hardcoded cols; check one known lag col exists"""
        result = add_lag_features(sample_grid_data, lags=[1, 3, 24])
        # Feature names are {col}_lag_{n}h for col in hardcoded lag_cols
        assert 'carbon_intensity_gco2_per_kwh_lag_1h' in result.columns
        assert 'carbon_intensity_gco2_per_kwh_lag_3h' in result.columns
        assert 'carbon_intensity_gco2_per_kwh_lag_24h' in result.columns

    def test_lag_values_correct(self):
        """Test that lag values shift correctly for a single zone."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'carbon_intensity_gco2_per_kwh': [10.0, 20.0, 30.0, 40.0, 50.0],
            'total_load_mw': [1.0] * 5,
            'temperature_2m_c': [1.0] * 5,
            'wind_speed_100m_ms': [1.0] * 5,
        })
        result = add_lag_features(df, lags=[1])
        expected_lag = [np.nan, 10.0, 20.0, 30.0, 40.0]
        pd.testing.assert_series_equal(
            result['carbon_intensity_gco2_per_kwh_lag_1h'],
            pd.Series(expected_lag, name='carbon_intensity_gco2_per_kwh_lag_1h'),
            check_dtype=False
        )


class TestAddRollingFeatures:
    """Test rolling feature creation"""

    def test_adds_rolling_features(self, sample_grid_data):
        """add_rolling_features(df, windows) uses hardcoded cols; check known cols exist"""
        result = add_rolling_features(sample_grid_data, windows=[4, 12, 24])
        assert 'carbon_intensity_gco2_per_kwh_mean_4h' in result.columns
        assert 'carbon_intensity_gco2_per_kwh_mean_12h' in result.columns
        assert 'carbon_intensity_gco2_per_kwh_mean_24h' in result.columns
        assert 'carbon_intensity_gco2_per_kwh_std_24h' in result.columns

    def test_rolling_mean_calculation(self):
        """Test rolling mean is calculated correctly for a single zone."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'carbon_intensity_gco2_per_kwh': [10.0, 20.0, 30.0, 40.0, 50.0],
            'total_load_mw': [1.0] * 5,
            'temperature_2m_c': [1.0] * 5,
        })
        result = add_rolling_features(df, windows=[3])
        # min_periods=1: [10, 15, 20, 30, 40]
        assert result['carbon_intensity_gco2_per_kwh_mean_3h'].iloc[0] == 10.0
        assert result['carbon_intensity_gco2_per_kwh_mean_3h'].iloc[2] == 20.0
        assert result['carbon_intensity_gco2_per_kwh_mean_3h'].iloc[4] == 40.0


class TestAddInteractionFeatures:
    """Test interaction feature creation"""

    def test_adds_solar_potential(self, sample_grid_data):
        """solar_potential requires is_daytime and cloud_cover_pct"""
        df = add_temporal_features(sample_grid_data)  # adds is_daytime
        result = add_interaction_features(df)
        assert 'solar_potential' in result.columns

    def test_solar_potential_calculation(self):
        """solar_potential = is_daytime * (100 - cloud_cover_pct) / 100"""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 12:00', periods=1, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'],
            'carbon_intensity_gco2_per_kwh': [300.0],
            'total_load_mw': [5000.0],
            'temperature_2m_c': [20.0],
            'wind_speed_100m_ms': [10.0],
            'cloud_cover_pct': [50.0],
            'carbon_free_energy_pct': [40.0],
            'renewable_energy_pct': [30.0],
        })
        df = add_temporal_features(df)  # is_daytime=1 at noon
        result = add_interaction_features(df)
        # is_daytime=1, cloud=50 -> 1 * (100-50)/100 = 0.5
        assert result['solar_potential'].iloc[0] == pytest.approx(0.5)


class TestHandleFeatureNulls:
    """Test feature null handling — maps to handle_missing_values in feature_engineering"""

    def test_fills_lag_feature_nulls(self):
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'carbon_intensity_gco2_per_kwh_lag_1h': [np.nan, 10.0, 20.0, 30.0, 40.0],
            'carbon_intensity_gco2_per_kwh_mean_4h': [10.0, 15.0, np.nan, 25.0, 30.0],
        })
        result = handle_feature_nulls(df)
        assert result['carbon_intensity_gco2_per_kwh_lag_1h'].isnull().sum() == 0
        assert result['carbon_intensity_gco2_per_kwh_mean_4h'].isnull().sum() == 0