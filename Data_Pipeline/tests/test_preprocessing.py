"""
Unit tests for preprocessing functions
Tests grid, weather, and merge/feature engineering logic
"""
import sys
import os
from pathlib import Path

# Add src directory to path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from grid_preprocessing import (
    remove_duplicates,
    fill_timeline_gaps,
    handle_missing_values,
    validate_and_clip,
    filter_training_window
)
from weather_preprocessing import (
    remove_duplicates as weather_remove_duplicates,
    fill_timeline_gaps as weather_fill_timeline_gaps,
    handle_missing_values as weather_handle_missing_values,
    validate_and_clip as weather_validate_and_clip
)
from merge_and_features import (
    merge_datasets
)


@pytest.fixture
def sample_grid_data():
    """Create sample grid data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC')
    return pd.DataFrame({
        'datetime': dates,
        'zone': ['US-MIDA-PJM'] * 100,
        'carbon_intensity_gco2_per_kwh': np.random.randint(200, 600, 100),
        'renewable_energy_pct': np.random.randint(20, 80, 100)
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
        'cloud_cover_pct': np.random.randint(0, 100, 100),
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
    dates_with_gaps = [dates[i] for i in range(len(dates)) if i % 10 != 5]  # Remove every 10th row
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
    values[10:15] = np.nan  # Add some nulls
    return pd.DataFrame({
        'datetime': dates,
        'zone': ['US-MIDA-PJM'] * 100,
        'value': values
    })


class TestRemoveDuplicates:
    """Test duplicate removal"""
    
    def test_removes_duplicates(self, data_with_duplicates):
        """Test that duplicates are removed"""
        result = remove_duplicates(data_with_duplicates)
        assert len(result) == 10  # Original 10 unique rows
        assert not result.duplicated(subset=['datetime', 'zone']).any()
    
    def test_no_duplicates_unchanged(self, sample_grid_data):
        """Test that data without duplicates remains unchanged"""
        result = remove_duplicates(sample_grid_data)
        assert len(result) == len(sample_grid_data)


class TestFillTimelineGaps:
    """Test timeline gap filling"""
    
    def test_fills_gaps(self, data_with_gaps):
        """Test that timeline gaps are filled"""
        zone = 'US-MIDA-PJM'
        result = fill_timeline_gaps(data_with_gaps, zone)
        
        # Check hourly frequency
        time_diffs = result['datetime'].diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all()
    
    def test_preserves_existing_data(self, data_with_gaps):
        """Test that existing data is preserved"""
        zone = 'US-MIDA-PJM'
        original_values = data_with_gaps[['datetime', 'value']].copy()
        result = fill_timeline_gaps(data_with_gaps, zone)
        
        # Check that original non-null values are preserved
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
        """Test that missing values are filled"""
        result = handle_missing_values(data_with_nulls, ['value'])
        assert result['value'].isnull().sum() == 0
    
    def test_ffill_bfill_logic(self):
        """Test forward fill and backward fill logic"""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'value': [1.0, np.nan, np.nan, 4.0, 5.0]
        })
        result = handle_missing_values(df, ['value'])
        
        # After ffill: [1, 1, 1, 4, 5]
        # After bfill: [1, 1, 1, 4, 5]
        expected = [1.0, 1.0, 1.0, 4.0, 5.0]
        assert result['value'].tolist() == expected


class TestValidateAndClip:
    """Test value validation and clipping"""
    
    def test_clips_outliers(self):
        """Test that outliers are clipped"""
        df = pd.DataFrame({
            'value': [50, 100, 150, 200, 250]
        })
        value_ranges = {'value': {'min': 75, 'max': 225}}
        
        result = validate_and_clip(df, value_ranges)
        
        assert result['value'].min() == 75   # 50 clipped to 75
        assert result['value'].max() == 225  # 250 clipped to 225
    
    def test_preserves_valid_values(self):
        """Test that valid values are preserved"""
        df = pd.DataFrame({
            'value': [100, 150, 200]
        })
        value_ranges = {'value': {'min': 50, 'max': 250}}
        
        result = validate_and_clip(df, value_ranges)
        
        assert result['value'].tolist() == [100, 150, 200]


class TestFilterTrainingWindow:
    """Test training window filtering"""
    
    def test_filters_by_date_range(self):
        """Test that data is filtered to training window"""
        dates = pd.date_range('2024-01-01', periods=365, freq='D', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'value': range(365)
        })
        
        result = filter_training_window(df, '2024-02-01', '2024-02-29')
        
        assert result['datetime'].min() >= pd.Timestamp('2024-02-01', tz='UTC')
        assert result['datetime'].max() <= pd.Timestamp('2024-02-29 23:59:59', tz='UTC')
        assert len(result) == 29  # February 2024 has 29 days


class TestMergeDatasets:
    """Test dataset merging"""
    
    def test_merge_grid_weather(self, sample_grid_data, sample_weather_data):
        """Test that grid and weather data merge correctly"""
        result = merge_datasets(sample_grid_data, sample_weather_data)
        
        # Check all rows merged
        assert len(result) == 100
        
        # Check columns from both datasets present
        assert 'carbon_intensity_gco2_per_kwh' in result.columns
        assert 'temperature_2m_c' in result.columns
    
    def test_merge_preserves_zones(self, sample_grid_data, sample_weather_data):
        """Test that zone information is preserved"""
        result = merge_datasets(sample_grid_data, sample_weather_data)
        assert 'zone' in result.columns
        assert len(result['zone'].unique()) >= 1


class TestAddTemporalFeatures:
    """Test temporal feature creation"""
    
    def test_adds_hour_features(self, sample_grid_data):
        """Test that hour-based features are added"""
        result = add_temporal_features(sample_grid_data)
        
        assert 'hour_of_day' in result.columns
        assert result['hour_of_day'].between(0, 23).all()
    
    def test_adds_cyclical_features(self, sample_grid_data):
        """Test that cyclical encoding is added"""
        result = add_temporal_features(sample_grid_data)
        
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        
        # Check cyclical properties
        assert result['hour_sin'].between(-1, 1).all()
        assert result['hour_cos'].between(-1, 1).all()
    
    def test_adds_weekend_flag(self, sample_grid_data):
        """Test that weekend flag is added"""
        result = add_temporal_features(sample_grid_data)
        
        assert 'is_weekend' in result.columns
        assert result['is_weekend'].isin([0, 1]).all()


class TestAddLagFeatures:
    """Test lag feature creation"""
    
    def test_adds_lag_features(self, sample_grid_data):
        """Test that lag features are created"""
        lags = [1, 3, 24]
        result = add_lag_features(sample_grid_data, 'carbon_intensity_gco2_per_kwh', lags)
        
        assert 'lag_1h' in result.columns
        assert 'lag_3h' in result.columns
        assert 'lag_24h' in result.columns
    
    def test_lag_values_correct(self):
        """Test that lag values are calculated correctly"""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'value': [10, 20, 30, 40, 50]
        })
        
        result = add_lag_features(df, 'value', [1])
        
        # lag_1h should be previous value
        expected_lag = [np.nan, 10, 20, 30, 40]
        pd.testing.assert_series_equal(
            result['lag_1h'],
            pd.Series(expected_lag, name='lag_1h'),
            check_dtype=False
        )


class TestAddRollingFeatures:
    """Test rolling feature creation"""
    
    def test_adds_rolling_features(self, sample_grid_data):
        """Test that rolling features are created"""
        windows = [4, 12, 24]
        result = add_rolling_features(sample_grid_data, 'carbon_intensity_gco2_per_kwh', windows)
        
        assert 'rolling_mean_4h' in result.columns
        assert 'rolling_mean_12h' in result.columns
        assert 'rolling_mean_24h' in result.columns
        assert 'rolling_std_24h' in result.columns
    
    def test_rolling_mean_calculation(self):
        """Test that rolling mean is calculated correctly"""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'value': [10, 20, 30, 40, 50]
        })
        
        result = add_rolling_features(df, 'value', [3])
        
        # rolling_mean_3h: [10, 15, 20, 30, 40]
        assert result['rolling_mean_3h'].iloc[0] == 10
        assert result['rolling_mean_3h'].iloc[2] == 20
        assert result['rolling_mean_3h'].iloc[4] == 40


class TestAddInteractionFeatures:
    """Test interaction feature creation"""
    
    def test_adds_solar_potential(self, sample_weather_data):
        """Test that solar potential is calculated"""
        merged_data = sample_weather_data.copy()
        result = add_interaction_features(merged_data)
        
        assert 'solar_potential' in result.columns
    
    def test_solar_potential_calculation(self):
        """Test solar potential formula"""
        df = pd.DataFrame({
            'shortwave_radiation_wm2': [1000],
            'cloud_cover_pct': [50]
        })
        
        result = add_interaction_features(df)
        
        # solar_potential = radiation * (1 - cloud_cover/100)
        # = 1000 * (1 - 0.5) = 500
        assert result['solar_potential'].iloc[0] == 500


class TestHandleFeatureNulls:
    """Test feature null handling"""
    
    def test_fills_lag_feature_nulls(self):
        """Test that nulls in lag features are filled"""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC'),
            'zone': ['US-MIDA-PJM'] * 5,
            'lag_1h': [np.nan, 10, 20, 30, 40],
            'rolling_mean_4h': [10, 15, np.nan, 25, 30]
        })
        
        result = handle_feature_nulls(df)
        
        assert result['lag_1h'].isnull().sum() == 0
        assert result['rolling_mean_4h'].isnull().sum() == 0
