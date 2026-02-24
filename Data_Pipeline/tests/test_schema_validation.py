"""
Unit tests for schema_validation.py
Tests core validation logic with TFDV
"""

import pytest
import pandas as pd
import tensorflow_data_validation as tfdv
import os
from pathlib import Path
from schema_validation import validate_dataset, detect_drift


@pytest.fixture
def sample_weather_data():
    """Create valid weather data matching actual schema"""
    return pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=100, freq='h', tz='UTC'),
        'zone': ['US-MIDA-PJM'] * 100,
        'temperature_2m_c': [15.5] * 100,
        'wind_speed_100m_ms': [10.2] * 100,
        'cloud_cover_pct': [50] * 100,
        'shortwave_radiation_wm2': [200.0] * 100,
        'rain_mm': [0.0] * 100,
        'snowfall_cm': [0.0] * 100,
        'weather_code': [1] * 100
    })


@pytest.fixture
def baseline_schema(sample_weather_data, tmp_path):
    """Generate baseline schema for testing"""
    stats = tfdv.generate_statistics_from_dataframe(sample_weather_data)
    schema = tfdv.infer_schema(statistics=stats)
    
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_path = schema_dir / "weather_schema.pbtxt"
    tfdv.write_schema_text(schema, str(schema_path))
    
    return str(schema_dir)


class TestValidateDataset:
    """Test suite for validate_dataset function"""
    
    def test_valid_data_passes(self, sample_weather_data, baseline_schema):
        """Test that valid data passes validation"""
        result = validate_dataset(
            sample_weather_data, 
            'weather', 
            schemas_dir=baseline_schema
        )
        assert result == True
    
    
    def test_missing_column_fails(self, sample_weather_data, baseline_schema):
        """Test that missing required column fails validation"""
        corrupted_data = sample_weather_data.drop(columns=['temperature_2m_c'])
        
        result = validate_dataset(
            corrupted_data, 
            'weather', 
            schemas_dir=baseline_schema
        )
        assert result == False
    
    
    def test_wrong_data_type_fails(self, sample_weather_data, baseline_schema):
        """Test that wrong data type fails validation"""
        corrupted_data = sample_weather_data.copy()
        corrupted_data['temperature_2m_c'] = corrupted_data['temperature_2m_c'].astype(str)
        
        result = validate_dataset(
            corrupted_data, 
            'weather', 
            schemas_dir=baseline_schema
        )
        assert result == False
    
    
    def test_unexpected_categorical_value_fails(self, sample_weather_data, baseline_schema):
        """Test that unexpected categorical value fails validation"""
        corrupted_data = sample_weather_data.copy()
        corrupted_data.loc[0, 'zone'] = 'UNKNOWN-ZONE'
        
        result = validate_dataset(
            corrupted_data, 
            'weather', 
            schemas_dir=baseline_schema
        )
        assert result == False
    
    
    def test_missing_schema_file_handles_gracefully(self, sample_weather_data):
        """Test that missing schema file is handled gracefully"""
        result = validate_dataset(
            sample_weather_data, 
            'nonexistent_dataset', 
            schemas_dir='nonexistent_directory'
        )
        assert result == True
    
    
    def test_empty_dataframe_fails(self, baseline_schema):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result = validate_dataset(
            empty_df, 
            'weather', 
            schemas_dir=baseline_schema
        )
        assert result == False


class TestDetectDrift:
    """Test suite for detect_drift function"""
    
    def test_detect_drift_runs_without_error(self, sample_weather_data, tmp_path):
        """Test that drift detection runs without crashing"""
        stats = tfdv.generate_statistics_from_dataframe(sample_weather_data)
        stats_path = tmp_path / "weather_stats.pbtxt"
        tfdv.write_stats_text(stats, str(stats_path))
        
        drifted_data = sample_weather_data.copy()
        drifted_data['temperature_2m_c'] = drifted_data['temperature_2m_c'] + 10
        
        result = detect_drift(
            drifted_data,
            str(stats_path),
            'weather'
        )
        
        assert result is None
    
    
    def test_detect_drift_with_missing_stats_file(self, sample_weather_data):
        """Test drift detection with missing baseline stats file"""
        with pytest.raises(Exception):
            detect_drift(
                sample_weather_data,
                'nonexistent_stats.pbtxt',
                'weather'
            )