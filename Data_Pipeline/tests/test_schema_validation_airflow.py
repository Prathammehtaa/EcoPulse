"""
Unit tests for schema_validation_task.py
Tests Airflow task wrapper that orchestrates validation
"""

import pytest
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from schema_validation_task import run_tfdv_schema_validation, _load_config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "local": {
            "processed": "data/processed"
        },
        "output": {
            "files": {
                "grid_processed": "grid_data_processed.parquet",
                "weather_processed": "weather_data_processed.parquet",
                "merged_processed": "merged_dataset.parquet",
                "feature_table": "feature_table.parquet"
            }
        }
    }


@pytest.fixture
def sample_dataframe():
    """Create a simple valid DataFrame for mocking"""
    return pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })


class TestRunTFDVSchemaValidation:
    """Test suite for run_tfdv_schema_validation function"""
    
    @patch('schema_validation_task.Path')
    @patch('schema_validation_task._load_config')
    @patch('schema_validation_task.pd.read_parquet')
    @patch('schema_validation_task.validate_dataset')
    def test_all_datasets_pass_validation(
        self,
        mock_validate,
        mock_read_parquet,
        mock_load_config,
        mock_path_class,
        mock_config,
        sample_dataframe
    ):
        """Test that function returns True when all datasets pass validation"""
        mock_load_config.return_value = mock_config
        mock_read_parquet.return_value = sample_dataframe
        mock_validate.return_value = True
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_class.return_value.resolve.return_value.parent.parent = mock_path
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.resolve.return_value = mock_path
        
        result = run_tfdv_schema_validation()
        
        assert result == True
        assert mock_validate.call_count == 4
    
    
    @patch('schema_validation_task.Path')
    @patch('schema_validation_task._load_config')
    @patch('schema_validation_task.pd.read_parquet')
    @patch('schema_validation_task.validate_dataset')
    def test_one_dataset_fails_raises_exception(
        self,
        mock_validate,
        mock_read_parquet,
        mock_load_config,
        mock_path_class,
        mock_config,
        sample_dataframe
    ):
        """Test that function raises ValueError when any dataset fails validation"""
        mock_load_config.return_value = mock_config
        mock_read_parquet.return_value = sample_dataframe
        mock_validate.side_effect = [True, False, True, True]
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_class.return_value.resolve.return_value.parent.parent = mock_path
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.resolve.return_value = mock_path
        
        with pytest.raises(ValueError, match="TFDV schema validation failed"):
            run_tfdv_schema_validation()
    
    
    @patch('schema_validation_task.Path')
    @patch('schema_validation_task._load_config')
    def test_missing_parquet_files_skipped(
        self,
        mock_load_config,
        mock_path_class,
        mock_config
    ):
        """Test that missing parquet files are skipped gracefully"""
        mock_load_config.return_value = mock_config
        
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_class.return_value.resolve.return_value.parent.parent = mock_path
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.resolve.return_value = mock_path
        
        result = run_tfdv_schema_validation()
        
        assert result == True
    
    
    @patch('schema_validation_task.Path')
    @patch('schema_validation_task._load_config')
    @patch('schema_validation_task.pd.read_parquet')
    @patch('schema_validation_task.validate_dataset')
    def test_validates_all_four_datasets(
        self,
        mock_validate,
        mock_read_parquet,
        mock_load_config,
        mock_path_class,
        mock_config,
        sample_dataframe
    ):
        """Test that all four datasets (grid, weather, merged, features) are validated"""
        mock_load_config.return_value = mock_config
        mock_read_parquet.return_value = sample_dataframe
        mock_validate.return_value = True
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_class.return_value.resolve.return_value.parent.parent = mock_path
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.resolve.return_value = mock_path
        
        run_tfdv_schema_validation()
        
        validated_datasets = []
        for call in mock_validate.call_args_list:
            if len(call.args) >= 2:
                validated_datasets.append(call.args[1])
            elif 'dataset_name' in call.kwargs:
                validated_datasets.append(call.kwargs['dataset_name'])
        
        assert 'grid' in validated_datasets
        assert 'weather' in validated_datasets
        assert 'merged' in validated_datasets
        assert 'features' in validated_datasets
        assert mock_validate.call_count == 4


class TestLoadConfig:
    """Test suite for _load_config function"""
    
    @patch('builtins.open', new_callable=mock_open, read_data="test: config")
    @patch('yaml.safe_load')
    def test_load_config_reads_yaml(self, mock_yaml_load, mock_file):
        """Test that config is loaded from YAML file"""
        mock_config = {'test': 'config'}
        mock_yaml_load.return_value = mock_config
        
        project_root = Path('/fake/project')
        result = _load_config(project_root)
        
        assert result == mock_config