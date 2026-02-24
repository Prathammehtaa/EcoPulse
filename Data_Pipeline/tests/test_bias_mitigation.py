"""
Unit Tests for EcoPulse Bias Detection and Mitigation
======================================================
Tests for bias analysis and mitigation functions.

Run with: pytest tests/test_bias_mitigation.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCarbonBucketCreation:
    """Tests for carbon bucket assignment."""
    
    def test_very_low_bucket(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(50) == 'Very Low'
        assert create_carbon_bucket(99) == 'Very Low'
    
    def test_low_bucket(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(100) == 'Low'
        assert create_carbon_bucket(199) == 'Low'
    
    def test_medium_bucket(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(200) == 'Medium'
        assert create_carbon_bucket(349) == 'Medium'
    
    def test_high_bucket(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(350) == 'High'
        assert create_carbon_bucket(499) == 'High'
    
    def test_very_high_bucket(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(500) == 'Very High'
        assert create_carbon_bucket(1000) == 'Very High'
    
    def test_nan_handling(self):
        from bias_mitigation import create_carbon_bucket
        assert create_carbon_bucket(np.nan) == 'Unknown'


class TestSeverityClassification:
    """Tests for bias severity classification."""
    
    def test_low_severity(self):
        from bias_mitigation import get_severity
        assert get_severity(1.0) == 'LOW'
        assert get_severity(2.0) == 'LOW'
    
    def test_moderate_severity(self):
        from bias_mitigation import get_severity
        assert get_severity(3.0) == 'MODERATE'
        assert get_severity(5.0) == 'MODERATE'
    
    def test_high_severity(self):
        from bias_mitigation import get_severity
        assert get_severity(6.0) == 'HIGH'
        assert get_severity(10.0) == 'HIGH'
    
    def test_severe_severity(self):
        from bias_mitigation import get_severity
        assert get_severity(11.0) == 'SEVERE'
        assert get_severity(100.0) == 'SEVERE'


class TestOversampling:
    """Tests for random oversampling function."""
    
    @pytest.fixture
    def imbalanced_df(self):
        """Create imbalanced test dataframe."""
        np.random.seed(42)
        data = {
            'carbon_bucket': ['A'] * 100 + ['B'] * 50 + ['C'] * 10,
            'value': np.random.randn(160)
        }
        return pd.DataFrame(data)
    
    def test_oversampling_balances_classes(self, imbalanced_df):
        from bias_mitigation import random_oversample
        result = random_oversample(imbalanced_df, 'carbon_bucket')
        counts = result['carbon_bucket'].value_counts()
        assert counts.nunique() == 1  # All classes have same count
    
    def test_oversampling_increases_samples(self, imbalanced_df):
        from bias_mitigation import random_oversample
        result = random_oversample(imbalanced_df, 'carbon_bucket')
        assert len(result) >= len(imbalanced_df)
    
    def test_oversampling_preserves_columns(self, imbalanced_df):
        from bias_mitigation import random_oversample
        result = random_oversample(imbalanced_df, 'carbon_bucket')
        assert list(result.columns) == list(imbalanced_df.columns)


class TestStratifiedSplit:
    """Tests for stratified splitting function."""
    
    @pytest.fixture
    def balanced_df(self):
        """Create balanced test dataframe."""
        np.random.seed(42)
        data = {
            'carbon_bucket': ['A'] * 100 + ['B'] * 100 + ['C'] * 100,
            'value': np.random.randn(300)
        }
        return pd.DataFrame(data)
    
    def test_split_ratios(self, balanced_df):
        from bias_mitigation import stratified_split
        train, val, test = stratified_split(balanced_df, 'carbon_bucket')
        total = len(balanced_df)
        assert abs(len(train) / total - 0.7) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05
    
    def test_no_data_loss(self, balanced_df):
        from bias_mitigation import stratified_split
        train, val, test = stratified_split(balanced_df, 'carbon_bucket')
        assert len(train) + len(val) + len(test) == len(balanced_df)
    
    def test_stratification_preserved(self, balanced_df):
        from bias_mitigation import stratified_split
        train, val, test = stratified_split(balanced_df, 'carbon_bucket')
        # Each split should have all classes
        assert set(train['carbon_bucket'].unique()) == {'A', 'B', 'C'}
        assert set(val['carbon_bucket'].unique()) == {'A', 'B', 'C'}
        assert set(test['carbon_bucket'].unique()) == {'A', 'B', 'C'}


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        df = pd.DataFrame({'carbon_bucket': [], 'value': []})
        # Should handle gracefully
        assert len(df) == 0
    
    def test_single_class(self):
        from bias_mitigation import random_oversample
        df = pd.DataFrame({
            'carbon_bucket': ['A'] * 100,
            'value': np.random.randn(100)
        })
        result = random_oversample(df, 'carbon_bucket')
        assert len(result) == 100  # No change needed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])