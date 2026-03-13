"""
Unit Tests for TFDV Bias Analysis
==================================
Run with: pytest test_tfdv_bias_analysis.py -v
"""

from tfdv_bias_analysis import (
    add_slice_features,
    analyze_slice_distribution,
    compute_target_stats_by_slice,
)
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def sample_data():
    """Create sample EcoPulse data."""
    np.random.seed(42)
    n = 500

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        'zone': ['US-NE-ISNE'] * n,
        'timestamp_utc': timestamps,
        'carbon_intensity_gco2_per_kwh': np.random.normal(200, 50, n),
        'carbon_free_energy_pct': np.random.uniform(30, 70, n),
    })


@pytest.fixture
def imbalanced_data():
    """Create imbalanced data for bias detection."""
    np.random.seed(42)

    # 90% weekday, 10% weekend
    weekday_ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) +
                  timedelta(hours=i) for i in range(450)]
    weekend_ts = [datetime(2024, 1, 6, tzinfo=timezone.utc) +
                  timedelta(hours=i) for i in range(50)]

    return pd.DataFrame({
        'zone': ['US-NE-ISNE'] * 500,
        'timestamp_utc': weekday_ts + weekend_ts,
        'carbon_intensity_gco2_per_kwh': np.random.normal(200, 50, 500),
    })


# ----------------------------
# Tests: Slice Features
# ----------------------------
class TestSliceFeatures:

    def test_adds_hour_of_day(self, sample_data):
        result = add_slice_features(sample_data)
        assert 'hour_of_day' in result.columns

    def test_adds_day_of_week(self, sample_data):
        result = add_slice_features(sample_data)
        assert 'day_of_week' in result.columns
        valid_days = ['Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday', 'Sunday']
        assert all(d in valid_days for d in result['day_of_week'].unique())

    def test_adds_season(self, sample_data):
        result = add_slice_features(sample_data)
        assert 'season' in result.columns

    def test_adds_weekend_flag(self, sample_data):
        result = add_slice_features(sample_data)
        assert 'is_weekend' in result.columns

    def test_adds_carbon_bucket(self, sample_data):
        result = add_slice_features(sample_data)
        assert 'carbon_intensity_bucket' in result.columns


# ----------------------------
# Tests: Distribution Analysis
# ----------------------------
class TestDistributionAnalysis:

    def test_calculates_counts(self, sample_data):
        df = add_slice_features(sample_data)
        dist = analyze_slice_distribution(df, 'day_of_week')

        assert 'count' in dist.columns
        assert dist['count'].sum() == len(df)

    def test_calculates_percentages(self, sample_data):
        df = add_slice_features(sample_data)
        dist = analyze_slice_distribution(df, 'day_of_week')

        assert 'percentage' in dist.columns
        assert abs(dist['percentage'].sum() - 100) < 1  # ~100%

    def test_flags_imbalanced_slices(self, imbalanced_data):
        df = add_slice_features(imbalanced_data)
        dist = analyze_slice_distribution(df, 'is_weekend')

        # Weekend should be underrepresented
        assert dist['is_underrepresented'].any(
        ) or dist['is_overrepresented'].any()

    def test_handles_missing_column(self, sample_data):
        dist = analyze_slice_distribution(sample_data, 'nonexistent')
        assert dist.empty


# ----------------------------
# Tests: Target Statistics
# ----------------------------
class TestTargetStats:

    def test_computes_mean(self, sample_data):
        df = add_slice_features(sample_data)
        stats = compute_target_stats_by_slice(df, 'day_of_week')

        assert 'mean' in stats.columns

    def test_computes_std(self, sample_data):
        df = add_slice_features(sample_data)
        stats = compute_target_stats_by_slice(df, 'day_of_week')

        assert 'std' in stats.columns

    def test_groups_by_slice(self, sample_data):
        df = add_slice_features(sample_data)
        stats = compute_target_stats_by_slice(df, 'day_of_week')

        # Should have stats for each day
        assert len(stats) <= 7


# ----------------------------
# Tests: Edge Cases
# ----------------------------
class TestEdgeCases:

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = add_slice_features(df)
        assert result.empty

    def test_single_row(self):
        df = pd.DataFrame({
            'zone': ['US-NE-ISNE'],
            'timestamp_utc': [datetime(2024, 6, 15, 12, tzinfo=timezone.utc)],
            'carbon_intensity_gco2_per_kwh': [200.0]
        })

        result = add_slice_features(df)
        assert len(result) == 1
        assert result['season'].iloc[0] == 'Summer'


# ----------------------------
# Tests: Season Logic
# ----------------------------
class TestSeasonLogic:

    @pytest.mark.parametrize("month,expected", [
        (1, 'Winter'), (2, 'Winter'), (12, 'Winter'),
        (3, 'Spring'), (4, 'Spring'), (5, 'Spring'),
        (6, 'Summer'), (7, 'Summer'), (8, 'Summer'),
        (9, 'Fall'), (10, 'Fall'), (11, 'Fall'),
    ])
    def test_season_mapping(self, month, expected):
        df = pd.DataFrame({
            'zone': ['US-NE-ISNE'],
            'timestamp_utc': [datetime(2024, month, 15, tzinfo=timezone.utc)],
            'carbon_intensity_gco2_per_kwh': [200.0]
        })

        result = add_slice_features(df)
        assert result['season'].iloc[0] == expected


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
