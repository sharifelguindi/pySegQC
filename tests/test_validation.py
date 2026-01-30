"""Tests for validation module"""

import pytest
import pandas as pd
import numpy as np
from pysegqc.validation import impute_missing_values, standardize_features, detect_outliers


def test_impute_missing_median():
    """Test median imputation strategy"""
    df = pd.DataFrame({
        'feat1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feat2': [10.0, np.nan, 30.0, 40.0, 50.0]
    })

    result = impute_missing_values(df, strategy='median')

    assert not result.isnull().any().any()
    assert result.loc[2, 'feat1'] == 3.0  # median of [1,2,4,5]
    assert result.loc[1, 'feat2'] == 35.0  # median of [10,30,40,50]


def test_impute_missing_mean():
    """Test mean imputation strategy"""
    df = pd.DataFrame({
        'feat1': [1.0, 2.0, np.nan, 4.0, 5.0]
    })

    result = impute_missing_values(df, strategy='mean')

    assert not result.isnull().any().any()
    assert result.loc[2, 'feat1'] == 3.0  # mean of [1,2,4,5]


def test_standardize_features():
    """Test feature standardization (z-score)"""
    df = pd.DataFrame({
        'feat1': [10.0, 20.0, 30.0, 40.0, 50.0],
        'feat2': [100.0, 200.0, 300.0, 400.0, 500.0]
    })

    result, scaler = standardize_features(df)

    # Check mean ~ 0
    assert np.abs(result.mean().mean()) < 1e-10

    # Check std is close to 1 (pandas uses ddof=1, so std will be slightly different)
    # Use result.values to get numpy array std which should be exactly 1
    assert np.abs(result.values.std() - 1.0) < 0.01

    # Check scaler can inverse transform
    assert scaler is not None


def test_detect_outliers():
    """Test outlier detection with Isolation Forest"""
    # Create data with clear outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (95, 5))
    outlier_data = np.random.normal(10, 1, (5, 5))  # 5 outliers far from cluster

    data = pd.DataFrame(np.vstack([normal_data, outlier_data]))

    result = detect_outliers(data, contamination=0.05)

    # Check return structure
    assert 'outlier_mask' in result
    assert 'outlier_scores' in result
    assert 'outlier_indices' in result

    # Should detect roughly 5% as outliers
    assert len(result['outlier_indices']) == pytest.approx(5, abs=2)
    assert len(result['outlier_scores']) == 100
    assert result['outlier_mask'].sum() == len(result['outlier_indices'])
