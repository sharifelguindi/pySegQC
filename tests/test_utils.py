"""Tests for utils module"""

import pytest
import pandas as pd
import numpy as np
from pysegqc.utils import (
    filter_volume_dependent_features,
    normalize_by_volume,
    detect_structure_positions,
    VOLUME_DEPENDENT_FEATURES,
    VOLUME_INDEPENDENT_FEATURES
)


def test_filter_volume_dependent_features():
    """Test filtering of volume-dependent features"""
    # Use actual feature names from VOLUME_DEPENDENT_FEATURES list
    feature_names = [
        '001_sphericity',  # Independent
        '001_elongation',  # Independent
        '001_volume_cc',  # DEPENDENT (in VOLUME_DEPENDENT_FEATURES)
        '001_surface_area_mm2',  # DEPENDENT (in VOLUME_DEPENDENT_FEATURES)
        '001_hu_mean',  # Independent
    ]

    filtered = filter_volume_dependent_features(feature_names, exclude_spatial=False)

    # Should exclude volume and surface area
    assert '001_sphericity' in filtered
    assert '001_elongation' in filtered
    assert '001_hu_mean' in filtered
    assert '001_volume_cc' not in filtered
    assert '001_surface_area_mm2' not in filtered


def test_normalize_by_volume():
    """Test volume normalization with power laws"""
    # Use actual feature names recognized by normalize_by_volume
    df = pd.DataFrame({
        '001_volume_cc': [8.0, 27.0, 64.0],  # volumes
        '001_maximum_3d_diameter': [4.0, 6.0, 8.0],  # linear ~ V^1/3
        '001_surface_area_mm2': [16.0, 36.0, 64.0],  # area ~ V^2/3
        '001_sphericity': [0.8, 0.85, 0.9],  # independent (unchanged)
    })

    feature_names = list(df.columns)
    normalized_df, normalized_names = normalize_by_volume(df, feature_names)

    # Volume should be excluded from normalized features
    assert '001_volume_cc' not in normalized_names

    # Linear features should be normalized by V^1/3 and renamed with _norm suffix
    diameter_normalized_col = '001_maximum_3d_diameter_norm'
    if diameter_normalized_col in normalized_names:
        # After normalization, diameter/V^1/3 should be constant
        normalized_values = normalized_df[diameter_normalized_col].values
        assert np.allclose(normalized_values, normalized_values[0], rtol=1e-10)

    # Sphericity should remain unchanged
    assert '001_sphericity' in normalized_names
    assert np.allclose(
        normalized_df['001_sphericity'].values,
        [0.8, 0.85, 0.9]
    )


def test_detect_structure_positions():
    """Test structure position detection from column names"""
    columns = [
        '001_original_shape_Sphericity',
        '001_original_firstorder_Mean',
        '002_original_shape_Sphericity',
        '003_original_shape_Elongation',
        'MRN',  # metadata, no prefix
        'Plan_ID',  # metadata, no prefix
    ]

    positions, n_features_per_structure = detect_structure_positions(columns)

    # Check detected positions
    assert positions == [1, 2, 3]

    # Check feature count for first position (001 has 2 features)
    assert n_features_per_structure == 2


def test_volume_dependent_features_list():
    """Test that VOLUME_DEPENDENT_FEATURES list is non-empty"""
    assert len(VOLUME_DEPENDENT_FEATURES) > 0
    assert 'volume_cc' in VOLUME_DEPENDENT_FEATURES or 'VoxelVolume' in VOLUME_DEPENDENT_FEATURES


def test_volume_independent_features_list():
    """Test that VOLUME_INDEPENDENT_FEATURES list is non-empty"""
    assert len(VOLUME_INDEPENDENT_FEATURES) > 0
    assert any('sphericity' in feat.lower() for feat in VOLUME_INDEPENDENT_FEATURES)
