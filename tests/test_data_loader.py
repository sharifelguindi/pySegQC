"""Tests for data_loader module"""

import pytest
from pysegqc.data_loader import load_and_preprocess_data


def test_load_pca_data_sheet(synthetic_radiomics_excel):
    """Test loading PCA_Data sheet from Excel"""
    metadata, features, feature_names, structure_info = load_and_preprocess_data(
        synthetic_radiomics_excel, 'PCA_Data', mode=None, position=None
    )

    # Check metadata
    assert 'MRN' in metadata.columns
    assert 'Plan_ID' in metadata.columns
    assert len(metadata) == 50

    # Check features
    assert len(feature_names) > 0
    assert all(fname.startswith('001_') or fname.startswith('002_') for fname in feature_names)

    # Check structure info
    assert 'positions' in structure_info
    assert 1 in structure_info['positions']
    assert 2 in structure_info['positions']


def test_load_position_mode(synthetic_radiomics_excel):
    """Test loading single position in position mode"""
    metadata, features, feature_names, structure_info = load_and_preprocess_data(
        synthetic_radiomics_excel, 'PCA_Data', mode='position', position=1
    )

    # Features should have position prefix stripped in position mode
    # (data loader removes '001_' prefix when analyzing single position)
    assert len(feature_names) > 0
    assert not any(fname.startswith('001_') for fname in feature_names)
    assert not any(fname.startswith('002_') for fname in feature_names)


def test_load_concat_mode(synthetic_radiomics_excel):
    """Test loading in concat mode (all structures concatenated)"""
    metadata, features, feature_names, structure_info = load_and_preprocess_data(
        synthetic_radiomics_excel, 'PCA_Data', mode='concat', position=None
    )

    # Features should include both 001_ and 002_ prefixes
    has_001 = any(fname.startswith('001_') for fname in feature_names)
    has_002 = any(fname.startswith('002_') for fname in feature_names)
    assert has_001 and has_002
