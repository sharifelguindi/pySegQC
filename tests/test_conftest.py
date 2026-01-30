"""Tests for conftest.py fixtures"""

import pandas as pd


def test_synthetic_radiomics_data(synthetic_radiomics_excel):
    """Test synthetic data generator creates valid Excel file"""
    df = pd.read_excel(synthetic_radiomics_excel, sheet_name='PCA_Data')

    # Check structure
    assert len(df) == 50  # 50 samples
    assert 'MRN' in df.columns
    assert 'Plan_ID' in df.columns

    # Check multi-structure features
    feature_cols = [c for c in df.columns if c.startswith('001_')]
    assert len(feature_cols) > 0

    # Check numeric features
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_temp_output_dir(temp_output_dir):
    """Test temporary output directory fixture"""
    assert temp_output_dir.exists()
    assert temp_output_dir.is_dir()


def test_trained_models_dir(trained_models_dir):
    """Test trained models directory fixture"""
    assert trained_models_dir.exists()
    models_file = trained_models_dir / 'trained_models.pkl'
    assert models_file.exists()
