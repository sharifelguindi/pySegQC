"""
Tests for training case selection export functionality.

This module tests the export.py functions related to training case selection,
which are currently uncovered (export.py: 54% coverage, missing lines 104-483).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pysegqc.export import export_results


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Suppress sklearn numerical warnings
def test_export_with_training_selection(tmp_path, synthetic_radiomics_excel, temp_output_dir):
    """Test export with training selection data."""
    from pysegqc.data_loader import load_and_preprocess_data
    from pysegqc.pca import perform_pca
    from pysegqc.clustering import perform_hierarchical_clustering
    from pysegqc.validation import impute_missing_values, standardize_features
    from pysegqc.training import select_multi_structure_training_cases

    # Load and process data
    metadata, features_df, feature_names, structure_info = load_and_preprocess_data(
        synthetic_radiomics_excel, 'PCA_Data', mode='position', position=1
    )

    features_imputed = impute_missing_values(features_df, strategy='median')
    features_scaled, scaler = standardize_features(features_imputed)

    pca_model, pca_data, explained_variance = perform_pca(
        features_scaled, n_components=5
    )

    clustering_model, cluster_labels = perform_hierarchical_clustering(
        pca_data, n_clusters=3
    )

    # Select training cases (using the actual function signature)
    training_selection_data = select_multi_structure_training_cases(
        excel_path=synthetic_radiomics_excel,
        sheet_name='PCA_Data',
        impute_strategy='median',
        mode='position',
        position=1
    )

    # Export with training selection (excel_path is the base name, _clustered.xlsx will be appended)
    output_excel_base = temp_output_dir / 'test_with_training.xlsx'
    export_results(
        metadata_df=metadata,
        features_df=features_df,
        pca_data=pca_data,
        labels=cluster_labels,
        pca_model=pca_model,
        feature_names=feature_names,
        output_dir=temp_output_dir,
        excel_path=output_excel_base,
        scaler=scaler,
        impute_strategy='median',
        training_selection_data=training_selection_data
    )

    # The actual file created has _clustered.xlsx appended
    actual_output = temp_output_dir / 'test_with_training_clustered.xlsx'
    assert actual_output.exists()

    # Load and verify sheets - should have training sheets
    with pd.ExcelFile(actual_output, engine='openpyxl') as xlsx:
        sheets = xlsx.sheet_names
        # Training sheets should be present when training_selection_data is provided
        assert any('Training' in sheet for sheet in sheets)


def test_export_without_training_selection(tmp_path, synthetic_radiomics_excel, temp_output_dir):
    """Test that export works without training selection data (baseline test)."""
    from pysegqc.data_loader import load_and_preprocess_data
    from pysegqc.pca import perform_pca
    from pysegqc.clustering import perform_hierarchical_clustering
    from pysegqc.validation import impute_missing_values, standardize_features

    # Load and process data
    metadata, features_df, feature_names, structure_info = load_and_preprocess_data(
        synthetic_radiomics_excel, 'PCA_Data', mode='position', position=1
    )

    features_imputed = impute_missing_values(features_df, strategy='median')
    features_scaled, scaler = standardize_features(features_imputed)

    pca_model, pca_data, explained_variance = perform_pca(
        features_scaled, n_components=5
    )

    clustering_model, cluster_labels = perform_hierarchical_clustering(
        pca_data, n_clusters=3
    )

    # Export WITHOUT training selection (excel_path is the base name, _clustered.xlsx will be appended)
    output_excel_base = temp_output_dir / 'test_no_training.xlsx'
    export_results(
        metadata_df=metadata,
        features_df=features_df,
        pca_data=pca_data,
        labels=cluster_labels,
        pca_model=pca_model,
        feature_names=feature_names,
        output_dir=temp_output_dir,
        excel_path=output_excel_base,
        scaler=scaler,
        impute_strategy='median',
        training_selection_data=None  # Explicitly None
    )

    # The actual file created has _clustered.xlsx appended
    actual_output = temp_output_dir / 'test_no_training_clustered.xlsx'
    assert actual_output.exists()

    # Load and verify standard sheets exist
    with pd.ExcelFile(actual_output, engine='openpyxl') as xlsx:
        sheets = xlsx.sheet_names
        # Standard sheets should exist
        assert 'Summary' in sheets
        assert 'Clustered_Data' in sheets
        assert 'Cluster_0' in sheets
