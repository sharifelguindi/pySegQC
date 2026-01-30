"""Test that all modules can be imported correctly."""

import pytest


def test_import_package():
    """Test that the main package can be imported."""
    import pysegqc
    assert pysegqc.__version__ == "0.1.0"


def test_import_modules():
    """Test that all modules can be imported."""
    from pysegqc import utils
    from pysegqc import data_loader
    from pysegqc import validation
    from pysegqc import pca
    from pysegqc import clustering
    from pysegqc import metrics
    from pysegqc import training
    from pysegqc import visualization
    from pysegqc import export
    from pysegqc import prediction
    from pysegqc import pipeline


def test_import_main_functions():
    """Test that main functions can be imported from package root."""
    from pysegqc import (
        load_and_preprocess_data,
        impute_missing_values,
        standardize_features,
        perform_pca,
        perform_hierarchical_clustering,
        perform_kmeans,
        find_optimal_clusters,
        export_results,
        predict_new_cases,
        run_analysis_pipeline,
    )


def test_import_constants():
    """Test that constants can be imported."""
    from pysegqc import CLUSTER_COLORS
    assert len(CLUSTER_COLORS) == 10
    assert all(isinstance(color, str) for color in CLUSTER_COLORS)
