"""
pySegQC: Quality Control and Clustering Analysis for Medical Imaging Segmentations

This package provides comprehensive tools for analyzing radiomics features from
medical imaging segmentations, including PCA dimensionality reduction, hierarchical
clustering, quality metrics, and intelligent training case selection.

Main Modules:
- utils: Helper functions and constants
- data_loader: Data loading and preprocessing
- validation: Data validation, imputation, and cleaning
- pca: PCA dimensionality reduction
- clustering: Hierarchical and k-means clustering
- metrics: Clustering quality metrics and evaluation
- training: Training case selection algorithms
- visualization: Comprehensive plotting and dashboard generation
- export: Excel export with conditional formatting
- prediction: Apply trained models to new cases
- pipeline: Main orchestration pipeline
"""

__version__ = "0.1.0"
__author__ = "pySegQC Contributors"

# Import commonly used functions for convenience
from .utils import (
    CLUSTER_COLORS,
    filter_volume_dependent_features,
    normalize_by_volume,
    detect_structure_positions,
    extract_view_scan_urls,
    get_plotly_click_handler_script,
)

from .data_loader import load_and_preprocess_data
from .validation import impute_missing_values, standardize_features, detect_outliers
from .pca import perform_pca
from .clustering import perform_hierarchical_clustering, perform_kmeans
from .metrics import (
    find_optimal_clusters,
    calculate_gap_statistic,
    assess_cluster_stability_bootstrap,
    perform_consensus_clustering,
)
from .training import (
    select_training_cases_from_clustering,
    select_multi_structure_training_cases,
    calculate_selection_coverage,
    calculate_representativeness_scores,
    calculate_redundancy_score,
)
from .export import export_results
from .prediction import predict_new_cases
from .pipeline import run_analysis_pipeline

# Utility functions (merged into parent modules)
from .visualization import build_hover_text, extract_urls, get_cluster_colors
from .export import apply_cluster_conditional_formatting, generate_cluster_summary
from .utils import extract_case_metadata, get_case_id, print_banner

__all__ = [
    # Utils
    "CLUSTER_COLORS",
    "filter_volume_dependent_features",
    "normalize_by_volume",
    "detect_structure_positions",
    "extract_view_scan_urls",
    "get_plotly_click_handler_script",
    # Data loading
    "load_and_preprocess_data",
    # Validation
    "impute_missing_values",
    "standardize_features",
    "detect_outliers",
    # PCA
    "perform_pca",
    # Clustering
    "perform_hierarchical_clustering",
    "perform_kmeans",
    # Metrics
    "find_optimal_clusters",
    "calculate_gap_statistic",
    "assess_cluster_stability_bootstrap",
    "perform_consensus_clustering",
    # Training
    "select_training_cases_from_clustering",
    "select_multi_structure_training_cases",
    "calculate_selection_coverage",
    "calculate_representativeness_scores",
    "calculate_redundancy_score",
    # Export
    "export_results",
    # Prediction
    "predict_new_cases",
    # Pipeline
    "run_analysis_pipeline",
    # Utility functions
    "build_hover_text",
    "extract_urls",
    "get_cluster_colors",
    "apply_cluster_conditional_formatting",
    "generate_cluster_summary",
    "extract_case_metadata",
    "get_case_id",
    "print_banner",
]
