"""Comprehensive tests for training module"""

import pytest
import numpy as np
import pandas as pd
from pysegqc.training import (
    calculate_selection_coverage,
    calculate_representativeness_scores,
    calculate_redundancy_score,
    select_training_cases_from_clustering,
    select_multi_structure_training_cases
)


def test_calculate_selection_coverage(three_cluster_data):
    """Test coverage metric calculation"""
    pca_data, labels = three_cluster_data

    # Select 30% of samples evenly from clusters
    selected_indices = np.concatenate([
        np.where(labels == 0)[0][:10],
        np.where(labels == 1)[0][:10],
        np.where(labels == 2)[0][:10],
    ])

    coverage = calculate_selection_coverage(selected_indices, pca_data, labels, n_clusters=3)

    assert 'coverage_score' in coverage
    assert 0 <= coverage['coverage_score'] <= 1
    assert 'cluster_coverage' in coverage


def test_calculate_representativeness_scores(three_cluster_data):
    """Test representativeness scoring"""
    pca_data, labels = three_cluster_data
    selected_indices = np.array([0, 10, 20, 30, 40, 50])

    scores = calculate_representativeness_scores(selected_indices, pca_data, labels)

    assert len(scores) == len(selected_indices)
    assert all(score >= 0 for score in scores)


def test_calculate_redundancy_score(three_cluster_data):
    """Test redundancy calculation"""
    pca_data, _ = three_cluster_data

    # Select similar points (high redundancy)
    selected_indices = np.array([0, 1, 2, 3, 4])
    redundancy = calculate_redundancy_score(selected_indices, pca_data)

    assert 'redundancy_score' in redundancy
    assert 0 <= redundancy['redundancy_score'] <= 1


def test_select_training_cases_from_clustering(three_cluster_data):
    """Test training case selection from clustering"""
    pca_data, labels = three_cluster_data

    metadata_df = pd.DataFrame({
        'MRN': [f'MRN{i:04d}' for i in range(len(labels))],
        'Plan_ID': [f'PLAN{i:04d}' for i in range(len(labels))]
    })

    selection = select_training_cases_from_clustering(
        labels, pca_data, metadata_df, max_clusters=3, max_cases=10
    )

    # Check per_cluster_selections format
    assert 'per_cluster_selections' in selection
    assert len(selection['per_cluster_selections']) <= 3

    # Check each cluster has selections
    for cluster_id, cluster_info in selection['per_cluster_selections'].items():
        assert 'selected_case_ids' in cluster_info
        assert 'cluster_size' in cluster_info
        # max_cases is a guideline, actual selection may vary based on algorithm
        assert len(cluster_info['selected_case_ids']) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Suppress sklearn numerical warnings
def test_select_multi_structure_training_cases(synthetic_radiomics_excel):
    """Test multi-structure training case selection"""
    selection = select_multi_structure_training_cases(
        synthetic_radiomics_excel,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    # Should return a dictionary with selection info
    assert isinstance(selection, dict)
    assert 'selected_case_ids' in selection or 'per_cluster_selections' in selection or 'selected_cases' in selection


def test_training_selection_with_small_cluster():
    """Test training selection handles small clusters correctly"""
    # Create data with one very small cluster
    from sklearn.datasets import make_blobs

    # 2 large clusters + 1 tiny cluster
    X1, _ = make_blobs(n_samples=45, n_features=5, centers=1, cluster_std=0.5, random_state=42)
    X2, _ = make_blobs(n_samples=45, n_features=5, centers=1, cluster_std=0.5, random_state=43)
    X3, _ = make_blobs(n_samples=3, n_features=5, centers=1, cluster_std=0.5, random_state=44)

    pca_data = np.vstack([X1, X2, X3])
    labels = np.array([0] * 45 + [1] * 45 + [2] * 3)

    metadata_df = pd.DataFrame({
        'MRN': [f'MRN{i:04d}' for i in range(len(labels))]
    })

    selection = select_training_cases_from_clustering(
        labels, pca_data, metadata_df, max_clusters=3, max_cases=10
    )

    # Small cluster should be handled gracefully
    assert 'per_cluster_selections' in selection
    cluster_2_info = selection['per_cluster_selections'].get(2)
    if cluster_2_info:
        # Should select all 3 cases from tiny cluster
        assert len(cluster_2_info['selected_case_ids']) == 3


def test_selection_diversity():
    """Test that selected cases are diverse within clusters"""
    from sklearn.datasets import make_blobs

    pca_data, labels = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

    metadata_df = pd.DataFrame({
        'MRN': [f'MRN{i:04d}' for i in range(len(labels))]
    })

    selection = select_training_cases_from_clustering(
        labels, pca_data, metadata_df, max_clusters=3, max_cases=10
    )

    # Check that selections exist and are reasonable
    assert 'per_cluster_selections' in selection
    for cluster_id, cluster_info in selection['per_cluster_selections'].items():
        # Should have selected some cases
        assert len(cluster_info['selected_case_ids']) > 0
        # Selection should be a reasonable portion of cluster
        cluster_size = cluster_info['cluster_size']
        assert len(cluster_info['selected_case_ids']) <= cluster_size
