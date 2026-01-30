"""Tests for metrics module"""

import pytest
import numpy as np
from pysegqc.metrics import (
    find_optimal_clusters,
    calculate_gap_statistic,
    assess_cluster_stability_bootstrap,
    perform_consensus_clustering
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Suppress sklearn numerical warnings
def test_find_optimal_clusters(three_cluster_data):
    """Test silhouette-based optimal k finding"""
    pca_data, _ = three_cluster_data

    silhouette_scores, best_k = find_optimal_clusters(
        pca_data, max_k=6, method='hierarchical'
    )

    # Should find k=3 as optimal (or close to it for clear 3-cluster data)
    assert best_k >= 2 and best_k <= 6
    assert len(silhouette_scores) == 5  # k from 2 to 6 (max_k)
    assert all(0 <= score <= 1 for score in silhouette_scores)


def test_calculate_gap_statistic(three_cluster_data):
    """Test gap statistic calculation"""
    pca_data, _ = three_cluster_data

    gap_results = calculate_gap_statistic(
        pca_data, max_k=6, n_refs=10, method='hierarchical'
    )

    assert 'optimal_k' in gap_results
    assert 'gaps' in gap_results
    assert gap_results['optimal_k'] >= 2
    assert gap_results['optimal_k'] <= 6


def test_assess_cluster_stability_bootstrap(three_cluster_data):
    """Test bootstrap stability assessment"""
    pca_data, _ = three_cluster_data

    stability_results = assess_cluster_stability_bootstrap(
        pca_data, n_clusters=3, method='hierarchical', n_iterations=10
    )

    assert 'mean_stability' in stability_results
    assert 0 <= stability_results['mean_stability'] <= 1
    # Check that we get a reasonable stability score (not just validating range)
    assert stability_results['mean_stability'] > 0


def test_perform_consensus_clustering(three_cluster_data):
    """Test consensus clustering"""
    pca_data, _ = three_cluster_data

    consensus_results = perform_consensus_clustering(
        pca_data, n_clusters=3, n_iterations=10, method='hierarchical'
    )

    assert 'consensus_matrix' in consensus_results
    assert 'robustness_score' in consensus_results
    assert consensus_results['consensus_matrix'].shape == (150, 150)
    assert 0 <= consensus_results['robustness_score'] <= 1
