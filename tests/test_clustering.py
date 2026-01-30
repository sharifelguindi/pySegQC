"""Test clustering functionality."""

import pytest
import numpy as np
from pysegqc import perform_hierarchical_clustering, perform_kmeans


@pytest.fixture
def sample_pca_data():
    """Create sample PCA data for testing."""
    np.random.seed(42)
    # Create 3 clear clusters
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([5, 5])
    cluster3 = np.random.randn(20, 2) + np.array([10, 0])
    return np.vstack([cluster1, cluster2, cluster3])


def test_hierarchical_clustering(sample_pca_data):
    """Test hierarchical clustering."""
    clusterer, labels = perform_hierarchical_clustering(sample_pca_data, n_clusters=3)

    assert len(labels) == 60
    assert len(np.unique(labels)) == 3
    assert all(label in [0, 1, 2] for label in labels)


def test_kmeans_clustering(sample_pca_data):
    """Test k-means clustering."""
    clusterer, labels = perform_kmeans(sample_pca_data, n_clusters=3)

    assert len(labels) == 60
    assert len(np.unique(labels)) == 3
    assert all(label in [0, 1, 2] for label in labels)


def test_clustering_reproducibility(sample_pca_data):
    """Test that hierarchical clustering is deterministic."""
    _, labels1 = perform_hierarchical_clustering(sample_pca_data, n_clusters=3)
    _, labels2 = perform_hierarchical_clustering(sample_pca_data, n_clusters=3)

    assert np.array_equal(labels1, labels2)


def test_invalid_n_clusters(sample_pca_data):
    """Test that invalid n_clusters raises appropriate error."""
    with pytest.raises((ValueError, Exception)):
        perform_hierarchical_clustering(sample_pca_data, n_clusters=100)
