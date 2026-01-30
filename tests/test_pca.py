"""Test PCA functionality."""

import pytest
import numpy as np
import pandas as pd
from pysegqc import perform_pca


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    np.random.seed(42)
    # Create correlated features
    n_samples = 100
    n_features = 50

    # Generate random data
    data = np.random.randn(n_samples, n_features)

    # Add some correlation structure
    data[:, 1] = data[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1
    data[:, 2] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2

    return pd.DataFrame(data)


def test_pca_basic(sample_features):
    """Test basic PCA functionality."""
    pca_model, pca_data, explained_var = perform_pca(sample_features, n_components=10)

    assert pca_data.shape == (100, 10)
    assert len(explained_var) == 10
    assert np.sum(explained_var) <= 1.0
    assert np.all(explained_var >= 0)


def test_pca_variance_explained(sample_features):
    """Test that explained variance is in descending order."""
    _, _, explained_var = perform_pca(sample_features, n_components=10)

    # Check that variance is in descending order
    for i in range(len(explained_var) - 1):
        assert explained_var[i] >= explained_var[i + 1]


def test_pca_auto_components(sample_features):
    """Test PCA with automatic component selection."""
    pca_model, pca_data, explained_var = perform_pca(sample_features, n_components=None)

    assert pca_data.shape[0] == 100
    assert pca_data.shape[1] <= 50  # At most all features


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Suppress sklearn numerical warnings
def test_pca_transform(sample_features):
    """Test that PCA model can transform data."""
    pca_model, pca_data, _ = perform_pca(sample_features, n_components=5)

    # Transform should give same result
    pca_data_2 = pca_model.transform(sample_features)
    np.testing.assert_array_almost_equal(pca_data, pca_data_2)
