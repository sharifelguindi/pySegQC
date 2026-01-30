import pytest
import pandas as pd
import numpy as np
from pysegqc.visualization import build_hover_text, extract_urls, get_cluster_colors


def test_build_hover_text_with_mrn():
    """Test hover text with MRN metadata."""
    metadata_df = pd.DataFrame({
        'MRN': ['12345', '67890'],
        'Patient_ID': ['PT001', 'PT002']
    })
    pca_data = np.array([[1.5, 2.3], [3.1, 4.2]])

    result = build_hover_text(
        index=0,
        metadata_df=metadata_df,
        pca_data=pca_data,
        label=1,
        pc_indices=[0, 1]
    )

    assert 'MRN: 12345' in result
    assert 'Cluster: 1' in result
    assert 'PC1: 1.50' in result
    assert 'PC2: 2.30' in result


def test_build_hover_text_without_metadata():
    """Test hover text fallback without metadata."""
    pca_data = np.array([[1.5, 2.3]])

    result = build_hover_text(
        index=0,
        metadata_df=None,
        pca_data=pca_data,
        label=2,
        pc_indices=[0, 1]
    )

    assert 'Sample: 0' in result
    assert 'Cluster: 2' in result
    assert 'PC1: 1.50' in result


def test_build_hover_text_with_url():
    """Test hover text with clickable URL instruction."""
    metadata_df = pd.DataFrame({
        'MRN': ['12345'],
        'View_Scan_URL': ['http://example.com/scan1']
    })
    pca_data = np.array([[1.0, 2.0]])

    result = build_hover_text(
        index=0,
        metadata_df=metadata_df,
        pca_data=pca_data,
        label=0,
        pc_indices=[0, 1]
    )

    assert 'Click to open scan viewer' in result


def test_build_hover_text_with_context():
    """Test hover text with context prefix (for radar plots)."""
    metadata_df = pd.DataFrame({'MRN': ['12345']})
    pca_data = np.array([[1.0, 2.0, 3.0]])

    result = build_hover_text(
        index=0,
        metadata_df=metadata_df,
        pca_data=pca_data,
        label=1,
        context="Centroid<br>",
        pc_indices=[0, 1, 2]
    )

    assert 'Centroid<br>MRN: 12345' in result
    assert 'PC3: 3.00' in result


def test_build_hover_text_3d():
    """Test hover text with 3 principal components."""
    metadata_df = pd.DataFrame({'MRN': ['12345']})
    pca_data = np.array([[1.0, 2.0, 3.0]])

    result = build_hover_text(
        index=0,
        metadata_df=metadata_df,
        pca_data=pca_data,
        label=0,
        pc_indices=[0, 1, 2]
    )

    assert 'PC1: 1.00' in result
    assert 'PC2: 2.00' in result
    assert 'PC3: 3.00' in result


def test_extract_urls_with_metadata():
    """Test URL extraction from metadata."""
    metadata_df = pd.DataFrame({
        'View_Scan_URL': ['http://example.com/1', 'http://example.com/2', None]
    })

    result = extract_urls([0, 1, 2], metadata_df)

    assert result == ['http://example.com/1', 'http://example.com/2', '']


def test_extract_urls_without_column():
    """Test URL extraction when column missing."""
    metadata_df = pd.DataFrame({'MRN': ['123']})

    result = extract_urls([0], metadata_df)

    assert result == ['']


def test_extract_urls_none_metadata():
    """Test URL extraction with None metadata."""
    result = extract_urls([0, 1], None)

    assert result == ['', '']


def test_get_cluster_colors():
    """Test cluster color palette generation."""
    colors = get_cluster_colors(5)

    assert len(colors) == 5
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith('#') or c.startswith('rgb') for c in colors)
