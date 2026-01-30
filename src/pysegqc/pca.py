"""
PCA dimensionality reduction for radiomics clustering analysis.

This module performs Principal Component Analysis on standardized features.
"""

import numpy as np
from sklearn.decomposition import PCA


def perform_pca(data, n_components=None):
    """Perform PCA on standardized data."""
    print(f"\n{'='*70}")
    print(f"Performing PCA")
    print(f"{'='*70}")

    if n_components is None:
        n_components = min(data.shape[0], data.shape[1])

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\nPCA Results:")
    print(f"  Components: {n_components}")
    print(f"  Variance explained by PC1: {explained_var[0]:.1%}")
    if len(explained_var) > 1:
        print(f"  Variance explained by PC1-2: {cumulative_var[1]:.1%}")
    if len(explained_var) > 2:
        print(f"  Variance explained by PC1-3: {cumulative_var[2]:.1%}")

    # Find number of components for 90% variance
    n_for_90 = np.argmax(cumulative_var >= 0.90) + 1
    print(f"  Components for 90% variance: {n_for_90}")

    return pca, transformed, explained_var
