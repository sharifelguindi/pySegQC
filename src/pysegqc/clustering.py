"""
Clustering algorithms for radiomics analysis.

This module implements hierarchical (Ward linkage) and k-means clustering.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def perform_hierarchical_clustering(pca_data, n_clusters):
    """
    Perform hierarchical clustering with Ward linkage on PCA data.

    Args:
        pca_data: PCA-transformed data
        n_clusters: Number of clusters to create

    Returns:
        tuple: (clustering_model, cluster_labels)
    """
    print(f"\n{'='*70}")
    print(f"Performing Hierarchical Clustering (Ward Linkage, k={n_clusters})")
    print(f"{'='*70}")

    # Perform hierarchical clustering with Ward linkage
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(pca_data)

    # Calculate silhouette score
    sil_score = silhouette_score(pca_data, labels)

    print(f"\nClustering Results:")
    print(f"  Method: Ward Linkage (deterministic)")
    print(f"  Silhouette Score: {sil_score:.3f}")

    # Show cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")

    return clusterer, labels


def perform_kmeans(pca_data, n_clusters):
    """
    Perform k-means clustering on PCA data.

    Note: K-Means is kept for backward compatibility.
    Hierarchical (Ward) is now the default method.
    """
    print(f"\n{'='*70}")
    print(f"Performing K-Means Clustering (k={n_clusters})")
    print(f"{'='*70}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(pca_data)

    inertia = kmeans.inertia_
    sil_score = silhouette_score(pca_data, labels)

    print(f"\nClustering Results:")
    print(f"  Inertia: {inertia:.0f}")
    print(f"  Silhouette Score: {sil_score:.3f}")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")

    return kmeans, labels
