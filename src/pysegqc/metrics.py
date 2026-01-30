"""
Clustering quality metrics and evaluation.

This module provides functions for finding optimal cluster numbers, calculating
gap statistics, assessing stability, and performing consensus clustering.
"""

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_optimal_clusters(pca_data, max_k=10, method='hierarchical'):
    """
    Find optimal number of clusters using silhouette score.

    Args:
        pca_data: PCA-transformed data
        max_k: Maximum number of clusters to try
        method: 'hierarchical' (default) or 'kmeans'

    Returns:
        tuple: (silhouette_scores_list, best_k)
    """
    print(f"\n{'='*70}")
    print(f"Finding optimal number of clusters (k=2 to {max_k}) - Method: {method.upper()}")
    print(f"{'='*70}")

    silhouette_scores_list = []
    k_range = range(2, min(max_k + 1, len(pca_data)))

    for k in k_range:
        if method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        else:
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = clusterer.fit_predict(pca_data)
        sil_score = silhouette_score(pca_data, labels)
        silhouette_scores_list.append(sil_score)

        print(f"  k={k}: Silhouette={sil_score:.3f}")

    best_k = list(k_range)[np.argmax(silhouette_scores_list)]
    print(f"\n✓ Best k by silhouette score: {best_k} (score: {max(silhouette_scores_list):.3f})")

    return silhouette_scores_list, best_k


def assess_cluster_stability_bootstrap(pca_data, n_clusters, method='hierarchical',
                                       n_iterations=100, sample_fraction=0.8):
    """
    Assess cluster stability using bootstrap resampling.

    Measures how consistently samples are assigned to the same cluster across
    multiple bootstrap samples. High stability indicates robust clustering.

    Args:
        pca_data: PCA-transformed data (N x n_components)
        n_clusters: Number of clusters
        method: 'hierarchical' or 'kmeans'
        n_iterations: Number of bootstrap iterations (default: 100)
        sample_fraction: Fraction of data to sample each iteration (default: 0.8)

    Returns:
        dict: {
            'stability_scores': array of stability score for each sample,
            'mean_stability': overall mean stability,
            'cluster_stability': stability score for each cluster
        }
    """
    logger.info(f"Assessing cluster stability with {n_iterations} bootstrap iterations...")

    n_samples = pca_data.shape[0]
    sample_size = int(n_samples * sample_fraction)

    # Store cluster assignments across iterations
    cluster_matrix = np.zeros((n_samples, n_iterations))
    cluster_matrix[:] = np.nan

    for i in tqdm(range(n_iterations), desc="Bootstrap iterations"):
        # Resample data
        indices = resample(np.arange(n_samples), n_samples=sample_size, replace=False)
        bootstrap_data = pca_data[indices]

        # Cluster bootstrap sample
        if method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=i)

        labels = clusterer.fit_predict(bootstrap_data)
        cluster_matrix[indices, i] = labels

    # Calculate stability: proportion of times each sample clusters with its most frequent partners
    stability_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Get iterations where this sample was included
        included = ~np.isnan(cluster_matrix[i, :])
        if included.sum() < 2:
            stability_scores[i] = 0.0
            continue

        # For each included iteration, find samples in the same cluster
        same_cluster_counts = np.zeros(n_samples)
        for iter_idx in np.where(included)[0]:
            same_cluster = (cluster_matrix[:, iter_idx] == cluster_matrix[i, iter_idx])
            same_cluster_counts += same_cluster.astype(int)

        # Stability = avg proportion of times sample clustered with same neighbors
        stability_scores[i] = same_cluster_counts.mean() / included.sum()

    # Calculate per-cluster stability
    final_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    final_labels = final_clusterer.fit_predict(pca_data)

    cluster_stability = {}
    for k in range(n_clusters):
        cluster_mask = final_labels == k
        cluster_stability[k] = stability_scores[cluster_mask].mean()

    logger.info(f"Mean stability score: {stability_scores.mean():.3f}")

    return {
        'stability_scores': stability_scores,
        'mean_stability': stability_scores.mean(),
        'cluster_stability': cluster_stability
    }


def calculate_gap_statistic(pca_data, max_k=10, n_refs=20, method='hierarchical'):
    """
    Calculate gap statistic to find optimal number of clusters.

    Compares within-cluster dispersion to that expected under null reference
    distribution. The optimal k maximizes the gap statistic.

    Args:
        pca_data: PCA-transformed data
        max_k: Maximum number of clusters to try
        n_refs: Number of reference datasets to generate
        method: 'hierarchical' or 'kmeans'

    Returns:
        dict: {
            'gaps': gap statistic for each k,
            'optimal_k': k that maximizes gap,
            'within_dispersions': within-cluster dispersions,
            'ref_dispersions': expected dispersions under null
        }
    """
    logger.info(f"Calculating gap statistic for k=2 to {max_k}...")

    n_samples, n_features = pca_data.shape
    gaps = np.zeros(max_k - 1)
    within_dispersions = np.zeros(max_k - 1)
    ref_dispersions = np.zeros(max_k - 1)

    # Get data range for generating reference data
    mins = pca_data.min(axis=0)
    maxs = pca_data.max(axis=0)

    for k_idx, k in enumerate(tqdm(range(2, max_k + 1), desc="Gap statistic")):
        # Cluster actual data
        if method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        else:
            clusterer = KMeans(n_clusters=k, random_state=42)

        labels = clusterer.fit_predict(pca_data)

        # Calculate within-cluster dispersion
        wk = 0
        for cluster_id in range(k):
            cluster_points = pca_data[labels == cluster_id]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                wk += np.sum((cluster_points - center) ** 2)

        within_dispersions[k_idx] = np.log(wk + 1e-10)

        # Generate reference datasets and calculate expected dispersion
        ref_wks = []
        for _ in range(n_refs):
            # Generate uniform random data in same range
            ref_data = np.random.uniform(mins, maxs, size=(n_samples, n_features))

            # Cluster reference data
            if method == 'hierarchical':
                ref_clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            else:
                ref_clusterer = KMeans(n_clusters=k, random_state=42)

            ref_labels = ref_clusterer.fit_predict(ref_data)

            # Calculate reference dispersion
            ref_wk = 0
            for cluster_id in range(k):
                cluster_points = ref_data[ref_labels == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    ref_wk += np.sum((cluster_points - center) ** 2)

            ref_wks.append(np.log(ref_wk + 1e-10))

        ref_dispersions[k_idx] = np.mean(ref_wks)
        gaps[k_idx] = ref_dispersions[k_idx] - within_dispersions[k_idx]

    # Find optimal k
    optimal_idx = np.argmax(gaps)
    optimal_k = optimal_idx + 2

    logger.info(f"Gap statistic optimal k: {optimal_k} (gap={gaps[optimal_idx]:.3f})")

    return {
        'gaps': gaps,
        'optimal_k': optimal_k,
        'within_dispersions': within_dispersions,
        'ref_dispersions': ref_dispersions
    }


def perform_consensus_clustering(pca_data, n_clusters, n_iterations=100,
                                  sample_fraction=0.8, method='hierarchical'):
    """
    Perform consensus clustering to assess robustness.

    Runs clustering multiple times on bootstrap samples and builds a consensus
    matrix showing how often pairs of samples cluster together.

    Args:
        pca_data: PCA-transformed data
        n_clusters: Number of clusters
        n_iterations: Number of clustering iterations
        sample_fraction: Fraction of data to sample each iteration
        method: 'hierarchical' or 'kmeans'

    Returns:
        dict: {
            'consensus_matrix': NxN matrix of co-clustering frequencies,
            'consensus_labels': cluster labels from consensus matrix,
            'robustness_score': overall robustness metric
        }
    """
    logger.info(f"Performing consensus clustering ({n_iterations} iterations)...")

    n_samples = pca_data.shape[0]
    sample_size = int(n_samples * sample_fraction)

    # Consensus matrix: count how often pairs cluster together
    consensus_matrix = np.zeros((n_samples, n_samples))
    indicator_matrix = np.zeros((n_samples, n_samples))  # Track which pairs were sampled together

    for i in tqdm(range(n_iterations), desc="Consensus clustering"):
        # Resample
        indices = resample(np.arange(n_samples), n_samples=sample_size, replace=False)
        bootstrap_data = pca_data[indices]

        # Cluster
        if method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=i)

        labels = clusterer.fit_predict(bootstrap_data)

        # Update consensus matrix
        for idx_i, sample_i in enumerate(indices):
            for idx_j, sample_j in enumerate(indices):
                if labels[idx_i] == labels[idx_j]:
                    consensus_matrix[sample_i, sample_j] += 1
                indicator_matrix[sample_i, sample_j] += 1

    # Normalize by number of times pairs were sampled together
    with np.errstate(divide='ignore', invalid='ignore'):
        consensus_matrix = np.divide(consensus_matrix, indicator_matrix)
        consensus_matrix = np.nan_to_num(consensus_matrix)

    # Cluster the consensus matrix to get final labels
    consensus_dist = 1 - consensus_matrix
    final_clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='precomputed'  # Explicitly specify we're passing a distance matrix
    )
    consensus_labels = final_clusterer.fit_predict(consensus_dist)

    # Calculate robustness: mean within-cluster consensus
    robustness_scores = []
    for k in range(n_clusters):
        cluster_mask = consensus_labels == k
        cluster_consensus = consensus_matrix[np.ix_(cluster_mask, cluster_mask)]
        robustness_scores.append(cluster_consensus.mean())

    robustness_score = np.mean(robustness_scores)
    logger.info(f"Consensus clustering robustness: {robustness_score:.3f}")

    return {
        'consensus_matrix': consensus_matrix,
        'consensus_labels': consensus_labels,
        'robustness_score': robustness_score,
        'cluster_robustness': robustness_scores
    }
