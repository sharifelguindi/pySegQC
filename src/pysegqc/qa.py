"""
QA outlier detection and verdict computation for radiomics clustering.

Combines distance-based z-scores (intra-cluster) with Isolation Forest
(global) to produce per-case QA verdicts: pass / review / fail.

Verdict rules:
    PASS   - Neither method flags the case
    REVIEW - Exactly one method flags (distance OR Isolation Forest)
    FAIL   - Both methods flag
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from .validation import detect_outliers

logger = logging.getLogger(__name__)


def _compute_risk_scores(distance_z_scores, iforest_scores, n_samples):
    """Combine distance z-scores and Isolation Forest scores into a 0-1 risk score."""
    dist_z_norm = np.clip(distance_z_scores / 5.0, 0.0, 1.0)

    if_min, if_max = iforest_scores.min(), iforest_scores.max()
    if if_max > if_min:
        if_norm = 1.0 - (iforest_scores - if_min) / (if_max - if_min)
    else:
        if_norm = np.zeros(n_samples)

    return 0.6 * dist_z_norm + 0.4 * if_norm


def _combine_verdicts(distance_outlier_mask, iforest_outlier_mask, n_samples):
    """Combine two outlier masks into pass/review/fail verdicts."""
    verdicts = np.array(['pass'] * n_samples, dtype=object)
    both_flag = distance_outlier_mask & iforest_outlier_mask
    either_flag = distance_outlier_mask ^ iforest_outlier_mask  # XOR = exactly one
    verdicts[either_flag] = 'review'
    verdicts[both_flag] = 'fail'
    return verdicts


def compute_qa_verdicts(
    pca_data: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    distance_sigma: float = 2.0,
    iforest_contamination: float = 0.1,
) -> Dict[str, Any]:
    """
    Combined outlier detection producing per-case QA verdicts.

    Args:
        pca_data: PCA-transformed data (n_samples, n_components)
        cluster_labels: Cluster assignment per sample
        centroids: Cluster centroids in PCA space (n_clusters, n_components)
        distance_sigma: Z-score threshold for distance-based flagging (default: 2.0)
        iforest_contamination: Expected outlier proportion for Isolation Forest (default: 0.1)

    Returns:
        Dictionary with:
            verdicts: array of 'pass' | 'review' | 'fail' per case
            qa_risk_scores: combined numeric risk score (0-1) per case
            distance_to_centroid: Euclidean distance per case
            distance_z_scores: within-cluster z-score per case
            distance_outlier_mask: boolean, True if z-score > sigma threshold
            iforest_outlier_mask: boolean, True if Isolation Forest flags
            iforest_scores: raw anomaly scores from Isolation Forest
            per_cluster_stats: dict with per-cluster distance mean/std (for prediction reuse)
    """
    n_samples = len(pca_data)
    n_clusters = len(centroids)

    # --- Distance-based detection (intra-cluster) ---
    distances = np.linalg.norm(pca_data - centroids[cluster_labels], axis=1)

    # Compute per-cluster mean and std for z-score
    cluster_stats = {}
    distance_z_scores = np.zeros(n_samples)

    for k in range(n_clusters):
        mask = cluster_labels == k
        if mask.sum() < 2:
            # Single-member cluster: can't compute std, mark as pass
            cluster_stats[k] = {'mean': distances[mask].mean() if mask.any() else 0.0, 'std': 0.0}
            distance_z_scores[mask] = 0.0
            continue

        cluster_distances = distances[mask]
        mu = cluster_distances.mean()
        sigma = cluster_distances.std()
        cluster_stats[k] = {'mean': float(mu), 'std': float(sigma)}

        if sigma > 0:
            distance_z_scores[mask] = (cluster_distances - mu) / sigma
        else:
            distance_z_scores[mask] = 0.0

    distance_outlier_mask = distance_z_scores > distance_sigma

    # --- Isolation Forest detection (global) ---
    iforest_results = detect_outliers(pca_data, contamination=iforest_contamination)
    iforest_outlier_mask = iforest_results['outlier_mask']
    iforest_scores = iforest_results['outlier_scores']

    # --- Combine into verdicts ---
    verdicts = _combine_verdicts(distance_outlier_mask, iforest_outlier_mask, n_samples)
    qa_risk_scores = _compute_risk_scores(distance_z_scores, iforest_scores, n_samples)

    # --- Summary logging ---
    n_pass = (verdicts == 'pass').sum()
    n_review = (verdicts == 'review').sum()
    n_fail = (verdicts == 'fail').sum()
    logger.info(f"QA verdicts: {n_pass} pass, {n_review} review, {n_fail} fail")
    logger.info(f"  Distance-based: {distance_outlier_mask.sum()} flagged (>{distance_sigma}sigma)")
    logger.info(f"  Isolation Forest: {iforest_outlier_mask.sum()} flagged")

    return {
        'verdicts': verdicts,
        'qa_risk_scores': qa_risk_scores,
        'distance_to_centroid': distances,
        'distance_z_scores': distance_z_scores,
        'distance_outlier_mask': distance_outlier_mask,
        'iforest_outlier_mask': iforest_outlier_mask,
        'iforest_scores': iforest_scores,
        'per_cluster_stats': cluster_stats,
    }


def create_neutral_qa_results(
    pca_data: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
) -> Dict[str, Any]:
    """
    Create all-pass QA results without running outlier detection.

    Used in training case selection mode where data is assumed clean,
    so expensive Isolation Forest / z-score flagging is unnecessary.
    The returned dict is structurally identical to compute_qa_verdicts()
    so all downstream consumers (dashboard, export, viewer) work unchanged.

    Real distance_to_centroid values are still computed — they're cheap
    and useful for training case ranking.
    """
    n_samples = len(pca_data)
    n_clusters = len(centroids)

    distances = np.linalg.norm(pca_data - centroids[cluster_labels], axis=1)

    logger.info("QA detection skipped (training case selection mode) — all cases marked pass")

    return {
        'verdicts': np.array(['pass'] * n_samples, dtype=object),
        'qa_risk_scores': np.zeros(n_samples),
        'distance_to_centroid': distances,
        'distance_z_scores': np.zeros(n_samples),
        'distance_outlier_mask': np.zeros(n_samples, dtype=bool),
        'iforest_outlier_mask': np.zeros(n_samples, dtype=bool),
        'iforest_scores': np.zeros(n_samples),
        'per_cluster_stats': {k: {'mean': 0.0, 'std': 0.0} for k in range(n_clusters)},
    }


def compute_prediction_qa_verdicts(
    pca_data: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    training_cluster_stats: Dict[int, Dict[str, float]],
    trained_iforest: Optional[Any] = None,
    distance_sigma: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute QA verdicts for new prediction cases using training-established thresholds.

    Unlike compute_qa_verdicts(), this uses pre-computed cluster statistics from
    training to ensure consistent thresholds between training and prediction.

    Args:
        pca_data: PCA-transformed new case data
        cluster_labels: Predicted cluster assignments
        centroids: Training cluster centroids
        training_cluster_stats: Per-cluster distance mean/std from training
        trained_iforest: Pre-fitted IsolationForest model (optional)
        distance_sigma: Z-score threshold (should match training)

    Returns:
        Same structure as compute_qa_verdicts()
    """
    n_samples = len(pca_data)

    # Distance-based using training stats
    distances = np.linalg.norm(pca_data - centroids[cluster_labels], axis=1)

    distance_z_scores = np.zeros(n_samples)
    for i in range(n_samples):
        k = cluster_labels[i]
        stats = training_cluster_stats.get(k, {'mean': 0.0, 'std': 0.0})
        if stats['std'] > 0:
            distance_z_scores[i] = (distances[i] - stats['mean']) / stats['std']

    distance_outlier_mask = distance_z_scores > distance_sigma

    # Isolation Forest using trained model (or skip if not available)
    if trained_iforest is not None:
        iforest_labels = trained_iforest.predict(pca_data)
        iforest_scores = trained_iforest.score_samples(pca_data)
        iforest_outlier_mask = iforest_labels == -1
    else:
        iforest_outlier_mask = np.zeros(n_samples, dtype=bool)
        iforest_scores = np.zeros(n_samples)

    # Combine verdicts and risk scores
    verdicts = _combine_verdicts(distance_outlier_mask, iforest_outlier_mask, n_samples)
    qa_risk_scores = _compute_risk_scores(distance_z_scores, iforest_scores, n_samples)

    return {
        'verdicts': verdicts,
        'qa_risk_scores': qa_risk_scores,
        'distance_to_centroid': distances,
        'distance_z_scores': distance_z_scores,
        'distance_outlier_mask': distance_outlier_mask,
        'iforest_outlier_mask': iforest_outlier_mask,
        'iforest_scores': iforest_scores,
        'per_cluster_stats': training_cluster_stats,
    }
