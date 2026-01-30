"""Tests for QA outlier detection and verdict computation."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from pysegqc.qa import compute_qa_verdicts, compute_prediction_qa_verdicts


@pytest.fixture
def clustered_data():
    """Create well-separated 3-cluster data with known outliers."""
    np.random.seed(42)

    # 3 tight clusters
    data, labels = make_blobs(n_samples=90, centers=3, n_features=5,
                              cluster_std=1.0, random_state=42)

    # Add 3 extreme outliers (one per cluster, way outside normal range)
    outlier_0 = data[labels == 0].mean(axis=0) + 15.0  # Very far from cluster 0
    outlier_1 = data[labels == 1].mean(axis=0) + 15.0
    outlier_2 = data[labels == 2].mean(axis=0) + 15.0

    data = np.vstack([data, outlier_0, outlier_1, outlier_2])
    labels = np.append(labels, [0, 1, 2])

    # Compute centroids
    n_clusters = 3
    centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

    return data, labels, centroids


def test_compute_qa_verdicts_returns_expected_keys(clustered_data):
    """Test that compute_qa_verdicts returns all expected keys."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)

    expected_keys = [
        'verdicts', 'qa_risk_scores', 'distance_to_centroid',
        'distance_z_scores', 'distance_outlier_mask',
        'iforest_outlier_mask', 'iforest_scores', 'per_cluster_stats'
    ]
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"


def test_compute_qa_verdicts_shapes(clustered_data):
    """Test that output arrays have correct shapes."""
    data, labels, centroids = clustered_data
    n_samples = len(data)

    results = compute_qa_verdicts(data, labels, centroids)

    assert len(results['verdicts']) == n_samples
    assert len(results['qa_risk_scores']) == n_samples
    assert len(results['distance_to_centroid']) == n_samples
    assert len(results['distance_z_scores']) == n_samples
    assert len(results['distance_outlier_mask']) == n_samples
    assert len(results['iforest_outlier_mask']) == n_samples
    assert len(results['iforest_scores']) == n_samples


def test_verdict_values(clustered_data):
    """Test that verdicts are only 'pass', 'review', or 'fail'."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)

    valid_verdicts = {'pass', 'review', 'fail'}
    actual_verdicts = set(results['verdicts'])
    assert actual_verdicts.issubset(valid_verdicts), f"Invalid verdicts: {actual_verdicts - valid_verdicts}"


def test_verdict_logic(clustered_data):
    """Test verdict combines distance and IF flags correctly."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)

    dist_flag = results['distance_outlier_mask']
    if_flag = results['iforest_outlier_mask']
    verdicts = results['verdicts']

    for i in range(len(data)):
        if dist_flag[i] and if_flag[i]:
            assert verdicts[i] == 'fail', f"Case {i}: both flagged but verdict={verdicts[i]}"
        elif dist_flag[i] != if_flag[i]:  # XOR: exactly one
            assert verdicts[i] == 'review', f"Case {i}: one flagged but verdict={verdicts[i]}"
        else:
            assert verdicts[i] == 'pass', f"Case {i}: neither flagged but verdict={verdicts[i]}"


def test_risk_scores_bounded(clustered_data):
    """Test that risk scores are in [0, 1] range."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)

    assert np.all(results['qa_risk_scores'] >= 0.0)
    assert np.all(results['qa_risk_scores'] <= 1.0)


def test_outliers_have_higher_risk(clustered_data):
    """Test that flagged cases have higher risk scores than unflagged."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)

    pass_mask = results['verdicts'] == 'pass'
    fail_mask = results['verdicts'] == 'fail'

    if fail_mask.any() and pass_mask.any():
        mean_pass_risk = results['qa_risk_scores'][pass_mask].mean()
        mean_fail_risk = results['qa_risk_scores'][fail_mask].mean()
        assert mean_fail_risk > mean_pass_risk, \
            f"Fail risk ({mean_fail_risk:.3f}) should be > pass risk ({mean_pass_risk:.3f})"


def test_per_cluster_stats(clustered_data):
    """Test per-cluster statistics are populated correctly."""
    data, labels, centroids = clustered_data

    results = compute_qa_verdicts(data, labels, centroids)
    stats = results['per_cluster_stats']

    n_clusters = len(centroids)
    assert len(stats) == n_clusters

    for k in range(n_clusters):
        assert 'mean' in stats[k]
        assert 'std' in stats[k]
        assert stats[k]['mean'] >= 0
        assert stats[k]['std'] >= 0


def test_single_member_cluster():
    """Test handling of single-member clusters (can't compute std)."""
    data = np.array([[0, 0], [10, 10], [20, 20]])
    labels = np.array([0, 1, 2])
    centroids = data.copy()

    results = compute_qa_verdicts(data, labels, centroids)

    # Single-member clusters: z-score = 0 (no dispersion to measure)
    assert np.all(results['distance_z_scores'] == 0.0)


def test_custom_sigma_threshold(clustered_data):
    """Test that stricter sigma catches more cases."""
    data, labels, centroids = clustered_data

    results_strict = compute_qa_verdicts(data, labels, centroids, distance_sigma=1.0)
    results_loose = compute_qa_verdicts(data, labels, centroids, distance_sigma=3.0)

    strict_flagged = results_strict['distance_outlier_mask'].sum()
    loose_flagged = results_loose['distance_outlier_mask'].sum()

    assert strict_flagged >= loose_flagged, \
        f"Stricter sigma should flag more: {strict_flagged} vs {loose_flagged}"


def test_compute_prediction_qa_verdicts(clustered_data):
    """Test prediction QA using training stats."""
    data, labels, centroids = clustered_data

    # First get training stats
    training_results = compute_qa_verdicts(data, labels, centroids)
    training_stats = training_results['per_cluster_stats']

    # Create "new" data (subset of training as test)
    new_data = data[:10]
    new_labels = labels[:10]

    pred_results = compute_prediction_qa_verdicts(
        new_data, new_labels, centroids, training_stats
    )

    assert len(pred_results['verdicts']) == 10
    assert set(pred_results['verdicts']).issubset({'pass', 'review', 'fail'})


def test_prediction_qa_without_iforest(clustered_data):
    """Test prediction QA gracefully handles missing IF model."""
    data, labels, centroids = clustered_data
    training_results = compute_qa_verdicts(data, labels, centroids)

    pred_results = compute_prediction_qa_verdicts(
        data[:5], labels[:5], centroids,
        training_results['per_cluster_stats'],
        trained_iforest=None  # No IF model
    )

    # Without IF, only distance-based flags should exist
    assert not pred_results['iforest_outlier_mask'].any()
    assert len(pred_results['verdicts']) == 5
