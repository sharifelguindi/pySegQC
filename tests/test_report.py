"""Tests for HTML dashboard and JSON report generation."""

import json
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from pysegqc.report import (
    generate_json_report,
    generate_html_dashboard,
    _get_case_id,
    _build_qa_summary_banner,
    _build_risk_table,
    _build_metrics_cards,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_qa_data():
    """Create minimal QA results matching compute_qa_verdicts output."""
    np.random.seed(42)
    n = 20
    labels = np.array([0] * 8 + [1] * 7 + [2] * 5)
    pca_data = np.random.randn(n, 5)
    centroids = np.array([pca_data[labels == k].mean(axis=0) for k in range(3)])

    verdicts = np.array(
        ['pass'] * 14 + ['review'] * 4 + ['fail'] * 2
    )
    risk_scores = np.random.uniform(0, 1, n)
    distances = np.random.uniform(0.5, 5.0, n)
    z_scores = np.random.uniform(0, 3.0, n)

    qa_results = {
        'verdicts': verdicts,
        'qa_risk_scores': risk_scores,
        'distance_to_centroid': distances,
        'distance_z_scores': z_scores,
        'distance_outlier_mask': np.array([False] * 16 + [True] * 4),
        'iforest_outlier_mask': np.array([False] * 18 + [True] * 2),
        'iforest_scores': np.random.uniform(-0.5, 0.5, n),
        'per_cluster_stats': {
            0: {'mean': 2.1, 'std': 0.8},
            1: {'mean': 1.9, 'std': 0.7},
            2: {'mean': 2.5, 'std': 1.1},
        },
    }

    metadata_df = pd.DataFrame({
        'Case_ID': [f'patient_{i:03d}' for i in range(n)],
        'MRN': [f'MRN{i:05d}' for i in range(n)],
    })

    analysis_config = {
        'method': 'hierarchical',
        'mode': 'concat',
        'volume_independent': True,
        'silhouette': 0.452,
        'stability': 0.91,
        'robustness': 0.88,
        'n_clusters': 3,
        'distance_sigma': 2.0,
    }

    return {
        'metadata_df': metadata_df,
        'pca_data': pca_data,
        'cluster_labels': labels,
        'centroids': centroids,
        'qa_results': qa_results,
        'analysis_config': analysis_config,
    }


@pytest.fixture
def sample_figures():
    """Create minimal Plotly figures for dashboard embedding."""
    figs = {}
    for name in ['pca_2d', 'pca_3d', 'scree', 'dendrogram', 'elbow']:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name=name))
        fig.update_layout(title=name)
        figs[name] = fig
    return figs


# ── JSON Report Tests ─────────────────────────────────────────────────


class TestJSONReport:

    def test_generates_valid_json(self, tmp_path, sample_qa_data):
        """Output is valid JSON with expected top-level keys."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)

        for key in ('version', 'generated_at', 'tool', 'tool_version',
                     'summary', 'cases', 'clusters', 'diagnostics'):
            assert key in loaded

    def test_summary_verdict_counts(self, tmp_path, sample_qa_data):
        """Summary pass/review/fail counts match input."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        s = report['summary']
        assert s['pass_count'] == 14
        assert s['review_count'] == 4
        assert s['fail_count'] == 2
        assert s['total_cases'] == 20
        assert s['verdict'] == 'FAIL'  # has fail cases

    def test_cases_length(self, tmp_path, sample_qa_data):
        """Per-case array has one entry per sample."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        assert len(report['cases']) == 20

    def test_case_entry_keys(self, tmp_path, sample_qa_data):
        """Each case entry has all expected fields."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        expected_keys = {
            'case_id', 'cluster', 'qa_verdict', 'qa_risk_score',
            'distance_to_centroid', 'distance_z_score',
            'distance_outlier', 'iforest_outlier', 'pc_coordinates',
        }
        for case in report['cases']:
            assert set(case.keys()) == expected_keys

    def test_clusters_count(self, tmp_path, sample_qa_data):
        """Cluster array has one entry per cluster."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        assert len(report['clusters']) == 3

    def test_diagnostics_outlier_detection(self, tmp_path, sample_qa_data):
        """Diagnostics include outlier detection info."""
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        od = report['diagnostics']['outlier_detection']
        assert od['distance_method']['sigma_threshold'] == 2.0
        assert od['combined']['pass'] == 14

    def test_prediction_info_optional(self, tmp_path, sample_qa_data):
        """Prediction info is included when provided."""
        out = tmp_path / 'report.json'
        pred_info = {'model_path': '/some/path', 'new_cases': 10}
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
            prediction_info=pred_info,
        )

        assert 'prediction' in report
        assert report['prediction']['new_cases'] == 10

    def test_overall_verdict_review(self, tmp_path, sample_qa_data):
        """Overall verdict is REVIEW when no fails but some reviews."""
        # Modify to have no fails
        sample_qa_data['qa_results']['verdicts'] = np.array(
            ['pass'] * 16 + ['review'] * 4
        )
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        assert report['summary']['verdict'] == 'REVIEW'

    def test_overall_verdict_pass(self, tmp_path, sample_qa_data):
        """Overall verdict is PASS when all pass."""
        sample_qa_data['qa_results']['verdicts'] = np.array(['pass'] * 20)
        out = tmp_path / 'report.json'
        report = generate_json_report(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['pca_data'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['centroids'],
            sample_qa_data['qa_results'],
            sample_qa_data['analysis_config'],
        )

        assert report['summary']['verdict'] == 'PASS'


# ── HTML Dashboard Tests ──────────────────────────────────────────────


class TestHTMLDashboard:

    def test_generates_html_file(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard writes a valid HTML file."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert out.exists()
        assert '<!DOCTYPE html>' in html
        assert 'plotly-2.26.0.min.js' in html

    def test_contains_qa_summary(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard includes QA summary banner with counts."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        # Card-based layout: stat-title and stat-value are in separate divs
        assert '>Pass<' in html
        assert '>14<' in html
        assert '>Review<' in html
        assert '>Fail<' in html

    def test_contains_risk_table(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard includes risk table with case IDs."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert 'risk-table' in html
        assert 'patient_000' in html

    def test_contains_plotly_divs(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard embeds Plotly figure divs."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert 'plot-pca_2d' in html
        assert 'plot-scree' in html

    def test_thumbnail_panel_present(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard includes thumbnail side panel."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert 'thumbnail-panel' in html
        assert 'panel-thumbnail' in html

    def test_handles_no_thumbnails(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard works without thumbnails."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
            thumbnails=None,
        )

        assert out.exists()
        assert '{}' in html  # empty thumbnail data

    def test_handles_thumbnails(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard includes thumbnail data when provided."""
        out = tmp_path / 'dashboard.html'
        thumbs = {0: 'AAAA', 5: 'BBBB'}
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
            thumbnails=thumbs,
        )

        assert 'AAAA' in html
        assert 'BBBB' in html

    def test_viewer_button_present_when_enabled(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard includes viewer link when has_viewer=True."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
            has_viewer=True,
        )

        assert 'panel-viewer-link' in html
        assert 'var hasViewer = true' in html

    def test_viewer_button_hidden_when_disabled(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard hides viewer link when has_viewer=False (default)."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert 'panel-viewer-link' in html  # element is always in DOM
        assert 'var hasViewer = false' in html

    def test_missing_figures_skipped(self, tmp_path, sample_qa_data):
        """Dashboard handles missing figure keys gracefully."""
        out = tmp_path / 'dashboard.html'
        # Only provide one figure
        figs = {'pca_2d': go.Figure().add_trace(go.Scatter(x=[1], y=[2]))}
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            figs,
            sample_qa_data['analysis_config'],
        )

        assert 'plot-pca_2d' in html
        assert 'plot-dendrogram' not in html  # not provided

    def test_metrics_cards(self, tmp_path, sample_qa_data, sample_figures):
        """Dashboard shows metrics cards."""
        out = tmp_path / 'dashboard.html'
        html = generate_html_dashboard(
            out,
            sample_qa_data['metadata_df'],
            sample_qa_data['cluster_labels'],
            sample_qa_data['qa_results'],
            sample_figures,
            sample_qa_data['analysis_config'],
        )

        assert '0.452' in html  # silhouette
        assert 'Silhouette Score' in html


# ── Helper Tests ──────────────────────────────────────────────────────


class TestHelpers:

    def test_get_case_id_with_case_id_col(self):
        """Uses Case_ID column when available."""
        df = pd.DataFrame({'Case_ID': ['pat_001'], 'MRN': ['12345']})
        assert _get_case_id(df, 0) == 'pat_001'

    def test_get_case_id_falls_back_to_mrn(self):
        """Falls back to MRN when Case_ID not present."""
        df = pd.DataFrame({'MRN': ['12345']})
        assert _get_case_id(df, 0) == '12345'

    def test_get_case_id_falls_back_to_index(self):
        """Falls back to index when no ID columns present."""
        df = pd.DataFrame({'Other': ['x']}, index=[42])
        assert _get_case_id(df, 0) == '42'

    def test_build_metrics_cards_empty(self):
        """Returns empty string when no metrics provided."""
        result = _build_metrics_cards({})
        assert result == ''

    def test_build_metrics_cards_partial(self):
        """Returns cards for available metrics only."""
        result = _build_metrics_cards({'silhouette': 0.5})
        assert '0.500' in result
        assert 'Stability' not in result
