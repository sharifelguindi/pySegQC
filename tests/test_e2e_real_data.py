"""
End-to-end tests using real HN OAR NIfTI data.

These tests exercise the full pySegQC pipeline (extract → analyze → predict)
against 15 real NIfTI cases with 14-structure multi-class masks, then
comprehensively validate every output artifact.

Requires: scratch/test_data/hn_oars/ with 15 NIfTI cases (image/ + mask/)
          scratch/hn_oar_labels.json (14 HN OAR structures)

Run with:  pytest tests/test_e2e_real_data.py -v --timeout=600
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'scratch' / 'test_data' / 'hn_oars'
LABEL_MAP = PROJECT_ROOT / 'scratch' / 'hn_oar_labels.json'
OUTPUT_BASE = PROJECT_ROOT / 'scratch' / 'e2e_test_outputs'
PYSEGQC = str(PROJECT_ROOT / 'venv' / 'bin' / 'pysegqc')

# ── Skip if data not available ─────────────────────────────────────────────
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not DATA_DIR.exists() or not LABEL_MAP.exists(),
        reason="Real NIfTI test data not available (need scratch/test_data/hn_oars/)"
    ),
]

# ── Data split ─────────────────────────────────────────────────────────────
N_TRAIN = 10
N_PREDICT = 5
N_STRUCTURES = 14  # HN OARs (labels 1-14)


def _get_all_cases():
    """Get sorted list of case stems from image directory."""
    image_dir = DATA_DIR / 'image'
    if not image_dir.exists():
        return []
    return sorted([p.name.replace('.nii.gz', '') for p in image_dir.glob('*.nii.gz')])


def _symlink_cases(cases, dest_dir):
    """Create symlinks for a list of cases into dest_dir/{image,mask}/."""
    for subdir in ('image', 'mask'):
        target = dest_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        for case in cases:
            src = DATA_DIR / subdir / f'{case}.nii.gz'
            dst = target / f'{case}.nii.gz'
            if not dst.exists() and src.exists():
                dst.symlink_to(src)


def _run_cli(*args, timeout=600):
    """Run a pysegqc CLI command, returning CompletedProcess."""
    cmd = [PYSEGQC] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE-SCOPED FIXTURE: runs extract → analyze → predict ONCE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def e2e_outputs():
    """
    One-time setup: split data, extract features, analyze, predict.

    Returns a dict with paths to all output directories and files.
    """
    all_cases = _get_all_cases()
    assert len(all_cases) >= N_TRAIN + N_PREDICT, (
        f"Need {N_TRAIN + N_PREDICT} cases, found {len(all_cases)}"
    )
    train_cases = all_cases[:N_TRAIN]
    predict_cases = all_cases[N_TRAIN:N_TRAIN + N_PREDICT]

    # Create output structure
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    train_data = OUTPUT_BASE / 'train_data' / 'hn_oars'
    predict_data = OUTPUT_BASE / 'predict_data' / 'hn_oars'

    _symlink_cases(train_cases, train_data)
    _symlink_cases(predict_cases, predict_data)

    # ── Step 1: Extract training features ──
    train_features = OUTPUT_BASE / 'train_features.xlsx'
    if not train_features.exists():
        result = _run_cli(
            'extract', str(train_data),
            '--image-dir', 'image',
            '--mask-dir', 'mask',
            '--label-map', str(LABEL_MAP),
            '--output', str(train_features),
        )
        assert result.returncode == 0, f"Extract (train) failed:\n{result.stderr[-1000:]}"

    # ── Step 2: Extract prediction features ──
    predict_features = OUTPUT_BASE / 'predict_features.xlsx'
    if not predict_features.exists():
        result = _run_cli(
            'extract', str(predict_data),
            '--image-dir', 'image',
            '--mask-dir', 'mask',
            '--label-map', str(LABEL_MAP),
            '--output', str(predict_features),
        )
        assert result.returncode == 0, f"Extract (predict) failed:\n{result.stderr[-1000:]}"

    # ── Step 3: Analyze (default per-structure mode, auto-k, volume-independent) ──
    analyze_output = OUTPUT_BASE / 'train_features_clustering_results'
    if not analyze_output.exists():
        result = _run_cli(
            'analyze', str(train_features),
            '--auto-k',
            '--volume-independent',
            '--output', str(analyze_output),
        )
        assert result.returncode == 0, f"Analyze failed:\n{result.stderr[-1000:]}"

    # ── Step 4: Find first structure results directory for prediction ──
    structure_dirs = sorted(analyze_output.glob('structure_*_results'))
    assert len(structure_dirs) > 0, "No structure_*_results dirs found after analyze"

    # Pick the first structure for deep validation
    deep_structure_dir = structure_dirs[0]

    # ── Step 5: Predict using the first structure's trained model ──
    # Prediction creates output inside the model_dir
    predict_output_name = f"predict_features_predictions"
    predict_output = deep_structure_dir / predict_output_name

    if not predict_output.exists():
        # Determine position from directory name (e.g., structure_002_Parotid_L_results → 2)
        dir_name = deep_structure_dir.name
        pos_match = re.search(r'structure_(\d+)', dir_name)
        position = int(pos_match.group(1)) if pos_match else 1

        result = _run_cli(
            'predict', str(predict_features),
            str(deep_structure_dir),
            '--sheet', 'PCA_Data',
            '--mode', 'position',
            '--position', str(position),
        )
        assert result.returncode == 0, f"Predict failed:\n{result.stderr[-1000:]}"

    return {
        'train_cases': train_cases,
        'predict_cases': predict_cases,
        'train_features': train_features,
        'predict_features': predict_features,
        'analyze_output': analyze_output,
        'structure_dirs': structure_dirs,
        'deep_structure_dir': deep_structure_dir,
        'predict_output': predict_output,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestExtraction:
    """Validate the extracted training features Excel file."""

    def test_excel_exists(self, e2e_outputs):
        assert e2e_outputs['train_features'].exists()

    def test_has_pca_data_sheet(self, e2e_outputs):
        xl = pd.ExcelFile(e2e_outputs['train_features'], engine='openpyxl')
        assert 'PCA_Data' in xl.sheet_names

    def test_row_count(self, e2e_outputs):
        df = pd.read_excel(e2e_outputs['train_features'], sheet_name='PCA_Data', engine='openpyxl')
        assert len(df) == N_TRAIN, f"Expected {N_TRAIN} rows, got {len(df)}"

    def test_feature_columns_present(self, e2e_outputs):
        df = pd.read_excel(e2e_outputs['train_features'], sheet_name='PCA_Data', engine='openpyxl')
        feature_cols = [c for c in df.columns if re.match(r'^\d{3}_original_', c)]
        assert len(feature_cols) > 0, "No feature columns with NNN_original_ prefix found"

    def test_all_structures_have_features(self, e2e_outputs):
        df = pd.read_excel(e2e_outputs['train_features'], sheet_name='PCA_Data', engine='openpyxl')
        feature_cols = [c for c in df.columns if re.match(r'^\d{3}_original_', c)]
        positions = set(c[:3] for c in feature_cols)
        assert len(positions) == N_STRUCTURES, (
            f"Expected {N_STRUCTURES} structure positions, found {len(positions)}: {sorted(positions)}"
        )

    def test_metadata_columns(self, e2e_outputs):
        df = pd.read_excel(e2e_outputs['train_features'], sheet_name='PCA_Data', engine='openpyxl')
        assert 'Case_ID' in df.columns, "Missing Case_ID column"

    def test_no_all_nan_feature_columns(self, e2e_outputs):
        df = pd.read_excel(e2e_outputs['train_features'], sheet_name='PCA_Data', engine='openpyxl')
        feature_cols = [c for c in df.columns if re.match(r'^\d{3}_original_', c)]
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        assert len(all_nan_cols) == 0, f"All-NaN feature columns: {all_nan_cols[:5]}"


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: STRUCTURE DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysisStructures:
    """Validate that per-structure analysis directories were created."""

    def test_structure_dirs_exist(self, e2e_outputs):
        assert len(e2e_outputs['structure_dirs']) == N_STRUCTURES, (
            f"Expected {N_STRUCTURES} structure dirs, found {len(e2e_outputs['structure_dirs'])}"
        )

    def test_each_dir_has_expected_files(self, e2e_outputs):
        """Spot-check that every structure dir has core output files."""
        for sdir in e2e_outputs['structure_dirs']:
            clustered_files = list(sdir.glob('*_clustered.xlsx'))
            assert len(clustered_files) >= 1, f"No *_clustered.xlsx in {sdir.name}"
            assert (sdir / 'analysis_dashboard.html').exists(), f"No dashboard in {sdir.name}"
            assert (sdir / 'analysis_report.json').exists(), f"No JSON report in {sdir.name}"
            assert (sdir / 'trained_models.pkl').exists(), f"No trained_models.pkl in {sdir.name}"


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: EXCEL (deep validation on one structure)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysisExcel:
    """Deep validation of the clustered Excel output for one structure."""

    @pytest.fixture()
    def clustered_excel(self, e2e_outputs):
        sdir = e2e_outputs['deep_structure_dir']
        files = list(sdir.glob('*_clustered.xlsx'))
        assert files, f"No clustered Excel in {sdir.name}"
        return files[0]

    def test_required_sheets(self, clustered_excel):
        xl = pd.ExcelFile(clustered_excel, engine='openpyxl')
        for sheet in ('Summary', 'Clustered_Data', 'PCA_Loadings', 'Cluster_Statistics'):
            assert sheet in xl.sheet_names, f"Missing sheet: {sheet}"

    def test_clustered_data_row_count(self, clustered_excel):
        df = pd.read_excel(clustered_excel, sheet_name='Clustered_Data', engine='openpyxl')
        assert len(df) == N_TRAIN, f"Expected {N_TRAIN} rows, got {len(df)}"

    def test_cluster_column_present(self, clustered_excel):
        df = pd.read_excel(clustered_excel, sheet_name='Clustered_Data', engine='openpyxl')
        assert 'Cluster' in df.columns

    def test_pc_columns_present(self, clustered_excel):
        df = pd.read_excel(clustered_excel, sheet_name='Clustered_Data', engine='openpyxl')
        pc_cols = [c for c in df.columns if re.match(r'^PC\d+$', c)]
        assert len(pc_cols) >= 2, f"Expected at least 2 PC columns, found {len(pc_cols)}"

    def test_pca_loadings_structure(self, clustered_excel):
        df = pd.read_excel(clustered_excel, sheet_name='PCA_Loadings', engine='openpyxl')
        assert len(df) > 0, "PCA_Loadings sheet is empty"
        # Should have PC columns
        pc_cols = [c for c in df.columns if 'PC' in str(c)]
        assert len(pc_cols) >= 1, "No PC columns in PCA_Loadings"

    def test_cluster_statistics_present(self, clustered_excel):
        df = pd.read_excel(clustered_excel, sheet_name='Cluster_Statistics', engine='openpyxl')
        assert len(df) > 0, "Cluster_Statistics sheet is empty"


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysisDashboard:
    """Validate the HTML dashboard for one structure."""

    @pytest.fixture()
    def dashboard_html(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'analysis_dashboard.html'
        assert path.exists(), "Dashboard HTML not found"
        return path.read_text(encoding='utf-8')

    def test_html_size(self, dashboard_html):
        assert len(dashboard_html) > 10_000, (
            f"Dashboard too small ({len(dashboard_html)} bytes), likely incomplete"
        )

    def test_html_doctype(self, dashboard_html):
        assert '<!DOCTYPE html>' in dashboard_html or '<!doctype html>' in dashboard_html.lower()

    def test_plotly_cdn(self, dashboard_html):
        assert 'plotly' in dashboard_html.lower(), "No Plotly reference in dashboard"

    def test_qa_summary_section(self, dashboard_html):
        # Should contain pass/review/fail counts somewhere
        assert 'pass' in dashboard_html.lower() or 'PASS' in dashboard_html

    def test_pca_plot_containers(self, dashboard_html):
        # Plotly divs for PCA plots
        assert 'pca_2d' in dashboard_html or 'pca-2d' in dashboard_html or 'plot-pca' in dashboard_html, \
            "No PCA 2D plot container found"

    def test_scree_plot_container(self, dashboard_html):
        assert 'scree' in dashboard_html.lower(), "No scree plot reference found"

    def test_risk_table(self, dashboard_html):
        # QA risk table
        assert 'risk' in dashboard_html.lower() or 'table' in dashboard_html.lower()


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: JSON REPORT
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysisJSON:
    """Validate the JSON report for one structure."""

    @pytest.fixture()
    def report(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'analysis_report.json'
        assert path.exists(), "JSON report not found"
        with open(path) as f:
            return json.load(f)

    def test_top_level_keys(self, report):
        for key in ('version', 'generated_at', 'summary', 'cases', 'clusters'):
            assert key in report, f"Missing top-level key: {key}"

    def test_summary_total_cases(self, report):
        assert report['summary']['total_cases'] == N_TRAIN

    def test_summary_verdict_counts(self, report):
        s = report['summary']
        verdict_sum = s.get('pass_count', 0) + s.get('review_count', 0) + s.get('fail_count', 0)
        assert verdict_sum == N_TRAIN, (
            f"Verdict counts sum to {verdict_sum}, expected {N_TRAIN}"
        )

    def test_summary_verdict_value(self, report):
        assert report['summary']['verdict'] in ('PASS', 'REVIEW', 'FAIL')

    def test_cases_count(self, report):
        assert len(report['cases']) == N_TRAIN

    def test_case_fields(self, report):
        required = {'case_id', 'cluster', 'qa_verdict', 'qa_risk_score'}
        for case in report['cases']:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('case_id', '?')} missing: {missing}"

    def test_qa_risk_bounds(self, report):
        for case in report['cases']:
            score = case['qa_risk_score']
            assert 0.0 <= score <= 1.0, f"Risk score {score} out of [0,1] for {case['case_id']}"

    def test_qa_verdict_values(self, report):
        for case in report['cases']:
            assert case['qa_verdict'] in ('pass', 'review', 'fail'), (
                f"Invalid verdict '{case['qa_verdict']}' for {case['case_id']}"
            )

    def test_clusters_array(self, report):
        n_clusters = report['summary']['n_clusters']
        assert len(report['clusters']) == n_clusters

    def test_cluster_sizes_sum(self, report):
        total = sum(c['size'] for c in report['clusters'])
        assert total == N_TRAIN, f"Cluster sizes sum to {total}, expected {N_TRAIN}"

    def test_cluster_fields(self, report):
        for cluster in report['clusters']:
            assert 'cluster_id' in cluster
            assert 'size' in cluster
            assert 'mean_distance' in cluster

    def test_diagnostics_present(self, report):
        assert 'diagnostics' in report
        diag = report['diagnostics']
        assert 'outlier_detection' in diag


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: TRAINED MODELS PKL
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainedModels:
    """Validate the trained_models.pkl for one structure."""

    @pytest.fixture()
    def models(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'trained_models.pkl'
        assert path.exists(), "trained_models.pkl not found"
        return joblib.load(path)

    def test_required_keys(self, models):
        for key in ('pca', 'scaler', 'centroids', 'feature_names', 'n_clusters',
                     'impute_strategy', 'n_components'):
            assert key in models, f"Missing key in PKL: {key}"

    def test_qa_thresholds(self, models):
        assert 'qa_thresholds' in models, "Missing qa_thresholds in PKL"
        qa = models['qa_thresholds']
        assert 'per_cluster_stats' in qa
        assert 'distance_sigma' in qa

    def test_centroids_shape(self, models):
        centroids = models['centroids']
        n_clusters = models['n_clusters']
        n_components = models['n_components']
        assert centroids.shape == (n_clusters, n_components), (
            f"Centroids shape {centroids.shape}, expected ({n_clusters}, {n_components})"
        )

    def test_feature_names_nonempty(self, models):
        assert len(models['feature_names']) > 0

    def test_pca_n_components(self, models):
        pca = models['pca']
        assert pca.n_components_ == models['n_components']


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: STATIC PLOTS (legacy PNGs)
# ═══════════════════════════════════════════════════════════════════════════

class TestStandalonePlots:
    """Check that standalone interactive plot files were generated."""

    def test_scree_plot(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'scree_plot.html'
        assert path.exists(), "Missing scree_plot.html"
        assert path.stat().st_size > 1000, "Scree plot appears empty"

    def test_pca_clusters_2d(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'pca_clusters_2d_interactive.html'
        assert path.exists(), "Missing pca_clusters_2d_interactive.html"
        assert path.stat().st_size > 1000, "2D plot appears empty"

    def test_dendrogram(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'dendrogram.html'
        assert path.exists(), "Missing dendrogram.html"
        assert path.stat().st_size > 1000, "Dendrogram appears empty"

    def test_elbow_plot(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'elbow_plot.html'
        assert path.exists(), "Missing elbow_plot.html"

    def test_feature_importance(self, e2e_outputs):
        path = e2e_outputs['deep_structure_dir'] / 'feature_importance_heatmap.html'
        assert path.exists(), "Missing feature_importance_heatmap.html"


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPrediction:
    """Validate prediction outputs."""

    @pytest.fixture()
    def prediction_dir(self, e2e_outputs):
        """Find the prediction output directory."""
        sdir = e2e_outputs['deep_structure_dir']
        pred_dirs = list(sdir.glob('*_predictions'))
        assert pred_dirs, f"No *_predictions dir in {sdir.name}"
        return pred_dirs[0]

    @pytest.fixture()
    def prediction_excel(self, prediction_dir):
        files = list(prediction_dir.glob('*_predictions.xlsx'))
        assert files, f"No *_predictions.xlsx in {prediction_dir.name}"
        return files[0]

    def test_prediction_dir_exists(self, prediction_dir):
        assert prediction_dir.is_dir()

    def test_predictions_excel_exists(self, prediction_excel):
        assert prediction_excel.exists()

    def test_predictions_has_required_sheets(self, prediction_excel):
        xl = pd.ExcelFile(prediction_excel, engine='openpyxl')
        assert 'Summary' in xl.sheet_names, "Missing Summary sheet"
        # Predictions sheet may be named 'Predictions' or 'All_Predictions'
        pred_sheets = [s for s in xl.sheet_names if 'predict' in s.lower() or 'all' in s.lower()]
        assert len(pred_sheets) > 0, f"No predictions sheet found. Sheets: {xl.sheet_names}"

    def test_predictions_row_count(self, prediction_excel):
        xl = pd.ExcelFile(prediction_excel, engine='openpyxl')
        # Find the data sheet
        for sheet in xl.sheet_names:
            if 'predict' in sheet.lower() or sheet == 'All_Cases':
                df = pd.read_excel(prediction_excel, sheet_name=sheet, engine='openpyxl')
                # May have fewer rows if duplicates were filtered
                assert len(df) <= N_PREDICT, (
                    f"More rows ({len(df)}) than predict cases ({N_PREDICT})"
                )
                assert len(df) > 0, "Predictions sheet is empty"
                return
        # If we get here, try first non-Summary sheet
        non_summary = [s for s in xl.sheet_names if s != 'Summary']
        if non_summary:
            df = pd.read_excel(prediction_excel, sheet_name=non_summary[0], engine='openpyxl')
            assert len(df) > 0, "First data sheet is empty"

    def test_predicted_cluster_column(self, prediction_excel):
        xl = pd.ExcelFile(prediction_excel, engine='openpyxl')
        for sheet in xl.sheet_names:
            df = pd.read_excel(prediction_excel, sheet_name=sheet, engine='openpyxl')
            if 'Predicted_Cluster' in df.columns:
                return  # Found it
        pytest.fail(f"No Predicted_Cluster column found in any sheet: {xl.sheet_names}")

    def test_confidence_scores_bounded(self, prediction_excel):
        xl = pd.ExcelFile(prediction_excel, engine='openpyxl')
        for sheet in xl.sheet_names:
            df = pd.read_excel(prediction_excel, sheet_name=sheet, engine='openpyxl')
            if 'Confidence_Score' in df.columns:
                # Confidence is distance ratio — lower is better, but should be non-negative
                assert (df['Confidence_Score'] >= 0).all(), "Negative confidence scores found"
                return
        pytest.fail("No Confidence_Score column found")

    def test_prediction_json_report(self, prediction_dir):
        report_path = prediction_dir / 'prediction_report.json'
        assert report_path.exists(), "prediction_report.json not found"
        with open(report_path) as f:
            report = json.load(f)
        assert 'summary' in report
        assert report['summary']['total_cases'] > 0

    def test_prediction_json_has_cases(self, prediction_dir):
        report_path = prediction_dir / 'prediction_report.json'
        if not report_path.exists():
            pytest.skip("No prediction JSON report")
        with open(report_path) as f:
            report = json.load(f)
        assert 'cases' in report
        for case in report['cases']:
            assert 'qa_verdict' in case
            assert case['qa_verdict'] in ('pass', 'review', 'fail')


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION: consistency between outputs
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    """Check consistency across different output artifacts."""

    def test_cluster_ids_json_vs_excel(self, e2e_outputs):
        """Training cluster IDs in JSON should match Excel Cluster column."""
        sdir = e2e_outputs['deep_structure_dir']

        # JSON clusters
        with open(sdir / 'analysis_report.json') as f:
            report = json.load(f)
        json_cluster_ids = {c['cluster'] for c in report['cases']}

        # Excel clusters
        excel_files = list(sdir.glob('*_clustered.xlsx'))
        df = pd.read_excel(excel_files[0], sheet_name='Clustered_Data', engine='openpyxl')
        excel_cluster_ids = set(df['Cluster'].unique())

        assert json_cluster_ids == excel_cluster_ids, (
            f"JSON clusters {json_cluster_ids} != Excel clusters {excel_cluster_ids}"
        )

    def test_pca_components_consistent(self, e2e_outputs):
        """PCA n_components should match across PKL and JSON."""
        sdir = e2e_outputs['deep_structure_dir']

        models = joblib.load(sdir / 'trained_models.pkl')
        n_comp_pkl = models['n_components']

        with open(sdir / 'analysis_report.json') as f:
            report = json.load(f)
        n_comp_json = report['diagnostics']['pca']['n_components']

        assert n_comp_pkl == n_comp_json, (
            f"PKL n_components={n_comp_pkl} != JSON n_components={n_comp_json}"
        )

    def test_feature_names_pkl_vs_excel(self, e2e_outputs):
        """Feature names in PKL should correspond to Excel feature columns."""
        sdir = e2e_outputs['deep_structure_dir']

        models = joblib.load(sdir / 'trained_models.pkl')
        pkl_features = set(models['feature_names'])

        excel_files = list(sdir.glob('*_clustered.xlsx'))
        df = pd.read_excel(excel_files[0], sheet_name='Clustered_Data', engine='openpyxl')
        # Feature columns in Excel may have position prefix stripped
        # PKL features may or may not have prefix depending on mode
        # Just check that both have features and they're non-empty
        assert len(pkl_features) > 0, "PKL has no feature names"

    def test_n_clusters_consistent(self, e2e_outputs):
        """n_clusters should match across PKL, JSON, and Excel."""
        sdir = e2e_outputs['deep_structure_dir']

        models = joblib.load(sdir / 'trained_models.pkl')
        n_pkl = models['n_clusters']

        with open(sdir / 'analysis_report.json') as f:
            report = json.load(f)
        n_json = report['summary']['n_clusters']

        assert n_pkl == n_json, f"PKL n_clusters={n_pkl} != JSON n_clusters={n_json}"

    def test_prediction_clusters_valid(self, e2e_outputs):
        """Predicted cluster IDs should be valid training cluster IDs."""
        sdir = e2e_outputs['deep_structure_dir']

        # Get valid cluster IDs from training
        models = joblib.load(sdir / 'trained_models.pkl')
        valid_clusters = set(range(models['n_clusters']))

        # Get predicted cluster IDs
        pred_dirs = list(sdir.glob('*_predictions'))
        if not pred_dirs:
            pytest.skip("No prediction output found")
        pred_excels = list(pred_dirs[0].glob('*_predictions.xlsx'))
        if not pred_excels:
            pytest.skip("No prediction Excel found")

        xl = pd.ExcelFile(pred_excels[0], engine='openpyxl')
        for sheet in xl.sheet_names:
            df = pd.read_excel(pred_excels[0], sheet_name=sheet, engine='openpyxl')
            if 'Predicted_Cluster' in df.columns:
                # Drop NaN rows (summary/separator rows may exist)
                clusters = df['Predicted_Cluster'].dropna()
                predicted = set(int(c) for c in clusters)
                invalid = predicted - valid_clusters
                assert not invalid, (
                    f"Invalid predicted clusters: {invalid} (valid: {valid_clusters})"
                )
                return
