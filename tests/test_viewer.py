"""Tests for the NiiVue viewer page generation module."""

import json

import numpy as np
import pandas as pd
import pytest

from pysegqc.viewer import generate_viewer_data, generate_viewer_html, _get_case_id


@pytest.fixture
def sample_viewer_data():
    """Create sample data for viewer tests."""
    metadata_df = pd.DataFrame({
        'Case_ID': ['patient001', 'patient002', 'patient003'],
        'Image_Path': ['/data/img1.nii.gz', '/data/img2.nii.gz', '/data/img3.nii.gz'],
        'Mask_Path': ['/data/mask1.nii.gz', '/data/mask2.nii.gz', '/data/mask3.nii.gz'],
    })
    cluster_labels = np.array([0, 1, 0])
    qa_results = {
        'verdicts': np.array(['pass', 'review', 'fail']),
        'qa_risk_scores': np.array([0.1, 0.55, 0.9]),
    }
    return metadata_df, cluster_labels, qa_results


class TestGetCaseId:
    """Tests for _get_case_id helper."""

    def test_uses_case_id_column(self):
        df = pd.DataFrame({'Case_ID': ['ABC123']})
        assert _get_case_id(df, 0) == 'ABC123'

    def test_falls_back_to_mrn(self):
        df = pd.DataFrame({'MRN': ['MRN001']})
        assert _get_case_id(df, 0) == 'MRN001'

    def test_falls_back_to_index(self):
        df = pd.DataFrame({'other_col': ['x']})
        assert _get_case_id(df, 0) == '0'


class TestGenerateViewerData:
    """Tests for JSON sidecar generation."""

    def test_writes_json_file(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        generate_viewer_data(output, metadata_df, cluster_labels, qa_results)
        assert output.exists()

    def test_json_structure(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)

        assert 'cases' in data
        assert len(data['cases']) == 3

    def test_case_fields(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)

        case = data['cases'][0]
        assert case['index'] == 0
        assert case['case_id'] == 'patient001'
        assert case['cluster'] == 0
        assert case['verdict'] == 'pass'
        assert isinstance(case['risk_score'], float)
        assert 'image_path' in case
        assert 'mask_path' in case

    def test_verdicts_preserved(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)

        verdicts = [c['verdict'] for c in data['cases']]
        assert verdicts == ['pass', 'review', 'fail']

    def test_json_parseable(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        generate_viewer_data(output, metadata_df, cluster_labels, qa_results)

        parsed = json.loads(output.read_text())
        assert len(parsed['cases']) == 3

    def test_no_image_path_column(self, tmp_path):
        metadata_df = pd.DataFrame({'Case_ID': ['a', 'b']})
        cluster_labels = np.array([0, 1])
        qa_results = {
            'verdicts': np.array(['pass', 'pass']),
            'qa_risk_scores': np.array([0.1, 0.2]),
        }
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)

        # Cases should still be generated, just without paths
        assert len(data['cases']) == 2
        assert 'image_path' not in data['cases'][0]

    def test_returns_data_dict(self, tmp_path, sample_viewer_data):
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        result = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)
        assert isinstance(result, dict)


class TestGenerateViewerHtml:
    """Tests for NiiVue HTML page generation."""

    def test_writes_html_file(self, tmp_path):
        output = tmp_path / 'viewer.html'
        generate_viewer_html(output)
        assert output.exists()

    def test_html_contains_niivue(self, tmp_path):
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'niivue' in html.lower()
        assert 'Niivue' in html

    def test_html_contains_canvas(self, tmp_path):
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert '<canvas' in html

    def test_html_contains_navigation(self, tmp_path):
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'Prev' in html
        assert 'Next' in html
        assert 'navigate(' in html

    def test_html_contains_viewer_data_fetch(self, tmp_path):
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'viewer_data.json' in html

    def test_html_returns_string(self, tmp_path):
        output = tmp_path / 'viewer.html'
        result = generate_viewer_html(output)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_inline_data_embedded(self, tmp_path):
        """When viewer_data is provided, it is inlined into the HTML."""
        output = tmp_path / 'viewer.html'
        data = {'cases': [{'index': 0, 'case_id': 'test123', 'cluster': 1,
                           'verdict': 'pass', 'risk_score': 0.05}]}
        html = generate_viewer_html(output, viewer_data=data)
        assert 'test123' in html
        assert '"cluster": 1' in html

    def test_null_without_data(self, tmp_path):
        """Without viewer_data, INLINE_DATA is null (falls back to fetch)."""
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'const INLINE_DATA = null;' in html

    def test_html_multiplanar_mode(self, tmp_path):
        """NiiVue is configured for multiplanar (all 3 planes)."""
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'sliceType: 3' in html
        assert 'multiplanarForceRender: true' in html
        assert 'show3Dcrosshair: false' in html
        assert 'crosshairWidth: 0' in html

    def test_html_contour_mode(self, tmp_path):
        """Mask is converted to boundary contour via edge detection."""
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'img.slice()' in html
        assert 'interior' in html

    def test_html_zoom_controls(self, tmp_path):
        """Zoom buttons are present."""
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'zoomIn' in html
        assert 'zoomOut' in html
        assert 'volScaleMultiplier' in html


class TestStructureLabelSupport:
    """Tests for single-structure mask display."""

    def test_structure_label_in_json(self, tmp_path, sample_viewer_data):
        """structure_label is written to JSON when provided."""
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(
            output, metadata_df, cluster_labels, qa_results,
            structure_label=2, structure_name='Brainstem',
        )
        assert data['structure_label'] == 2
        assert data['structure_name'] == 'Brainstem'
        # Also verify persisted JSON
        parsed = json.loads(output.read_text())
        assert parsed['structure_label'] == 2
        assert parsed['structure_name'] == 'Brainstem'

    def test_no_structure_label_by_default(self, tmp_path, sample_viewer_data):
        """structure_label is absent from JSON when not provided."""
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(output, metadata_df, cluster_labels, qa_results)
        assert 'structure_label' not in data
        assert 'structure_name' not in data

    def test_voxel_zeroing_in_html(self, tmp_path, sample_viewer_data):
        """HTML JS contains voxel zeroing logic for single-structure masks."""
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output_json = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(
            output_json, metadata_df, cluster_labels, qa_results,
            structure_label=3, structure_name='Parotid_L',
        )
        output_html = tmp_path / 'viewer.html'
        html = generate_viewer_html(output_html, viewer_data=data)
        assert 'Math.round(img[i])' in html
        assert 'updateGLVolume' in html
        assert '"structure_label": 3' in html
        assert 'Parotid_L' in html

    def test_window_level_controls_in_html(self, tmp_path):
        """HTML contains window/level preset buttons and custom inputs."""
        output = tmp_path / 'viewer.html'
        html = generate_viewer_html(output)
        assert 'setWL(' in html
        assert 'wl-level' in html
        assert 'wl-width' in html
        assert 'Soft Tissue' in html
        assert 'Bone' in html
        assert 'Brain' in html
        assert 'Lung' in html
        assert 'applyCustomWL' in html

    def test_structure_name_displayed_in_html(self, tmp_path, sample_viewer_data):
        """Structure name row is present in HTML when provided."""
        metadata_df, cluster_labels, qa_results = sample_viewer_data
        output_json = tmp_path / 'viewer_data.json'
        data = generate_viewer_data(
            output_json, metadata_df, cluster_labels, qa_results,
            structure_label=1, structure_name='SpinalCord',
        )
        output_html = tmp_path / 'viewer.html'
        html = generate_viewer_html(output_html, viewer_data=data)
        assert 'meta-structure' in html
        assert 'SpinalCord' in html
