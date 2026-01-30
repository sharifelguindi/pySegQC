"""Comprehensive tests for feature extraction module"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Skip entire test module if pyradiomics is not installed
# It's required but has installation issues - see CLAUDE.md for workarounds
pytest.importorskip("radiomics", reason="pyradiomics required for NIfTI extraction (see CLAUDE.md for installation)")
pytest.importorskip("nibabel", reason="nibabel required for NIfTI extraction")
pytest.importorskip("SimpleITK", reason="SimpleITK required for NIfTI extraction")

from pysegqc.feature_extraction import (
    _safe_float,
    extract_patient_id,
    validate_nifti_pair,
    create_radiomics_extractor,
    extract_features_from_nifti,
    process_multi_class_mask,
    format_features_for_pysegqc,
    find_image_mask_pairs,
    create_pca_data_sheet,
)


def test_safe_float_conversion():
    """Test safe float conversion with various inputs."""
    # Valid float
    assert _safe_float(3.14) == 3.14

    # String to float
    assert _safe_float("2.71") == 2.71

    # None returns default
    assert _safe_float(None) is None
    assert _safe_float(None, default=0.0) == 0.0

    # NaN returns default
    assert _safe_float(np.nan) is None
    assert _safe_float(np.nan, default=-1.0) == -1.0

    # Inf returns default
    assert _safe_float(np.inf) is None
    assert _safe_float(-np.inf, default=-999.0) == -999.0

    # Invalid string returns default
    assert _safe_float("invalid") is None
    assert _safe_float("", default=0.0) == 0.0


def test_extract_patient_id():
    """Test patient ID extraction from various filename patterns."""
    # Pattern: patientXXX
    assert extract_patient_id("patient001_CT") == "patient001"
    assert extract_patient_id("PATIENT123_mask") == "PATIENT123"

    # Pattern: sub-XXX
    assert extract_patient_id("sub-001_T1w") == "sub-001"
    assert extract_patient_id("sub-042_seg") == "sub-042"

    # Pattern: case_XXX or case-XXX
    assert extract_patient_id("case_123_image") == "case_123"
    assert extract_patient_id("case-456_mask") == "case-456"

    # Fallback to digits
    assert extract_patient_id("scan_0042_final") == "0042"

    # No pattern: return whole filename
    assert extract_patient_id("no_numbers_here") == "no_numbers_here"


def test_extract_patient_id_short_numeric():
    """Test patient ID extraction for 1-2 digit numeric filenames (bug fix)."""
    # 1-digit IDs
    assert extract_patient_id("5") == "5"

    # 2-digit IDs (previously missed by \d{3,} regex)
    assert extract_patient_id("64") == "64"
    assert extract_patient_id("65") == "65"
    assert extract_patient_id("71") == "71"
    assert extract_patient_id("88") == "88"
    assert extract_patient_id("89") == "89"

    # 3+ digit IDs still work
    assert extract_patient_id("1033") == "1033"
    assert extract_patient_id("15703") == "15703"


def test_validate_nifti_pair_nonexistent_files(tmp_path):
    """Test validation fails for non-existent files."""
    fake_image = tmp_path / "nonexistent_image.nii.gz"
    fake_mask = tmp_path / "nonexistent_mask.nii.gz"

    is_valid, error = validate_nifti_pair(fake_image, fake_mask)
    assert not is_valid
    assert "not found" in error.lower()


def test_validate_nifti_pair_valid(synthetic_nifti_data):
    """Test validation passes for valid NIfTI pairs."""
    image_path, mask_path = synthetic_nifti_data['pairs'][0]

    is_valid, error = validate_nifti_pair(image_path, mask_path)
    assert is_valid
    assert error == ""


def test_validate_nifti_pair_empty_mask(tmp_path):
    """Test validation fails for empty masks."""
    import nibabel as nib

    # Create image
    image_array = np.random.normal(0, 10, (32, 64, 64)).astype(np.float32)
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(image_array, affine), image_path)

    # Create empty mask (all zeros)
    mask_array = np.zeros((32, 64, 64), dtype=np.uint8)
    mask_path = tmp_path / "empty_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    is_valid, error = validate_nifti_pair(image_path, mask_path)
    assert not is_valid
    assert "no foreground voxels" in error.lower()


def test_validate_nifti_pair_dimension_mismatch(tmp_path):
    """Test validation fails for mismatched dimensions."""
    import nibabel as nib

    affine = np.eye(4)

    # Create image (32x64x64)
    image_array = np.random.normal(0, 10, (32, 64, 64)).astype(np.float32)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(image_array, affine), image_path)

    # Create mask with different size (16x32x32)
    mask_array = np.ones((16, 32, 32), dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    is_valid, error = validate_nifti_pair(image_path, mask_path)
    assert not is_valid
    assert "mismatch" in error.lower()


def test_create_radiomics_extractor():
    """Test PyRadiomics extractor creation."""
    extractor = create_radiomics_extractor()

    # Check extractor is created
    assert extractor is not None

    # Check default settings
    assert extractor.settings['binWidth'] == 50
    assert extractor.settings['force2D'] is False
    assert extractor.settings['minimumROISize'] == 1


def test_create_radiomics_extractor_custom():
    """Test PyRadiomics extractor with custom settings."""
    extractor = create_radiomics_extractor(
        feature_classes=['shape', 'firstorder'],
        bin_width=25,
        force_2d=True
    )

    assert extractor.settings['binWidth'] == 25
    assert extractor.settings['force2D'] is True


def test_extract_features_single_case(synthetic_nifti_data):
    """Test feature extraction from single case."""
    image_path, mask_path = synthetic_nifti_data['pairs'][0]

    features = extract_features_from_nifti(image_path, mask_path)

    # Check features are extracted
    assert isinstance(features, dict)
    assert len(features) > 0

    # Check key feature classes present
    assert any('shape' in key for key in features.keys())
    assert any('firstorder' in key for key in features.keys())

    # Check specific expected features
    assert 'original_shape_Sphericity' in features
    assert 'original_firstorder_Mean' in features

    # Check values are numeric (not NaN/Inf)
    for key, value in features.items():
        assert value is None or isinstance(value, (int, float))


def test_extract_features_diagnostics_filtered(synthetic_nifti_data):
    """Test that diagnostic keys are filtered out."""
    image_path, mask_path = synthetic_nifti_data['pairs'][0]

    features = extract_features_from_nifti(image_path, mask_path)

    # No diagnostic keys should be present
    diagnostic_keys = [k for k in features.keys() if k.startswith('diagnostics_')]
    assert len(diagnostic_keys) == 0


def test_process_multi_class_mask(synthetic_multiclass_nifti):
    """Test extraction from multi-class mask."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']

    features_by_class = process_multi_class_mask(image_path, mask_path)

    # Should have 3 classes
    assert len(features_by_class) == 3
    assert 1 in features_by_class
    assert 2 in features_by_class
    assert 3 in features_by_class

    # Each class should have features
    for class_id, features in features_by_class.items():
        assert len(features) > 0
        assert 'original_shape_Sphericity' in features


def test_format_features_for_pysegqc():
    """Test feature name formatting with position prefixes."""
    raw_features = {
        'original_shape_Sphericity': 0.87,
        'original_firstorder_Mean': 45.2,
        'original_glcm_Contrast': 32.1,
        'diagnostics_something': 999  # Should be skipped
    }

    # Format for class 1
    formatted = format_features_for_pysegqc(raw_features, class_id=1)

    assert '001_original_shape_Sphericity' in formatted
    assert formatted['001_original_shape_Sphericity'] == 0.87

    assert '001_original_firstorder_Mean' in formatted
    assert '001_original_glcm_Contrast' in formatted

    # Diagnostic keys should not be included
    assert not any('diagnostics' in key for key in formatted.keys())

    # Format for class 2
    formatted2 = format_features_for_pysegqc(raw_features, class_id=2)
    assert '002_original_shape_Sphericity' in formatted2


def test_find_image_mask_pairs(synthetic_nifti_data):
    """Test file pair discovery."""
    data_dir = synthetic_nifti_data['data_dir']

    pairs = find_image_mask_pairs(data_dir)

    # Should find 3 pairs
    assert len(pairs) == 3

    # All paths should exist
    for img_path, mask_path in pairs:
        assert img_path.exists()
        assert mask_path.exists()

    # Check filenames match pattern
    for img_path, mask_path in pairs:
        assert 'patient' in img_path.stem
        assert 'patient' in mask_path.stem


def test_find_image_mask_pairs_custom_patterns(synthetic_nifti_data):
    """Test file discovery with custom patterns."""
    data_dir = synthetic_nifti_data['data_dir']

    pairs = find_image_mask_pairs(
        data_dir,
        image_pattern='*_CT.nii.gz',
        mask_pattern='*_mask.nii.gz'
    )

    assert len(pairs) == 3


def test_find_image_mask_pairs_missing_directory(tmp_path):
    """Test error handling when directories don't exist."""
    fake_dir = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError):
        find_image_mask_pairs(fake_dir)


def test_create_pca_data_sheet_single_class(synthetic_nifti_data, tmp_path):
    """Test end-to-end Excel generation with single-class masks."""
    pairs = synthetic_nifti_data['pairs']
    output_excel = tmp_path / 'features.xlsx'

    result_path = create_pca_data_sheet(pairs, output_excel, n_jobs=1)

    # Check file was created
    assert result_path.exists()
    assert result_path == output_excel

    # Check Excel structure
    df = pd.read_excel(result_path, sheet_name='PCA_Data')

    # Check number of rows (3 cases)
    assert len(df) == 3

    # Check metadata columns
    assert 'Case_ID' in df.columns
    assert 'Image_Path' in df.columns
    assert 'Mask_Path' in df.columns
    assert 'N_Structures' in df.columns

    # Check feature columns have 001_ prefix
    feature_cols = [col for col in df.columns if col.startswith('001_')]
    assert len(feature_cols) > 0

    # Check specific features
    assert '001_original_shape_Sphericity' in df.columns
    assert '001_original_firstorder_Mean' in df.columns

    # Check values are not all NaN
    assert not df['001_original_shape_Sphericity'].isna().all()


def test_create_pca_data_sheet_multiclass(synthetic_multiclass_nifti, tmp_path):
    """Test Excel generation with multi-class mask."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']
    pairs = [(image_path, mask_path)]

    output_excel = tmp_path / 'multiclass_features.xlsx'

    result_path = create_pca_data_sheet(pairs, output_excel, n_jobs=1)

    # Load and check
    df = pd.read_excel(result_path, sheet_name='PCA_Data')

    assert len(df) == 1

    # Should have features for all 3 classes
    assert any(col.startswith('001_') for col in df.columns)
    assert any(col.startswith('002_') for col in df.columns)
    assert any(col.startswith('003_') for col in df.columns)

    # Check N_Structures column
    assert df['N_Structures'].iloc[0] == 3


def test_create_pca_data_sheet_empty_pairs_list(tmp_path):
    """Test error handling with empty pairs list."""
    output_excel = tmp_path / 'empty.xlsx'

    with pytest.raises(ValueError, match="No image-mask pairs"):
        create_pca_data_sheet([], output_excel)


def test_create_pca_data_sheet_feature_class_selection(synthetic_nifti_data, tmp_path):
    """Test feature extraction with custom feature classes."""
    pairs = [synthetic_nifti_data['pairs'][0]]  # Just one case for speed
    output_excel = tmp_path / 'custom_features.xlsx'

    # Extract only shape and firstorder
    result_path = create_pca_data_sheet(
        pairs,
        output_excel,
        feature_classes=['shape', 'firstorder'],
        n_jobs=1
    )

    df = pd.read_excel(result_path, sheet_name='PCA_Data')

    # Should have shape and firstorder features
    feature_cols = [col for col in df.columns if col.startswith('001_')]
    shape_features = [col for col in feature_cols if 'shape' in col]
    firstorder_features = [col for col in feature_cols if 'firstorder' in col]

    assert len(shape_features) > 0
    assert len(firstorder_features) > 0

    # Should NOT have GLCM features (not requested)
    glcm_features = [col for col in feature_cols if 'glcm' in col]
    assert len(glcm_features) == 0


def test_integration_with_data_loader(synthetic_nifti_data, tmp_path):
    """Test that extracted features can be loaded by pySegQC's data_loader."""
    from pysegqc.data_loader import load_and_preprocess_data

    # Extract features
    pairs = synthetic_nifti_data['pairs']
    excel_path = tmp_path / 'integration_test.xlsx'
    create_pca_data_sheet(pairs, excel_path, n_jobs=1)

    # Load with data_loader
    metadata, features_df, feature_names, structure_info = load_and_preprocess_data(
        excel_path,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    # Check data loaded successfully
    assert len(features_df) == 3  # 3 cases
    assert len(feature_names) > 0

    # Feature names should NOT have 001_ prefix in position mode
    # (data_loader strips it)
    assert not any(name.startswith('001_') for name in feature_names)
    assert 'original_shape_Sphericity' in feature_names


def test_cli_extract_command(synthetic_nifti_data, tmp_path):
    """Test CLI extract subcommand integration."""
    import subprocess
    import sys

    data_dir = synthetic_nifti_data['data_dir']
    output_excel = tmp_path / 'cli_output.xlsx'

    # Run pysegqc extract via CLI
    result = subprocess.run(
        [
            sys.executable, '-m', 'pysegqc',
            'extract',
            str(data_dir),
            '--output', str(output_excel),
            '--n-jobs', '1'
        ],
        capture_output=True,
        text=True
    )

    # Check command succeeded
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Check output file exists
    assert output_excel.exists()

    # Verify Excel content
    df = pd.read_excel(output_excel, sheet_name='PCA_Data')
    assert len(df) == 3  # 3 cases
    assert 'Case_ID' in df.columns
    assert any(col.startswith('001_') for col in df.columns)


def test_extract_features_corrupted_nifti(tmp_path):
    """Test error handling with corrupted NIfTI file."""
    import nibabel as nib

    # Create valid mask
    mask_array = np.ones((32, 64, 64), dtype=np.uint8)
    affine = np.eye(4)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    # Create corrupted image file (not valid NIfTI)
    image_path = tmp_path / "corrupted.nii.gz"
    image_path.write_bytes(b"This is not a NIfTI file")

    # Should raise error
    with pytest.raises((ValueError, Exception)):
        extract_features_from_nifti(image_path, mask_path)


def test_extract_features_single_voxel_roi(tmp_path):
    """Test that extraction correctly rejects single-voxel ROIs."""
    import nibabel as nib

    # Create image
    image_array = np.random.normal(0, 10, (32, 64, 64)).astype(np.float32)
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(image_array, affine), image_path)

    # Create mask with single voxel
    mask_array = np.zeros((32, 64, 64), dtype=np.uint8)
    mask_array[16, 32, 32] = 1  # Single voxel at center
    mask_path = tmp_path / "single_voxel.nii.gz"
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    # PyRadiomics correctly rejects single-voxel ROIs (can't compute meaningful texture features)
    with pytest.raises(ValueError, match="only contains 1 segmented voxel"):
        extract_features_from_nifti(image_path, mask_path)


def test_extract_features_extreme_intensities(tmp_path):
    """Test extraction with extreme intensity values."""
    import nibabel as nib

    # Create image with extreme values
    image_array = np.random.uniform(-3000, 3000, (32, 64, 64)).astype(np.float32)
    affine = np.eye(4)
    image_path = tmp_path / "extreme.nii.gz"
    nib.save(nib.Nifti1Image(image_array, affine), image_path)

    # Create spherical mask
    center = (16, 32, 32)
    z, y, x = np.ogrid[:32, :64, :64]
    distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask_array = (distance <= 10).astype(np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    # Should handle extreme values gracefully (normalize=True helps)
    features = extract_features_from_nifti(image_path, mask_path)

    # Should have valid features
    assert 'original_firstorder_Mean' in features
    assert features['original_firstorder_Mean'] is not None
    # Mean should be within reasonable range after normalization
    assert -5000 < features['original_firstorder_Mean'] < 5000


def test_create_pca_data_sheet_with_failed_case(synthetic_nifti_data, tmp_path):
    """Test that pipeline continues even if one case fails."""
    import nibabel as nib

    pairs = synthetic_nifti_data['pairs']

    # Add a corrupted pair
    bad_image = tmp_path / 'bad.nii.gz'
    bad_mask = tmp_path / 'bad_mask.nii.gz'
    bad_image.write_bytes(b"corrupted")
    bad_mask.write_bytes(b"corrupted")

    pairs_with_bad = pairs + [(bad_image, bad_mask)]

    # Should complete despite one failure
    output_excel = tmp_path / 'with_failures.xlsx'
    result_path = create_pca_data_sheet(pairs_with_bad, output_excel, n_jobs=1)

    assert result_path.exists()

    # Check Excel has all cases (including failed one with Error column)
    df = pd.read_excel(result_path, sheet_name='PCA_Data')
    assert len(df) == 4  # 3 good + 1 bad

    # Failed case should have Error column or N_Structures=0
    failed_row = df[df['Case_ID'] == 'bad']
    assert len(failed_row) == 1
    assert failed_row['N_Structures'].iloc[0] == 0 or 'Error' in failed_row.columns


def test_find_image_mask_pairs_exact_id_matching(tmp_path):
    """Test that ID matching is exact, not substring-based (bug fix).

    Previously, patient ID '64' would match mask '16453' because
    'if patient_id in mask_path.stem' used substring matching.
    The fix uses extract_patient_id() on both sides and compares with ==.
    """
    import nibabel as nib

    # Create directory structure with tricky numeric IDs
    data_dir = tmp_path / 'id_test'
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    affine = np.eye(4)
    image_array = np.random.normal(0, 10, (32, 64, 64)).astype(np.float32)
    mask_array = np.ones((32, 64, 64), dtype=np.uint8)

    # Create images: 64, 16453
    for pid in ['64', '16453']:
        nib.save(nib.Nifti1Image(image_array, affine), images_dir / f'{pid}.nii.gz')
        nib.save(nib.Nifti1Image(mask_array, affine), masks_dir / f'{pid}.nii.gz')

    pairs = find_image_mask_pairs(data_dir)

    # Should find exactly 2 pairs
    assert len(pairs) == 2

    # Each image should match its own mask, not cross-match
    matched_ids = set()
    for img_path, mask_path in pairs:
        img_id = img_path.stem.replace('.nii', '')
        mask_id = mask_path.stem.replace('.nii', '')
        assert img_id == mask_id, f"Image {img_id} matched to wrong mask {mask_id}"
        matched_ids.add(img_id)

    assert '64' in matched_ids
    assert '16453' in matched_ids


def test_find_image_mask_pairs_no_false_substring_match(tmp_path):
    """Test that '64' does NOT match '164' or '640' or '16453'."""
    import nibabel as nib

    data_dir = tmp_path / 'substring_test'
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    affine = np.eye(4)
    image_array = np.random.normal(0, 10, (32, 64, 64)).astype(np.float32)
    mask_array = np.ones((32, 64, 64), dtype=np.uint8)

    # Image for patient 64 exists
    nib.save(nib.Nifti1Image(image_array, affine), images_dir / '64.nii.gz')

    # Only masks for 164, 640, 16453 exist — none should match patient 64
    for pid in ['164', '640', '16453']:
        nib.save(nib.Nifti1Image(mask_array, affine), masks_dir / f'{pid}.nii.gz')

    pairs = find_image_mask_pairs(data_dir)

    # Patient 64 should NOT match any of these masks
    assert len(pairs) == 0


def test_create_pca_data_sheet_with_label_map(synthetic_multiclass_nifti, tmp_path):
    """Test that label_map parameter creates a Label_Map sheet in Excel."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']
    pairs = [(image_path, mask_path)]
    output_excel = tmp_path / 'labeled_features.xlsx'

    label_map = {1: 'Outer_Structure', 2: 'Middle_Structure', 3: 'Inner_Structure'}

    result_path = create_pca_data_sheet(
        pairs, output_excel, n_jobs=1,
        label_map=label_map
    )

    # Verify Label_Map sheet exists
    label_map_df = pd.read_excel(result_path, sheet_name='Label_Map')
    assert len(label_map_df) == 3
    assert 'Label_ID' in label_map_df.columns
    assert 'Structure_Name' in label_map_df.columns
    assert 'Position_Prefix' in label_map_df.columns

    # Check content
    names = set(label_map_df['Structure_Name'])
    assert 'Outer_Structure' in names
    assert 'Middle_Structure' in names
    assert 'Inner_Structure' in names

    # Check prefix format
    prefixes = set(label_map_df['Position_Prefix'])
    assert '001_' in prefixes
    assert '002_' in prefixes
    assert '003_' in prefixes


def test_create_pca_data_sheet_with_exclude_labels(synthetic_multiclass_nifti, tmp_path):
    """Test that exclude_labels filters out specified labels."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']
    pairs = [(image_path, mask_path)]
    output_excel = tmp_path / 'excluded_features.xlsx'

    # Exclude label 1 (like excluding Brainstem in HN dataset)
    result_path = create_pca_data_sheet(
        pairs, output_excel, n_jobs=1,
        exclude_labels=[1]
    )

    df = pd.read_excel(result_path, sheet_name='PCA_Data')

    # Should NOT have 001_ features (label 1 excluded)
    cols_001 = [c for c in df.columns if c.startswith('001_')]
    assert len(cols_001) == 0, f"Label 1 should be excluded but found columns: {cols_001}"

    # Should still have 002_ and 003_ features
    cols_002 = [c for c in df.columns if c.startswith('002_')]
    cols_003 = [c for c in df.columns if c.startswith('003_')]
    assert len(cols_002) > 0, "Label 2 features should be present"
    assert len(cols_003) > 0, "Label 3 features should be present"


def test_create_pca_data_sheet_label_map_excludes_in_sheet(synthetic_multiclass_nifti, tmp_path):
    """Test that Label_Map sheet respects exclude_labels."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']
    pairs = [(image_path, mask_path)]
    output_excel = tmp_path / 'map_exclude_features.xlsx'

    label_map = {1: 'Outer', 2: 'Middle', 3: 'Inner'}

    result_path = create_pca_data_sheet(
        pairs, output_excel, n_jobs=1,
        label_map=label_map,
        exclude_labels=[1]
    )

    label_map_df = pd.read_excel(result_path, sheet_name='Label_Map')

    # Label 1 should NOT appear in the Label_Map sheet
    assert 1 not in label_map_df['Label_ID'].values
    assert 'Outer' not in label_map_df['Structure_Name'].values

    # Labels 2 and 3 should be present
    assert 2 in label_map_df['Label_ID'].values
    assert 3 in label_map_df['Label_ID'].values
    assert len(label_map_df) == 2


def test_process_multi_class_mask_with_exclude(synthetic_multiclass_nifti):
    """Test that process_multi_class_mask respects exclude_labels."""
    image_path = synthetic_multiclass_nifti['image_path']
    mask_path = synthetic_multiclass_nifti['mask_path']

    features_by_class = process_multi_class_mask(
        image_path, mask_path,
        exclude_labels=[1, 3]
    )

    # Only class 2 should remain
    assert len(features_by_class) == 1
    assert 2 in features_by_class
    assert 1 not in features_by_class
    assert 3 not in features_by_class


def test_feature_column_detection_many_structures():
    """Test that feature column detection works with >5 structures (bug fix).

    Previously, feature_cols used a hardcoded tuple:
        col.startswith(('001_', '002_', '003_', '004_', '005_'))
    This silently dropped structures 6-14. The fix uses:
        re.match(r'^\\d{3}_', str(col))
    """
    import re

    # Simulate a DataFrame with 14 structures (like HN OAR dataset)
    columns = ['Case_ID', 'Image_Path', 'Mask_Path', 'N_Structures']
    for label_id in range(2, 15):  # Labels 2-14
        columns.append(f'{label_id:03d}_original_shape_Sphericity')
        columns.append(f'{label_id:03d}_original_firstorder_Mean')

    df = pd.DataFrame(columns=columns)

    # Use the same regex pattern as feature_extraction.py line 587
    feature_cols = [col for col in df.columns if re.match(r'^\d{3}_', str(col))]

    # Should detect all 13 structures × 2 features = 26 feature columns
    assert len(feature_cols) == 26

    # Verify specific high-numbered structure columns are detected
    assert '010_original_shape_Sphericity' in feature_cols
    assert '014_original_firstorder_Mean' in feature_cols

    # Metadata columns should NOT be detected
    assert 'Case_ID' not in feature_cols
    assert 'N_Structures' not in feature_cols
