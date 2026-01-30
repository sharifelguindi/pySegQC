"""Tests for NIfTI thumbnail generation."""

import base64
import numpy as np
import pandas as pd
import pytest

from pysegqc.thumbnail import (
    _apply_ct_window,
    _find_mask_centroid,
    generate_thumbnail,
    generate_thumbnail_b64,
    generate_thumbnails_batch,
)


# ── Pure numpy helpers (no NIfTI needed) ──────────────────────────────


def test_apply_ct_window_default():
    """Default soft-tissue window: center=40, width=400 → range [-160, 240]."""
    arr = np.array([-1000, -160, 40, 240, 3000], dtype=np.float32)
    result = _apply_ct_window(arr)

    assert result.dtype == np.uint8
    assert result[0] == 0       # below window → black
    assert result[1] == 0       # at lower bound → black
    assert result[2] == 127     # center → mid-gray (approx)
    assert result[3] == 255     # at upper bound → white
    assert result[4] == 255     # above window → white


def test_apply_ct_window_custom():
    """Custom bone window: center=400, width=1500."""
    arr = np.array([-350, 400, 1150], dtype=np.float32)
    result = _apply_ct_window(arr, window_center=400, window_width=1500)

    assert result[0] == 0       # at lower bound
    assert result[2] == 255     # at upper bound
    assert 100 < result[1] < 160  # center should be ~128


def test_apply_ct_window_shape_preserved():
    """Output shape matches input shape."""
    arr = np.random.randn(10, 20, 30).astype(np.float32) * 500
    result = _apply_ct_window(arr)
    assert result.shape == arr.shape


def test_find_mask_centroid_basic():
    """Centroid of a small cube mask."""
    mask = np.zeros((10, 10, 10), dtype=np.int32)
    mask[3:7, 3:7, 3:7] = 1  # centered cube at (4.5, 4.5, 4.5)
    cz, cy, cx = _find_mask_centroid(mask)

    # Should be near center of the cube
    assert 3 <= cz <= 6
    assert 3 <= cy <= 6
    assert 3 <= cx <= 6


def test_find_mask_centroid_empty():
    """Empty mask falls back to volume center."""
    mask = np.zeros((20, 30, 40), dtype=np.int32)
    cz, cy, cx = _find_mask_centroid(mask)

    assert cz == 10  # 20 // 2
    assert cy == 15  # 30 // 2
    assert cx == 20  # 40 // 2


def test_find_mask_centroid_multiclass():
    """Centroid considers all nonzero labels."""
    mask = np.zeros((20, 20, 20), dtype=np.int32)
    mask[2:5, 2:5, 2:5] = 1   # label 1 near (3, 3, 3)
    mask[15:18, 15:18, 15:18] = 2  # label 2 near (16, 16, 16)
    cz, cy, cx = _find_mask_centroid(mask)

    # Centroid should be between the two regions
    assert 5 < cz < 15
    assert 5 < cy < 15
    assert 5 < cx < 15


# ── NIfTI-dependent tests ─────────────────────────────────────────────

@pytest.fixture
def simple_nifti_pair(tmp_path):
    """Create minimal NIfTI image+mask pair using nibabel."""
    nib = pytest.importorskip("nibabel")

    # Small CT-like volume
    image_array = np.random.normal(0, 200, (16, 32, 32)).astype(np.float32)

    # Single-label mask with sphere
    mask_array = np.zeros((16, 32, 32), dtype=np.int32)
    center = (8, 16, 16)
    z, y, x = np.ogrid[:16, :32, :32]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask_array[dist < 6] = 1

    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    img_path = tmp_path / "test_image.nii.gz"
    mask_path = tmp_path / "test_mask.nii.gz"

    nib.save(nib.Nifti1Image(image_array, affine), str(img_path))
    nib.save(nib.Nifti1Image(mask_array, affine), str(mask_path))

    return img_path, mask_path


@pytest.fixture
def multiclass_nifti_pair(tmp_path):
    """Create NIfTI pair with multi-class mask."""
    nib = pytest.importorskip("nibabel")

    image_array = np.random.normal(40, 150, (16, 32, 32)).astype(np.float32)

    mask_array = np.zeros((16, 32, 32), dtype=np.int32)
    z, y, x = np.ogrid[:16, :32, :32]
    center = (8, 16, 16)
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask_array[dist < 10] = 1
    mask_array[dist < 6] = 2
    mask_array[dist < 3] = 3

    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    img_path = tmp_path / "multi_image.nii.gz"
    mask_path = tmp_path / "multi_mask.nii.gz"

    nib.save(nib.Nifti1Image(image_array, affine), str(img_path))
    nib.save(nib.Nifti1Image(mask_array, affine), str(mask_path))

    return img_path, mask_path


@pytest.fixture
def anisotropic_nifti_pair(tmp_path):
    """Create NIfTI pair with anisotropic voxel spacing (0.5, 0.5, 2.0 mm)."""
    nib = pytest.importorskip("nibabel")

    image_array = np.random.normal(0, 200, (16, 32, 32)).astype(np.float32)

    mask_array = np.zeros((16, 32, 32), dtype=np.int32)
    center = (8, 16, 16)
    z, y, x = np.ogrid[:16, :32, :32]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask_array[dist < 6] = 1

    # Anisotropic: 0.5mm x 0.5mm in-plane, 2.0mm slice thickness
    affine = np.diag([0.5, 0.5, 2.0, 1.0])

    img_path = tmp_path / "aniso_image.nii.gz"
    mask_path = tmp_path / "aniso_mask.nii.gz"

    nib.save(nib.Nifti1Image(image_array, affine), str(img_path))
    nib.save(nib.Nifti1Image(mask_array, affine), str(mask_path))

    return img_path, mask_path


@pytest.mark.skipif(
    not pytest.importorskip("SimpleITK", reason="SimpleITK not installed"),
    reason="SimpleITK required"
)
class TestGenerateThumbnail:
    """Tests requiring NIfTI I/O (SimpleITK + nibabel)."""

    def test_returns_png_bytes(self, simple_nifti_pair):
        """Output is valid PNG bytes (starts with PNG magic bytes)."""
        img_path, mask_path = simple_nifti_pair
        result = generate_thumbnail(img_path, mask_path)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # PNG magic bytes
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_reasonable_size(self, simple_nifti_pair):
        """PNG should be reasonably sized (< 100KB for small synthetic data)."""
        img_path, mask_path = simple_nifti_pair
        result = generate_thumbnail(img_path, mask_path)

        assert len(result) < 100_000  # 100KB max for tiny synthetic

    def test_multiclass_mask(self, multiclass_nifti_pair):
        """Multi-class mask renders without error."""
        img_path, mask_path = multiclass_nifti_pair
        result = generate_thumbnail(img_path, mask_path)

        assert isinstance(result, bytes)
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_custom_window(self, simple_nifti_pair):
        """Custom CT window produces different output."""
        img_path, mask_path = simple_nifti_pair

        soft_tissue = generate_thumbnail(img_path, mask_path,
                                         window_center=40, window_width=400)
        bone = generate_thumbnail(img_path, mask_path,
                                  window_center=400, window_width=1500)

        # Different windows should produce different images
        assert soft_tissue != bone

    def test_file_not_found(self, tmp_path):
        """Missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            generate_thumbnail(tmp_path / "missing.nii.gz",
                               tmp_path / "also_missing.nii.gz")

    def test_b64_output(self, simple_nifti_pair):
        """Base64 wrapper returns valid base64 string."""
        img_path, mask_path = simple_nifti_pair
        result = generate_thumbnail_b64(img_path, mask_path)

        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'

    def test_batch_generation(self, simple_nifti_pair):
        """Batch generation processes DataFrame correctly."""
        img_path, mask_path = simple_nifti_pair

        metadata_df = pd.DataFrame({
            'Image_Path': [str(img_path), str(img_path)],
            'Mask_Path': [str(mask_path), str(mask_path)],
        }, index=[0, 1])

        result = generate_thumbnails_batch(metadata_df, max_workers=1)

        assert len(result) == 2
        assert 0 in result
        assert 1 in result
        # Each entry is valid base64
        for b64_str in result.values():
            decoded = base64.b64decode(b64_str)
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n'

    def test_batch_missing_columns(self):
        """Batch gracefully handles missing columns."""
        df = pd.DataFrame({'Other': [1, 2]})
        result = generate_thumbnails_batch(df)
        assert result == {}

    def test_batch_handles_failures(self, simple_nifti_pair, tmp_path):
        """Batch skips failed cases and logs warning."""
        img_path, mask_path = simple_nifti_pair

        metadata_df = pd.DataFrame({
            'Image_Path': [str(img_path), str(tmp_path / "bad.nii.gz")],
            'Mask_Path': [str(mask_path), str(tmp_path / "bad_mask.nii.gz")],
        }, index=[0, 1])

        result = generate_thumbnails_batch(metadata_df, max_workers=1)

        # Case 0 should succeed, case 1 should be skipped
        assert 0 in result
        assert 1 not in result

    def test_batch_empty_dataframe(self):
        """Batch with no valid rows returns empty dict."""
        df = pd.DataFrame({
            'Image_Path': [None, None],
            'Mask_Path': [None, None],
        })
        result = generate_thumbnails_batch(df)
        assert result == {}

    def test_anisotropic_voxels(self, anisotropic_nifti_pair):
        """Anisotropic voxels (e.g. 0.5x0.5x2.0mm) render without errors."""
        img_path, mask_path = anisotropic_nifti_pair
        result = generate_thumbnail(img_path, mask_path)
        assert isinstance(result, bytes)
        assert result[:8] == b'\x89PNG\r\n\x1a\n'
