"""
NIfTI slice thumbnail generation for QA dashboard.

Renders 3-view orthogonal slices (axial, coronal, sagittal) at the mask
centroid with CT windowing and semi-transparent mask overlay. Output is
PNG bytes suitable for base64 embedding in HTML dashboards.

Uses matplotlib for raster compositing (the one remaining matplotlib use)
and SimpleITK for NIfTI I/O (consistent with feature_extraction.py).
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy dependencies
_MPL_AVAILABLE = None
_SITK_AVAILABLE = None


def _check_matplotlib():
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    return _MPL_AVAILABLE


def _check_simpleitk():
    global _SITK_AVAILABLE
    if _SITK_AVAILABLE is None:
        try:
            import SimpleITK
            _SITK_AVAILABLE = True
        except ImportError:
            _SITK_AVAILABLE = False
    return _SITK_AVAILABLE


def _apply_ct_window(image_array: np.ndarray,
                     window_center: float = 40,
                     window_width: float = 400) -> np.ndarray:
    """
    Apply CT windowing to convert HU values to 0-255 display range.

    Args:
        image_array: Raw HU values
        window_center: Center of the display window in HU
        window_width: Width of the display window in HU

    Returns:
        uint8 array clipped and scaled to [0, 255]
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed


def _find_mask_centroid(mask_array: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the center of mass of all nonzero voxels in the mask.

    Returns:
        (z, y, x) indices of the centroid, clamped to array bounds.
    """
    nonzero = np.argwhere(mask_array > 0)
    if len(nonzero) == 0:
        # Fallback to volume center if mask is empty
        return tuple(s // 2 for s in mask_array.shape)
    centroid = nonzero.mean(axis=0).astype(int)
    # Clamp to array bounds
    centroid = np.clip(centroid, 0, np.array(mask_array.shape) - 1)
    return tuple(centroid)


def generate_thumbnail(
    image_path,
    mask_path,
    window_center: float = 40,
    window_width: float = 400,
    figsize: Tuple[float, float] = (9, 3),
    dpi: int = 100,
    mask_alpha: float = 0.35,
) -> bytes:
    """
    Generate 3-panel orthogonal thumbnail as PNG bytes.

    Renders axial, coronal, and sagittal slices through the mask centroid
    with CT windowing and semi-transparent mask overlay.

    Args:
        image_path: Path to CT NIfTI file
        mask_path: Path to mask NIfTI file
        window_center: CT window center in HU (default: 40, soft tissue)
        window_width: CT window width in HU (default: 400)
        figsize: Figure size in inches (width, height)
        dpi: Resolution (default: 72 for web display)
        mask_alpha: Mask overlay transparency (default: 0.35)

    Returns:
        PNG image as raw bytes

    Raises:
        ImportError: If matplotlib or SimpleITK not installed
        FileNotFoundError: If image or mask file doesn't exist
    """
    if not _check_matplotlib():
        raise ImportError("matplotlib is required for thumbnail generation")
    if not _check_simpleitk():
        raise ImportError("SimpleITK is required for thumbnail generation")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import SimpleITK as sitk

    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Load NIfTI volumes
    sitk_image = sitk.ReadImage(str(image_path))
    sitk_mask = sitk.ReadImage(str(mask_path))

    image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    mask_array = sitk.GetArrayFromImage(sitk_mask).astype(np.int32)

    # Voxel spacing for aspect ratio correction (anisotropic voxels)
    # GetSpacing() returns (x, y, z) in mm; array axes are (z, y, x)
    spacing = sitk_image.GetSpacing()
    sp_x, sp_y, sp_z = spacing[0], spacing[1], spacing[2]
    # imshow renders rows=axis0, cols=axis1 → aspect = row_spacing / col_spacing
    aspect_ratios = [
        sp_y / sp_x,   # Axial    [z,:,:] → rows=y, cols=x
        sp_z / sp_x,   # Coronal  [:,y,:] → rows=z, cols=x
        sp_z / sp_y,   # Sagittal [:,:,x] → rows=z, cols=y
    ]

    # Apply CT windowing
    windowed = _apply_ct_window(image_array, window_center, window_width)

    # Find mask centroid for slice positions
    cz, cy, cx = _find_mask_centroid(mask_array)

    # Extract orthogonal slices
    axial_img = windowed[cz, :, :]
    coronal_img = windowed[:, cy, :]
    sagittal_img = windowed[:, :, cx]

    axial_mask = mask_array[cz, :, :]
    coronal_mask = mask_array[:, cy, :]
    sagittal_mask = mask_array[:, :, cx]

    # Build color overlay for multi-label masks
    unique_labels = np.unique(mask_array)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background

    # Use tab20 colormap for distinct label colors
    if len(unique_labels) > 0:
        cmap = plt.colormaps.get_cmap('tab20').resampled(max(len(unique_labels), 2))
        label_colors = {label: cmap(i)[:3] for i, label in enumerate(unique_labels)}
    else:
        label_colors = {}

    # Render figure
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    titles = ['Axial', 'Coronal', 'Sagittal']
    slices_img = [axial_img, coronal_img, sagittal_img]
    slices_mask = [axial_mask, coronal_mask, sagittal_mask]

    for ax, title, img_slice, mask_slice, aspect in zip(
        axes, titles, slices_img, slices_mask, aspect_ratios
    ):
        ax.imshow(img_slice, cmap='gray', aspect=aspect)

        # Overlay each mask label with its color
        if label_colors:
            overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            for label, color in label_colors.items():
                region = mask_slice == label
                if region.any():
                    overlay[region, :3] = color
                    overlay[region, 3] = mask_alpha
            ax.imshow(overlay, aspect=aspect)

        ax.set_title(title, fontsize=8, pad=2)
        ax.axis('off')

    fig.tight_layout(pad=0.5)

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05,
                facecolor='black', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_thumbnail_b64(
    image_path,
    mask_path,
    **kwargs,
) -> str:
    """
    Generate thumbnail and return as base64-encoded string.

    Convenience wrapper around generate_thumbnail() for HTML embedding.
    The returned string can be used directly in an <img> tag:
        <img src="data:image/png;base64,{result}">

    Args:
        image_path: Path to CT NIfTI file
        mask_path: Path to mask NIfTI file
        **kwargs: Passed to generate_thumbnail()

    Returns:
        Base64-encoded PNG string
    """
    png_bytes = generate_thumbnail(image_path, mask_path, **kwargs)
    return base64.b64encode(png_bytes).decode('ascii')


def generate_thumbnails_batch(
    metadata_df,
    image_col: str = 'Image_Path',
    mask_col: str = 'Mask_Path',
    max_workers: int = 4,
    max_cases_warn: int = 500,
    **kwargs,
) -> Dict[int, str]:
    """
    Batch generate base64 thumbnails for all cases with image/mask paths.

    Args:
        metadata_df: DataFrame with image and mask path columns
        image_col: Column name for image paths
        mask_col: Column name for mask paths
        max_workers: Number of parallel workers for I/O
        max_cases_warn: Warn if more than this many cases (default: 500)
        **kwargs: Passed to generate_thumbnail()

    Returns:
        Dict mapping case index to base64 thumbnail string.
        Cases that fail are logged and skipped.
    """
    if image_col not in metadata_df.columns or mask_col not in metadata_df.columns:
        logger.warning(f"Columns '{image_col}' and/or '{mask_col}' not found — skipping thumbnails")
        return {}

    # Filter to rows with valid paths
    valid = metadata_df[[image_col, mask_col]].dropna()
    n_cases = len(valid)

    if n_cases == 0:
        logger.info("No cases with image/mask paths — skipping thumbnails")
        return {}

    if n_cases > max_cases_warn:
        logger.warning(
            f"{n_cases} cases detected — thumbnail generation may be slow. "
            f"Consider --no-thumbnails for large datasets."
        )

    logger.info(f"Generating thumbnails for {n_cases} cases (workers={max_workers})...")

    thumbnails = {}

    def _generate_one(case_id, image_path, mask_path):
        return case_id, generate_thumbnail_b64(image_path, mask_path, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for case_id, row in valid.iterrows():
            future = executor.submit(
                _generate_one, case_id, row[image_col], row[mask_col]
            )
            futures[future] = case_id

        for future in as_completed(futures):
            case_id = futures[future]
            try:
                _, b64_str = future.result()
                thumbnails[case_id] = b64_str
            except Exception as e:
                logger.warning(f"Thumbnail failed for case {case_id}: {e}")

    logger.info(f"Generated {len(thumbnails)}/{n_cases} thumbnails successfully")
    return thumbnails
