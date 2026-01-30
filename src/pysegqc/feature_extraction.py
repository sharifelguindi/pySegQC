"""
NIfTI-based radiomics feature extraction using pyradiomics.

This module provides functionality to extract radiomics features directly
from NIfTI files, generating the PCA_Data Excel sheet for pySegQC analysis.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import gc

if TYPE_CHECKING:
    from radiomics.featureextractor import RadiomicsFeatureExtractor

import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm

# Try to import pyradiomics - it's required but may fail to install due to packaging issues
try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
    # Suppress verbose PyRadiomics logging
    logging.getLogger('radiomics').setLevel(logging.ERROR)
except ImportError:
    RADIOMICS_AVAILABLE = False
    featureextractor = None

logger = logging.getLogger(__name__)


# Default feature classes to extract (matches task_metrics.py patterns)
DEFAULT_FEATURE_CLASSES = [
    'firstorder',   # 18 features (Mean, Std, Median, etc.)
    'shape',        # 14 features (Volume, Sphericity, Elongation, etc.)
    'glcm',         # 24 features (Contrast, Correlation, etc.)
    'glrlm',        # 16 features (Run length patterns)
    'glszm',        # 16 features (Zone size patterns)
    'ngtdm',        # 5 features (Texture complexity)
]

# PyRadiomics configuration constants (memory-efficient settings)
DEFAULT_BIN_WIDTH = 50          # 75% memory reduction vs default (200)
DEFAULT_MINIMUM_ROI_SIZE = 1    # Process even small structures
DEFAULT_NORMALIZE = True        # Normalize image intensities
DEFAULT_NORMALIZE_SCALE = 100   # Normalization scale factor


def _safe_float(value, default=None):
    """
    Convert PyRadiomics output to float, handling NaN/Inf/None.

    Critical for distinguishing "zero value" from "no value". Pattern from task_metrics.py.

    Args:
        value: Value to convert (can be None, string, float, etc.)
        default: Value to return if conversion fails or result is NaN/Inf

    Returns:
        float or default value
    """
    if value is None or (isinstance(value, str) and value == ''):
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def extract_patient_id(filename: str) -> str:
    """
    Extract patient/case ID from filename using common patterns.

    Args:
        filename: Filename (without extension) to parse

    Returns:
        Extracted patient ID or full filename if no pattern matches

    Examples:
        - 'patient001_CT' → 'patient001'
        - 'sub-001_T1w' → 'sub-001'
        - 'case_123_mask' → 'case_123'
    """
    # Common medical imaging ID patterns
    patterns = [
        r'(patient\d+)',
        r'(sub-\d+)',
        r'(case[_-]\d+)',
        r'(\d+)',  # Any numeric ID as fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback: use whole filename
    return filename


def validate_nifti_pair(image_path: Path, mask_path: Path) -> Tuple[bool, str]:
    """
    Validate that image and mask files exist and are compatible.

    Args:
        image_path: Path to image NIfTI file
        mask_path: Path to mask NIfTI file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file existence
    if not image_path.exists():
        return False, f"Image file not found: {image_path}"
    if not mask_path.exists():
        return False, f"Mask file not found: {mask_path}"

    try:
        # Load both files
        sitk_image = sitk.ReadImage(str(image_path))
        sitk_mask = sitk.ReadImage(str(mask_path))

        # Check dimensions match
        image_size = sitk_image.GetSize()
        mask_size = sitk_mask.GetSize()

        if image_size != mask_size:
            return False, f"Dimension mismatch: image {image_size} vs mask {mask_size}"

        # Check mask is not empty
        mask_array = sitk.GetArrayFromImage(sitk_mask)
        if np.sum(mask_array > 0) == 0:
            return False, "Mask contains no foreground voxels"

        return True, ""

    except Exception as e:
        return False, f"Failed to load NIfTI files: {str(e)}"


def create_radiomics_extractor(
    feature_classes: Optional[List[str]] = None,
    bin_width: int = DEFAULT_BIN_WIDTH,
    force_2d: bool = False
) -> "RadiomicsFeatureExtractor":
    """
    Create configured PyRadiomics extractor.

    Configuration based on task_metrics.py best practices for memory efficiency
    and robust feature extraction.

    Args:
        feature_classes: List of feature classes to enable (default: all standard classes)
        bin_width: Bin width for discretization (lower = less memory, default: 50)
        force_2d: If True, compute 2D features instead of 3D (default: False)

    Returns:
        Configured RadiomicsFeatureExtractor

    Raises:
        ImportError: If pyradiomics is not installed
    """
    if not RADIOMICS_AVAILABLE:
        raise ImportError(
            "pyradiomics is required for NIfTI feature extraction but is not installed. "
            "This package has known installation issues. See CLAUDE.md for workarounds:\n"
            "  Option 1: conda install -c conda-forge pyradiomics\n"
            "  Option 2: pip install --no-build-isolation pyradiomics\n"
            "Visit: https://pyradiomics.readthedocs.io/en/latest/installation.html"
        )

    if feature_classes is None:
        feature_classes = DEFAULT_FEATURE_CLASSES

    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Disable all, then enable selected (explicit control)
    extractor.disableAllFeatures()
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class)

    # Memory-efficient settings (from task_metrics.py)
    extractor.settings['binWidth'] = bin_width
    extractor.settings['force2D'] = force_2d
    extractor.settings['minimumROISize'] = DEFAULT_MINIMUM_ROI_SIZE
    extractor.settings['normalize'] = DEFAULT_NORMALIZE
    extractor.settings['normalizeScale'] = DEFAULT_NORMALIZE_SCALE

    return extractor


def extract_features_from_nifti(
    image_path: Path,
    mask_path: Path,
    feature_classes: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Extract radiomics features from a single image-mask pair.

    Args:
        image_path: Path to NIfTI image file
        mask_path: Path to NIfTI mask file (binary, single class)
        feature_classes: List of feature classes to extract (default: all standard)

    Returns:
        Dictionary of feature name → value, with safe float conversion

    Raises:
        ValueError: If files are incompatible
        FileNotFoundError: If files don't exist
    """
    # Validate inputs
    is_valid, error_msg = validate_nifti_pair(image_path, mask_path)
    if not is_valid:
        raise ValueError(error_msg)

    try:
        # Load images using SimpleITK (preserves spatial metadata)
        sitk_image = sitk.ReadImage(str(image_path))
        sitk_mask = sitk.ReadImage(str(mask_path))

        # Ensure mask is binary (convert multi-class if needed)
        mask_array = sitk.GetArrayFromImage(sitk_mask)
        binary_mask_array = (mask_array > 0).astype(np.uint8)

        sitk_binary_mask = sitk.GetImageFromArray(binary_mask_array)
        sitk_binary_mask.CopyInformation(sitk_image)  # CRITICAL: copy spatial metadata

        # Create extractor
        extractor = create_radiomics_extractor(feature_classes)

        # Extract features
        features_raw = extractor.execute(sitk_image, sitk_binary_mask)

        # Convert to safe floats and filter diagnostics
        features = {}
        for key, value in features_raw.items():
            # Skip diagnostic keys
            if key.startswith('diagnostics_'):
                continue
            features[key] = _safe_float(value)

        # Clean up memory
        del extractor, sitk_image, sitk_binary_mask, mask_array, binary_mask_array
        gc.collect()

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed for {image_path.name}: {str(e)}")
        raise


def process_multi_class_mask(
    image_path: Path,
    mask_path: Path,
    feature_classes: Optional[List[str]] = None,
    exclude_labels: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Extract features from multi-class mask (one set per class).

    For a mask with classes [1, 2, 3], extracts features for each class separately.
    This is the key function for supporting multi-structure analysis.

    The function automatically detects unique labels in the mask (excluding background=0)
    and extracts radiomics features for each structure independently.

    Args:
        image_path: Path to NIfTI image file
        mask_path: Path to NIfTI mask file (multi-class)
        feature_classes: List of feature classes to extract (default: all standard)

    Returns:
        Dictionary mapping class_id → features_dict

    Example:
        >>> features = process_multi_class_mask(
        ...     Path('patient001_CT.nii.gz'),
        ...     Path('patient001_mask.nii.gz')  # Contains labels 1, 2, 3
        ... )
        >>> features
        {
            1: {'original_shape_Sphericity': 0.87, 'original_firstorder_Mean': 45.2, ...},
            2: {'original_shape_Sphericity': 0.91, 'original_firstorder_Mean': 38.5, ...},
            3: {'original_shape_Sphericity': 0.76, 'original_firstorder_Mean': 52.1, ...}
        }

    Raises:
        ValueError: If mask contains no foreground labels
    """
    # Load mask to detect classes
    sitk_mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(sitk_mask)

    # Get unique labels (exclude background = 0)
    unique_labels = np.unique(mask_array)
    unique_labels = unique_labels[unique_labels > 0]

    # Filter excluded labels
    if exclude_labels:
        unique_labels = unique_labels[~np.isin(unique_labels, exclude_labels)]

    if len(unique_labels) == 0:
        raise ValueError(f"Mask {mask_path.name} contains no foreground labels (after exclusion)")

    logger.info(f"Detected {len(unique_labels)} classes in {mask_path.name}: {unique_labels}")

    # Extract features for each class
    features_by_class = {}

    for class_id in unique_labels:
        try:
            # Create binary mask for this class
            binary_mask_array = (mask_array == class_id).astype(np.uint8)

            # Save to temporary path for extraction
            temp_mask = sitk.GetImageFromArray(binary_mask_array)
            temp_mask.CopyInformation(sitk_mask)  # Copy spatial metadata

            # Create extractor and extract
            sitk_image = sitk.ReadImage(str(image_path))
            extractor = create_radiomics_extractor(feature_classes)
            features_raw = extractor.execute(sitk_image, temp_mask)

            # Convert to safe floats
            features = {}
            for key, value in features_raw.items():
                if not key.startswith('diagnostics_'):
                    features[key] = _safe_float(value)

            features_by_class[int(class_id)] = features

            # Clean up
            del extractor, temp_mask, binary_mask_array
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to extract features for class {class_id}: {str(e)}")
            features_by_class[int(class_id)] = {}

    return features_by_class


def format_features_for_pysegqc(
    features_dict: Dict[str, float],
    class_id: int = 1
) -> Dict[str, float]:
    """
    Format PyRadiomics features for pySegQC PCA_Data sheet.

    Adds position prefix (001_, 002_, 003_) to match pySegQC's multi-structure format.

    Args:
        features_dict: Raw features from PyRadiomics
        class_id: Class ID for multi-class masks (1-based, default: 1)

    Returns:
        Dict with position-prefixed feature names
        Example: 'original_shape_Sphericity' → '001_original_shape_Sphericity'
    """
    formatted = {}
    prefix = f"{class_id:03d}_"

    for key, value in features_dict.items():
        # Skip diagnostic keys (should already be filtered, but double-check)
        if key.startswith('diagnostics_'):
            continue

        # Add position prefix
        formatted_key = f"{prefix}{key}"
        formatted[formatted_key] = value

    return formatted


def find_image_mask_pairs(
    root_dir: Path,
    image_dir: str = 'images',
    mask_dir: str = 'masks',
    image_pattern: str = '*.nii*',
    mask_pattern: str = '*.nii*'
) -> List[Tuple[Path, Path]]:
    """
    Find matching image-mask pairs in directory tree.

    Supports paired directory structure:
        root_dir/
        ├── images/
        │   ├── patient001_CT.nii.gz
        │   └── patient002_CT.nii.gz
        └── masks/
            ├── patient001_mask.nii.gz
            └── patient002_mask.nii.gz

    Args:
        root_dir: Root directory containing image and mask subdirectories
        image_dir: Name of image subdirectory (default: 'images')
        mask_dir: Name of mask subdirectory (default: 'masks')
        image_pattern: Glob pattern for image files (default: '*.nii*')
        mask_pattern: Glob pattern for mask files (default: '*.nii*')

    Returns:
        List of (image_path, mask_path) tuples

    Raises:
        FileNotFoundError: If directories don't exist
    """
    root = Path(root_dir)
    images_path = root / image_dir
    masks_path = root / mask_dir

    if not images_path.exists():
        raise FileNotFoundError(f"Image directory not found: {images_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_path}")

    # Find all image and mask files
    image_files = sorted(images_path.glob(image_pattern))
    mask_files = sorted(masks_path.glob(mask_pattern))

    logger.info(f"Found {len(image_files)} image files and {len(mask_files)} mask files")

    # Match by patient ID
    pairs = []
    unmatched_images = []

    for img_path in image_files:
        # Extract patient ID from image filename
        patient_id = extract_patient_id(img_path.stem.replace('.nii', ''))

        # Find corresponding mask (exact ID comparison to avoid substring false matches)
        matched = False
        for mask_path in mask_files:
            mask_patient_id = extract_patient_id(mask_path.stem.replace('.nii', ''))
            if patient_id == mask_patient_id:
                pairs.append((img_path, mask_path))
                matched = True
                break

        if not matched:
            unmatched_images.append(img_path.name)

    if unmatched_images:
        logger.warning(f"Failed to find masks for {len(unmatched_images)} images: {unmatched_images[:5]}")

    logger.info(f"Matched {len(pairs)} image-mask pairs")

    return pairs


def create_pca_data_sheet(
    image_mask_pairs: List[Tuple[Path, Path]],
    output_excel: Path,
    feature_classes: Optional[List[str]] = None,
    n_jobs: int = 1,
    label_map: Optional[Dict[int, str]] = None,
    exclude_labels: Optional[List[int]] = None
) -> Path:
    """
    Create PCA_Data Excel sheet from NIfTI files.

    This is the main entry point that orchestrates the full pipeline:
    - Feature extraction with progress tracking
    - Automatic multi-class mask detection and handling
    - Excel generation with pySegQC-compatible formatting

    The output Excel file has the PCA_Data sheet with:
    - Metadata columns: Case_ID, Image_Path, Mask_Path, N_Structures
    - Feature columns: 001_original_*, 002_original_*, etc. (position prefixes)

    Args:
        image_mask_pairs: List of (image_path, mask_path) tuples from find_image_mask_pairs()
        output_excel: Path for output Excel file
        feature_classes: Feature classes to extract (default: all 6 standard classes)
        n_jobs: Number of parallel jobs (1 = serial, -1 = all CPUs, currently unused)
        label_map: Optional dict mapping integer label IDs to structure names
        exclude_labels: Optional list of integer label IDs to skip during extraction

    Returns:
        Path to created Excel file

    Example:
        >>> from pathlib import Path
        >>> pairs = find_image_mask_pairs(Path('/data/nifti'))
        >>> output = create_pca_data_sheet(
        ...     pairs,
        ...     Path('features.xlsx'),
        ...     feature_classes=['shape', 'firstorder']
        ... )
        >>> print(f"Created {output}")
        Created features.xlsx

    Raises:
        ValueError: If image_mask_pairs is empty
    """
    if len(image_mask_pairs) == 0:
        raise ValueError("No image-mask pairs provided")

    print(f"\n{'='*70}")
    print("EXTRACTING RADIOMICS FEATURES FROM NIFTI FILES")
    print(f"{'='*70}")
    print(f"\nCases to process: {len(image_mask_pairs)}")
    print(f"Feature classes: {feature_classes or DEFAULT_FEATURE_CLASSES}")
    print(f"Output: {output_excel}")

    # Extract features for all cases
    all_case_data = []

    for img_path, mask_path in tqdm(image_mask_pairs, desc="Extracting features"):
        case_id = extract_patient_id(img_path.stem.replace('.nii', ''))

        try:
            # Check if multi-class mask
            sitk_mask = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(sitk_mask)
            unique_labels = np.unique(mask_array)
            unique_labels = unique_labels[unique_labels > 0]

            # Filter excluded labels
            if exclude_labels:
                unique_labels = unique_labels[~np.isin(unique_labels, exclude_labels)]
                if len(unique_labels) == 0:
                    logger.warning(f"All labels excluded for {case_id}, skipping")
                    continue

            if len(unique_labels) > 1:
                # Multi-class: extract for each class (with exclusion filtering)
                features_by_class = process_multi_class_mask(
                    img_path, mask_path, feature_classes,
                    exclude_labels=exclude_labels
                )

                # Build row with all classes
                row_data = {
                    'Case_ID': case_id,
                    'Image_Path': str(img_path),
                    'Mask_Path': str(mask_path),
                    'N_Structures': len(features_by_class)
                }

                # Add features with position prefixes
                for class_id, features in features_by_class.items():
                    formatted_features = format_features_for_pysegqc(features, class_id)
                    row_data.update(formatted_features)

                all_case_data.append(row_data)

            else:
                # Single class: extract normally
                features = extract_features_from_nifti(img_path, mask_path, feature_classes)
                formatted_features = format_features_for_pysegqc(features, class_id=1)

                row_data = {
                    'Case_ID': case_id,
                    'Image_Path': str(img_path),
                    'Mask_Path': str(mask_path),
                    'N_Structures': 1
                }
                row_data.update(formatted_features)
                all_case_data.append(row_data)

        except Exception as e:
            logger.error(f"Failed to process {case_id}: {str(e)}")
            # Add row with NaN values to preserve case in dataset
            all_case_data.append({
                'Case_ID': case_id,
                'Image_Path': str(img_path),
                'Mask_Path': str(mask_path),
                'N_Structures': 0,
                'Error': str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(all_case_data)

    # Reorder columns: metadata first, then feature columns
    metadata_cols = ['Case_ID', 'Image_Path', 'Mask_Path', 'N_Structures']
    feature_cols = [col for col in df.columns if re.match(r'^\d{3}_', str(col))]
    other_cols = [col for col in df.columns if col not in metadata_cols and col not in feature_cols]

    ordered_cols = metadata_cols + sorted(feature_cols) + other_cols
    ordered_cols = [col for col in ordered_cols if col in df.columns]  # Filter to existing
    df = df[ordered_cols]

    # Save to Excel (with optional Label_Map sheet)
    output_excel = Path(output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='PCA_Data', index=False)

        # Write Label_Map sheet if label map is provided
        if label_map:
            excluded_set = set(exclude_labels) if exclude_labels else set()
            label_map_rows = [
                {'Label_ID': k, 'Structure_Name': v, 'Position_Prefix': f'{k:03d}_'}
                for k, v in sorted(label_map.items())
                if k not in excluded_set
            ]
            if label_map_rows:
                label_map_df = pd.DataFrame(label_map_rows)
                label_map_df.to_excel(writer, sheet_name='Label_Map', index=False)

    print(f"\n✓ Feature extraction complete!")
    print(f"  Cases processed: {len(all_case_data)}")
    print(f"  Features per structure: {len([c for c in feature_cols if c.startswith('001_')])}")
    print(f"  Output saved to: {output_excel}")
    if label_map:
        print(f"  Label map saved to 'Label_Map' sheet")

    return output_excel
