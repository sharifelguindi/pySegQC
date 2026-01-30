# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pySegQC** is a medical imaging radiomics quality control package that performs PCA-based clustering analysis on segmentation feature data. The tool identifies quality patterns in medical image segmentations (e.g., from radiation therapy contours) using hierarchical clustering, supports multi-structure analysis, and can select diverse training cases for model development.

**Core workflow**: Excel radiomics data → PCA dimensionality reduction → Ward hierarchical clustering → Interactive visualizations → Training case selection

## CLI Commands

### Feature Extraction from NIfTI Files (New!)

```bash
# Extract features from NIfTI images and masks
pysegqc extract /path/to/data/ --output features.xlsx

# Custom directory structure
pysegqc extract /data/ --image-dir ct --mask-dir seg

# Parallel processing
pysegqc extract /data/ --n-jobs 8 --output features.xlsx

# Custom file patterns
pysegqc extract /data/ \
  --image-pattern "*_CT.nii.gz" \
  --mask-pattern "*_mask.nii.gz" \
  --output features.xlsx
```

**Expected Directory Structure:**
```
data/
├── images/
│   ├── patient001_CT.nii.gz
│   ├── patient002_CT.nii.gz
│   └── patient003_CT.nii.gz
└── masks/
    ├── patient001_mask.nii.gz
    ├── patient002_mask.nii.gz
    └── patient003_mask.nii.gz
```

**Multi-Class Mask Support:** If masks contain multiple labels (1, 2, 3...), features are automatically extracted for each class and labeled with 001_, 002_, 003_ prefixes.

### Clustering Analysis

```bash
# Analyze extracted features (or pre-computed Excel)
pysegqc analyze features.xlsx --auto-k --volume-independent

# Multi-structure analysis modes
pysegqc analyze data.xlsx --mode concat --auto-k
pysegqc analyze data.xlsx --mode position --position 1 --n-clusters 3

# With training case selection
pysegqc analyze data.xlsx --auto-k --select-training-cases
```

### Prediction

```bash
# Predict clusters for new cases
pysegqc predict new_cases.xlsx results/ --sheet PCA_Data
```

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_pca.py

# Run with coverage report (HTML report in htmlcov/)
pytest --cov=pysegqc --cov-report=html --cov-report=term

# Run single test function
pytest tests/test_clustering.py::test_hierarchical_clustering -v
```

### Code Quality
```bash
# Format code (line length: 119)
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Installation

**IMPORTANT: Python 3.12 Required**

- **Supported:** Python 3.8 - 3.12
- **Not Supported:** Python 3.13 (pyradiomics compatibility issues)
- **Recommended:** Python 3.12

**Creating Python 3.12 Environment:**
```bash
# Install Python 3.12 (if needed)
brew install python@3.12  # macOS with Homebrew

# Create virtual environment with Python 3.12
rm -rf venv
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
```

**Standard Installation:**
```bash
# Basic installation (includes NIfTI extraction support)
# CPU-only mode (no GPU required)
DISABLE_CUDA_EXTENSIONS=1 pip install -e .

# With development dependencies (recommended for contributors)
DISABLE_CUDA_EXTENSIONS=1 pip install -e ".[dev]"
```

**Note on NIfTI Dependencies:**
- Uses `pyradiomics-cuda` (drop-in replacement for pyradiomics)
- Supports Python 3.9-3.13 including Python 3.12
- `DISABLE_CUDA_EXTENSIONS=1` ensures CPU-only installation (no GPU needed)
- Feature extraction results are identical to original pyradiomics
- Clean single-command installation (no conda or build workarounds needed)

**Alternative (if environment variable doesn't work):**
```bash
export DISABLE_CUDA_EXTENSIONS=1
pip install -e ".[dev]"
```

### Running the CLI

**Extract features from NIfTI files:**
```bash
# Extract radiomics features from NIfTI images and masks
pysegqc extract /path/to/data/ --output features.xlsx

# With custom directory structure
pysegqc extract /data/ --image-dir ct --mask-dir seg --n-jobs 8
```

**Analyze extracted features:**
```bash
# Basic multi-structure analysis (per-structure mode is default)
pysegqc analyze features.xlsx --auto-k

# Volume-independent clustering (recommended for shape-based QC)
pysegqc analyze features.xlsx --auto-k --volume-independent

# Concatenated multi-structure analysis (combine all structures)
pysegqc analyze data.xlsx --mode concat --auto-k

# Single position analysis
pysegqc analyze data.xlsx --mode position --position 1 --n-clusters 3

# With training case selection
pysegqc analyze data.xlsx --mode concat --auto-k --select-training-cases
```

**Predict clusters for new cases:**
```bash
pysegqc predict new_cases.xlsx results/ --sheet PCA_Data
```

## Architecture

### Module Organization

The codebase follows a functional pipeline architecture with clear separation of concerns:

- **`__main__.py`**: CLI with subcommands (extract, analyze, predict) and multi-structure orchestration logic
- **`feature_extraction.py`**: **NEW!** PyRadiomics-based feature extraction from NIfTI files (images and masks)
- **`pipeline.py`**: Main analysis orchestration - coordinates all processing steps
- **`data_loader.py`**: Excel loading, metadata extraction, feature column detection
- **`validation.py`**: Imputation, standardization, outlier detection
- **`utils.py`**: Feature classification (volume-dependent vs independent), structure position detection
- **`pca.py`**: PCA transformation and variance analysis
- **`clustering.py`**: Hierarchical (Ward) and k-means clustering implementations
- **`metrics.py`**: Quality metrics (silhouette, gap statistic, stability, consensus clustering)
- **`training.py`**: Training case selection algorithms (diversity-based sampling from clusters)
- **`visualization.py`**: Static plots (matplotlib/seaborn) and interactive HTML dashboards (Plotly)
- **`export.py`**: Excel export with conditional formatting by cluster
- **`prediction.py`**: Apply trained models to new cases
- **`excel_utils.py`**: Shared Excel formatting utilities (conditional formatting, cluster summary generation)
- **`plotly_utils.py`**: Plotly visualization utilities (hover text, URL extraction, color palettes)
- **`metadata_utils.py`**: Metadata extraction and display utilities (case metadata, console banners)

### NIfTI Feature Extraction (feature_extraction.py)

**NEW in v0.2**: pySegQC can now extract radiomics features directly from NIfTI medical images using PyRadiomics, eliminating the need for pre-computed feature Excel files.

**Key Functions:**

1. **`create_pca_data_sheet()`** - Main orchestrator
   - Accepts list of (image_path, mask_path) tuples
   - Handles multi-class masks automatically (extracts each class as separate structure)
   - Generates PCA_Data Excel sheet compatible with existing pipeline
   - Shows progress bar (tqdm) during extraction

2. **`find_image_mask_pairs()`** - File discovery
   - Matches image-mask pairs from paired directory structure
   - Extracts patient IDs using regex patterns (patient001, sub-001, case_123, etc.)
   - Supports custom directory names and glob patterns

3. **`extract_features_from_nifti()`** - Single-case extraction
   - Loads NIfTI with SimpleITK (preserves spatial metadata)
   - Validates dimensions match between image and mask
   - Configures PyRadiomics with memory-efficient settings (binWidth=50)
   - Extracts 6 feature classes: firstorder, shape, glcm, glrlm, glszm, ngtdm
   - Returns ~93 features per structure

4. **`process_multi_class_mask()`** - Multi-structure support
   - Detects unique labels in mask (excludes background=0)
   - Extracts binary mask for each class
   - Returns dict: {class_id: features_dict}
   - Example: 3-class mask → {1: {...}, 2: {...}, 3: {...}}

5. **`format_features_for_pysegqc()`** - Position prefix formatting
   - Adds 001_, 002_, 003_ prefixes to match pySegQC convention
   - Example: `original_shape_Sphericity` → `001_original_shape_Sphericity`

**PyRadiomics Configuration:**
```python
DEFAULT_FEATURE_CLASSES = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'ngtdm']

# Memory-efficient settings
binWidth = 50          # 75% memory reduction vs default
minimumROISize = 1     # Process even small structures
normalize = True       # Normalize image intensities
```

**Spatial Metadata Handling (CRITICAL):**
```python
sitk_mask = sitk.GetImageFromArray(mask_array)
sitk_mask.CopyInformation(sitk_image)  # DO NOT SKIP - copies spacing/origin/direction
```

Without `CopyInformation()`, PyRadiomics will fail or produce incorrect results.

**Multi-Class Mask Workflow:**
```
Input: mask.nii.gz with labels [1, 2, 3]
↓
Extract class 1 (binary) → Extract features → Prefix with 001_
Extract class 2 (binary) → Extract features → Prefix with 002_
Extract class 3 (binary) → Extract features → Prefix with 003_
↓
Single Excel row: [Case_ID, 001_feat1, 001_feat2, ..., 002_feat1, 002_feat2, ..., 003_feat1, ...]
```

**Integration with Existing Pipeline:**

The extracted PCA_Data Excel sheet has identical format to manually-computed features:
- Metadata columns: Case_ID, Image_Path, Mask_Path, N_Structures
- Feature columns: 001_original_*, 002_original_*, etc.
- Compatible with all downstream analysis modes (default, concat, position)

**Example Workflow:**
```bash
# 1. Extract features from NIfTI files
pysegqc extract /data/ct_scans/ --output extracted_features.xlsx

# 2. Analyze with volume-independent clustering
pysegqc analyze extracted_features.xlsx --auto-k --volume-independent

# 3. Predict on new NIfTI cases
pysegqc extract /data/new_scans/ --output new_features.xlsx
pysegqc predict new_features.xlsx results/
```

**Dependencies:**
- `nibabel`: NIfTI file I/O (required)
- `SimpleITK`: Medical image processing, spatial metadata handling (required)
- `pyradiomics-cuda`: Radiomics feature extraction (required, drop-in replacement for pyradiomics)
- `joblib`: Parallel processing utilities (required)

**Installation:**

NIfTI feature extraction is a **core feature** of pySegQC, included in the standard installation:

```bash
# Standard installation includes all NIfTI dependencies (CPU-only, no GPU required)
DISABLE_CUDA_EXTENSIONS=1 pip install -e .
```

**Python Version Requirement:**
- **Supported:** Python 3.8 - 3.12
- **Not Supported:** Python 3.13 (see note at top of Installation section)
- **Recommended:** Python 3.12

**Note:** If you have pre-computed radiomics features in Excel format, you can directly use the `pysegqc analyze` command and skip the `pysegqc extract` step.

### Data Flow

1. **Input**: Excel file with sheet `PCA_Data` containing:
   - Metadata columns (no position prefix): `MRN`, `Patient_ID`, etc.
   - Feature columns (with position prefix): `001_feature_name`, `002_feature_name`, etc.

2. **Preprocessing**:
   - Detect structure positions from column prefixes (`001_`, `002_`, ...)
   - Separate metadata from numeric features
   - Handle three analysis modes:
     - **Default**: Per-structure (loop through each position independently)
     - **Concat**: Concatenate all structure features into single patient-level row
     - **Position**: Analyze single specified position

3. **Volume Independence** (critical for shape-based clustering):
   - `--volume-independent`: Filter out volume-dependent features (recommended)
   - `--volume-normalize`: Normalize by volume powers (alternative approach)
   - See `utils.VOLUME_DEPENDENT_FEATURES` for complete classification

4. **PCA → Clustering → Export**:
   - Impute missing values (default: median)
   - Standardize features (z-score)
   - PCA with configurable components (default: 10)
   - Ward hierarchical clustering with auto k-selection (silhouette scores)
   - Export clustered data to Excel with per-cluster sheets
   - Generate interactive HTML dashboard

5. **Training Selection** (optional):
   - Select diverse, representative cases from each cluster
   - Create union summary across structures for multi-structure data
   - Export selected cases ready for model training

### Multi-Structure Handling

**Default Behavior**: When multi-structure data is detected (columns like `001_*`, `002_*`), the CLI automatically runs independent analyses for each structure in separate subfolders (`structure_001_results/`, etc.) unless `--mode concat` or `--mode position` is specified.

**Concat Mode**: Combines features from all structures into a single row per patient: `[001_feat1, 001_feat2, 002_feat1, 002_feat2, ...]` → enables patient-level clustering across all structures.

### Key Design Patterns

**Model Serialization**: Trained models are saved via `joblib` in `trained_models.pkl` containing:
- PCA model
- Scaler
- Clustering model
- Feature names
- Imputation strategy
- Analysis mode/position

This enables `--predict` mode to apply the exact same transformations to new cases.

**Interactive Visualizations**: The package generates both static PNG plots and interactive Plotly HTML dashboards. The dashboard (`analysis_dashboard.html`) consolidates all key visualizations with clickable data points that link to scan URLs (if available in metadata).

**Training Case Selection**: Uses distance-from-centroid ranking within each cluster to select diverse, representative cases. For multi-structure data, creates a union of selections across structures to maximize coverage.

## Important Implementation Notes

### Volume-Dependent Features

**Critical**: Medical radiomics features often correlate with structure volume, which can dominate clustering even after standardization. The `utils.py` module contains comprehensive classification of volume-dependent vs independent features. Always consider using `--volume-independent` for shape-based quality control.

### Feature Column Detection

The system uses regex pattern `r'^\d{3}_'` to detect position-prefixed features. When adding new data loading logic, ensure this pattern is consistently applied.

### Cluster Color Consistency

`utils.CLUSTER_COLORS` defines a fixed palette for up to 10 clusters. This ensures visual consistency across all plots and Excel exports. When extending visualizations, use this palette.

### Excel Export Structure

The `export.py` module creates multi-sheet Excel files:
- **Summary**: Overall statistics and cluster sizes
- **All_Cases**: Complete dataset with cluster assignments
- **Data_Cluster_X**: Separate sheets per cluster with conditional formatting
- **PCA_Loadings**: Feature importance for each principal component

This structure is expected by downstream analysis workflows.

### Logging Configuration

The package uses Python's `logging` module. Key messages use `logger.info()` for progress and `logger.warning()` for potential issues. When adding new functionality, follow this convention rather than using `print()` statements in library code (CLI output in `__main__.py` uses `print()` for user-facing messages).

## Testing Notes

### Test Infrastructure

The test suite uses **synthetic radiomics data** generated programmatically via fixtures in `tests/conftest.py`. This approach:
- Eliminates dependency on large committed test files
- Creates realistic 3-cluster patterns for validation
- Supports multi-structure data with position prefixes (001_, 002_)
- Generates 50 samples with volume-dependent and independent features

### Key Fixtures

- `synthetic_radiomics_excel(tmp_path)` - Main fixture creating Excel with PCA_Data sheet
- `temp_output_dir(tmp_path)` - Temporary directory for test outputs
- `trained_models_dir(tmp_path, synthetic_radiomics_excel)` - Pre-trained models for prediction tests
- `three_cluster_data()` - sklearn make_blobs data for metrics testing

### Current Coverage

**Overall: ~63% (~100 tests)**

Excellent coverage modules:
- `validation.py`: 100%
- `clustering.py`: 100%
- `pca.py`: 100%
- `__init__.py`: 100%
- `metrics.py`: 95%
- `training.py`: 82%
- `prediction.py`: 80%
- `pipeline.py`: 77%
- `utils.py`: 68%

Moderate coverage (opportunities for improvement):
- `data_loader.py`: 54%
- `export.py`: 54%
- `visualization.py`: 54%
- `excel_utils.py`: ~80% (core formatting utilities)
- `plotly_utils.py`: ~90% (hover text, URL extraction)
- `metadata_utils.py`: ~90% (metadata extraction)

Not covered:
- `__main__.py`: 0% (CLI, tested manually)

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures (Excel, NIfTI synthetic data generators)
├── test_conftest.py         # Tests for fixtures themselves
├── test_data_loader.py      # Data loading with different modes
├── test_validation.py       # Imputation, standardization, outlier detection
├── test_utils.py            # Volume independence, feature filtering
├── test_metrics.py          # Cluster quality metrics
├── test_clustering.py       # Hierarchical and k-means clustering
├── test_pca.py              # PCA transformation
├── test_feature_extraction.py # NIfTI feature extraction (21 tests)
├── test_prediction.py       # Prediction on new cases (80% coverage)
├── test_training.py         # Training case selection (82% coverage)
├── test_export_training.py  # Training selection export
├── test_pipeline.py         # Integration tests (exercises multiple modules)
├── test_imports.py          # Basic import smoke tests
├── test_excel_utils.py      # Excel formatting utilities (conditional formatting, summaries)
├── test_plotly_utils.py     # Plotly visualization utilities
├── test_metadata_utils.py   # Metadata extraction utilities
└── test_insufficient_data.py # Edge cases for insufficient data handling (graceful degradation)
```

**New NIfTI Test Fixtures:**
- `synthetic_nifti_data(tmp_path)`: Creates 3 image-mask pairs in paired directories (images/, masks/)
- `synthetic_multiclass_nifti(tmp_path)`: Creates single case with 3-class mask for multi-structure testing

**test_feature_extraction.py covers:**
- Helper functions (_safe_float, extract_patient_id, validate_nifti_pair)
- PyRadiomics extractor configuration
- Single-case and multi-class feature extraction
- File discovery and patient ID matching
- Excel generation and format validation
- Integration with existing data_loader pipeline

### Running Tests

```bash
# All tests with coverage
pytest --cov=pysegqc --cov-report=html

# Specific module
pytest tests/test_pipeline.py -v

# Specific test
pytest tests/test_utils.py::test_filter_volume_dependent_features -v

# Coverage report location
open htmlcov/index.html
```

### Adding New Tests

When adding tests for new modules:
1. Use `synthetic_radiomics_excel` fixture for radiomics data
2. Use `temp_output_dir` for file outputs (auto-cleanup)
3. Test both happy paths and edge cases
4. Follow naming convention: `test_<functionality>` not `test_<implementation_detail>`
5. Keep tests focused and independent

## Scratch Directory

**All manual testing, data analysis, and experimental outputs must go in `scratch/`.** This directory is git-ignored and serves as the working area for:

- NIfTI test data (copied or symlinked patient files)
- Extracted feature Excel files (`*.xlsx`)
- Clustering results directories (`*_clustering_results/`)
- Label map JSON files
- Prediction outputs
- Any other ad-hoc analysis artifacts

```bash
# Example: run extraction into scratch/
pysegqc extract /path/to/data/ --output scratch/features.xlsx

# Example: run analysis with output in scratch/
pysegqc analyze scratch/features.xlsx --auto-k --volume-independent

# Example: predict into scratch/
pysegqc predict scratch/test_data/hn_oars/ scratch/clustering_results/ \
    --image-dir image --mask-dir mask
```

**Never place data files, Excel outputs, or analysis results in the project root.** The project root should only contain source code, config, and documentation.

## Common Gotchas

1. **Missing dependencies**: The package requires openpyxl for Excel I/O. Don't use xlrd/xlwt.

2. **Cluster count edge cases**: The auto k-selection (`--auto-k`) may fail if silhouette scores are ambiguous. Always validate with dendrogram visualization.

3. **Memory for large datasets**: PCA and clustering operate in-memory. For datasets >10,000 samples, consider batch processing or dimensionality reduction before PCA.

4. **Excel sheet names**: Default sheet name is `PCA_Data`. If input data uses different sheet names, must specify with `--sheet`.

5. **Position prefix format**: Must be exactly 3 digits (`001_`, `002_`, not `1_` or `0001_`). The regex pattern is strict.

## Package Distribution

- Package name: `pysegqc`
- Entry point: `pysegqc` command (defined in `pyproject.toml`)
- Version: Managed in `src/pysegqc/__init__.py` and `pyproject.toml`
- Build system: setuptools (PEP 517)
- License: MIT
