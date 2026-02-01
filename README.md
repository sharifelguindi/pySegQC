# pySegQC

**Quality Control and Clustering Analysis for Medical Image Segmentation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8-3.12](https://img.shields.io/badge/python-3.8--3.12-blue.svg)](https://www.python.org/downloads/)

A Python package for automated quality control of medical image segmentations using radiomics features, PCA dimensionality reduction, hierarchical clustering, and dual-method outlier detection. Includes a built-in NiiVue-based viewer for interactive case review.

## Features

- **NIfTI Feature Extraction**: Extract ~93 radiomics features per structure using PyRadiomics
- **Automated Clustering**: PCA + Ward hierarchical clustering with automatic k-selection
- **Multi-Structure Support**: Analyze multiple structures independently or combined
- **Multi-Class Masks**: Automatic detection and processing of multi-label segmentation masks
- **QA Verdict System**: Dual-method outlier detection (distance z-score + Isolation Forest)
- **NiiVue Viewer**: Built-in multiplanar NIfTI viewer with W/L controls and mask overlay
- **Training Case Selection**: Select diverse, representative cases from each cluster
- **Interactive Dashboards**: Plotly-based HTML dashboards with clickable data points
- **Prediction Mode**: Apply trained models to new cases with consistent QA thresholds
- **Volume Independence**: Filter or normalize volume-dependent features
- **CLI & Python API**: Full command-line interface and importable library

## Installation

**Python 3.12 recommended** (supports 3.8-3.12; Python 3.13 not supported due to pyradiomics).

### From Source

```bash
git clone https://github.com/sharifelguindi/pySegQC.git
cd pySegQC

# CPU-only installation (no GPU required)
DISABLE_CUDA_EXTENSIONS=1 pip install -e .

# With development dependencies
DISABLE_CUDA_EXTENSIONS=1 pip install -e ".[dev]"
```

### Creating a Python 3.12 Environment

```bash
brew install python@3.12  # macOS
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
DISABLE_CUDA_EXTENSIONS=1 pip install -e ".[dev]"
```

## Quick Start

### End-to-End Workflow

```bash
# 1. Extract radiomics features from NIfTI files
pysegqc extract /path/to/data/ --image-dir image --mask-dir mask --output features.xlsx

# 2. Cluster and analyze
pysegqc analyze features.xlsx --auto-k --volume-independent

# 3. Predict on new cases (generates viewer + QA report)
pysegqc predict /path/to/new_data/ features_clustering_results/ \
    --image-dir image --mask-dir mask
```

### Python API

```python
from pysegqc import run_analysis_pipeline
from pathlib import Path

class Args:
    auto_k = True
    volume_independent = True
    n_components = 10
    method = 'hierarchical'
    impute = 'median'
    select_training_cases = False

run_analysis_pipeline(Args(), Path("features.xlsx"), Path("output/"))
```

## QA Scoring & Verdicts

pySegQC uses a **dual-method outlier detection** system to assign quality verdicts to each case.

### How It Works

Each case is evaluated by two independent methods:

1. **Distance Z-Score** (intra-cluster): After PCA + clustering, each case's Euclidean distance from its cluster centroid is computed. The z-score normalizes this distance within the cluster (how many standard deviations from the cluster's typical distance). Cases with z > 2.0 are flagged.

2. **Isolation Forest** (global): An unsupervised anomaly detector trained on all cases in PCA space. Uses 10% contamination rate to identify global outliers regardless of cluster assignment.

### Verdict Logic

| Distance Flagged (z > 2σ) | Isolation Forest Flagged | Verdict |
|:---:|:---:|:---:|
| No | No | **PASS** |
| Yes | No | **REVIEW** |
| No | Yes | **REVIEW** |
| Yes | Yes | **FAIL** |

- **PASS**: Case is consistent with its cluster — no anomalies detected
- **REVIEW**: One method flagged the case — warrants manual inspection
- **FAIL**: Both methods independently flagged the case — high confidence anomaly

### QA Risk Score

A continuous 0-1 score combining both methods:

```
risk = 0.6 × clip(z_score / 5, 0, 1) + 0.4 × normalized_iforest_score
```

Higher scores indicate greater deviation from expected patterns. The 60/40 weighting emphasizes the distance-based method since it's cluster-aware.

### Prediction Mode

When predicting on new cases, pySegQC reuses the **training cluster statistics** (per-cluster mean/std distances) so that verdicts are consistent between training and prediction. New cases are scored against the same thresholds established during training.

## Analysis Modes

### 1. Per-Structure (Default)

Analyzes each structure independently in separate subfolders.

```bash
pysegqc analyze features.xlsx --auto-k
# Creates: structure_001_results/, structure_002_results/, etc.
```

### 2. Concatenated Multi-Structure

Combines all structure features into a single patient-level analysis.

```bash
pysegqc analyze features.xlsx --mode concat --auto-k
```

### 3. Single Position

Analyzes only one structure position.

```bash
pysegqc analyze features.xlsx --mode position --position 1 --n-clusters 3
```

## CLI Reference

### Extract Command

```bash
pysegqc extract DATA_DIR [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output PATH` | Output Excel file | `extracted_features.xlsx` |
| `--image-dir NAME` | Image subdirectory name | `images` |
| `--mask-dir NAME` | Mask subdirectory name | `masks` |
| `--label-map JSON` | Label map file for structure naming | None |
| `--exclude-labels N [N ...]` | Labels to exclude from extraction | None |
| `--n-jobs N` | Parallel jobs (-1 for all CPUs) | 1 |
| `--subset N` | Process only first N cases | None |

### Analyze Command

```bash
pysegqc analyze INPUT_FILE [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--sheet NAME` | Excel sheet name | `PCA_Data` |
| `--mode {concat,position}` | Analysis mode | per-structure |
| `--n-clusters N` | Fixed cluster count | None |
| `--auto-k` | Automatically find optimal k | False |
| `--volume-independent` | Filter volume-dependent features | False |
| `--select-training-cases` | Select diverse training cases | False |

### Predict Command

```bash
pysegqc predict INPUT MODEL_DIR [OPTIONS]
```

Input can be an Excel file or a NIfTI data directory. When a directory is provided, features are extracted on-the-fly.

| Option | Description | Default |
|--------|-------------|---------|
| `--image-dir NAME` | Image subdirectory (NIfTI mode) | `images` |
| `--mask-dir NAME` | Mask subdirectory (NIfTI mode) | `masks` |
| `--label-map JSON` | Label map for structure naming | None |

## Output Files

### Analysis Output

```
features_clustering_results/
├── structure_001_Brainstem_results/
│   ├── trained_models.pkl                 # Serialized models for prediction
│   ├── features_clustered.xlsx            # Results with per-cluster sheets
│   ├── pca_loadings.csv                   # Feature importance per PC
│   ├── cluster_statistics.csv             # Mean/std per cluster
│   ├── dendrogram.png                     # Hierarchical tree
│   ├── scree_plot.png                     # Variance explained
│   ├── cluster_selection.png              # Silhouette scores
│   ├── pca_clusters_2d.png                # 2D static plot
│   ├── feature_importance_heatmap.png     # Feature loadings
│   ├── pca_clusters_2d_interactive.html   # Interactive Plotly plot
│   └── analysis_dashboard.html            # Comprehensive dashboard
├── structure_002_Parotid_L_results/
│   └── ...
└── structure_003_Parotid_R_results/
    └── ...
```

### Prediction Output

```
model_dir/structure_001_results/_nifti_prediction_features_predictions/
├── prediction_report.json     # QA verdicts, risk scores, cluster assignments
├── viewer.html                # NiiVue multiplanar viewer
└── viewer_data.json           # Case manifest for viewer
```

### NiiVue Viewer

The built-in viewer provides:
- **Multiplanar display**: Axial, coronal, and sagittal views simultaneously
- **Mask overlay**: Single-structure mask display with automatic voxel filtering
- **W/L presets**: Brain, Bone, Lung, Pelvis presets plus custom level/width
- **Case navigation**: Previous/Next buttons with verdict badges and risk scores
- **File serving**: Open via `cd / && python -m http.server 8080`

## Volume Independence

Medical radiomics features often correlate with structure volume. pySegQC provides two strategies:

```bash
# Strategy 1: Filter volume-dependent features (recommended)
pysegqc analyze features.xlsx --volume-independent --auto-k

# Strategy 2: Normalize by volume powers
pysegqc analyze features.xlsx --volume-normalize --auto-k
```

## Data Format

pySegQC expects Excel files with a `PCA_Data` sheet:

| MRN | Case_ID | 001_shape_Volume | 001_shape_Sphericity | 002_shape_Volume | ... |
|-----|---------|------------------|---------------------|------------------|-----|
| 123 | PT001   | 45.2             | 0.87                | 32.1             | ... |

- **Metadata columns**: No position prefix (`MRN`, `Case_ID`, `Image_Path`, `Mask_Path`)
- **Feature columns**: 3-digit position prefix (`001_`, `002_`, `003_`)

## Package Structure

```
pySegQC/
├── src/pysegqc/
│   ├── __init__.py              # Public API exports
│   ├── __main__.py              # CLI entry point (extract, analyze, predict)
│   ├── pipeline.py              # Analysis orchestration
│   ├── feature_extraction.py    # NIfTI radiomics extraction (PyRadiomics)
│   ├── data_loader.py           # Excel loading, feature column detection
│   ├── validation.py            # Imputation, standardization, outlier detection
│   ├── utils.py                 # Volume independence, feature classification
│   ├── pca.py                   # PCA transformation
│   ├── clustering.py            # Hierarchical and k-means clustering
│   ├── metrics.py               # Silhouette, gap statistic, stability
│   ├── qa.py                    # QA verdicts and risk scoring
│   ├── training.py              # Training case selection
│   ├── visualization.py         # Static plots (matplotlib/seaborn)
│   ├── export.py                # Excel export with formatting
│   ├── prediction.py            # Apply models to new cases
│   ├── viewer.py                # NiiVue HTML viewer generation
│   ├── excel_utils.py           # Excel formatting utilities
│   ├── plotly_utils.py          # Plotly visualization utilities
│   └── metadata_utils.py        # Metadata extraction utilities
├── tests/                       # 185+ unit tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Dependencies

Core:
- numpy, pandas, scikit-learn, scipy
- matplotlib, seaborn, plotly
- openpyxl, tqdm, pyyaml, jinja2

NIfTI extraction:
- nibabel, SimpleITK
- pyradiomics-cuda (CPU-only with `DISABLE_CUDA_EXTENSIONS=1`)

## Development

```bash
git clone https://github.com/sharifelguindi/pySegQC.git
cd pySegQC
DISABLE_CUDA_EXTENSIONS=1 pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=pysegqc --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{pysegqc2025,
  title = {pySegQC: Quality Control and Clustering Analysis for Medical Image Segmentation},
  author = {Sharif Elguindi},
  year = {2025},
  url = {https://github.com/sharifelguindi/pySegQC}
}
```
