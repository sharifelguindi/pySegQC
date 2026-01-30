"""
Command-line interface for pySegQC.

Provides CLI access to all pySegQC functionality including feature extraction,
training/clustering analysis, and prediction.
"""

import argparse
import sys
import copy
import re
from pathlib import Path
import pandas as pd

from .pipeline import run_analysis_pipeline
from .prediction import predict_new_cases
from .utils import detect_structure_positions
from .feature_extraction import find_image_mask_pairs, create_pca_data_sheet


def create_extract_parser(subparsers):
    """Create parser for 'extract' subcommand."""
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract radiomics features from NIfTI files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extract radiomics features from NIfTI image and mask files',
        epilog="""
Examples:
  # Extract features from paired directories
  pysegqc extract /data/ct_scans/ --output features.xlsx

  # Custom directory names
  pysegqc extract /data/ --image-dir ct --mask-dir seg --output features.xlsx

  # Custom file patterns
  pysegqc extract /data/ --image-pattern "*_CT.nii.gz" --mask-pattern "*_mask.nii.gz"

  # Parallel processing
  pysegqc extract /data/ --n-jobs 8 --output features.xlsx

Directory Structure Expected:
  data_dir/
  ├── images/           (or custom --image-dir)
  │   ├── patient001_CT.nii.gz
  │   └── patient002_CT.nii.gz
  └── masks/            (or custom --mask-dir)
      ├── patient001_mask.nii.gz
      └── patient002_mask.nii.gz

Multi-Class Masks:
  If masks contain multiple labels (1, 2, 3...), features will be extracted
  for each class separately and labeled with 001_, 002_, 003_ prefixes.
        """
    )

    # Required positional argument
    extract_parser.add_argument(
        'data_dir',
        type=str,
        help='Root directory containing image and mask subdirectories'
    )

    # Output
    extract_parser.add_argument(
        '-o', '--output',
        type=str,
        default='extracted_features.xlsx',
        help='Output Excel file path (default: extracted_features.xlsx)'
    )

    # Directory structure
    extract_parser.add_argument(
        '--image-dir',
        type=str,
        default='images',
        help='Name of image subdirectory (default: images)'
    )
    extract_parser.add_argument(
        '--mask-dir',
        type=str,
        default='masks',
        help='Name of mask subdirectory (default: masks)'
    )

    # File patterns
    extract_parser.add_argument(
        '--image-pattern',
        type=str,
        default='*.nii*',
        help='Glob pattern for image files (default: *.nii*)'
    )
    extract_parser.add_argument(
        '--mask-pattern',
        type=str,
        default='*.nii*',
        help='Glob pattern for mask files (default: *.nii*)'
    )

    # Processing
    extract_parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (1=serial, -1=all CPUs, default: 1)'
    )
    extract_parser.add_argument(
        '--feature-classes',
        type=str,
        nargs='+',
        default=None,
        help='Feature classes to extract (default: all standard classes)'
    )

    # Label mapping and filtering
    extract_parser.add_argument(
        '--label-map',
        type=str,
        default=None,
        help='JSON file or inline JSON mapping label IDs to structure names. '
             'Example: \'{"2":"Parotid_L","3":"Parotid_R"}\' or path to .json file'
    )
    extract_parser.add_argument(
        '--exclude-labels',
        type=int,
        nargs='+',
        default=None,
        help='Label IDs to exclude from extraction (e.g., --exclude-labels 1 for Brainstem)'
    )
    extract_parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Extract only first N patients (sorted, for testing)'
    )

    extract_parser.set_defaults(func=extract_command)
    return extract_parser


def create_analyze_parser(subparsers):
    """Create parser for 'analyze' subcommand (main clustering analysis)."""
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Perform PCA + clustering analysis on radiomics data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='PCA + Hierarchical (Ward) Clustering Analysis for Segmentation QC',
        epilog="""
Examples:
  # DEFAULT: Multi-structure data analyzed per-structure (independent analyses)
  pysegqc analyze metrics.xlsx --auto-k
  # Creates: structure_001_results/, structure_002_results/, etc.

  # CONCAT MODE: Combine all structures into single analysis
  pysegqc analyze metrics.xlsx --mode concat --auto-k
  # Concatenates features: [001_*, 002_*, ...] → single patient-level analysis

  # CONCAT MODE: With training case selection
  pysegqc analyze metrics.xlsx --mode concat --auto-k --select-training-cases

  # POSITION MODE: Analyze single structure only
  pysegqc analyze metrics.xlsx --mode position --position 1 --n-clusters 3

  # Volume-independent clustering (recommended for shape-based QC)
  pysegqc analyze metrics.xlsx --auto-k --volume-independent

Analysis Modes:
  (default):  Per-structure independent analysis (creates subfolders for each structure)
  concat:     Concatenate all structure features (e.g., 001_*, 002_*) for combined analysis
  position:   Analyze single structure position only

Clustering Methods:
  hierarchical: Ward linkage (default) - deterministic, robust
  kmeans:       K-Means clustering - for comparison or legacy workflows
        """
    )

    # Required positional argument
    analyze_parser.add_argument(
        'input',
        type=str,
        help='Path to Excel file with radiomics features'
    )

    # Data loading
    analyze_parser.add_argument(
        '--sheet',
        default='PCA_Data',
        type=str,
        help='Sheet name to analyze (default: PCA_Data)'
    )

    # Analysis mode
    analyze_parser.add_argument(
        '--mode',
        default=None,
        choices=['concat', 'position'],
        help='Analysis mode: concat (combine all structures) or position (single structure)'
    )
    analyze_parser.add_argument(
        '--position',
        type=int,
        default=1,
        help='Structure position to use if mode=position (default: 1)'
    )

    # PCA settings
    analyze_parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of PCA components (default: 10)'
    )

    # Clustering settings
    analyze_parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (default: auto-detect)'
    )
    analyze_parser.add_argument(
        '--auto-k',
        action='store_true',
        help='Automatically find optimal k using silhouette scores'
    )
    analyze_parser.add_argument(
        '--max-k',
        type=int,
        default=10,
        help='Max k for auto-detection (default: 10)'
    )
    analyze_parser.add_argument(
        '--method',
        default='hierarchical',
        choices=['hierarchical', 'kmeans'],
        help='Clustering method (default: hierarchical)'
    )

    # Preprocessing
    analyze_parser.add_argument(
        '--impute',
        default='median',
        choices=['mean', 'median', 'most_frequent'],
        help='Missing value imputation strategy (default: median)'
    )

    # Volume independence
    analyze_parser.add_argument(
        '--volume-independent',
        action='store_true',
        help='Filter out volume-dependent features (ensures shape-based clustering)'
    )
    analyze_parser.add_argument(
        '--volume-normalize',
        action='store_true',
        help='Normalize volume-dependent features by volume powers'
    )

    # Training case selection
    analyze_parser.add_argument(
        '--select-training-cases',
        action='store_true',
        help='Select diverse training cases from clustering results'
    )

    # QA settings
    analyze_parser.add_argument(
        '--qa-sigma',
        type=float,
        default=2.0,
        help='Distance z-score threshold for QA outlier detection (default: 2.0)'
    )
    analyze_parser.add_argument(
        '--qa-contamination',
        type=float,
        default=0.1,
        help='Isolation Forest contamination rate (default: 0.1)'
    )

    # Thumbnail settings
    analyze_parser.add_argument(
        '--thumbnails',
        action='store_true',
        default=True,
        help='Generate NIfTI thumbnails in dashboard (default: enabled)'
    )
    analyze_parser.add_argument(
        '--no-thumbnails',
        dest='thumbnails',
        action='store_false',
        help='Disable NIfTI thumbnail generation'
    )
    analyze_parser.add_argument(
        '--thumbnail-window',
        type=str,
        default='40/400',
        help='CT window center/width for thumbnails (default: 40/400)'
    )

    # Output
    analyze_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: {input_stem}_clustering_results)'
    )

    analyze_parser.set_defaults(func=analyze_command)
    return analyze_parser


def create_predict_parser(subparsers):
    """Create parser for 'predict' subcommand."""
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict clusters for new cases using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Classify new cases using trained clustering models',
        epilog="""
Examples:
  # Predict from Excel (existing workflow)
  pysegqc predict new_cases.xlsx results/

  # Predict directly from NIfTI files (auto-detects directory input)
  pysegqc predict /path/to/nifti_data/ results/ --image-dir image --mask-dir mask

  # Per-structure prediction (auto-detected from model directory)
  pysegqc predict /data/ clustering_results/ --image-dir image --mask-dir mask

The model directory should contain 'trained_models.pkl' (single structure)
or 'structure_*_results/' subdirectories (multi-structure).
        """
    )

    # Required positional arguments
    predict_parser.add_argument(
        'input',
        type=str,
        help='Path to Excel file or NIfTI data directory'
    )
    predict_parser.add_argument(
        'model_dir',
        type=str,
        help='Directory containing trained_models.pkl (or parent with structure_*_results/ subdirs)'
    )

    # Data loading
    predict_parser.add_argument(
        '--sheet',
        default='PCA_Data',
        type=str,
        help='Sheet name in input file (default: PCA_Data)'
    )

    # Mode (must match training mode)
    predict_parser.add_argument(
        '--mode',
        default=None,
        choices=['concat', 'position'],
        help='Analysis mode (must match training mode)'
    )
    predict_parser.add_argument(
        '--position',
        type=int,
        default=1,
        help='Structure position if mode=position (default: 1)'
    )

    # NIfTI-specific options (used when input is a directory)
    predict_parser.add_argument(
        '--image-dir',
        default='images',
        type=str,
        help='Subdirectory name for images (default: images)'
    )
    predict_parser.add_argument(
        '--mask-dir',
        default='masks',
        type=str,
        help='Subdirectory name for masks (default: masks)'
    )
    predict_parser.add_argument(
        '--label-map',
        default=None,
        type=str,
        help='JSON file mapping label IDs to structure names'
    )

    predict_parser.set_defaults(func=predict_command)
    return predict_parser


def _parse_label_map(label_map_arg: str) -> dict:
    """Parse label map from JSON file path or inline JSON string.

    Args:
        label_map_arg: Path to .json file or inline JSON string

    Returns:
        Dictionary mapping integer label IDs to structure name strings
    """
    import json
    path = Path(label_map_arg)
    if path.exists() and path.suffix == '.json':
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = json.loads(label_map_arg)
    return {int(k): v for k, v in raw.items()}


def extract_command(args):
    """Execute feature extraction from NIfTI files."""
    print(f"\n{'='*70}")
    print("PYSEGQC: FEATURE EXTRACTION MODE")
    print(f"{'='*70}")

    data_dir = Path(args.data_dir)

    # Validate directory
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Find image-mask pairs
    try:
        pairs = find_image_mask_pairs(
            data_dir,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_pattern=args.image_pattern,
            mask_pattern=args.mask_pattern
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    if len(pairs) == 0:
        print(f"❌ Error: No matching image-mask pairs found in {data_dir}")
        print(f"   Looked for:")
        print(f"     - Images in: {data_dir / args.image_dir} (pattern: {args.image_pattern})")
        print(f"     - Masks in: {data_dir / args.mask_dir} (pattern: {args.mask_pattern})")
        sys.exit(1)

    # Parse label map if provided
    label_map = None
    if args.label_map:
        try:
            label_map = _parse_label_map(args.label_map)
            print(f"\n   Label map: {len(label_map)} structures defined")
            for label_id, name in sorted(label_map.items()):
                print(f"     {label_id:3d}: {name}")
        except Exception as e:
            print(f"❌ Error parsing label map: {e}")
            sys.exit(1)

    # Report excluded labels
    if args.exclude_labels:
        excluded_names = []
        for label_id in args.exclude_labels:
            name = label_map.get(label_id, f"label_{label_id}") if label_map else f"label_{label_id}"
            excluded_names.append(f"{label_id} ({name})")
        print(f"   Excluding labels: {', '.join(excluded_names)}")

    # Apply subset if requested
    if args.subset and args.subset < len(pairs):
        pairs = pairs[:args.subset]
        print(f"   Using subset: first {args.subset} patients (sorted)")

    # Extract features
    output_path = Path(args.output)
    try:
        result_path = create_pca_data_sheet(
            pairs,
            output_path,
            feature_classes=args.feature_classes,
            n_jobs=args.n_jobs,
            label_map=label_map,
            exclude_labels=args.exclude_labels
        )

        print(f"\n✅ Feature extraction complete!")
        print(f"   Output: {result_path}")
        print(f"\n📊 Next steps:")
        print(f"   # Analyze the extracted features")
        print(f"   pysegqc analyze {result_path} --auto-k --volume-independent")

    except Exception as e:
        print(f"\n❌ Error during feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def analyze_command(args):
    """Execute clustering analysis (main training workflow)."""
    print(f"\n{'='*70}")
    print("PYSEGQC: ANALYSIS MODE")
    print(f"{'='*70}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_clustering_results"

    # Detect structures from input file to determine analysis strategy
    df = pd.read_excel(input_path, sheet_name=args.sheet)
    feature_pattern = re.compile(r'^\d{3}_')
    original_feature_cols = [col for col in df.columns if feature_pattern.match(str(col))]

    has_multi_structure = bool(original_feature_cols)

    # Try to load label map from Excel (written by extract command)
    label_map = None
    try:
        label_map_df = pd.read_excel(input_path, sheet_name='Label_Map')
        label_map = dict(zip(
            label_map_df['Label_ID'].astype(int),
            label_map_df['Structure_Name']
        ))
        print(f"\n   Label map loaded: {len(label_map)} structures")
    except Exception:
        pass  # No label map sheet, use numeric positions

    # Determine analysis mode
    if args.mode == 'concat':
        # Concat mode: single analysis with all structures concatenated
        if not has_multi_structure:
            print("\n❌ Error: --mode concat requires multi-structure data (format: 001_feature_name)")
            sys.exit(1)

        print(f"\n{'='*70}")
        print("CONCATENATED MULTI-STRUCTURE ANALYSIS")
        print(f"{'='*70}")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_analysis_pipeline(args, input_path, output_dir)

    elif args.mode == 'position':
        # Position mode: single structure only
        print(f"\n{'='*70}")
        print("SINGLE POSITION ANALYSIS")
        print(f"{'='*70}")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_analysis_pipeline(args, input_path, output_dir)

    elif has_multi_structure:
        # DEFAULT for multi-structure: per-structure independent analysis
        positions, _ = detect_structure_positions(original_feature_cols)

        print(f"\n{'='*70}")
        print("PER-STRUCTURE INDEPENDENT ANALYSIS (DEFAULT)")
        print(f"{'='*70}")
        if label_map:
            print(f"\nDetected {len(positions)} structures:")
            for p in positions:
                name = label_map.get(p, 'Unknown')
                print(f"  {p:03d}: {name}")
        else:
            print(f"\nDetected {len(positions)} structures: {[f'{p:03d}' for p in positions]}")
        print("Each structure will be analyzed independently in separate subfolders")
        print("\nTip: Use --mode concat to analyze all structures together")

        # Create main output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect training selection data from each structure (if requested)
        structure_training_data = {}

        # Loop through each structure
        for pos in positions:
            try:
                # Build display name with label map if available
                struct_name = label_map.get(pos, None) if label_map else None
                display_name = f"{pos:03d}_{struct_name}" if struct_name else f"{pos:03d}"

                print(f"\n{'='*70}")
                print(f"PROCESSING STRUCTURE {display_name}")
                print(f"{'='*70}")

                # Create subfolder for this structure (named if label map available)
                structure_output_dir = output_dir / f"structure_{display_name}_results"
                structure_output_dir.mkdir(parents=True, exist_ok=True)

                # Create modified args for this structure
                modified_args = copy.deepcopy(args)
                modified_args.mode = 'position'
                modified_args.position = pos
                modified_args.structure_name = struct_name
                modified_args.output = str(structure_output_dir)

                # Run analysis for this structure
                training_data = run_analysis_pipeline(modified_args, input_path, structure_output_dir)

                # Collect training selection data if available
                if training_data is not None:
                    structure_training_data[pos] = training_data

            except ValueError as e:
                # Handle insufficient data gracefully
                print(f"\n⚠️  WARNING: Structure {display_name} skipped due to insufficient data")
                print(f"   Reason: {str(e)}")
                print(f"   Continuing with remaining structures...")
                continue

        # Create union summary if training cases were selected
        if structure_training_data and args.select_training_cases:
            _create_training_union(
                structure_training_data,
                output_dir,
                input_path,
                args.sheet
            )

        # Print final summary
        print(f"\n{'='*70}")
        print("✅ ALL STRUCTURES ANALYZED!")
        print(f"{'='*70}")
        print(f"\n📁 Main output directory: {output_dir}")
        print(f"📊 Structures processed: {len(positions)}")
        for pos in positions:
            name = label_map.get(pos, None) if label_map else None
            display = f"{pos:03d}_{name}" if name else f"{pos:03d}"
            print(f"  - structure_{display}_results/")

    else:
        # Single-structure data: standard analysis
        print(f"\n{'='*70}")
        print("STANDARD ANALYSIS (SINGLE STRUCTURE)")
        print(f"{'='*70}")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_analysis_pipeline(args, input_path, output_dir)


def predict_command(args):
    """Execute prediction on new cases (Excel or NIfTI directory)."""
    print(f"\n{'='*70}")
    print("PYSEGQC: PREDICTION MODE")
    print(f"{'='*70}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input not found: {input_path}")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Auto-detect input type: directory = NIfTI, file = Excel
    is_nifti = input_path.is_dir()

    if is_nifti:
        predict_path = _predict_from_nifti(args, input_path, model_dir)
    else:
        predict_path = input_path

    # Detect per-structure model directories
    structure_dirs = sorted(model_dir.glob('structure_*_results'))
    has_per_structure = len(structure_dirs) > 0 and any(
        (d / 'trained_models.pkl').exists() for d in structure_dirs
    )

    if has_per_structure:
        _predict_per_structure(args, predict_path, model_dir, structure_dirs)
    else:
        # Single model prediction (original behavior)
        model_file = model_dir / 'trained_models.pkl'
        if not model_file.exists():
            print(f"❌ Error: No trained_models.pkl found in {model_dir}")
            sys.exit(1)
        try:
            predict_new_cases(predict_path, model_dir, args.sheet, args.mode, args.position)
            print(f"\n✅ Prediction complete!")
        except Exception as e:
            print(f"\n❌ Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def _predict_from_nifti(args, input_path, model_dir):
    """Extract features from NIfTI directory, return path to temp Excel."""
    import tempfile

    print(f"\n   Input type: NIfTI directory")
    print(f"   Image dir: {args.image_dir}")
    print(f"   Mask dir: {args.mask_dir}")

    image_dir_name = getattr(args, 'image_dir', 'images')
    mask_dir_name = getattr(args, 'mask_dir', 'masks')

    pairs = find_image_mask_pairs(
        input_path,
        image_dir=image_dir_name,
        mask_dir=mask_dir_name,
    )

    if not pairs:
        print(f"❌ Error: No image-mask pairs found in {input_path}")
        print(f"   Looked for images in '{image_dir_name}/' and masks in '{mask_dir_name}/'")
        sys.exit(1)

    print(f"   Cases found: {len(pairs)}")

    # Parse label map if provided
    label_map = None
    if getattr(args, 'label_map', None):
        label_map = _parse_label_map(args.label_map)
        print(f"   Label map: {len(label_map)} structures")

    # Extract to temp Excel file in the model directory
    temp_excel = model_dir / '_nifti_prediction_features.xlsx'

    print(f"\n{'='*70}")
    print("EXTRACTING FEATURES FROM NIFTI FILES")
    print(f"{'='*70}")

    create_pca_data_sheet(
        pairs,
        output_excel=str(temp_excel),
        label_map=label_map,
    )

    print(f"\n✓ Features extracted to: {temp_excel}")
    return temp_excel


def _predict_per_structure(args, predict_path, model_dir, structure_dirs):
    """Run prediction for each structure subdirectory and print summary."""
    print(f"\n{'='*70}")
    print("PER-STRUCTURE PREDICTION")
    print(f"{'='*70}")
    print(f"   Found {len(structure_dirs)} structure model directories")

    summary = []  # (display_name, cluster, confidence)

    for struct_dir in structure_dirs:
        model_file = struct_dir / 'trained_models.pkl'
        if not model_file.exists():
            continue

        # Extract position number from directory name (e.g., structure_001_Brainstem_results → 1)
        match = re.search(r'structure_(\d{3})(?:_(.+))?_results', struct_dir.name)
        if not match:
            continue

        pos = int(match.group(1))
        struct_name = match.group(2) or f"{pos:03d}"
        display_name = f"{pos:03d}_{struct_name}" if match.group(2) else f"{pos:03d}"

        print(f"\n{'='*70}")
        print(f"PREDICTING STRUCTURE {display_name}")
        print(f"{'='*70}")

        try:
            predict_new_cases(predict_path, struct_dir, args.sheet, 'position', pos,
                              structure_name=struct_name)
            summary.append((display_name, True, None))
        except Exception as e:
            print(f"  ⚠️  Skipped {display_name}: {str(e)}")
            summary.append((display_name, False, str(e)))

    # Print consolidated summary
    print(f"\n{'='*70}")
    print("✅ PER-STRUCTURE PREDICTION SUMMARY")
    print(f"{'='*70}")
    succeeded = sum(1 for _, ok, _ in summary if ok)
    print(f"   Structures predicted: {succeeded}/{len(summary)}")
    for name, ok, err in summary:
        status = "✓" if ok else f"⚠️  ({err})"
        print(f"   {name}: {status}")
    print(f"\n   Results saved in per-structure subdirectories of: {model_dir}")


def _create_training_union(structure_training_data, output_dir, input_path, sheet_name):
    """
    Create a union summary of training cases selected across all structures.

    Parameters
    ----------
    structure_training_data : dict
        Dictionary mapping structure position to training selection data
    output_dir : Path
        Main output directory for the union file
    input_path : Path
        Path to the original input Excel file
    sheet_name : str
        Name of the sheet containing the PCA data
    """
    print(f"\n{'='*70}")
    print("CREATING TRAINING CASES UNION")
    print(f"{'='*70}")

    # Collect all selected case IDs with cluster info
    all_cases = {}  # case_id -> {structures: [...], cluster_info: [...]}
    for pos, data in structure_training_data.items():
        # Handle both new per-cluster format and legacy format
        if 'per_cluster_selections' in data:
            # NEW FORMAT: Iterate through all clusters
            for cluster_id, cluster_info in data['per_cluster_selections'].items():
                for case_id in cluster_info['selected_case_ids']:
                    if case_id not in all_cases:
                        all_cases[case_id] = {
                            'structures': [],
                            'cluster_ids': [],
                            'cluster_sizes': [],
                            'distances': []
                        }
                    all_cases[case_id]['structures'].append(f"{pos:03d}")

                    # Get cluster info for this case
                    details = cluster_info['case_details'][case_id]
                    all_cases[case_id]['cluster_ids'].append(str(cluster_id))
                    all_cases[case_id]['cluster_sizes'].append(str(cluster_info['cluster_size']))
                    all_cases[case_id]['distances'].append(f"{details['Distance_From_Centroid']:.4f}")
        else:
            # LEGACY FORMAT
            for case_id in data['selected_case_ids']:
                if case_id not in all_cases:
                    all_cases[case_id] = {
                        'structures': [],
                        'cluster_ids': [],
                        'cluster_sizes': [],
                        'distances': []
                    }
                all_cases[case_id]['structures'].append(f"{pos:03d}")

                # Get cluster info for this structure's selection
                details = data['case_details'][case_id]
                all_cases[case_id]['cluster_ids'].append(str(details['cluster_id']))
                all_cases[case_id]['cluster_sizes'].append(str(details['cluster_size']))
                all_cases[case_id]['distances'].append(f"{details['distance']:.4f}")

    # Create union summary
    union_data = []
    for case_id in sorted(all_cases.keys()):
        case_info = all_cases[case_id]
        structures = case_info['structures']

        # Extract metadata from first structure
        first_pos = list(structure_training_data.keys())[0]
        metadata_df = structure_training_data[first_pos].get('metadata_df', None)

        # Build case data
        case_data = {'Case_ID': case_id}

        # Add metadata if available
        if metadata_df is not None and case_id in metadata_df.index:
            for col in ['MRN', 'Plan_ID', 'Session_ID', 'All_Structure_Names']:
                if col in metadata_df.columns:
                    case_data[col if col != 'All_Structure_Names' else 'Structure_Names'] = \
                        metadata_df.loc[case_id, col]

        # Add structure and cluster info
        case_data.update({
            'Structures_Represented': ', '.join(structures),
            'Structure_Count': len(structures),
            'Cluster_IDs': ', '.join(case_info['cluster_ids']),
            'Cluster_Sizes': ', '.join(case_info['cluster_sizes']),
            'Distances': ', '.join(case_info['distances'])
        })

        union_data.append(case_data)

    union_df = pd.DataFrame(union_data)

    # Load original data for selected cases
    print("\n  Loading original data for selected cases...")
    pca_data_df = pd.read_excel(input_path, sheet_name=sheet_name)
    selected_case_indices = sorted(all_cases.keys())
    selected_pca_rows = pca_data_df.iloc[selected_case_indices].copy()

    # Organize cases by cluster (use primary cluster)
    cases_by_cluster = {}
    case_to_cluster = {}
    for case_id, case_info in all_cases.items():
        primary_cluster = case_info['cluster_ids'][0]
        case_to_cluster[case_id] = primary_cluster
        if primary_cluster not in cases_by_cluster:
            cases_by_cluster[primary_cluster] = []
        cases_by_cluster[primary_cluster].append(case_id)

    # Add cluster information to selected_data_df
    if 'Case_ID' not in selected_pca_rows.columns:
        selected_pca_rows.insert(0, 'Case_ID', selected_case_indices)

    selected_pca_rows['Cluster_ID'] = [case_to_cluster.get(idx, 'Unknown')
                                        for idx in selected_case_indices]

    # Add cluster info columns from union_df
    cluster_info_cols = ['Cluster_IDs', 'Cluster_Sizes', 'Distances', 'Structures_Represented']
    for col in cluster_info_cols:
        if col in union_df.columns:
            col_map = dict(zip(union_df['Case_ID'], union_df[col]))
            selected_pca_rows[col] = selected_pca_rows['Case_ID'].map(col_map)

    # Sort by Cluster_ID
    selected_pca_rows = selected_pca_rows.sort_values('Cluster_ID').reset_index(drop=True)

    # Save to Excel with separate data sheets per cluster
    union_path = output_dir / 'training_cases_union.xlsx'
    with pd.ExcelWriter(union_path, engine='openpyxl') as writer:
        # Create summary sheet
        summary_data = []
        for cluster_id in sorted(cases_by_cluster.keys(), key=int):
            cluster_cases = cases_by_cluster[cluster_id]
            summary_data.append({
                'Cluster_ID': cluster_id,
                'Cases_Count': len(cluster_cases),
                'Case_IDs': ', '.join([str(cid) for cid in sorted(cluster_cases)[:10]]) +
                           ('...' if len(cluster_cases) > 10 else '')
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Create separate data sheets per cluster
        for cluster_id in sorted(cases_by_cluster.keys(), key=int):
            cluster_cases = cases_by_cluster[cluster_id]
            cluster_data_df = selected_pca_rows[selected_pca_rows['Case_ID'].isin(cluster_cases)].copy()
            cluster_data_df = cluster_data_df.sort_values('Case_ID').reset_index(drop=True)
            sheet_name = f'Data_Cluster_{cluster_id}'
            cluster_data_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Write 'All_Cases_Info' sheet
        union_df.to_excel(writer, sheet_name='All_Cases_Info', index=False)

    print(f"\n  ✓ Union of training cases across all structures:")
    print(f"    Total unique cases: {len(all_cases)}")
    print(f"    Cases in multiple structures: {sum(1 for info in all_cases.values() if len(info['structures']) > 1)}")
    print(f"    Clusters found: {len(cases_by_cluster)}")
    print(f"    Saved to: {union_path}")
    print(f"      - 'Summary' sheet: Cluster breakdown overview")
    for cluster_id, cluster_cases in sorted(cases_by_cluster.items(), key=lambda x: int(x[0])):
        print(f"      - 'Data_Cluster_{cluster_id}' sheet: {len(cluster_cases)} cases (ready for training)")
    print(f"      - 'All_Cases_Info' sheet: Metadata and cluster info (reference)")

    # Print top multi-structure cases
    multi_structure = [(cid, info['structures']) for cid, info in all_cases.items()
                      if len(info['structures']) > 1]
    if multi_structure:
        print("\n  Multi-structure cases (valuable for training):")
        for case_id, structures in sorted(multi_structure, key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"    Case {case_id}: {len(structures)} structures ({', '.join(structures)})")


def main():
    """Main CLI entry point for pySegQC with subcommands."""
    parser = argparse.ArgumentParser(
        description='pySegQC: Quality control and clustering analysis for medical image segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  extract    Extract radiomics features from NIfTI files
  analyze    Perform PCA + clustering analysis on radiomics data
  predict    Predict clusters for new cases using trained models

For help on a specific subcommand:
  pysegqc extract --help
  pysegqc analyze --help
  pysegqc predict --help

Example Workflow:
  1. Extract features from NIfTI files:
     pysegqc extract /data/ct_scans/ --output features.xlsx

  2. Analyze and cluster:
     pysegqc analyze features.xlsx --auto-k --volume-independent

  3. Predict on new cases:
     pysegqc predict new_cases.xlsx results/ --sheet PCA_Data
        """
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title='Available commands',
        dest='command',
        help='Use "pysegqc <command> --help" for more information'
    )

    # Create subcommand parsers
    create_extract_parser(subparsers)
    create_analyze_parser(subparsers)
    create_predict_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, print help
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    args.func(args)


if __name__ == '__main__':
    main()
