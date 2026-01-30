"""
Data loading and preprocessing for radiomics clustering analysis.

This module handles loading data from Excel files and preprocessing it for analysis.
"""

import re
from pathlib import Path
import pandas as pd
from .utils import detect_structure_positions, extract_view_scan_urls


def load_and_preprocess_data(excel_path, sheet_name='PCA_Data', mode='all', position=None):
    """
    Load PCA_Data sheet and preprocess for analysis.

    Args:
        excel_path: Path to Excel file with metrics
        sheet_name: Name of sheet to load (default: 'PCA_Data')
        mode: Analysis mode:
            - 'all': Use all structures as separate samples
            - 'average': Average features across structures per patient
            - 'position': Use specific structure position only
        position: Structure position to use if mode='position' (e.g., 1 for 001_)

    Returns:
        tuple: (metadata_df, features_df, feature_names, structure_info)
    """
    print(f"\n{'='*70}")
    print(f"Loading data from {Path(excel_path).name}")
    print(f"{'='*70}")

    # Load the PCA_Data sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"Loaded {len(df)} rows (patients) and {len(df.columns)} total columns")

    # Separate metadata columns from feature columns
    # Feature columns: anything starting with position prefix (001_, 002_, etc.)
    feature_pattern = re.compile(r'^\d{3}_')
    feature_cols = [col for col in df.columns if feature_pattern.match(str(col))]

    # Metadata columns: everything else
    metadata_cols = [col for col in df.columns if col not in feature_cols]

    if len(metadata_cols) == 0:
        print(f"\n⚠️  Warning: No metadata columns detected. Adding default Row_Number.")
        df['Row_Number'] = range(1, len(df) + 1)
        metadata_cols = ['Row_Number']

    print(f"\n📋 Detected {len(metadata_cols)} metadata column(s): {metadata_cols[:5]}{'...' if len(metadata_cols) > 5 else ''}")

    # Detect structure positions
    try:
        positions, n_features_per = detect_structure_positions(feature_cols)
        n_structures = len(positions)

        print(f"\n📊 Structure Detection:")
        print(f"   Detected {n_structures} structure positions: {positions}")
        print(f"   Features per structure: {n_features_per}")
        print(f"   Total feature columns: {len(feature_cols)} ({n_structures} × {n_features_per})")

        structure_info = {
            'positions': positions,
            'n_structures': n_structures,
            'n_features_per_structure': n_features_per
        }
    except ValueError as e:
        print(f"\n⚠️  Warning: {e}")
        print("   Treating all columns as single feature set")
        structure_info = {'positions': [1], 'n_structures': 1, 'n_features_per_structure': len(feature_cols)}

    # Extract metadata
    metadata_df = df[metadata_cols].copy()

    # Join with results and Structure_Mapping sheets for database IDs and structure names
    try:
        # Read results sheet for database IDs
        results_df = pd.read_excel(excel_path, sheet_name='results')

        # Extract View Scan URLs from hyperlinks
        view_scan_urls = extract_view_scan_urls(excel_path)

        # Join on Row_Number (PCA_Data) = index (results)
        if 'Row_Number' in df.columns and 'index' in results_df.columns:
            # Merge to get database IDs
            db_info_cols = ['index', 'plan_c_id', 'session_id', 'contours_found']

            # Add view_scan URL if available
            if view_scan_urls is not None:
                results_df['View_Scan_URL'] = results_df['index'].map(view_scan_urls)
                db_info_cols.append('View_Scan_URL')

            db_info = results_df[db_info_cols].copy()
            metadata_with_ids = metadata_df.merge(db_info, left_on='Row_Number', right_on='index', how='left')

            # Add database ID columns to metadata_df
            metadata_df['Plan_ID'] = metadata_with_ids['plan_c_id']
            metadata_df['Session_ID'] = metadata_with_ids['session_id']
            metadata_df['All_Structure_Names'] = metadata_with_ids['contours_found']

            # Add View Scan URL if available
            if view_scan_urls is not None:
                metadata_df['View_Scan_URL'] = metadata_with_ids['View_Scan_URL']
                print(f"\n✓ Joined with 'results' sheet - added database IDs and View Scan URLs")
            else:
                print(f"\n✓ Joined with 'results' sheet - added database IDs")

            # If analyzing specific position, get structure name from Structure_Mapping
            if mode == 'position' and position is not None:
                try:
                    mapping_df = pd.read_excel(excel_path, sheet_name='Structure_Mapping')
                    pos_col_name = f"{position:03d}_name"

                    if 'Row_#' in mapping_df.columns and pos_col_name in mapping_df.columns:
                        # Merge to get structure name for this position
                        struct_mapping = mapping_df[['Row_#', pos_col_name]].copy()
                        metadata_with_struct = metadata_df.merge(struct_mapping, left_on='Row_Number', right_on='Row_#', how='left')
                        metadata_df['Structure_Name'] = metadata_with_struct[pos_col_name]
                        print(f"✓ Joined with 'Structure_Mapping' sheet - added structure name for position {position:03d}")
                    else:
                        print(f"⚠️  Warning: Expected columns not found in Structure_Mapping sheet")
                        metadata_df['Structure_Name'] = None
                except Exception as e:
                    print(f"⚠️  Warning: Could not read Structure_Mapping sheet: {e}")
                    metadata_df['Structure_Name'] = None
        else:
            print(f"⚠️  Warning: Required columns (Row_Number, index) not found for joining")
            metadata_df['Plan_ID'] = None
            metadata_df['Session_ID'] = None
            metadata_df['All_Structure_Names'] = None
            if mode == 'position':
                metadata_df['Structure_Name'] = None
    except Exception as e:
        print(f"⚠️  Warning: Could not read 'results' sheet: {e}")
        print("   Continuing without database IDs")
        metadata_df['Plan_ID'] = None
        metadata_df['Session_ID'] = None
        metadata_df['All_Structure_Names'] = None
        if mode == 'position':
            metadata_df['Structure_Name'] = None

    # Process features based on mode
    if mode == 'concat' and structure_info['n_structures'] > 1:
        print(f"\n🔗 Analysis Mode: CONCATENATED")
        print(f"   Using all {len(feature_cols)} features ({n_structures} structures concatenated)")
        print(f"   Samples: {len(df)} patients")

        # Use all feature columns with their position prefixes
        features_df = df[feature_cols].copy()
        feature_names = feature_cols.copy()

    elif mode == 'position' and position is not None:
        print(f"\n📍 Analysis Mode: SINGLE POSITION")
        print(f"   Using only structure position {position:03d}")

        # Extract features for specified position only
        pos_prefix = f"{position:03d}_"
        position_cols = [col for col in feature_cols if col.startswith(pos_prefix)]

        features_df = df[position_cols].copy()
        # Remove position prefix from column names for clarity
        features_df.columns = [col[4:] for col in features_df.columns]
        feature_names = list(features_df.columns)

    else:
        # Default: use all features as-is (for single-structure data or when no mode specified)
        print(f"\n📊 Analysis Mode: DEFAULT")
        print(f"   Using {len(feature_cols)} feature columns as-is")

        features_df = df[feature_cols].copy()
        feature_names = feature_cols.copy()

    print(f"\n📈 Final Dataset:")
    print(f"   Samples: {len(features_df)}")
    print(f"   Features: {len(features_df.columns)}")

    # Report missing data
    missing_pct = (features_df.isnull().sum() / len(features_df) * 100)
    cols_with_missing = missing_pct[missing_pct > 0]

    if len(cols_with_missing) > 0:
        print(f"\n⚠️  Missing Data:")
        print(f"   Columns with missing: {len(cols_with_missing)}/{len(features_df.columns)}")
        print(f"   Average missing: {missing_pct.mean():.1f}%")
        print(f"   Max missing: {missing_pct.max():.1f}%")

        # Remove columns with >80% missing
        cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
        if cols_to_drop:
            print(f"   Dropping {len(cols_to_drop)} columns with >80% missing")
            features_df = features_df.drop(columns=cols_to_drop)
            feature_names = [f for f in feature_names if f not in cols_to_drop]

            # Validate that features remain after dropping
            if len(features_df.columns) == 0:
                raise ValueError(
                    "Insufficient data: All features dropped due to >80% missing values. "
                    "Please check your input data quality."
                )
    else:
        print("\n✓ No missing data found")

    return metadata_df, features_df, feature_names, structure_info
