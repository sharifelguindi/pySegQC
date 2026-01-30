"""
Prediction on new cases using trained clustering models.

This module provides functionality for applying trained PCA and clustering
models to new cases and formatting prediction results.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .data_loader import load_and_preprocess_data
from .validation import impute_missing_values
from .export import generate_cluster_summary, apply_cluster_conditional_formatting
from .qa import compute_prediction_qa_verdicts
from .report import generate_json_report, generate_prediction_dashboard
from .thumbnail import generate_thumbnails_batch
from .visualization import plot_prediction_with_training

logger = logging.getLogger(__name__)


def predict_new_cases(predict_path, model_dir, sheet_name='PCA_Data', mode='all', position=None,
                      structure_name=None):
    """
    Predict cluster assignments for new cases using trained models.

    Args:
        predict_path: Path to Excel file with new cases
        model_dir: Directory containing trained_models.pkl
        sheet_name: Sheet name in Excel file
        mode: Data mode ('all', 'average', 'position')
        position: Structure position to use if mode='position' (e.g., 1 for 001_)

    Returns:
        None (saves results to Excel)
    """
    print(f"\n{'='*70}")
    print("PREDICTING CLUSTER ASSIGNMENTS FOR NEW CASES")
    print(f"{'='*70}")

    # Load trained models
    model_path = Path(model_dir) / 'trained_models.pkl'
    if not model_path.exists():
        print(f"❌ ERROR: Trained models not found at {model_path}")
        print(f"   Please run training first to generate trained_models.pkl")
        sys.exit(1)

    print(f"\nLoading trained models from: {model_path}")
    trained_models = joblib.load(model_path)

    scaler = trained_models['scaler']
    pca_model = trained_models['pca']
    centroids = trained_models['centroids']
    feature_names = trained_models['feature_names']
    n_clusters = trained_models['n_clusters']
    impute_strategy = trained_models['impute_strategy']
    n_components = trained_models['n_components']

    print(f"✓ Loaded models:")
    print(f"  - {n_clusters} cluster centroids")
    print(f"  - PCA model ({n_components} components)")
    print(f"  - {len(feature_names)} features")

    # Load new cases
    print(f"\n{'='*70}")
    print(f"Loading new cases from {predict_path}")
    print(f"{'='*70}")

    predict_path = Path(predict_path)
    metadata_df, features_df, new_feature_names, structure_info = load_and_preprocess_data(
        predict_path, sheet_name, mode, position=position
    )

    # Validate features match
    if set(new_feature_names) != set(feature_names):
        missing = set(feature_names) - set(new_feature_names)
        extra = set(new_feature_names) - set(feature_names)
        print(f"\n❌ ERROR: Feature mismatch!")
        if missing:
            print(f"  Missing features: {missing}")
        if extra:
            print(f"  Extra features: {extra}")
        sys.exit(1)

    # Reorder features to match training
    features_df = features_df[feature_names]
    print(f"✓ Features validated ({len(feature_names)} features)")

    # Preprocess new data (same pipeline as training)
    print(f"\n{'='*70}")
    print("Preprocessing new cases")
    print(f"{'='*70}")

    # Impute
    features_imputed = impute_missing_values(features_df, strategy=impute_strategy)

    # Standardize using saved scaler
    print(f"\nStandardizing features using trained scaler")
    features_standardized = scaler.transform(features_imputed)
    features_standardized = pd.DataFrame(
        features_standardized,
        columns=features_df.columns,
        index=features_df.index
    )
    print("✓ Features standardized")

    # PCA transform
    print(f"\nTransforming to PCA space ({n_components} components)")
    pca_data = pca_model.transform(features_standardized)
    print("✓ PCA transformation complete")

    # Predict clusters using nearest centroid
    print(f"\n{'='*70}")
    print(f"Assigning to {n_clusters} clusters")
    print(f"{'='*70}")

    # Calculate distances to all centroids
    distances = np.linalg.norm(
        pca_data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )

    # Assign to nearest cluster
    predicted_labels = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)

    # Calculate confidence score (ratio of nearest to 2nd nearest distance)
    sorted_distances = np.sort(distances, axis=1)
    confidence_scores = sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-10)

    # Print cluster assignment summary
    print(f"\nPrediction Results:")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} cases ({count/len(predicted_labels)*100:.1f}%)")

    avg_confidence = confidence_scores.mean()
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"  (Lower is better, <0.5 = high confidence)")

    # Create output directory in training folder
    output_dir = Path(model_dir) / f"{predict_path.stem}_predictions"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load training metadata early for deduplication check
    print(f"\n{'='*70}")
    print("LOADING TRAINING METADATA FOR DEDUPLICATION CHECK")
    print(f"{'='*70}")

    training_metadata_for_dedup = None
    clustered_files = list(Path(model_dir).glob('*_clustered.xlsx'))
    if not clustered_files:
        clustered_files = list(Path(model_dir).parent.glob('*_clustered.xlsx'))

    if clustered_files:
        try:
            with pd.ExcelFile(clustered_files[0], engine='openpyxl') as xlsx:
                clustered_df_dedup = pd.read_excel(xlsx, sheet_name='Clustered_Data')
            metadata_cols = ['Plan_ID', 'Session_ID', 'MRN']
            available_cols = [col for col in metadata_cols if col in clustered_df_dedup.columns]
            if available_cols:
                training_metadata_for_dedup = clustered_df_dedup[available_cols].copy()
                print(f"  Loaded training metadata: {len(training_metadata_for_dedup)} cases")
        except Exception as e:
            print(f"  ⚠️  Could not load training metadata: {e}")

    # Check for and remove duplicate cases
    if training_metadata_for_dedup is not None and 'Plan_ID' in training_metadata_for_dedup.columns and 'Session_ID' in training_metadata_for_dedup.columns and 'Plan_ID' in metadata_df.columns and 'Session_ID' in metadata_df.columns:
        print(f"\n  Checking for duplicates using Plan_ID + Session_ID...")

        # Create composite keys
        training_keys = set(zip(training_metadata_for_dedup['Plan_ID'].astype(str),
                                training_metadata_for_dedup['Session_ID'].astype(str)))
        prediction_keys = list(zip(metadata_df['Plan_ID'].astype(str),
                                   metadata_df['Session_ID'].astype(str)))

        # Find duplicates
        is_duplicate = np.array([key in training_keys for key in prediction_keys])
        n_duplicates = is_duplicate.sum()

        if n_duplicates > 0:
            print(f"  ⚠️  Found {n_duplicates} duplicate cases (already in training data)")

            # Filter out duplicates
            keep_mask = ~is_duplicate

            metadata_df = metadata_df[keep_mask].reset_index(drop=True)
            pca_data = pca_data[keep_mask]
            predicted_labels = predicted_labels[keep_mask]
            confidence_scores = confidence_scores[keep_mask]
            min_distances = min_distances[keep_mask]
            features_df = features_df[keep_mask].reset_index(drop=True)

            print(f"  ✓ Filtered to {len(metadata_df)} unique new cases")
        else:
            print(f"  ✓ No duplicates found - all {len(metadata_df)} cases are new")
    else:
        print(f"  ⚠️  Cannot check for duplicates (Plan_ID/Session_ID not available)")
        print(f"  Proceeding with all {len(metadata_df)} prediction cases")

    # Create results dataframe (same format as training)
    print(f"\n{'='*70}")
    print("Creating prediction results")
    print(f"{'='*70}")

    results_df = metadata_df.copy()
    results_df['Predicted_Cluster'] = predicted_labels
    results_df['Confidence_Score'] = confidence_scores
    results_df['Distance_to_Centroid'] = min_distances

    # Add PC scores
    for i in range(min(10, n_components)):
        results_df[f'PC{i+1}'] = pca_data[:, i]

    # Add original features
    results_df = pd.concat([results_df, features_df], axis=1)

    # Sort by predicted cluster
    results_df = results_df.sort_values('Predicted_Cluster').reset_index(drop=True)

    # Prepare Excel file with multiple sheets
    output_excel = output_dir / f"{predict_path.stem}_predictions.xlsx"

    # Create Excel writer
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 1. Create Summary Sheet
        print("  Creating summary sheet...")
        summary_df = generate_cluster_summary(
            results_df,
            feature_names=feature_names,
            cluster_column='Predicted_Cluster',
            n_top_features=10,
            include_confidence=True
        )
        # Add Avg_Distance column manually (not in standard excel_utils)
        for cluster in range(n_clusters):
            mask = results_df['Predicted_Cluster'] == cluster
            summary_df.loc[cluster, 'Avg_Distance'] = f"{results_df[mask]['Distance_to_Centroid'].mean():.3f}"

        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 2. Write main predictions sheet
        print("  Writing predictions sheet...")
        results_df.to_excel(writer, sheet_name='Predictions', index=False)

        # 3. Write separate sheets per cluster
        print("  Creating per-cluster sheets...")
        for cluster in range(n_clusters):
            cluster_data = results_df[results_df['Predicted_Cluster'] == cluster].copy()
            if len(cluster_data) > 0:
                sheet_name = f'Cluster_{cluster}'
                cluster_data.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"    ✓ Sheet '{sheet_name}': {len(cluster_data)} cases")

        # 4. Create AI Segmentation Sample sheet (~25 cases, proportional to cluster size)
        print("  Creating AI segmentation sample sheet...")
        target_total = 25
        total_cases = len(results_df)

        # Calculate proportional samples per cluster
        sample_selection = []
        for cluster in range(n_clusters):
            cluster_mask = results_df['Predicted_Cluster'] == cluster
            cluster_size = cluster_mask.sum()

            if cluster_size > 0:
                # Proportional allocation
                n_samples = max(1, round(target_total * (cluster_size / total_cases)))

                # Get indices for this cluster
                cluster_indices = results_df[cluster_mask].index.tolist()

                # Randomly sample (or take all if cluster smaller than sample size)
                n_to_sample = min(n_samples, len(cluster_indices))
                sampled_indices = np.random.choice(cluster_indices, size=n_to_sample, replace=False)

                sample_selection.extend(sampled_indices)
                print(f"    Cluster {cluster}: selected {n_to_sample}/{cluster_size} cases")

        # Ensure we have exactly 25 (or close to it)
        if len(sample_selection) > target_total:
            sample_selection = np.random.choice(sample_selection, size=target_total, replace=False)
        elif len(sample_selection) < target_total and len(sample_selection) < total_cases:
            # Add a few more random cases if we're short
            remaining_indices = list(set(results_df.index) - set(sample_selection))
            n_needed = min(target_total - len(sample_selection), len(remaining_indices))
            if n_needed > 0:
                additional = np.random.choice(remaining_indices, size=n_needed, replace=False)
                sample_selection = np.concatenate([sample_selection, additional])

        # Create sample dataframe
        sample_df = results_df.loc[sample_selection].copy()

        # Select relevant columns for AI segmentation
        seg_columns = ['Plan_ID', 'Session_ID', 'Predicted_Cluster']

        # Add MRN if available
        if 'MRN' in sample_df.columns:
            seg_columns.insert(0, 'MRN')

        # Add structure name (actual structure name like "CTV", "Bladder")
        if 'Structure_Name' in sample_df.columns:
            seg_columns.insert(len(seg_columns) - 1, 'Structure_Name')  # Insert before Predicted_Cluster

        # Add structure position
        if mode == 'position' and position is not None:
            sample_df['Structure'] = f"{position:03d}"
            seg_columns.insert(len(seg_columns) - 1, 'Structure')  # Insert before Predicted_Cluster

        seg_sample_df = sample_df[seg_columns].reset_index(drop=True)

        # Write to Excel
        seg_sample_df.to_excel(writer, sheet_name='AI_Segmentation_Sample', index=False)
        print(f"    ✓ Sheet 'AI_Segmentation_Sample': {len(seg_sample_df)} cases selected")

    # Apply conditional formatting (using 'Predicted_Cluster' column)
    print("  Applying conditional formatting...")
    _apply_prediction_formatting(output_excel, n_clusters)

    print(f"✓ Saved predictions: {output_excel}")
    print(f"  - Summary sheet with prediction statistics")
    print(f"  - Main data sorted by cluster with color coding")
    print(f"  - {n_clusters} separate sheets (one per cluster)")

    # Generate visualization with training context (if training data available)
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION WITH TRAINING CONTEXT")
    print(f"{'='*70}")

    training_pca_data = trained_models.get('training_pca_data')
    training_labels = trained_models.get('training_labels')
    training_metadata = trained_models.get('training_metadata')

    # Load training data from clustered Excel file (primary method)
    if training_pca_data is None or training_labels is None:
        print(f"  Loading training data from clustered Excel file...")

        # Look for *_clustered.xlsx in model directory and parent directory
        clustered_files = list(Path(model_dir).glob('*_clustered.xlsx'))

        # If not found in model_dir, check parent directory (for per-structure analysis)
        if not clustered_files:
            clustered_files = list(Path(model_dir).parent.glob('*_clustered.xlsx'))

        if clustered_files:
            clustered_file = clustered_files[0]
            print(f"  Found: {clustered_file.name}")

            try:
                # Load the Clustered_Data sheet (use context manager to avoid resource warnings)
                with pd.ExcelFile(clustered_file, engine='openpyxl') as xlsx:
                    clustered_df = pd.read_excel(xlsx, sheet_name='Clustered_Data')
                print(f"  Loaded {len(clustered_df)} training samples")

                # Extract cluster labels
                if 'Cluster' in clustered_df.columns:
                    training_labels = clustered_df['Cluster'].values

                    # Extract PCA data (PC1, PC2, PC3, etc.)
                    pc_cols = [col for col in clustered_df.columns if col.startswith('PC') and col[2:].isdigit()]
                    pc_cols_sorted = sorted(pc_cols, key=lambda x: int(x[2:]))

                    if pc_cols_sorted:
                        training_pca_data = clustered_df[pc_cols_sorted].values
                        print(f"  Extracted {training_pca_data.shape[1]} PC components")

                        # Extract metadata (MRN if available)
                        metadata_cols = ['MRN', 'Plan_ID', 'Session_ID', 'Patient_Name', 'View_Scan_URL']
                        available_metadata_cols = [col for col in metadata_cols if col in clustered_df.columns]
                        if available_metadata_cols:
                            training_metadata = clustered_df[available_metadata_cols].copy()
                        else:
                            training_metadata = None

                        print(f"  ✓ Successfully loaded training data from Excel file")
                    else:
                        print(f"  ⚠️  No PC columns found in clustered file")
                else:
                    print(f"  ⚠️  No 'Cluster' column found in clustered file")
            except Exception as e:
                print(f"  ⚠️  Error loading clustered file: {e}")
                training_pca_data = None
                training_labels = None
        else:
            print(f"  ⚠️  No clustered Excel file found in {model_dir}")

    prediction_figures = {}
    if training_pca_data is not None and training_labels is not None:
        print(f"\n  Generating visualization with {len(training_pca_data)} training samples...")
        prediction_figures = plot_prediction_with_training(
            training_pca_data=training_pca_data,
            training_labels=training_labels,
            training_metadata=training_metadata,
            prediction_pca_data=pca_data,
            prediction_labels=predicted_labels,
            prediction_metadata=metadata_df,
            n_clusters=n_clusters,
            output_dir=output_dir,
            confidence_scores=confidence_scores
        ) or {}
    else:
        print(f"  ⚠️  Could not load training data")
        print(f"  Skipping visualization with training context")

    # ─── QA verdict computation ───
    print(f"\n{'='*70}")
    print("COMPUTING QA VERDICTS FOR NEW CASES")
    print(f"{'='*70}")

    qa_thresholds = trained_models.get('qa_thresholds', {})
    training_cluster_stats = qa_thresholds.get('per_cluster_stats', {})
    qa_sigma = qa_thresholds.get('distance_sigma', 2.0)

    qa_results = compute_prediction_qa_verdicts(
        pca_data, predicted_labels, centroids,
        training_cluster_stats=training_cluster_stats,
        distance_sigma=qa_sigma,
    )

    n_pass = (qa_results['verdicts'] == 'pass').sum()
    n_review = (qa_results['verdicts'] == 'review').sum()
    n_fail = (qa_results['verdicts'] == 'fail').sum()
    print(f"  QA verdicts: {n_pass} pass, {n_review} review, {n_fail} fail")

    # ─── Generate JSON report for prediction ───
    analysis_config = {
        'method': 'prediction',
        'mode': mode or 'default',
        'volume_independent': False,
        'n_clusters': n_clusters,
        'silhouette': 0.0,
        'stability': 0.0,
        'robustness': 0.0,
        'distance_sigma': qa_sigma,
    }

    generate_json_report(
        output_path=output_dir / 'prediction_report.json',
        metadata_df=metadata_df,
        pca_data=pca_data,
        cluster_labels=predicted_labels,
        centroids=centroids,
        qa_results=qa_results,
        analysis_config=analysis_config,
        prediction_info={
            'model_path': str(model_path),
            'new_cases': len(pca_data),
            'avg_confidence': float(avg_confidence),
        },
    )
    print(f"  ✓ Saved prediction report: {output_dir / 'prediction_report.json'}")

    # ─── Generate thumbnails for prediction cases ───
    thumbnails = {}
    try:
        thumbnails = generate_thumbnails_batch(metadata_df)
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")

    # ─── NiiVue viewer (if NIfTI paths available) ───
    has_viewer = False
    if 'Image_Path' in metadata_df.columns:
        try:
            from .viewer import generate_viewer_data, generate_viewer_html
            viewer_data = generate_viewer_data(
                output_dir / 'viewer_data.json',
                metadata_df, predicted_labels, qa_results,
                structure_label=position if mode == 'position' else None,
                structure_name=structure_name,
            )
            generate_viewer_html(output_dir / 'viewer.html', viewer_data=viewer_data)
            has_viewer = True
        except Exception as e:
            logger.warning(f"Viewer generation failed: {e}")

    # ─── Generate prediction HTML dashboard ───
    if prediction_figures:
        generate_prediction_dashboard(
            output_path=output_dir / 'prediction_dashboard.html',
            metadata_df=metadata_df,
            predicted_labels=predicted_labels,
            qa_results=qa_results,
            figures=prediction_figures,
            prediction_config={
                'n_clusters': n_clusters,
                'avg_confidence': float(avg_confidence),
                'model_path': str(model_path),
                'n_features': len(feature_names),
                'mode': mode or 'default',
            },
            thumbnails=thumbnails or None,
            has_viewer=has_viewer,
        )
        print(f"  ✓ Saved prediction dashboard: {output_dir / 'prediction_dashboard.html'}")

    print(f"\n{'='*70}")
    print("✅ PREDICTION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results: {output_dir}")
    print(f"📊 Cases analyzed: {len(results_df)}")
    print(f"📊 Clusters: {n_clusters}")
    print(f"📊 Average confidence: {avg_confidence:.3f}")
    print(f"📊 QA: {n_pass} pass / {n_review} review / {n_fail} fail")
    if prediction_figures:
        print(f"\n🌐 Open prediction_dashboard.html in your browser for interactive results!")


def _apply_prediction_formatting(excel_path, n_clusters):
    """Apply color coding to prediction results sheets."""
    # Use excel_utils for consistent formatting
    apply_cluster_conditional_formatting(
        excel_path,
        n_clusters=n_clusters,
        main_sheet_name='Predictions',
        cluster_column_name='Predicted_Cluster'
    )
