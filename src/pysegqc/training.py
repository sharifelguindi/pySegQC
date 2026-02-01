"""
Training case selection logic for radiomics clustering.

This module implements intelligent training case selection algorithms based on
clustering results, including coverage, representativeness, and redundancy metrics.
"""

import logging
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_loader import load_and_preprocess_data
from .utils import detect_structure_positions, extract_case_metadata

logger = logging.getLogger(__name__)


def calculate_selection_coverage(selected_indices, pca_data, labels, n_clusters):
    """
    Calculate how well selected training cases cover the feature space.

    Measures diversity and representativeness of selected cases.

    Args:
        selected_indices: Indices of selected training cases
        pca_data: Full PCA-transformed data
        labels: Cluster labels
        n_clusters: Number of clusters

    Returns:
        dict: {
            'coverage_score': overall coverage metric (0-1),
            'cluster_coverage': coverage per cluster,
            'feature_space_coverage': proportion of feature space covered,
            'diversity_score': mean pairwise distance between selected cases
        }
    """
    logger.info("Calculating selection coverage metrics...")

    selected_pca = pca_data[selected_indices]
    selected_labels = labels[selected_indices]

    # Cluster coverage: proportion of clusters represented
    selected_clusters = set(selected_labels)
    cluster_coverage = len(selected_clusters) / n_clusters

    # Per-cluster coverage: how well each cluster is represented
    cluster_coverage_scores = {}
    for k in range(n_clusters):
        cluster_mask = labels == k
        cluster_points = pca_data[cluster_mask]

        selected_cluster_mask = selected_labels == k
        if selected_cluster_mask.any():
            selected_cluster_points = selected_pca[selected_cluster_mask]

            # Calculate coverage as: selected cases span cluster's variance
            cluster_variance = np.var(cluster_points, axis=0).sum()
            selected_variance = np.var(selected_cluster_points, axis=0).sum()
            coverage = min(selected_variance / (cluster_variance + 1e-10), 1.0)
            cluster_coverage_scores[k] = coverage
        else:
            cluster_coverage_scores[k] = 0.0

    # Diversity score: mean pairwise distance between selected cases
    if len(selected_indices) > 1:
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(selected_pca, metric='euclidean')
        diversity_score = pairwise_dists.mean()
    else:
        diversity_score = 0.0

    # Feature space coverage: volume of convex hull relative to full data
    # (simplified: ratio of selected to full data range across PCs)
    full_ranges = pca_data.max(axis=0) - pca_data.min(axis=0)
    selected_ranges = selected_pca.max(axis=0) - selected_pca.min(axis=0)
    feature_space_coverage = (selected_ranges / (full_ranges + 1e-10)).mean()

    # Overall coverage score (weighted average)
    coverage_score = (
        0.3 * cluster_coverage +
        0.3 * np.mean(list(cluster_coverage_scores.values())) +
        0.2 * feature_space_coverage +
        0.2 * min(diversity_score / 10, 1.0)  # Normalize diversity
    )

    logger.info(f"Coverage score: {coverage_score:.3f}")

    return {
        'coverage_score': coverage_score,
        'cluster_coverage': cluster_coverage_scores,
        'feature_space_coverage': feature_space_coverage,
        'diversity_score': diversity_score
    }


def calculate_representativeness_scores(selected_indices, pca_data, labels):
    """
    Calculate representativeness score for each selected case.

    A case is representative if it's close to its cluster centroid and
    in a dense region (many neighbors).

    Args:
        selected_indices: Indices of selected training cases
        pca_data: Full PCA-transformed data
        labels: Cluster labels

    Returns:
        dict: {case_id: {'centroid_distance': float, 'density': float, 'representativeness': float}}
    """
    logger.info("Calculating representativeness scores...")

    representativeness = {}

    for idx in selected_indices:
        cluster_id = labels[idx]
        cluster_mask = labels == cluster_id
        cluster_points = pca_data[cluster_mask]

        # Distance to cluster centroid
        centroid = cluster_points.mean(axis=0)
        centroid_distance = euclidean(pca_data[idx], centroid)

        # Normalize by cluster size (larger clusters have larger distances)
        max_distance = np.max([euclidean(p, centroid) for p in cluster_points])
        normalized_distance = centroid_distance / (max_distance + 1e-10)

        # Density: number of neighbors within threshold distance
        threshold = np.percentile(
            [euclidean(p, pca_data[idx]) for p in cluster_points],
            25  # 25th percentile = close neighbors
        )
        density = np.sum([euclidean(p, pca_data[idx]) < threshold for p in cluster_points])
        normalized_density = density / len(cluster_points)

        # Representativeness: high density, low distance from centroid
        representativeness_score = (
            0.6 * (1 - normalized_distance) +  # Closer to centroid is better
            0.4 * normalized_density            # Higher density is better
        )

        representativeness[idx] = {
            'centroid_distance': centroid_distance,
            'normalized_distance': normalized_distance,
            'density': density,
            'normalized_density': normalized_density,
            'representativeness': representativeness_score
        }

    logger.info(f"Mean representativeness: {np.mean([r['representativeness'] for r in representativeness.values()]):.3f}")

    return representativeness


def calculate_redundancy_score(selected_indices, pca_data):
    """
    Calculate redundancy among selected training cases.

    High redundancy means selected cases are too similar to each other.

    Args:
        selected_indices: Indices of selected training cases
        pca_data: Full PCA-transformed data

    Returns:
        dict: {
            'redundancy_score': overall redundancy (0-1, lower is better),
            'min_distance': minimum pairwise distance,
            'mean_distance': mean pairwise distance,
            'pairwise_distances': matrix of distances
        }
    """
    logger.info("Calculating redundancy score...")

    if len(selected_indices) < 2:
        return {
            'redundancy_score': 0.0,
            'min_distance': 0.0,
            'mean_distance': 0.0,
            'pairwise_distances': np.array([[0.0]])
        }

    selected_pca = pca_data[selected_indices]

    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    pairwise_dists = squareform(pdist(selected_pca, metric='euclidean'))

    # Redundancy: high if cases are very close to each other
    # Use inverse of mean pairwise distance
    mean_distance = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)].mean()
    min_distance = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)].min()

    # Normalize by typical distance in full dataset
    full_distances = pdist(pca_data, metric='euclidean')
    typical_distance = np.median(full_distances)

    # Redundancy score: 0 if well-separated, 1 if very close
    redundancy_score = 1 - min(mean_distance / (typical_distance + 1e-10), 1.0)

    logger.info(f"Redundancy score: {redundancy_score:.3f} (lower is better)")

    return {
        'redundancy_score': redundancy_score,
        'min_distance': min_distance,
        'mean_distance': mean_distance,
        'typical_distance': typical_distance,
        'pairwise_distances': pairwise_dists
    }


def select_training_cases_from_clustering(cluster_labels, pca_data, metadata_df, max_clusters=3, max_cases=10):
    """
    Select core training cases from each cluster separately (for concat mode).

    Creates separate training selections for EACH cluster using core-member strategy.
    User can then pick which cluster's training set to use for AI model training.

    Args:
        cluster_labels: Cluster assignments for all samples
        pca_data: PCA-transformed data (N x n_components)
        metadata_df: Metadata dataframe with sample info
        max_clusters: DEPRECATED - kept for backward compatibility (now selects from ALL clusters)
        max_cases: DEPRECATED - kept for backward compatibility

    Returns:
        dict: {
            'per_cluster_selections': dict mapping cluster_id to {
                'selected_case_ids': list of case IDs,
                'case_details': dict mapping case_id to metadata,
                'cluster_size': int,
                'selection_strategy': str
            },
            'metadata_df': metadata dataframe
        }
    """
    print(f"\n{'='*70}")
    print("PER-CLUSTER TRAINING CASE SELECTION")
    print(f"{'='*70}")

    n_samples = len(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    print(f"\n  Total samples: {n_samples}")
    print(f"  Clusters found: {n_clusters}")
    print(f"\n  Strategy: Creating separate training selections for EACH cluster")
    print(f"  Selection: Core members only (closest to centroid)")
    print(f"  Output: One training set per cluster - pick the one you want to use!")

    # Process ALL clusters (not just max_clusters)
    per_cluster_selections = {}

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        print(f"\n  Cluster {cluster_id} ({cluster_size} samples):")

        # Get cluster data in PCA space
        cluster_pca = pca_data[cluster_mask]

        # Compute centroid in PCA space
        centroid = cluster_pca.mean(axis=0)

        # Calculate distances from centroid
        distances = np.linalg.norm(cluster_pca - centroid, axis=1)

        # Create dataframe for easy selection
        distance_df = pd.DataFrame({
            'sample_idx': cluster_indices,
            'distance': distances
        })

        # Core-member selection strategy based on cluster size
        if cluster_size <= 5:
            # Small clusters: Select all
            n_select = cluster_size
            selected_idx = cluster_indices.tolist()
            strategy = f"All {n_select} cases (small cluster)"
        elif cluster_size <= 15:
            # Medium clusters: Select 5 closest to centroid
            n_select = 5
            selected_idx = distance_df.nsmallest(n_select, 'distance')['sample_idx'].tolist()
            strategy = f"{n_select} core members (closest to centroid)"
        else:
            # Large clusters: Select 10-15 closest to centroid
            n_select = min(15, cluster_size)
            selected_idx = distance_df.nsmallest(n_select, 'distance')['sample_idx'].tolist()
            strategy = f"{n_select} core members (closest to centroid)"

        print(f"    Strategy: {strategy}")

        # Build case details for this cluster
        case_details = {}
        for idx in selected_idx:
            dist = distance_df[distance_df['sample_idx'] == idx]['distance'].values[0]

            case_metadata = extract_case_metadata(idx, metadata_df)

            case_metadata['Distance_From_Centroid'] = float(dist)
            case_details[idx] = case_metadata

        # Store this cluster's selection
        per_cluster_selections[int(cluster_id)] = {
            'selected_case_ids': selected_idx,
            'case_details': case_details,
            'cluster_size': cluster_size,
            'selection_strategy': strategy,
            'n_selected': len(selected_idx)
        }

    print(f"\n{'='*70}")
    print(f"✅ SELECTION COMPLETE")
    print(f"{'='*70}")

    total_selected = sum(s['n_selected'] for s in per_cluster_selections.values())
    print(f"\n  Summary:")
    print(f"    Clusters analyzed: {n_clusters}")
    print(f"    Total cases across all clusters: {total_selected}")

    for cluster_id, info in sorted(per_cluster_selections.items()):
        print(f"    Cluster {cluster_id}: {info['n_selected']}/{info['cluster_size']} cases selected")

    print(f"\n  Next step: Review per-cluster sheets and pick which cluster to use for training!")

    return {
        'per_cluster_selections': per_cluster_selections,
        'metadata_df': metadata_df,
        'mode': 'per_cluster'  # Flag for Excel export to use new format
    }


def select_multi_structure_training_cases(excel_path, sheet_name='PCA_Data', impute_strategy='median',
                                          mode='all', position=None):
    """
    Analyze each structure independently and select representative training cases.

    For each structure (001_, 002_, etc.):
    - Perform hierarchical clustering
    - Find optimal k using silhouette score
    - Select edge + center cases from each cluster:
        - Size ≤2: select all
        - Size 3-9: select 2 edge + 1 center (3 total)
        - Size ≥10: select 4 edge + 1 center (5 total)

    Returns union of selected cases across all structures with metadata.

    Args:
        excel_path: Path to Excel file with original data
        sheet_name: Sheet name to read from
        impute_strategy: Strategy for missing value imputation
        mode: Analysis mode ('all', 'average', 'position')
        position: Structure position to use if mode='position'

    Returns:
        dict: {
            'selected_case_ids': list of case IDs selected for training,
            'case_details': dict mapping case_id to {structures, count, type},
            'structure_details': dict mapping structure to {clusters, silhouette, cases, case_ids},
            'cluster_mapping': list of dicts with keys [Case_ID, Structure, Cluster_ID, Cluster_Size, Distance_From_Centroid]
        }
    """
    print(f"\n{'='*70}")
    print("MULTI-STRUCTURE TRAINING CASE SELECTION")
    print(f"{'='*70}")

    # Load original data
    print(f"\n  Loading original data from {excel_path}...")
    print(f"  Mode: {mode.upper()}" + (f", Position: {position:03d}" if mode == 'position' and position else ""))

    metadata_df, features_df, feature_names, _ = load_and_preprocess_data(
        excel_path, sheet_name, mode=mode, position=position
    )

    # Detect structure positions from ORIGINAL feature columns
    # Need to read the Excel directly to get original column names
    df_original = pd.read_excel(excel_path, sheet_name=sheet_name)
    feature_pattern = re.compile(r'^\d{3}_')
    original_feature_cols = [col for col in df_original.columns if feature_pattern.match(str(col))]

    if not original_feature_cols:
        print("\n  ⚠️  Warning: No multi-structure data detected (no 001_, 002_, etc. prefixes)")
        print("  This feature requires multiple structure positions to work.")
        print("  Skipping training case selection.\n")
        return None

    positions, _ = detect_structure_positions(original_feature_cols)
    all_positions = positions

    # Filter to single position if mode='position'
    if mode == 'position' and position is not None:
        if position not in positions:
            print(f"\n  ⚠️  Warning: Position {position:03d} not found in data")
            print(f"  Available positions: {[f'{p:03d}' for p in positions]}")
            return None
        positions = [position]
        print(f"\n  Detected {len(all_positions)} structures in data: {[f'{p:03d}' for p in all_positions]}")
        print(f"  Analyzing only position: {position:03d}")
    else:
        print(f"\n  Detected {len(positions)} structures: {[f'{p:03d}' for p in positions]}")

    n_structures = len(positions)

    structure_selections = {}  # structure -> list of selected case indices
    structure_metadata = {}    # structure -> clustering metadata
    cluster_mapping = []       # detailed cluster mapping for all cases

    # Analyze each structure independently
    for pos in positions:
        structure_name = f"{pos:03d}"
        print(f"\n  Processing structure {structure_name}...")

        # Extract features for this structure from ORIGINAL data
        structure_cols = [col for col in original_feature_cols if col.startswith(f"{pos:03d}_")]
        structure_features = df_original[structure_cols].copy()

        # Check if we have enough samples
        n_samples = len(structure_features)
        if n_samples < 3:
            print(f"    ⚠️  Warning: Only {n_samples} samples - selecting all")
            structure_selections[structure_name] = list(structure_features.index)
            structure_metadata[structure_name] = {
                'clusters': 1,
                'silhouette': 0.0,
                'cases_selected': n_samples,
                'case_ids': list(structure_features.index)
            }
            # Add cluster mapping for these cases (all in cluster 0)
            for case_id in structure_features.index:
                cluster_mapping.append({
                    'Case_ID': case_id,
                    'Structure': structure_name,
                    'Cluster_ID': 0,
                    'Cluster_Size': n_samples,
                    'Distance_From_Centroid': 0.0
                })
            continue

        # Drop columns with >80% missing first
        missing_pct = structure_features.isnull().mean()
        valid_cols = missing_pct[missing_pct <= 0.8].index.tolist()

        if len(valid_cols) < len(structure_features.columns):
            print(f"    Dropping {len(structure_features.columns) - len(valid_cols)} columns with >80% missing")
            structure_features = structure_features[valid_cols]

        # Impute remaining missing values
        if structure_features.isnull().any().any():
            imputer = SimpleImputer(strategy=impute_strategy)
            structure_features_imputed = pd.DataFrame(
                imputer.fit_transform(structure_features),
                columns=structure_features.columns,
                index=structure_features.index
            )
        else:
            structure_features_imputed = structure_features.copy()

        # Standardize
        scaler = StandardScaler()
        structure_features_scaled = scaler.fit_transform(structure_features_imputed)

        # Find optimal k using silhouette score (max 3 clusters)
        max_k = min(3, n_samples - 1)
        best_k = 2
        best_silhouette = -1

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clustering.fit_predict(structure_features_scaled)

            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                silhouette = silhouette_score(structure_features_scaled, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k

        print(f"    Optimal clusters: {best_k} (silhouette: {best_silhouette:.3f})")

        # Cluster with optimal k
        clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
        labels = clustering.fit_predict(structure_features_scaled)

        # Select representative cases from each cluster
        selected_indices = []

        for cluster_id in range(best_k):
            cluster_mask = labels == cluster_id
            cluster_indices = structure_features.index[cluster_mask].tolist()
            cluster_size = len(cluster_indices)

            if cluster_size == 0:
                continue

            # Get cluster data and compute centroid
            cluster_data = structure_features_scaled[cluster_mask]
            centroid = cluster_data.mean(axis=0)

            # Calculate distances from centroid
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            distance_df = pd.DataFrame({
                'index': cluster_indices,
                'distance': distances
            })

            # Selection strategy based on cluster size
            if cluster_size <= 2:
                # Select all
                selected = cluster_indices
            elif cluster_size <= 9:
                # Select 2 edge + 1 center = 3 cases
                edges = distance_df.nlargest(2, 'distance')['index'].tolist()
                center = distance_df.nsmallest(1, 'distance')['index'].tolist()
                selected = edges + center
            else:
                # Select 4 edge + 1 center = 5 cases
                edges = distance_df.nlargest(4, 'distance')['index'].tolist()
                center = distance_df.nsmallest(1, 'distance')['index'].tolist()
                selected = edges + center

            selected_indices.extend(selected)

            # Store cluster mapping ONLY for selected cases
            for case_id in selected:
                # Find distance for this case
                case_distance = distance_df[distance_df['index'] == case_id]['distance'].values[0]
                cluster_mapping.append({
                    'Case_ID': case_id,
                    'Structure': structure_name,
                    'Cluster_ID': cluster_id,
                    'Cluster_Size': cluster_size,
                    'Distance_From_Centroid': case_distance
                })

        structure_selections[structure_name] = selected_indices
        structure_metadata[structure_name] = {
            'clusters': best_k,
            'silhouette': best_silhouette,
            'cases_selected': len(selected_indices),
            'case_ids': sorted(selected_indices)
        }

        print(f"    Selected {len(selected_indices)} representative cases")

    # Compute union of selected cases across all structures
    all_selected_cases = set()
    for cases in structure_selections.values():
        all_selected_cases.update(cases)

    all_selected_cases = sorted(all_selected_cases)

    print(f"\n  Union of selected cases across all structures: {len(all_selected_cases)} unique patients")
    print(f"  Case IDs: {all_selected_cases[:20]}" + (" ..." if len(all_selected_cases) > 20 else ""))

    # Reduce to max 10 cases globally if needed
    if len(all_selected_cases) > 10:
        print(f"\n  Reducing from {len(all_selected_cases)} to 10 most diverse cases globally...")

        # Calculate diversity score for each case (average distance across structures)
        patient_diversity_scores = {}

        for case_id in all_selected_cases:
            total_distance = 0
            count = 0

            for entry in cluster_mapping:
                if entry['Case_ID'] == case_id:
                    total_distance += entry['Distance_From_Centroid']
                    count += 1

            # Average distance across structures this patient appears in
            patient_diversity_scores[case_id] = total_distance / count if count > 0 else 0

        # Sort by diversity score (higher = more diverse) and keep top 10
        sorted_cases = sorted(patient_diversity_scores.items(), key=lambda x: x[1], reverse=True)
        all_selected_cases = [case_id for case_id, score in sorted_cases[:10]]

        # Update cluster_mapping to only include final selected cases
        cluster_mapping = [entry for entry in cluster_mapping if entry['Case_ID'] in all_selected_cases]

        print(f"  Kept 10 most diverse cases: {all_selected_cases}")
        print(f"  Final selection: {len(all_selected_cases)} unique patients")
    else:
        print(f"  No reduction needed ({len(all_selected_cases)} ≤ 10)")

    # Build case details (which structures each case represents)
    case_details = {}
    for case_id in all_selected_cases:
        structures_represented = []
        for structure_name, selected_cases in structure_selections.items():
            if case_id in selected_cases:
                structures_represented.append(structure_name)

        case_details[case_id] = {
            'structures': structures_represented,
            'count': len(structures_represented)
        }

    print(f"\n{'='*70}")
    print(f"✅ SELECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Total training cases selected: {len(all_selected_cases)}")
    print(f"  Structures analyzed: {n_structures}")
    print(f"\n  Cases representing multiple structures:")
    multi_structure_cases = {k: v for k, v in case_details.items() if v['count'] > 1}
    if multi_structure_cases:
        for case_id, details in sorted(multi_structure_cases.items(),
                                       key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"    Case {case_id}: {details['count']} structures ({', '.join(details['structures'])})")
    else:
        print(f"    (No cases represent multiple structures)")

    return {
        'selected_case_ids': all_selected_cases,
        'case_details': case_details,
        'structure_details': structure_metadata,
        'cluster_mapping': cluster_mapping,
        'metadata_df': metadata_df
    }
