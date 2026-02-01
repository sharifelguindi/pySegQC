"""
Main orchestration pipeline for radiomics clustering analysis.

This module contains the main analysis pipeline that coordinates data loading,
preprocessing, PCA, clustering, QA detection, visualization, and export.
"""

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .data_loader import load_and_preprocess_data
from .validation import impute_missing_values, standardize_features
from .utils import filter_volume_dependent_features, normalize_by_volume
from .pca import perform_pca
from .clustering import perform_hierarchical_clustering, perform_kmeans
from .metrics import find_optimal_clusters, assess_cluster_stability_bootstrap, calculate_gap_statistic, perform_consensus_clustering
from .visualization import (
    create_scree_figure, create_dendrogram_figure, create_cluster_metrics_figure,
    create_pca_2d_figure, create_pca_3d_figure,
    create_elbow_figure, create_feature_importance_figure,
    create_radar_figure, create_distance_heatmap_figure,
    # Legacy wrappers still used for standalone file output
    plot_scree, plot_dendrogram, plot_cluster_metrics, plot_pca_clusters,
    plot_elbow, plot_feature_importance_heatmap, plot_cluster_profiles_radar,
    plot_cluster_distance_heatmap,
    plot_selection_quality,
)
from .qa import compute_qa_verdicts, create_neutral_qa_results
from .report import generate_html_dashboard, generate_json_report
from .thumbnail import generate_thumbnails_batch
from .export import export_results
from .training import (
    select_training_cases_from_clustering,
    calculate_selection_coverage,
    calculate_representativeness_scores,
    calculate_redundancy_score,
)

logger = logging.getLogger(__name__)


def run_analysis_pipeline(args, input_path, output_dir):
    """
    Run the complete analysis pipeline for a single analysis unit.

    This function encapsulates the full analysis workflow:
    - Load and preprocess data
    - Impute and standardize
    - PCA transformation
    - Clustering
    - QA outlier detection
    - Visualization (figure collection + standalone files)
    - Dashboard and JSON report generation
    - Training case selection (if requested)
    - Export results

    Args:
        args: Parsed command-line arguments
        input_path: Path to input Excel file
        output_dir: Directory for output files
    """
    print(f"\n{'='*70}")
    print(f"RADIOMICS PCA + HIERARCHICAL CLUSTERING ANALYSIS")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Input: {input_path}")
    mode_display = args.mode.upper() if args.mode else "DEFAULT"
    print(f"  Mode: {mode_display}")
    print(f"  Method: {args.method.upper()}")
    if args.mode == 'position':
        print(f"  Position: {args.position:03d}")
    print(f"  Output: {output_dir}")

    # Load and preprocess
    metadata_df, features_df, feature_names, structure_info = load_and_preprocess_data(
        input_path, args.sheet, args.mode, args.position
    )

    # CRITICAL: Apply volume independence if requested
    original_feature_count = len(feature_names)
    if args.volume_independent and args.volume_normalize:
        logger.warning("Both --volume-independent and --volume-normalize specified. Using --volume-independent (filtering).")
        args.volume_normalize = False

    if args.volume_independent:
        logger.info("\n" + "="*70)
        logger.info("VOLUME INDEPENDENCE MODE: FILTERING")
        logger.info("="*70)
        logger.info("Excluding volume-dependent features to ensure SHAPE-based clustering")

        filtered_features = filter_volume_dependent_features(feature_names, exclude_spatial=False)
        features_df = features_df[filtered_features]
        feature_names = filtered_features

        if len(feature_names) == 0:
            raise ValueError(
                "Insufficient data: All features are volume-dependent. "
                "Cannot perform volume-independent clustering with this dataset. "
                "Try running without --volume-independent flag."
            )

        logger.info(f"Feature reduction: {original_feature_count} -> {len(feature_names)} features")

    elif args.volume_normalize:
        logger.info("\n" + "="*70)
        logger.info("VOLUME INDEPENDENCE MODE: NORMALIZATION")
        logger.info("="*70)

        features_df, feature_names = normalize_by_volume(features_df, feature_names)
        logger.info(f"Feature transformation: {original_feature_count} -> {len(feature_names)} features")

    # Impute and standardize
    features_imputed = impute_missing_values(features_df, strategy=args.impute)
    features_standardized, scaler = standardize_features(features_imputed)

    # PCA
    pca_model, pca_data, explained_variance = perform_pca(
        features_standardized, n_components=args.n_components
    )

    # ─── Collect figures into dict for dashboard embedding ───
    figures = {}
    figures['scree'] = create_scree_figure(explained_variance)
    figures['feature_heatmap'] = create_feature_importance_figure(
        pca_model, feature_names, n_components=5
    )

    if args.method == 'hierarchical':
        figures['dendrogram'] = create_dendrogram_figure(pca_data)

    # Elbow plot
    max_k = min(args.max_k, len(pca_data) - 1)
    inertias = []
    for k in range(2, max_k + 1):
        if args.method == 'kmeans':
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(pca_data)
            inertias.append(km.inertia_)
        else:
            ac = AgglomerativeClustering(n_clusters=k, linkage='ward')
            ac.fit(pca_data)
            inertias.append(-calinski_harabasz_score(pca_data, ac.labels_))
    figures['elbow'] = create_elbow_figure(inertias, max_k, args.method)

    # Find optimal k or use specified
    silhouette_scores_result = None
    if args.auto_k or args.n_clusters is None:
        silhouette_scores_result, best_k = find_optimal_clusters(
            pca_data, max_k=args.max_k, method=args.method
        )
        figures['silhouette'] = create_cluster_metrics_figure(silhouette_scores_result)

        logger.info("Running gap statistic analysis for additional validation...")
        gap_results = calculate_gap_statistic(pca_data, max_k=args.max_k, n_refs=20, method=args.method)
        logger.info(f"Gap statistic suggests k={gap_results['optimal_k']}, using silhouette-based k={best_k}")

        n_clusters = best_k
    else:
        n_clusters = args.n_clusters

    # Perform clustering
    if args.method == 'hierarchical':
        clustering_model, cluster_labels = perform_hierarchical_clustering(pca_data, n_clusters)
    else:
        clustering_model, cluster_labels = perform_kmeans(pca_data, n_clusters)

    # Calculate silhouette score
    sil_score = silhouette_score(pca_data, cluster_labels)

    # ─── QA outlier detection ───
    centroids = np.array([pca_data[cluster_labels == k].mean(axis=0) for k in range(n_clusters)])

    if getattr(args, 'select_training_cases', False):
        qa_results = create_neutral_qa_results(pca_data, cluster_labels, centroids)
    else:
        qa_sigma = getattr(args, 'qa_sigma', 2.0)
        qa_contamination = getattr(args, 'qa_contamination', 0.1)
        qa_results = compute_qa_verdicts(
            pca_data, cluster_labels, centroids,
            distance_sigma=qa_sigma, iforest_contamination=qa_contamination
        )
        logger.info(
            f"QA verdicts: {np.sum(qa_results['verdicts'] == 'pass')} pass, "
            f"{np.sum(qa_results['verdicts'] == 'review')} review, "
            f"{np.sum(qa_results['verdicts'] == 'fail')} fail"
        )

    # ─── Thumbnail generation (if NIfTI paths available) ───
    thumbnails = {}
    generate_thumbs = getattr(args, 'thumbnails', True)
    if generate_thumbs and 'Image_Path' in metadata_df.columns and 'Mask_Path' in metadata_df.columns:
        try:
            # Parse custom window setting
            window_str = getattr(args, 'thumbnail_window', '40/400')
            parts = window_str.split('/')
            ct_window = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (40, 400)
            thumbnails = generate_thumbnails_batch(
                metadata_df, max_workers=4,
                window_center=ct_window[0], window_width=ct_window[1]
            )
        except ImportError:
            logger.info("SimpleITK/matplotlib not available — skipping thumbnails")
        except Exception as e:
            logger.warning(f"Thumbnail generation failed: {e}")

    # ─── NiiVue viewer (if NIfTI paths available) ───
    has_viewer = False
    if 'Image_Path' in metadata_df.columns:
        try:
            from .viewer import generate_viewer_data, generate_viewer_html
            viewer_data = generate_viewer_data(
                output_dir / 'viewer_data.json',
                metadata_df, cluster_labels, qa_results,
                structure_label=args.position if args.mode == 'position' else None,
                structure_name=getattr(args, 'structure_name', None),
            )
            generate_viewer_html(output_dir / 'viewer.html', viewer_data=viewer_data)
            has_viewer = True
        except Exception as e:
            logger.warning(f"Viewer generation failed: {e}")

    # ─── QA-aware PCA figures ───
    figures['pca_2d'] = create_pca_2d_figure(
        pca_data, cluster_labels, n_clusters, metadata_df, qa_results=qa_results
    )
    figures['pca_3d'] = create_pca_3d_figure(
        pca_data, cluster_labels, n_clusters, metadata_df, qa_results=qa_results
    )

    # Additional visualizations
    figures['radar'] = create_radar_figure(
        cluster_labels, features_standardized, n_features=8
    )
    figures['distance_heatmap'] = create_distance_heatmap_figure(pca_data, cluster_labels)

    # Also write standalone files via legacy wrappers (backward compat)
    plot_scree(explained_variance, output_dir)
    plot_feature_importance_heatmap(pca_model, feature_names, n_components=5, output_dir=output_dir)
    if args.method == 'hierarchical':
        plot_dendrogram(pca_data, output_dir)
    plot_elbow(pca_data, max_k=args.max_k, method=args.method, output_dir=output_dir)
    if silhouette_scores_result is not None:
        plot_cluster_metrics(silhouette_scores_result, output_dir)
    plot_pca_clusters(pca_data, cluster_labels, n_clusters, output_dir, metadata_df=metadata_df)
    plot_cluster_profiles_radar(pca_data, cluster_labels, feature_names,
                                features_standardized, n_features=8, output_dir=output_dir)
    plot_cluster_distance_heatmap(pca_data, cluster_labels, output_dir=output_dir)

    # ─── Cluster stability analysis ───
    logger.info("Assessing clustering quality with bootstrap and consensus methods...")
    stability_results = assess_cluster_stability_bootstrap(
        pca_data, n_clusters, method=args.method, n_iterations=50
    )
    consensus_results = perform_consensus_clustering(
        pca_data, n_clusters, n_iterations=50, method=args.method
    )

    # ─── Training case selection (if requested) ───
    training_selection_data = None
    if args.select_training_cases:
        if args.mode == 'concat' or args.mode == 'position':
            training_selection_data = select_training_cases_from_clustering(
                cluster_labels, pca_data, metadata_df, max_clusters=3, max_cases=10
            )
            training_selection_data['mode'] = args.mode

            logger.info("Calculating enhanced training case selection metrics...")

            if 'per_cluster_selections' in training_selection_data:
                all_selected_case_ids = []
                for cluster_info in training_selection_data['per_cluster_selections'].values():
                    all_selected_case_ids.extend(cluster_info['selected_case_ids'])
            else:
                all_selected_case_ids = training_selection_data['selected_case_ids']

            selected_indices = np.array([i for i, row in enumerate(metadata_df.itertuples())
                                       if metadata_df.index[i] in all_selected_case_ids])

            if len(selected_indices) > 0:
                coverage_metrics = calculate_selection_coverage(
                    selected_indices, pca_data, cluster_labels, n_clusters
                )
                representativeness = calculate_representativeness_scores(
                    selected_indices, pca_data, cluster_labels
                )
                redundancy = calculate_redundancy_score(selected_indices, pca_data)

                plot_selection_quality(
                    selected_indices, pca_data, cluster_labels, metadata_df,
                    coverage_metrics, representativeness, redundancy,
                    output_dir=output_dir
                )

                logger.info(f"Selection quality: Coverage={coverage_metrics['coverage_score']:.3f}, "
                          f"Redundancy={redundancy['redundancy_score']:.3f}")
        else:
            logger.warning("--select-training-cases only supported with concat or position modes. Skipping.")

    # ─── Export Excel results ───
    export_results(metadata_df, features_df, pca_data, cluster_labels,
                   pca_model, feature_names, output_dir, input_path,
                   scaler=scaler, impute_strategy=args.impute,
                   training_selection_data=training_selection_data,
                   qa_results=qa_results)

    # ─── Analysis config for reports ───
    analysis_config = {
        'method': args.method,
        'mode': args.mode or 'default',
        'volume_independent': getattr(args, 'volume_independent', False),
        'n_clusters': n_clusters,
        'n_features': len(feature_names),
        'silhouette': sil_score,
        'stability': stability_results['mean_stability'],
        'robustness': consensus_results['robustness_score'],
        'distance_sigma': getattr(args, 'qa_sigma', 2.0),
    }

    # ─── Generate HTML dashboard ───
    generate_html_dashboard(
        output_path=output_dir / 'analysis_dashboard.html',
        metadata_df=metadata_df,
        cluster_labels=cluster_labels,
        qa_results=qa_results,
        figures=figures,
        analysis_config=analysis_config,
        thumbnails=thumbnails or None,
        has_viewer=has_viewer,
    )

    # ─── Generate JSON report ───
    generate_json_report(
        output_path=output_dir / 'analysis_report.json',
        metadata_df=metadata_df,
        pca_data=pca_data,
        cluster_labels=cluster_labels,
        centroids=centroids,
        qa_results=qa_results,
        analysis_config=analysis_config,
        pca_diagnostics={
            'n_components': pca_model.n_components_,
            'explained_variance_ratio': [round(float(v), 4) for v in explained_variance],
            'total_features': original_feature_count,
            'features_used': len(feature_names),
        },
    )

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"{'='*70}")
    logger.info(f"\nResults: {output_dir}")
    logger.info(f"\nSummary:")
    logger.info(f"   Method: {args.method.upper()}")
    logger.info(f"   Samples analyzed: {len(pca_data)}")
    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Clusters: {n_clusters}")
    logger.info(f"   Silhouette score: {sil_score:.3f}")
    logger.info(f"   Cluster stability: {stability_results['mean_stability']:.3f}")
    logger.info(f"   Consensus robustness: {consensus_results['robustness_score']:.3f}")
    logger.info(f"   QA: {np.sum(qa_results['verdicts'] == 'pass')} pass / "
              f"{np.sum(qa_results['verdicts'] == 'review')} review / "
              f"{np.sum(qa_results['verdicts'] == 'fail')} fail")
    if args.method == 'hierarchical':
        logger.info(f"\nWard linkage used - deterministic, robust clustering")

    logger.info(f"\nOpen analysis_dashboard.html in your browser for interactive results!")

    # Return training selection data for union computation
    return training_selection_data
