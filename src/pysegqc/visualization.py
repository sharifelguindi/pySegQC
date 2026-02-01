"""
Visualization functions for radiomics clustering analysis.

All public functions follow the `create_X_figure()` naming convention and
return a `plotly.graph_objects.Figure` — no file I/O. The pipeline collects
figures into a dict that the dashboard generator embeds.

Legacy `plot_X()` wrappers are retained for backward compatibility during
the transition; they delegate to the new functions and write files.

Plotly utility helpers (merged from plotly_utils.py) live at the top.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


# =============================================================================
# MAITE Dark Theme
# =============================================================================

def _pad_axis_range(values, pad_fraction=0.08):
    """Compute padded axis range so edge points aren't clipped by the plot frame."""
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    span = vmax - vmin
    if span == 0:
        span = 1.0
    margin = span * pad_fraction
    return [vmin - margin, vmax + margin]


def _apply_2d_padding(fig: go.Figure) -> None:
    """Add axis padding to a 2D scatter so no markers are clipped."""
    xs = np.concatenate([np.asarray(t.x) for t in fig.data if t.x is not None])
    ys = np.concatenate([np.asarray(t.y) for t in fig.data if t.y is not None])
    fig.update_xaxes(range=_pad_axis_range(xs))
    fig.update_yaxes(range=_pad_axis_range(ys))


def _apply_3d_padding(fig: go.Figure) -> None:
    """Add axis padding to a 3D scatter so no markers are clipped."""
    xs = np.concatenate([np.asarray(t.x) for t in fig.data if t.x is not None])
    ys = np.concatenate([np.asarray(t.y) for t in fig.data if t.y is not None])
    zs = np.concatenate([np.asarray(t.z) for t in fig.data if t.z is not None])
    fig.update_layout(scene=dict(
        xaxis=dict(range=_pad_axis_range(xs)),
        yaxis=dict(range=_pad_axis_range(ys)),
        zaxis=dict(range=_pad_axis_range(zs)),
    ))


def apply_maite_theme(fig: go.Figure) -> go.Figure:
    """Apply MAITE dark theme to a Plotly figure.

    Makes figures look correct both when embedded in the dashboard (transparent
    background over card) and when opened as standalone HTML files (dark bg).
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', color='#fbfbfb'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    try:
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.10)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.10)')
    except Exception:
        pass
    return fig


# =============================================================================
# Plotly Utilities (merged from plotly_utils.py)
# =============================================================================

def build_hover_text(
    index: int,
    metadata_df: Optional[pd.DataFrame],
    pca_data: np.ndarray,
    label: int,
    context: str = "",
    pc_indices: List[int] = [0, 1],
) -> str:
    """
    Build standardized hover text for Plotly plots.

    Args:
        index: Sample index in the dataset
        metadata_df: Optional DataFrame containing case metadata
        pca_data: PCA-transformed data array
        label: Cluster assignment for this sample
        context: Optional prefix text (e.g., "Training<br>")
        pc_indices: List of PC indices to display

    Returns:
        HTML-formatted hover text string
    """
    parts = []

    if metadata_df is not None and 'MRN' in metadata_df.columns:
        identifier = f"MRN: {metadata_df.iloc[index].get('MRN', 'N/A')}"
    else:
        identifier = f"Sample: {index}"

    parts.append(f"{context}{identifier}" if context else identifier)
    parts.append(f"Cluster: {label}")

    for i, pc_idx in enumerate(pc_indices, 1):
        parts.append(f"PC{i}: {pca_data[index, pc_idx]:.2f}")

    hover_text = "<br>".join(parts)

    if metadata_df is not None and 'View_Scan_URL' in metadata_df.columns:
        url = metadata_df.iloc[index].get('View_Scan_URL')
        if url and pd.notna(url):
            hover_text += "<br><br><i>Click to open scan viewer</i>"

    return hover_text


def extract_urls(
    indices: List[int],
    metadata_df: Optional[pd.DataFrame],
) -> List[str]:
    """Extract View_Scan URLs for specified indices."""
    if metadata_df is None or 'View_Scan_URL' not in metadata_df.columns:
        return [''] * len(indices)

    urls = []
    for i in indices:
        url = metadata_df.iloc[i].get('View_Scan_URL', '')
        urls.append(str(url) if pd.notna(url) else '')
    return urls


def get_cluster_colors(n_clusters: int) -> List[str]:
    """Get consistent Plotly color palette for clusters."""
    return px.colors.qualitative.Set3[:n_clusters]


# =============================================================================
# QA marker helpers
# =============================================================================

_VERDICT_MARKER = {
    'pass': dict(symbol='circle', size=10),
    'review': dict(symbol='diamond', size=12),
    'fail': dict(symbol='x', size=14),
}

_VERDICT_BORDER = {
    'pass': dict(color='black', width=1),
    'review': dict(color='orange', width=2),
    'fail': dict(color='red', width=2),
}


# =============================================================================
# Figure factory functions — all return go.Figure, zero I/O
# =============================================================================

def create_scree_figure(explained_variance: np.ndarray) -> go.Figure:
    """
    Scree plot: per-component variance (bars) + cumulative variance (line).

    Args:
        explained_variance: Array of variance ratios from PCA

    Returns:
        Plotly Figure with dual-panel subplot
    """
    n = len(explained_variance)
    pcs = list(range(1, n + 1))
    cumulative = np.cumsum(explained_variance) * 100
    x_max = min(20, n)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Variance per Component',
                                        'Cumulative Variance Explained'))

    fig.add_trace(
        go.Bar(x=pcs[:x_max], y=(explained_variance * 100)[:x_max],
               marker_color='steelblue', name='Variance %'),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=pcs[:x_max], y=cumulative[:x_max],
                   mode='lines+markers', name='Cumulative',
                   line=dict(width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=90, line_dash='dash', line_color='red',
                  annotation_text='90%', row=1, col=2)

    fig.update_xaxes(title_text='Principal Component', row=1, col=1)
    fig.update_xaxes(title_text='Number of Components', row=1, col=2)
    fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Variance (%)', row=1, col=2)

    fig.update_layout(
        title='PCA Scree Plot',
        showlegend=False,
        template='plotly_white',
        height=450,
    )
    return fig


def create_dendrogram_figure(pca_data: np.ndarray,
                             max_display: int = 30) -> go.Figure:
    """
    Dendrogram of Ward hierarchical clustering.

    Uses scipy linkage + manual rendering because plotly's
    figure_factory.create_dendrogram re-orders the data and lacks
    truncation support.

    Args:
        pca_data: PCA-transformed data
        max_display: Maximum leaf nodes shown

    Returns:
        Plotly Figure showing dendrogram
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    Z = linkage(pca_data, method='ward')

    # Compute dendrogram coordinates (don't plot with matplotlib)
    truncate = len(pca_data) > max_display
    dn = scipy_dendrogram(
        Z,
        truncate_mode='lastp' if truncate else 'none',
        p=max_display if truncate else 0,
        no_plot=True,
    )

    fig = go.Figure()
    for xs, ys in zip(dn['icoord'], dn['dcoord']):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color='steelblue', width=1.5),
            showlegend=False,
        ))

    suffix = f' (Truncated to {max_display})' if truncate else ''
    fig.update_layout(
        title=f'Hierarchical Clustering Dendrogram (Ward){suffix}',
        xaxis_title='Sample Index (or Cluster Size)',
        yaxis_title='Ward Distance',
        template='plotly_white',
        height=500,
    )
    return fig


def create_cluster_metrics_figure(silhouette_scores: list) -> go.Figure:
    """
    Line chart of silhouette scores by k with best-k marker.

    Args:
        silhouette_scores: Silhouette scores for k=2,3,...

    Returns:
        Plotly Figure
    """
    k_range = list(range(2, len(silhouette_scores) + 2))
    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_range[best_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_range, y=silhouette_scores,
        mode='lines+markers', name='Silhouette',
        line=dict(color='green', width=2),
        marker=dict(size=8),
    ))
    fig.add_vline(x=best_k, line_dash='dash', line_color='red',
                  annotation_text=f'Best k={best_k}')

    fig.update_layout(
        title='Silhouette Score by Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        template='plotly_white',
        height=450,
    )
    return fig


def create_elbow_figure(inertias: list, max_k: int = 10,
                        method: str = 'hierarchical') -> go.Figure:
    """
    Elbow plot: within-cluster sum of squares vs k.

    Args:
        inertias: Pre-computed inertia values for k=2..max_k
        max_k: Maximum k value
        method: Clustering method label

    Returns:
        Plotly Figure
    """
    k_values = list(range(2, len(inertias) + 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=inertias,
        mode='lines+markers',
        marker=dict(size=8, color='steelblue'),
        line=dict(width=2),
    ))

    fig.update_layout(
        title=f'Elbow Plot ({method.capitalize()} Clustering)',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Within-Cluster Sum of Squares',
        template='plotly_white',
        height=450,
    )
    return fig


def create_feature_importance_figure(
    pca_model,
    feature_names: List[str],
    n_components: int = 5,
) -> go.Figure:
    """
    Heatmap of PCA loadings (top features × components).

    Args:
        pca_model: Fitted PCA model with .components_
        feature_names: Feature names matching loadings columns
        n_components: Number of PCs to display

    Returns:
        Plotly Figure with go.Heatmap
    """
    loadings = pca_model.components_[:n_components, :]
    n_top = min(20, len(feature_names))

    total_contribution = np.abs(loadings).sum(axis=0)
    top_idx = np.argsort(total_contribution)[-n_top:]

    subset = loadings[:, top_idx]
    feat_subset = [feature_names[i] for i in top_idx]

    fig = go.Figure(go.Heatmap(
        z=subset,
        x=feat_subset,
        y=[f'PC{i+1}' for i in range(n_components)],
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Loading'),
    ))

    fig.update_layout(
        title=f'Feature Importance: Top {n_top} Features Across {n_components} PCs',
        xaxis_title='Feature',
        yaxis_title='Principal Component',
        template='plotly_white',
        height=max(400, n_top * 20),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def create_radar_figure(
    labels: np.ndarray,
    features_df: pd.DataFrame,
    n_features: int = 8,
) -> go.Figure:
    """
    Radar chart showing cluster z-score profiles.

    Features are z-score normalized; the chart shows how each cluster
    deviates from the population mean on the most discriminative features.

    Args:
        labels: Cluster assignments
        features_df: Standardized feature DataFrame (z-scores)
        n_features: Number of features to display

    Returns:
        Plotly Figure with Scatterpolar traces
    """
    n_clusters = len(np.unique(labels))
    colors = get_cluster_colors(n_clusters)

    # Select features with largest cross-cluster spread
    cluster_means = pd.DataFrame([
        features_df.iloc[labels == k].mean()
        for k in range(n_clusters)
    ])
    spread = cluster_means.max() - cluster_means.min()
    top_features = spread.nlargest(n_features).index.tolist()

    fig = go.Figure()
    for k in range(n_clusters):
        cluster_mean_z = features_df[top_features].iloc[labels == k].mean()
        n_in_cluster = (labels == k).sum()

        fig.add_trace(go.Scatterpolar(
            r=cluster_mean_z.values,
            theta=top_features,
            fill='toself',
            name=f'Cluster {k} (n={n_in_cluster})',
            line=dict(color=colors[k]),
            opacity=0.7,
        ))

    fig.update_layout(
        title='Cluster Morphological Profiles (Z-Score)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=['-3σ', '-2σ', '-1σ', 'μ', '+1σ', '+2σ', '+3σ'],
            ),
        ),
        template='plotly_white',
        height=550,
    )
    return fig


def create_distance_heatmap_figure(
    pca_data: np.ndarray,
    labels: np.ndarray,
) -> go.Figure:
    """
    Heatmap of pairwise Euclidean distances between cluster centroids.

    Args:
        pca_data: PCA-transformed data
        labels: Cluster assignments

    Returns:
        Plotly Figure with go.Heatmap
    """
    n_clusters = len(np.unique(labels))
    centroids = np.array([pca_data[labels == k].mean(axis=0)
                          for k in range(n_clusters)])
    distances = squareform(pdist(centroids, metric='euclidean'))

    cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]

    fig = go.Figure(go.Heatmap(
        z=distances,
        x=cluster_labels,
        y=cluster_labels,
        colorscale='YlOrRd',
        text=np.round(distances, 2),
        texttemplate='%{text}',
        colorbar=dict(title='Euclidean Distance'),
    ))

    fig.update_layout(
        title='Pairwise Cluster Centroid Distances',
        template='plotly_white',
        height=500,
        width=550,
    )
    return fig


def create_pca_2d_figure(
    pca_data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    metadata_df: Optional[pd.DataFrame] = None,
    qa_results: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Interactive 2D PCA scatter plot colored by cluster.

    If qa_results is provided, marker shape/border encodes verdict:
      - pass: circle, black border
      - review: diamond, orange border
      - fail: x, red border

    Args:
        pca_data: PCA-transformed data (n_samples, n_components)
        labels: Cluster labels
        n_clusters: Number of clusters
        metadata_df: Optional metadata for hover text
        qa_results: Optional dict from compute_qa_verdicts()

    Returns:
        Plotly Figure
    """
    colors = get_cluster_colors(n_clusters)
    verdicts = qa_results['verdicts'] if qa_results else None

    fig = go.Figure()

    for k in range(n_clusters):
        cluster_mask = labels == k

        if verdicts is not None:
            # Sub-group by verdict within this cluster
            for verdict in ('pass', 'review', 'fail'):
                verdict_mask = cluster_mask & (verdicts == verdict)
                if not verdict_mask.any():
                    continue

                indices = np.where(verdict_mask)[0]
                hover = [build_hover_text(i, metadata_df, pca_data, k, pc_indices=[0, 1])
                         for i in indices]
                urls = extract_urls(indices.tolist(), metadata_df)

                marker_cfg = _VERDICT_MARKER[verdict].copy()
                marker_cfg['color'] = colors[k]
                marker_cfg['line'] = _VERDICT_BORDER[verdict]

                fig.add_trace(go.Scatter(
                    x=pca_data[verdict_mask, 0],
                    y=pca_data[verdict_mask, 1],
                    mode='markers',
                    name=f'Cluster {k} - {verdict.capitalize()}',
                    marker=marker_cfg,
                    text=hover,
                    hovertemplate='%{text}<extra></extra>',
                    customdata=urls,
                ))
        else:
            indices = np.where(cluster_mask)[0]
            hover = [build_hover_text(i, metadata_df, pca_data, k, pc_indices=[0, 1])
                     for i in indices]
            urls = extract_urls(indices.tolist(), metadata_df)

            fig.add_trace(go.Scatter(
                x=pca_data[cluster_mask, 0],
                y=pca_data[cluster_mask, 1],
                mode='markers',
                name=f'Cluster {k}',
                marker=dict(size=10, color=colors[k],
                            line=dict(color='black', width=1)),
                text=hover,
                hovertemplate='%{text}<extra></extra>',
                customdata=urls,
            ))

    fig.update_layout(
        title=f'PCA Clustering Results (k={n_clusters})',
        xaxis_title='PC1',
        yaxis_title='PC2',
        hovermode='closest',
        template='plotly_white',
        height=700,
    )
    _apply_2d_padding(fig)
    return fig


def create_pca_3d_figure(
    pca_data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    metadata_df: Optional[pd.DataFrame] = None,
    qa_results: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Interactive 3D PCA scatter plot colored by cluster.

    Args:
        pca_data: PCA-transformed data (needs >= 3 components)
        labels: Cluster labels
        n_clusters: Number of clusters
        metadata_df: Optional metadata for hover text
        qa_results: Optional dict from compute_qa_verdicts()

    Returns:
        Plotly Figure (3D scatter)
    """
    if pca_data.shape[1] < 3:
        logger.warning("Need >= 3 PCA components for 3D plot")
        return go.Figure()

    colors = get_cluster_colors(n_clusters)
    verdicts = qa_results['verdicts'] if qa_results else None

    fig = go.Figure()

    for k in range(n_clusters):
        cluster_mask = labels == k

        if verdicts is not None:
            for verdict in ('pass', 'review', 'fail'):
                verdict_mask = cluster_mask & (verdicts == verdict)
                if not verdict_mask.any():
                    continue

                indices = np.where(verdict_mask)[0]
                hover = [build_hover_text(i, metadata_df, pca_data, k, pc_indices=[0, 1, 2])
                         for i in indices]
                urls = extract_urls(indices.tolist(), metadata_df)

                marker_cfg = _VERDICT_MARKER[verdict].copy()
                marker_cfg['color'] = colors[k]
                marker_cfg['line'] = _VERDICT_BORDER[verdict]
                # Scatter3d uses different size scale
                marker_cfg['size'] = max(4, marker_cfg['size'] - 4)

                fig.add_trace(go.Scatter3d(
                    x=pca_data[verdict_mask, 0],
                    y=pca_data[verdict_mask, 1],
                    z=pca_data[verdict_mask, 2],
                    mode='markers',
                    name=f'Cluster {k} - {verdict.capitalize()}',
                    marker=marker_cfg,
                    text=hover,
                    hovertemplate='%{text}<extra></extra>',
                    customdata=urls,
                ))
        else:
            indices = np.where(cluster_mask)[0]
            hover = [build_hover_text(i, metadata_df, pca_data, k, pc_indices=[0, 1, 2])
                     for i in indices]
            urls = extract_urls(indices.tolist(), metadata_df)

            fig.add_trace(go.Scatter3d(
                x=pca_data[cluster_mask, 0],
                y=pca_data[cluster_mask, 1],
                z=pca_data[cluster_mask, 2],
                mode='markers',
                name=f'Cluster {k}',
                marker=dict(size=6, color=colors[k],
                            line=dict(color='black', width=0.5)),
                text=hover,
                hovertemplate='%{text}<extra></extra>',
                customdata=urls,
            ))

    fig.update_layout(
        title=f'PCA Clustering Results 3D (k={n_clusters})',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
        ),
        height=800,
    )
    _apply_3d_padding(fig)
    return fig


def create_prediction_2d_figure(
    training_pca: np.ndarray,
    training_labels: np.ndarray,
    training_metadata: Optional[pd.DataFrame],
    prediction_pca: np.ndarray,
    prediction_labels: np.ndarray,
    prediction_metadata: Optional[pd.DataFrame],
    n_clusters: int,
    confidence_scores: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    2D prediction scatter with training context (grey background).

    Args:
        training_pca: Training PCA data
        training_labels: Training cluster labels
        training_metadata: Training metadata
        prediction_pca: New case PCA data
        prediction_labels: Predicted cluster labels
        prediction_metadata: New case metadata
        n_clusters: Number of clusters
        confidence_scores: Optional per-case confidence

    Returns:
        Plotly Figure
    """
    colors = get_cluster_colors(n_clusters)
    fig = go.Figure()

    # Training background
    for k in range(n_clusters):
        mask = training_labels == k
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        hover = [build_hover_text(i, training_metadata, training_pca, k,
                                  context="Training<br>", pc_indices=[0, 1])
                 for i in indices]
        urls = extract_urls(indices.tolist(), training_metadata)

        fig.add_trace(go.Scatter(
            x=training_pca[mask, 0], y=training_pca[mask, 1],
            mode='markers',
            name=f'Training - Cluster {k}',
            marker=dict(size=7, color=colors[k], opacity=0.35,
                        line=dict(color='black', width=0.8)),
            text=hover,
            hovertemplate='%{text}<extra></extra>',
            customdata=urls,
        ))

    # Predicted foreground
    for k in range(n_clusters):
        mask = prediction_labels == k
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        hover = []
        for i in indices:
            h = build_hover_text(i, prediction_metadata, prediction_pca, k,
                                 context="<b>Predicted</b><br>", pc_indices=[0, 1])
            if confidence_scores is not None:
                h += f"<br>Confidence: {confidence_scores[i]:.3f}"
            hover.append(h)
        urls = extract_urls(indices.tolist(), prediction_metadata)

        fig.add_trace(go.Scatter(
            x=prediction_pca[mask, 0], y=prediction_pca[mask, 1],
            mode='markers',
            name=f'Predicted - Cluster {k}',
            marker=dict(size=12, color=colors[k], opacity=1.0,
                        line=dict(color='black', width=1.5)),
            text=hover,
            hovertemplate='%{text}<extra></extra>',
            customdata=urls,
        ))

    fig.update_layout(
        title=f'Prediction Results with Training Context (k={n_clusters})',
        xaxis_title='PC1', yaxis_title='PC2',
        hovermode='closest',
        template='plotly_white',
        height=700,
    )
    _apply_2d_padding(fig)
    return fig


def create_prediction_3d_figure(
    training_pca: np.ndarray,
    training_labels: np.ndarray,
    training_metadata: Optional[pd.DataFrame],
    prediction_pca: np.ndarray,
    prediction_labels: np.ndarray,
    prediction_metadata: Optional[pd.DataFrame],
    n_clusters: int,
    confidence_scores: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    3D prediction scatter with training context.

    Args:
        Same as create_prediction_2d_figure

    Returns:
        Plotly Figure (3D)
    """
    if prediction_pca.shape[1] < 3 or training_pca.shape[1] < 3:
        logger.warning("Need >= 3 components for 3D prediction plot")
        return go.Figure()

    colors = get_cluster_colors(n_clusters)
    fig = go.Figure()

    # Training background
    for k in range(n_clusters):
        mask = training_labels == k
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        hover = [build_hover_text(i, training_metadata, training_pca, k,
                                  context="Training<br>", pc_indices=[0, 1, 2])
                 for i in indices]
        urls = extract_urls(indices.tolist(), training_metadata)

        fig.add_trace(go.Scatter3d(
            x=training_pca[mask, 0], y=training_pca[mask, 1],
            z=training_pca[mask, 2],
            mode='markers',
            name=f'Training - Cluster {k}',
            marker=dict(size=5, color=colors[k], opacity=0.35,
                        line=dict(color='black', width=0.6)),
            text=hover, hovertemplate='%{text}<extra></extra>',
            customdata=urls,
        ))

    # Predicted foreground
    for k in range(n_clusters):
        mask = prediction_labels == k
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        hover = []
        for i in indices:
            h = build_hover_text(i, prediction_metadata, prediction_pca, k,
                                 context="<b>Predicted</b><br>", pc_indices=[0, 1, 2])
            if confidence_scores is not None:
                h += f"<br>Confidence: {confidence_scores[i]:.3f}"
            hover.append(h)
        urls = extract_urls(indices.tolist(), prediction_metadata)

        fig.add_trace(go.Scatter3d(
            x=prediction_pca[mask, 0], y=prediction_pca[mask, 1],
            z=prediction_pca[mask, 2],
            mode='markers',
            name=f'Predicted - Cluster {k}',
            marker=dict(size=8, color=colors[k], opacity=1.0,
                        line=dict(color='black', width=1.0)),
            text=hover, hovertemplate='%{text}<extra></extra>',
            customdata=urls,
        ))

    fig.update_layout(
        title=f'Prediction Results 3D (k={n_clusters})',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        height=800,
    )
    _apply_3d_padding(fig)
    return fig


def create_selection_quality_figure(
    selected_indices: list,
    pca_data: np.ndarray,
    labels: np.ndarray,
    coverage_metrics: dict,
    representativeness: dict,
    redundancy: dict,
) -> go.Figure:
    """
    Training case selection quality: PCA scatter + coverage bars + summary.

    Args:
        selected_indices: Indices of selected training cases
        pca_data: PCA-transformed data
        labels: Cluster labels
        coverage_metrics: From calculate_selection_coverage()
        representativeness: From calculate_representativeness_scores()
        redundancy: From calculate_redundancy_score()

    Returns:
        Plotly Figure with 3-panel subplot
    """
    n_clusters = len(np.unique(labels))
    colors = get_cluster_colors(n_clusters)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Selected Cases (size=repr.)', 'Coverage by Cluster',
                        'Quality Metrics'),
        column_widths=[0.4, 0.3, 0.3],
        specs=[[{'type': 'scatter'}, {'type': 'bar'}, {'type': 'table'}]],
    )

    # Panel 1: PCA scatter — background + selected
    for k in range(n_clusters):
        mask = labels == k
        fig.add_trace(go.Scatter(
            x=pca_data[mask, 0], y=pca_data[mask, 1],
            mode='markers', showlegend=False,
            marker=dict(size=5, color=colors[k], opacity=0.2),
        ), row=1, col=1)

    for idx in selected_indices:
        rep = representativeness[idx]['representativeness']
        fig.add_trace(go.Scatter(
            x=[pca_data[idx, 0]], y=[pca_data[idx, 1]],
            mode='markers', showlegend=False,
            marker=dict(size=rep * 30 + 8, color=colors[labels[idx]],
                        line=dict(color='black', width=2)),
        ), row=1, col=1)

    # Panel 2: Coverage bars
    cluster_ids = sorted(coverage_metrics['cluster_coverage'].keys())
    coverages = [coverage_metrics['cluster_coverage'][k] for k in cluster_ids]
    fig.add_trace(go.Bar(
        x=[f'C{k}' for k in cluster_ids], y=coverages,
        marker_color=[colors[k] for k in cluster_ids],
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', row=1, col=2)

    # Panel 3: Summary table
    mean_rep = np.mean([r['representativeness'] for r in representativeness.values()])
    overall = (
        coverage_metrics['coverage_score'] * 0.4
        + (1 - redundancy['redundancy_score']) * 0.3
        + mean_rep * 0.3
    )
    quality_label = 'EXCELLENT' if overall > 0.7 else ('GOOD' if overall > 0.5 else 'NEEDS IMPROVEMENT')

    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value']),
        cells=dict(values=[
            ['Coverage', 'Feature Space', 'Diversity', 'Redundancy',
             'Mean Repr.', 'Selected', 'Overall', 'Rating'],
            [f"{coverage_metrics['coverage_score']:.3f}",
             f"{coverage_metrics['feature_space_coverage']:.3f}",
             f"{coverage_metrics['diversity_score']:.2f}",
             f"{redundancy['redundancy_score']:.3f}",
             f"{mean_rep:.3f}",
             f"{len(selected_indices)}/{len(pca_data)}",
             f"{overall:.3f}",
             quality_label],
        ]),
    ), row=1, col=3)

    fig.update_layout(
        title='Training Case Selection Quality',
        template='plotly_white',
        height=500,
    )
    return fig


# =============================================================================
# Legacy wrappers — backward compat during transition (removed in Phase 6)
# =============================================================================

def plot_scree(explained_variance, output_dir):
    """Legacy wrapper: creates + saves scree plot."""
    fig = create_scree_figure(explained_variance)
    output_path = output_dir / 'scree_plot.png'
    try:
        fig.write_image(str(output_path), width=1200, height=450, scale=2)
    except Exception:
        fig.write_html(str(output_dir / 'scree_plot.html'))
    logger.info(f"Saved scree plot: {output_path}")


def plot_dendrogram(pca_data, output_dir, max_display=30):
    """Legacy wrapper."""
    fig = create_dendrogram_figure(pca_data, max_display)
    output_path = output_dir / 'dendrogram.png'
    try:
        fig.write_image(str(output_path), width=1200, height=500, scale=2)
    except Exception:
        fig.write_html(str(output_dir / 'dendrogram.html'))
    logger.info(f"Saved dendrogram: {output_path}")
    # Return linkage matrix for backward compat
    return linkage(pca_data, method='ward')


def plot_cluster_metrics(silhouette_scores, output_dir):
    """Legacy wrapper."""
    fig = create_cluster_metrics_figure(silhouette_scores)
    output_path = output_dir / 'cluster_selection.png'
    try:
        fig.write_image(str(output_path), width=1000, height=450, scale=2)
    except Exception:
        fig.write_html(str(output_dir / 'cluster_selection.html'))
    logger.info(f"Saved cluster selection plot: {output_path}")


def plot_elbow(pca_data, max_k=10, method='hierarchical', output_dir=None):
    """Legacy wrapper: computes inertias then delegates."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from tqdm import tqdm

    inertias = []
    for k in tqdm(range(2, max_k + 1), desc="Elbow plot"):
        if method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labs = clusterer.fit_predict(pca_data)
            inertia = sum(
                np.sum((pca_data[labs == c] - pca_data[labs == c].mean(axis=0)) ** 2)
                for c in range(k)
            )
        else:
            clusterer = KMeans(n_clusters=k, random_state=42)
            clusterer.fit(pca_data)
            inertia = clusterer.inertia_
        inertias.append(inertia)

    fig = create_elbow_figure(inertias, max_k, method)
    if output_dir:
        output_path = output_dir / 'elbow_plot.png'
        try:
            fig.write_image(str(output_path), width=1000, height=450, scale=2)
        except Exception:
            fig.write_html(str(output_dir / 'elbow_plot.html'))
        logger.info(f"Saved elbow plot: {output_path}")

    return inertias


def plot_feature_importance_heatmap(pca_model, feature_names, n_components=5,
                                    output_dir=None):
    """Legacy wrapper."""
    fig = create_feature_importance_figure(pca_model, feature_names, n_components)
    if output_dir:
        output_path = output_dir / 'feature_importance_heatmap.png'
        try:
            fig.write_image(str(output_path), width=1200, height=600, scale=2)
        except Exception:
            fig.write_html(str(output_dir / 'feature_importance_heatmap.html'))
        logger.info(f"Saved feature importance heatmap: {output_path}")


def plot_cluster_profiles_radar(pca_data, labels, feature_names, features_df,
                                n_features=8, output_dir=None):
    """Legacy wrapper."""
    fig = create_radar_figure(labels, features_df, n_features)
    if output_dir:
        output_path = output_dir / 'cluster_radar_profiles.png'
        try:
            fig.write_image(str(output_path), width=900, height=550, scale=2)
        except Exception:
            fig.write_html(str(output_dir / 'cluster_radar_profiles.html'))
        logger.info(f"Saved cluster radar profiles: {output_path}")


def plot_cluster_distance_heatmap(pca_data, labels, output_dir=None):
    """Legacy wrapper."""
    fig = create_distance_heatmap_figure(pca_data, labels)
    if output_dir:
        output_path = output_dir / 'cluster_distance_heatmap.png'
        try:
            fig.write_image(str(output_path), width=550, height=500, scale=2)
        except Exception:
            fig.write_html(str(output_dir / 'cluster_distance_heatmap.html'))
        logger.info(f"Saved cluster distance heatmap: {output_path}")

    n_clusters = len(np.unique(labels))
    centroids = np.array([pca_data[labels == k].mean(axis=0)
                          for k in range(n_clusters)])
    return squareform(pdist(centroids, metric='euclidean'))


def plot_pca_clusters(pca_data, labels, n_clusters, output_dir,
                      metadata_df=None):
    """Legacy wrapper returning (div_2d, div_3d) for dashboard."""
    from .utils import get_plotly_click_handler_script

    fig_2d = create_pca_2d_figure(pca_data, labels, n_clusters, metadata_df)

    # Save standalone HTML
    fig_2d.write_html(
        str(output_dir / 'pca_clusters_2d_interactive.html'),
        post_script=get_plotly_click_handler_script(),
    )
    try:
        fig_2d.write_image(str(output_dir / 'pca_clusters_2d.png'),
                           width=1200, height=800)
    except Exception:
        pass

    div_2d = fig_2d.to_html(
        full_html=False, include_plotlyjs=False, div_id='pca-2d-plot',
        config={'displayModeBar': True, 'responsive': True, 'displaylogo': False},
    )

    div_3d = None
    if pca_data.shape[1] >= 3:
        fig_3d = create_pca_3d_figure(pca_data, labels, n_clusters, metadata_df)
        fig_3d.write_html(
            str(output_dir / 'pca_clusters_3d_interactive.html'),
            post_script=get_plotly_click_handler_script(),
        )
        try:
            fig_3d.write_image(str(output_dir / 'pca_clusters_3d.png'),
                               width=1200, height=900)
        except Exception:
            pass

        div_3d = fig_3d.to_html(
            full_html=False, include_plotlyjs=False, div_id='pca-3d-plot',
            config={'displayModeBar': True, 'responsive': True, 'displaylogo': False},
        )

    return (div_2d, div_3d)


def plot_selection_quality(selected_indices, pca_data, labels, metadata_df,
                           coverage_metrics, representativeness, redundancy,
                           output_dir=None):
    """Legacy wrapper."""
    fig = create_selection_quality_figure(
        selected_indices, pca_data, labels,
        coverage_metrics, representativeness, redundancy,
    )
    if output_dir:
        output_path = output_dir / 'selection_quality_assessment.png'
        try:
            fig.write_image(str(output_path), width=1400, height=500, scale=2)
        except Exception:
            fig.write_html(str(output_dir / 'selection_quality_assessment.html'))
        logger.info(f"Saved selection quality plot: {output_path}")


def plot_training_case_selection(selected_case_ids, pca_data, labels,
                                metadata_df, output_dir):
    """Legacy wrapper — renders training selection as 2D PCA scatter."""
    n_clusters = len(np.unique(labels))
    colors = get_cluster_colors(n_clusters)
    selected_mask = metadata_df.index.isin(selected_case_ids)

    fig = go.Figure()

    # Background: all cases in grey
    fig.add_trace(go.Scatter(
        x=pca_data[:, 0], y=pca_data[:, 1],
        mode='markers', name='All samples',
        marker=dict(size=7, color='lightgrey', opacity=0.4),
    ))

    # Selected cases colored by cluster
    for k in range(n_clusters):
        mask = (labels == k) & selected_mask
        if mask.any():
            fig.add_trace(go.Scatter(
                x=pca_data[mask, 0], y=pca_data[mask, 1],
                mode='markers', name=f'Selected - Cluster {k}',
                marker=dict(size=14, color=colors[k],
                            line=dict(color='black', width=1.5)),
            ))

    fig.update_layout(
        title='Selected Training Cases - PCA Space',
        xaxis_title='PC1', yaxis_title='PC2',
        template='plotly_white', height=600,
    )
    output_path = output_dir / 'training_case_selection_pca_2d.html'
    fig.write_html(str(output_path))
    try:
        fig.write_image(str(output_dir / 'training_case_selection_pca_2d.png'),
                        width=1200, height=600)
    except Exception:
        pass
    logger.info(f"Saved training case PCA plot: {output_path}")


def plot_prediction_with_training(training_pca_data, training_labels,
                                   training_metadata, prediction_pca_data,
                                   prediction_labels, prediction_metadata,
                                   n_clusters, output_dir,
                                   confidence_scores=None):
    """Legacy wrapper — writes standalone HTML files AND returns figures dict."""
    from .utils import get_plotly_click_handler_script

    figures = {}

    fig_2d = create_prediction_2d_figure(
        training_pca_data, training_labels, training_metadata,
        prediction_pca_data, prediction_labels, prediction_metadata,
        n_clusters, confidence_scores,
    )
    apply_maite_theme(fig_2d)
    fig_2d.write_html(
        str(output_dir / 'prediction_with_context_2d.html'),
        post_script=get_plotly_click_handler_script(),
    )
    figures['pca_2d'] = fig_2d

    if prediction_pca_data.shape[1] >= 3 and training_pca_data.shape[1] >= 3:
        fig_3d = create_prediction_3d_figure(
            training_pca_data, training_labels, training_metadata,
            prediction_pca_data, prediction_labels, prediction_metadata,
            n_clusters, confidence_scores,
        )
        apply_maite_theme(fig_3d)
        fig_3d.write_html(
            str(output_dir / 'prediction_with_context_3d.html'),
            post_script=get_plotly_click_handler_script(),
        )
        figures['pca_3d'] = fig_3d

    logger.info(f"Saved prediction visualizations: {output_dir}")
    return figures

