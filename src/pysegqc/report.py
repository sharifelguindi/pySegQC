"""
HTML dashboard and JSON report generation.

This module produces the two main report outputs of pySegQC:
1. A self-contained HTML dashboard with QA summary, risk table,
   interactive Plotly visualizations, and optional NIfTI thumbnails.
2. A machine-readable JSON report with summary verdicts and per-case diagnostics.

Both outputs are designed to be standalone — no external CSS, no local image
files — everything is inlined or loaded from CDN (Plotly.js).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_case_id as _get_case_id

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Report
# =============================================================================


def generate_json_report(
    output_path: Path,
    metadata_df: pd.DataFrame,
    pca_data: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    qa_results: Dict[str, Any],
    analysis_config: Dict[str, Any],
    pca_diagnostics: Optional[Dict[str, Any]] = None,
    clustering_diagnostics: Optional[Dict[str, Any]] = None,
    prediction_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured JSON report with summary, per-case, and diagnostic data.

    Args:
        output_path: File path for the JSON output.
        metadata_df: DataFrame with case metadata.
        pca_data: PCA-transformed data (n_samples, n_components).
        cluster_labels: Cluster assignments per sample.
        centroids: Cluster centroids in PCA space.
        qa_results: Output from compute_qa_verdicts().
        analysis_config: Dict with keys like 'method', 'mode',
            'volume_independent', 'n_features', 'tool_version'.
        pca_diagnostics: Optional PCA info (explained_variance_ratio, etc.).
        clustering_diagnostics: Optional clustering info (silhouette_scores_by_k, etc.).
        prediction_info: Optional prediction-mode info.

    Returns:
        The report dict (also written to output_path).
    """
    from . import __version__

    verdicts = qa_results['verdicts']
    n_total = len(verdicts)

    # Verdict counts
    pass_count = int(np.sum(verdicts == 'pass'))
    review_count = int(np.sum(verdicts == 'review'))
    fail_count = int(np.sum(verdicts == 'fail'))

    # Overall verdict
    if fail_count > 0:
        overall_verdict = 'FAIL'
    elif review_count > 0:
        overall_verdict = 'REVIEW'
    else:
        overall_verdict = 'PASS'

    n_clusters = len(centroids)

    # Build summary
    summary = {
        'verdict': overall_verdict,
        'total_cases': n_total,
        'pass_count': pass_count,
        'review_count': review_count,
        'fail_count': fail_count,
        'pass_rate': round(pass_count / max(n_total, 1), 3),
        'n_clusters': n_clusters,
        'silhouette_score': round(float(analysis_config.get('silhouette', 0)), 3),
        'analysis_mode': analysis_config.get('mode', 'default'),
        'volume_independent': analysis_config.get('volume_independent', False),
    }

    # Build per-case entries
    cases = []
    for i in range(n_total):
        case_id = _get_case_id(metadata_df, i)
        entry = {
            'case_id': str(case_id),
            'cluster': int(cluster_labels[i]),
            'qa_verdict': str(verdicts[i]),
            'qa_risk_score': round(float(qa_results['qa_risk_scores'][i]), 4),
            'distance_to_centroid': round(float(qa_results['distance_to_centroid'][i]), 4),
            'distance_z_score': round(float(qa_results['distance_z_scores'][i]), 4),
            'distance_outlier': bool(qa_results['distance_outlier_mask'][i]),
            'iforest_outlier': bool(qa_results['iforest_outlier_mask'][i]),
            'pc_coordinates': [round(float(v), 4) for v in pca_data[i, :min(3, pca_data.shape[1])]],
        }
        cases.append(entry)

    # Build per-cluster entries
    clusters = []
    for k in range(n_clusters):
        mask = cluster_labels == k
        cluster_verdicts = verdicts[mask]
        cluster_stats = qa_results.get('per_cluster_stats', {}).get(k, {'mean': 0.0, 'std': 0.0})
        cluster_entry = {
            'cluster_id': k,
            'size': int(mask.sum()),
            'proportion': round(float(mask.sum()) / max(n_total, 1), 3),
            'centroid_pc': [round(float(v), 4) for v in centroids[k, :min(3, centroids.shape[1])]],
            'mean_distance': round(float(cluster_stats['mean']), 4),
            'std_distance': round(float(cluster_stats['std']), 4),
            'pass_count': int(np.sum(cluster_verdicts == 'pass')),
            'review_count': int(np.sum(cluster_verdicts == 'review')),
            'fail_count': int(np.sum(cluster_verdicts == 'fail')),
        }
        clusters.append(cluster_entry)

    # Build diagnostics
    diagnostics = {}
    if pca_diagnostics:
        diagnostics['pca'] = pca_diagnostics
    if clustering_diagnostics:
        diagnostics['clustering'] = clustering_diagnostics

    diagnostics['outlier_detection'] = {
        'distance_method': {
            'sigma_threshold': float(analysis_config.get('distance_sigma', 2.0)),
            'flagged_count': int(qa_results['distance_outlier_mask'].sum()),
        },
        'isolation_forest': {
            'contamination': 0.1,
            'flagged_count': int(qa_results['iforest_outlier_mask'].sum()),
        },
        'combined': {
            'pass': pass_count,
            'review': review_count,
            'fail': fail_count,
        },
    }

    report = {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'tool': 'pySegQC',
        'tool_version': __version__,
        'summary': summary,
        'cases': cases,
        'clusters': clusters,
        'diagnostics': diagnostics,
    }

    if prediction_info:
        report['prediction'] = prediction_info

    # Write
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"JSON report saved: {output_path}")
    return report


# =============================================================================
# HTML Dashboard
# =============================================================================

def generate_html_dashboard(
    output_path: Path,
    metadata_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    qa_results: Dict[str, Any],
    figures: Dict[str, go.Figure],
    analysis_config: Dict[str, Any],
    thumbnails: Optional[Dict[int, str]] = None,
    has_viewer: bool = False,
) -> str:
    """
    Generate a self-contained HTML dashboard.

    Args:
        output_path: File path for the HTML output.
        metadata_df: DataFrame with case metadata.
        cluster_labels: Cluster assignments.
        qa_results: Output from compute_qa_verdicts().
        figures: Dict mapping figure names to Plotly Figure objects.
            Expected keys: 'pca_2d', 'pca_3d', 'scree', 'dendrogram',
            'elbow', 'silhouette', 'feature_heatmap', 'radar',
            'distance_heatmap'. Missing keys are silently skipped.
        analysis_config: Dict with method, mode, n_clusters, silhouette, etc.
        thumbnails: Optional dict mapping case index to base64 PNG string.
        has_viewer: If True, adds "Open in NiiVue Viewer" link to side panel.

    Returns:
        The HTML string (also written to output_path).
    """
    n_total = len(qa_results['verdicts'])
    verdicts = qa_results['verdicts']

    pass_count = int(np.sum(verdicts == 'pass'))
    review_count = int(np.sum(verdicts == 'review'))
    fail_count = int(np.sum(verdicts == 'fail'))
    pass_rate = pass_count / max(n_total, 1) * 100

    n_clusters = int(analysis_config.get('n_clusters', len(np.unique(cluster_labels))))
    method = analysis_config.get('method', 'hierarchical').upper()
    mode = analysis_config.get('mode', 'default').upper()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build sections
    sections = []
    sections.append(_build_qa_summary_banner(
        pass_count, review_count, fail_count, pass_rate,
        n_clusters, method, mode, n_total, timestamp,
        analysis_config,
    ))
    sections.append(_build_risk_table(metadata_df, cluster_labels, qa_results, thumbnails))
    sections.append(_build_plotly_section(
        'PCA Exploration', ['pca_2d', 'pca_3d'], figures
    ))
    sections.append(_build_plotly_section(
        'Cluster Validation', ['dendrogram', 'elbow', 'silhouette', 'distance_heatmap'], figures
    ))
    sections.append(_build_plotly_section(
        'Feature Analysis', ['scree', 'feature_heatmap', 'radar'], figures
    ))
    sections.append(_build_metrics_cards(analysis_config))

    body = '\n'.join(s for s in sections if s)

    # Thumbnail data for JS click handler
    thumb_json = json.dumps(thumbnails or {}, default=str)

    html = _DASHBOARD_TEMPLATE.format(
        title='pySegQC Analysis Dashboard',
        body=body,
        thumbnail_data=thumb_json,
        has_viewer_js='true' if has_viewer else 'false',
        timestamp=timestamp,
    )

    output_path = Path(output_path)
    output_path.write_text(html, encoding='utf-8')
    logger.info(f"HTML dashboard saved: {output_path}")
    return html


# =============================================================================
# Dashboard Components
# =============================================================================


def _build_qa_summary_banner(
    pass_count, review_count, fail_count, pass_rate,
    n_clusters, method, mode, n_total, timestamp,
    analysis_config,
    title='pySegQC Analysis Dashboard',
):
    """QA summary banner with stat cards and verdict badges."""
    sil = analysis_config.get('silhouette', 0)
    stability = analysis_config.get('stability', None)
    robustness = analysis_config.get('robustness', None)
    avg_confidence = analysis_config.get('avg_confidence', None)

    # Build metric chips for secondary info
    chips = []
    chips.append(f'<span class="chip">Method: {method}</span>')
    chips.append(f'<span class="chip">Mode: {mode}</span>')
    if sil:
        chips.append(f'<span class="chip">Silhouette: {sil:.3f}</span>')
    if stability is not None:
        chips.append(f'<span class="chip">Stability: {stability:.3f}</span>')
    if robustness is not None:
        chips.append(f'<span class="chip">Robustness: {robustness:.3f}</span>')
    if avg_confidence is not None:
        chips.append(f'<span class="chip">Avg Confidence: {avg_confidence:.3f}</span>')
    chips_html = '\n'.join(chips)

    return f'''
    <div class="card hero-card">
      <div class="card-header">
        <div class="card-title" style="font-size:1.5rem">{title}</div>
        <div class="card-description">Generated: {timestamp}</div>
      </div>
      <div class="card-content">
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-title">Pass</div>
            <div class="stat-value" style="color:var(--pass)">{pass_count}</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Review</div>
            <div class="stat-value" style="color:var(--review)">{review_count}</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Fail</div>
            <div class="stat-value" style="color:var(--fail)">{fail_count}</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Total Cases</div>
            <div class="stat-value">{n_total}</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Pass Rate</div>
            <div class="stat-value">{pass_rate:.1f}%</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Clusters</div>
            <div class="stat-value">{n_clusters}</div>
          </div>
        </div>
        <div class="chips-row">
          {chips_html}
        </div>
      </div>
    </div>
    '''


def _build_risk_table(metadata_df, cluster_labels, qa_results, thumbnails):
    """Sortable QA risk table in MAITE card style, sorted worst-first."""
    verdicts = qa_results['verdicts']
    risk_scores = qa_results['qa_risk_scores']

    # Sort by risk score descending
    order = np.argsort(-risk_scores)

    rows = []
    for idx in order:
        case_id = _get_case_id(metadata_df, idx)
        cluster = int(cluster_labels[idx])
        verdict = str(verdicts[idx])
        risk = float(risk_scores[idx])
        dist = float(qa_results['distance_to_centroid'][idx])
        z = float(qa_results['distance_z_scores'][idx])

        badge_cls = f'badge badge-{verdict}'
        thumb_attr = f'data-thumb="{thumbnails[idx]}"' if thumbnails and idx in thumbnails else ''

        rows.append(f'''
        <tr data-case-idx="{idx}" {thumb_attr}>
          <td>{case_id}</td>
          <td>{cluster}</td>
          <td><span class="{badge_cls}">{verdict.upper()}</span></td>
          <td>{risk:.3f}</td>
          <td>{dist:.2f}</td>
          <td>{z:.2f}</td>
        </tr>''')

    return f'''
    <div class="card">
      <div class="card-header">
        <div class="card-title">QA Risk Table</div>
        <div class="card-description">Cases sorted by risk score (worst first). Click a row to view thumbnail.</div>
      </div>
      <div class="card-content">
        <div class="table-container">
          <table class="risk-table" id="risk-table">
            <thead>
              <tr>
                <th onclick="sortTable(0)">Case ID</th>
                <th onclick="sortTable(1)">Cluster</th>
                <th onclick="sortTable(2)">Verdict</th>
                <th onclick="sortTable(3)">Risk Score</th>
                <th onclick="sortTable(4)">Distance</th>
                <th onclick="sortTable(5)">Z-Score</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    '''


def _build_plotly_section(title, figure_keys, figures):
    """Embed Plotly figures as cards in a responsive grid."""
    divs = []
    for key in figure_keys:
        fig = figures.get(key)
        if fig is None:
            continue
        # Apply transparent background for card embedding
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='system-ui, -apple-system, sans-serif', color='#fbfbfb'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
        )
        # Update axes styling for dark theme (only if axes exist)
        try:
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.10)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.10)')
        except Exception:
            pass
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f'plot-{key}',
            config={'displayModeBar': True, 'responsive': True, 'displaylogo': False},
        )
        divs.append(f'<div class="plot-card">{div_html}</div>')

    if not divs:
        return ''

    return f'''
    <div class="card">
      <div class="card-header">
        <div class="card-title">{title}</div>
      </div>
      <div class="card-content">
        <div class="plot-grid">
          {''.join(divs)}
        </div>
      </div>
    </div>
    '''


def _build_metrics_cards(analysis_config):
    """Metrics summary cards in MAITE StatCard pattern."""
    cards = []
    metric_defs = [
        ('Silhouette Score', 'silhouette', '.3f'),
        ('Cluster Stability', 'stability', '.3f'),
        ('Consensus Robustness', 'robustness', '.3f'),
    ]
    for label, key, fmt in metric_defs:
        val = analysis_config.get(key)
        if val is not None:
            cards.append(f'''
          <div class="stat-card">
            <div class="stat-title">{label}</div>
            <div class="stat-value" style="color:var(--accent)">{val:{fmt}}</div>
          </div>''')

    if not cards:
        return ''

    return f'''
    <div class="card">
      <div class="card-header">
        <div class="card-title">Clustering Quality Metrics</div>
      </div>
      <div class="card-content">
        <div class="stats-grid">
          {''.join(cards)}
        </div>
      </div>
    </div>
    '''


# =============================================================================
# Prediction Dashboard
# =============================================================================


def generate_prediction_dashboard(
    output_path: Path,
    metadata_df: pd.DataFrame,
    predicted_labels: np.ndarray,
    qa_results: Dict[str, Any],
    figures: Dict[str, go.Figure],
    prediction_config: Dict[str, Any],
    thumbnails: Optional[Dict[int, str]] = None,
    has_viewer: bool = False,
) -> str:
    """Generate a MAITE-styled HTML dashboard for prediction results.

    This solves the "blank prediction plots" problem: instead of bare
    Plotly HTML files, predictions now get a full dashboard with QA
    summary, risk table, and embedded interactive plots.

    Args:
        output_path: File path for the HTML output.
        metadata_df: DataFrame with new-case metadata.
        predicted_labels: Predicted cluster assignments per sample.
        qa_results: Output from compute_prediction_qa_verdicts().
        figures: Dict mapping figure names to Plotly Figure objects.
            Expected keys: 'pca_2d', 'pca_3d'. Missing keys skipped.
        prediction_config: Dict with keys like 'n_clusters',
            'avg_confidence', 'model_path', 'n_features'.
        thumbnails: Optional dict mapping case index to base64 PNG.

    Returns:
        The HTML string (also written to output_path).
    """
    n_total = len(qa_results['verdicts'])
    verdicts = qa_results['verdicts']

    pass_count = int(np.sum(verdicts == 'pass'))
    review_count = int(np.sum(verdicts == 'review'))
    fail_count = int(np.sum(verdicts == 'fail'))
    pass_rate = pass_count / max(n_total, 1) * 100

    n_clusters = int(prediction_config.get('n_clusters', len(np.unique(predicted_labels))))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build analysis_config-like dict for the banner
    banner_config = {
        'method': 'PREDICTION',
        'mode': prediction_config.get('mode', 'default').upper(),
        'silhouette': None,
        'stability': None,
        'robustness': None,
        'avg_confidence': prediction_config.get('avg_confidence'),
    }

    sections = []
    sections.append(_build_qa_summary_banner(
        pass_count, review_count, fail_count, pass_rate,
        n_clusters, 'PREDICTION', banner_config.get('mode', 'DEFAULT'),
        n_total, timestamp,
        banner_config,
        title='pySegQC Prediction Dashboard',
    ))
    sections.append(_build_risk_table(metadata_df, predicted_labels, qa_results, thumbnails))
    sections.append(_build_plotly_section(
        'Prediction PCA (with Training Context)', ['pca_2d', 'pca_3d'], figures
    ))

    # Prediction-specific info card
    model_path = prediction_config.get('model_path', 'N/A')
    n_features = prediction_config.get('n_features', 'N/A')
    avg_conf = prediction_config.get('avg_confidence')
    info_cards = []
    info_cards.append(f'''
      <div class="stat-card">
        <div class="stat-title">Model Directory</div>
        <div class="stat-value" style="font-size:0.9rem;word-break:break-all">{model_path}</div>
      </div>''')
    info_cards.append(f'''
      <div class="stat-card">
        <div class="stat-title">Features Used</div>
        <div class="stat-value" style="color:var(--accent)">{n_features}</div>
      </div>''')
    info_cards.append(f'''
      <div class="stat-card">
        <div class="stat-title">Clusters</div>
        <div class="stat-value" style="color:var(--accent)">{n_clusters}</div>
      </div>''')
    if avg_conf is not None:
        info_cards.append(f'''
      <div class="stat-card">
        <div class="stat-title">Avg Confidence</div>
        <div class="stat-value" style="color:var(--accent)">{avg_conf:.3f}</div>
      </div>''')

    sections.append(f'''
    <div class="card">
      <div class="card-header">
        <div class="card-title">Prediction Info</div>
      </div>
      <div class="card-content">
        <div class="stats-grid">
          {''.join(info_cards)}
        </div>
      </div>
    </div>
    ''')

    body = '\n'.join(s for s in sections if s)
    thumb_json = json.dumps(thumbnails or {}, default=str)

    html = _DASHBOARD_TEMPLATE.format(
        title='pySegQC Prediction Dashboard',
        body=body,
        thumbnail_data=thumb_json,
        has_viewer_js='true' if has_viewer else 'false',
        timestamp=timestamp,
    )

    output_path = Path(output_path)
    output_path.write_text(html, encoding='utf-8')
    logger.info(f"Prediction dashboard saved: {output_path}")
    return html


# =============================================================================
# HTML Template
# =============================================================================

_DASHBOARD_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    /* ── MAITE Design System (Dark Mode) ─────────────────────────── */
    :root {{
      --bg: oklch(0.22 0.01 250);
      --fg: oklch(0.985 0 0);
      --card: oklch(0.25 0.01 250);
      --card-fg: oklch(0.985 0 0);
      --secondary: oklch(0.29 0.01 250);
      --muted-fg: oklch(0.708 0 0);
      --border: rgba(255,255,255,0.10);
      --accent: oklch(0.488 0.243 264.376);
      --pass: #22c55e;
      --review: #eab308;
      --fail: #ef4444;
      --radius: 0.75rem;
      --shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06);
    }}
    *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html {{ font-size: 14px; }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg); color: var(--fg); line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}

    /* ── Page layout ─────────────────────────────────────────────── */
    .page {{
      max-width: 1400px; margin: 0 auto; padding: 1.5rem;
      display: flex; flex-direction: column; gap: 1.5rem;
    }}

    /* ── Card (matches MAITE card.tsx) ────────────────────────────── */
    .card {{
      background: var(--card); color: var(--card-fg);
      border: 1px solid var(--border); border-radius: var(--radius);
      box-shadow: var(--shadow);
      display: flex; flex-direction: column; gap: 1.5rem;
      padding: 1.5rem 0;
    }}
    .card-header {{
      display: flex; flex-direction: column; gap: 0.375rem;
      padding: 0 1.5rem;
    }}
    .card-title {{
      font-weight: 600; font-size: 1rem; line-height: 1;
      color: var(--fg);
    }}
    .card-description {{
      font-size: 0.857rem; color: var(--muted-fg);
    }}
    .card-content {{
      padding: 0 1.5rem;
    }}

    /* ── Stat cards grid (matches MAITE SummaryCards) ─────────────── */
    .stats-grid {{
      display: grid; gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .stat-card {{
      background: var(--secondary); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 1rem 1.25rem;
      box-shadow: var(--shadow);
    }}
    .stat-title {{
      font-size: 0.857rem; font-weight: 500; color: var(--muted-fg);
      margin-bottom: 0.25rem;
    }}
    .stat-value {{
      font-size: 1.5rem; font-weight: 700; color: var(--fg);
    }}

    /* ── Chips row ────────────────────────────────────────────────── */
    .chips-row {{
      display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;
    }}
    .chip {{
      display: inline-flex; align-items: center;
      background: var(--secondary); border: 1px solid var(--border);
      padding: 0.25rem 0.625rem; border-radius: 9999px;
      font-size: 0.786rem; font-weight: 500; color: var(--muted-fg);
      white-space: nowrap;
    }}

    /* ── Badge (matches MAITE badge.tsx) ──────────────────────────── */
    .badge {{
      display: inline-flex; align-items: center; justify-content: center;
      border-radius: 9999px; border: 1px solid transparent;
      padding: 0.125rem 0.5rem; font-size: 0.786rem; font-weight: 500;
      white-space: nowrap;
    }}
    .badge-pass {{ background: var(--pass); color: #fff; }}
    .badge-review {{ background: var(--review); color: #000; }}
    .badge-fail {{ background: var(--fail); color: #fff; }}

    /* ── Table (matches MAITE table.tsx) ──────────────────────────── */
    .table-container {{ width: 100%; overflow-x: auto; }}
    .risk-table {{
      width: 100%; border-collapse: collapse; font-size: 0.857rem;
      caption-side: bottom;
    }}
    .risk-table thead tr {{
      border-bottom: 1px solid var(--border);
    }}
    .risk-table th {{
      height: 2.5rem; padding: 0 0.5rem; text-align: left;
      font-weight: 500; color: var(--fg); white-space: nowrap;
      cursor: pointer; user-select: none; vertical-align: middle;
    }}
    .risk-table th:hover {{ color: var(--accent); }}
    .risk-table tbody tr {{
      border-bottom: 1px solid var(--border);
      transition: background-color 0.15s;
    }}
    .risk-table tbody tr:hover {{ background: var(--secondary); }}
    .risk-table td {{
      padding: 0.5rem; vertical-align: middle; white-space: nowrap;
    }}

    /* ── Plot grid (one plot per row to avoid clipping) ─────────── */
    .plot-grid {{
      display: flex; flex-direction: column; gap: 1.5rem;
    }}
    .plot-card {{
      background: var(--secondary); border: 1px solid var(--border);
      border-radius: calc(var(--radius) - 2px); padding: 0.5rem;
      overflow: hidden; width: 100%;
    }}

    /* ── Thumbnail side panel ────────────────────────────────────── */
    #thumbnail-panel {{
      position: fixed; right: -360px; top: 5rem; width: 340px;
      background: var(--card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 1.25rem;
      transition: right 0.3s ease; z-index: 1000;
      box-shadow: -4px 0 20px rgba(0,0,0,0.3); max-height: 80vh;
      overflow-y: auto;
    }}
    #thumbnail-panel.visible {{ right: 1.25rem; }}
    #thumbnail-panel h3 {{ color: var(--fg); margin-bottom: 0.5rem; font-weight: 600; }}
    #thumbnail-panel .close-btn {{
      position: absolute; top: 0.5rem; right: 0.75rem; cursor: pointer;
      color: var(--muted-fg); font-size: 1.25rem; background: none; border: none;
    }}
    #thumbnail-panel .close-btn:hover {{ color: var(--fg); }}
    #panel-thumbnail {{
      width: 100%; max-height: 300px; object-fit: contain;
      border-radius: calc(var(--radius) - 4px); margin: 0.5rem 0;
      background: #000; min-height: 80px;
    }}
    #panel-metadata {{ font-size: 0.857rem; color: var(--muted-fg); }}
    #panel-verdict-badge {{
      display: inline-flex; align-items: center; padding: 0.2rem 0.625rem;
      border-radius: 9999px; font-weight: 600; color: #fff;
      font-size: 0.857rem; margin: 0.25rem 0;
    }}
    .no-thumbnail {{ color: var(--muted-fg); font-style: italic; padding: 0.75rem 0; }}
    .viewer-btn {{
      display: inline-flex; align-items: center; gap: 0.375rem;
      background: var(--accent); color: #fff;
      padding: 0.5rem 1rem; border-radius: var(--radius);
      text-decoration: none; font-weight: 600; font-size: 0.857rem;
      margin: 0.5rem 0; transition: opacity 0.15s;
    }}
    .viewer-btn:hover {{ opacity: 0.85; }}

    /* ── Responsive ──────────────────────────────────────────────── */
    @media (max-width: 768px) {{
      .page {{ padding: 1rem; gap: 1rem; }}
      .plot-grid {{ grid-template-columns: 1fr; }}
      .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    {body}
  </div>

  <!-- Thumbnail side panel -->
  <div id="thumbnail-panel">
    <button class="close-btn" onclick="closeThumbnailPanel()">&times;</button>
    <h3>Case Details</h3>
    <div id="panel-case-id"></div>
    <div id="panel-verdict-badge"></div>
    <img id="panel-thumbnail" style="display:none" />
    <a id="panel-viewer-link" class="viewer-btn" href="#" target="_blank" style="display:none;">Open in NiiVue Viewer</a>
    <div id="panel-no-thumb" class="no-thumbnail" style="display:none">No thumbnail available</div>
    <div id="panel-metadata"></div>
  </div>

  <script>
    // Thumbnail data (base64 PNGs keyed by case index)
    var thumbnailData = {thumbnail_data};
    var hasViewer = {has_viewer_js};

    // Table sorting
    function sortTable(colIdx) {{
      var table = document.getElementById('risk-table');
      if (!table) return;
      var tbody = table.tBodies[0];
      var rows = Array.from(tbody.rows);
      var asc = table.dataset.sortCol == colIdx && table.dataset.sortDir == 'asc';
      rows.sort(function(a, b) {{
        var va = a.cells[colIdx].textContent.trim();
        var vb = b.cells[colIdx].textContent.trim();
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
      table.dataset.sortCol = colIdx;
      table.dataset.sortDir = asc ? 'desc' : 'asc';
    }}

    // Thumbnail panel
    function showThumbnailPanel(caseIdx, caseId, verdict, riskScore) {{
      var panel = document.getElementById('thumbnail-panel');
      document.getElementById('panel-case-id').textContent = 'Case: ' + caseId;
      var badge = document.getElementById('panel-verdict-badge');
      var colors = {{'pass': '#22c55e', 'review': '#eab308', 'fail': '#ef4444'}};
      badge.textContent = verdict.toUpperCase();
      badge.style.background = colors[verdict] || '#6c757d';
      badge.style.color = verdict === 'review' ? '#000' : '#fff';

      var img = document.getElementById('panel-thumbnail');
      var noThumb = document.getElementById('panel-no-thumb');
      var thumbB64 = thumbnailData[String(caseIdx)];
      if (thumbB64) {{
        img.src = 'data:image/png;base64,' + thumbB64;
        img.style.display = 'block';
        noThumb.style.display = 'none';
      }} else {{
        img.style.display = 'none';
        noThumb.style.display = 'block';
      }}

      var viewerLink = document.getElementById('panel-viewer-link');
      if (hasViewer) {{
        viewerLink.href = './viewer.html#case=' + caseIdx;
        viewerLink.style.display = 'inline-flex';
      }} else {{
        viewerLink.style.display = 'none';
      }}

      var metaEl = document.getElementById('panel-metadata');
      while (metaEl.firstChild) metaEl.removeChild(metaEl.firstChild);
      var riskLine = document.createElement('div');
      riskLine.textContent = 'Risk Score: ' + riskScore;
      metaEl.appendChild(riskLine);
      var idxLine = document.createElement('div');
      idxLine.textContent = 'Index: ' + caseIdx;
      metaEl.appendChild(idxLine);

      panel.classList.add('visible');
    }}

    function closeThumbnailPanel() {{
      document.getElementById('thumbnail-panel').classList.remove('visible');
    }}

    // Row click handler for risk table
    document.querySelectorAll('.risk-table tbody tr').forEach(function(row) {{
      row.style.cursor = 'pointer';
      row.addEventListener('click', function() {{
        var idx = this.dataset.caseIdx;
        var caseId = this.cells[0].textContent;
        var badge = this.querySelector('.badge');
        var verdict = badge ? badge.textContent.toLowerCase() : 'pass';
        var risk = this.cells[3].textContent;
        showThumbnailPanel(idx, caseId, verdict, risk);
      }});
    }});
  </script>
</body>
</html>
'''
