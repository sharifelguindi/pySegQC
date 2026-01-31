"""
NiiVue-based NIfTI viewer page generation.

Generates a standalone HTML viewer page and companion JSON sidecar
for interactive orthogonal viewing of NIfTI images with mask overlays.
The viewer uses NiiVue (WebGL2) for proper voxel spacing and orientation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_case_id(metadata_df: pd.DataFrame, index: int) -> str:
    """Extract best available case identifier from metadata."""
    for col in ('Case_ID', 'MRN', 'Patient_ID'):
        if col in metadata_df.columns:
            val = metadata_df.iloc[index].get(col)
            if val is not None and pd.notna(val):
                return str(val)
    return str(metadata_df.index[index])


def generate_viewer_data(
    output_path: Path,
    metadata_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    qa_results: Dict[str, Any],
    structure_label: Optional[int] = None,
    structure_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate viewer_data.json with case manifest for NiiVue viewer.

    Args:
        output_path: File path for the JSON output.
        metadata_df: DataFrame with case metadata (must have Image_Path/Mask_Path).
        cluster_labels: Cluster assignments per sample.
        qa_results: Output from compute_qa_verdicts().
        structure_label: Optional integer label for the structure being reviewed
            (e.g., 2 for Brainstem). Used to filter multi-class masks.
        structure_name: Optional display name (e.g., "Brainstem").

    Returns:
        The data dict that was written to JSON.
    """
    cases = []
    for i in range(len(metadata_df)):
        case = {
            'index': i,
            'case_id': _get_case_id(metadata_df, i),
            'cluster': int(cluster_labels[i]),
            'verdict': str(qa_results['verdicts'][i]),
            'risk_score': round(float(qa_results['qa_risk_scores'][i]), 4),
        }
        # Add file paths if available
        if 'Image_Path' in metadata_df.columns:
            val = metadata_df.iloc[i].get('Image_Path')
            if val is not None and pd.notna(val):
                case['image_path'] = str(Path(val).resolve())
        if 'Mask_Path' in metadata_df.columns:
            val = metadata_df.iloc[i].get('Mask_Path')
            if val is not None and pd.notna(val):
                case['mask_path'] = str(Path(val).resolve())
        cases.append(case)

    data = {'cases': cases}
    if structure_label is not None:
        data['structure_label'] = int(structure_label)
    if structure_name is not None:
        data['structure_name'] = str(structure_name)
    output_path = Path(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    logger.info(f"Viewer data saved: {output_path} ({len(cases)} cases)")
    return data


def generate_viewer_html(
    output_path: Path,
    viewer_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate NiiVue viewer HTML page.

    When viewer_data is provided the case manifest is inlined into the HTML
    so the page works when opened via file:// (no fetch needed).  A separate
    viewer_data.json is still written by generate_viewer_data() for tooling.

    Args:
        output_path: File path for the HTML output.
        viewer_data: Optional case manifest dict (output of generate_viewer_data).
            If provided, the data is inlined into the HTML template.

    Returns:
        The HTML string (also written to output_path).
    """
    inline_json = json.dumps(viewer_data, indent=2) if viewer_data else 'null'
    html = _VIEWER_TEMPLATE.replace('__INLINE_VIEWER_DATA__', inline_json)
    output_path = Path(output_path)
    output_path.write_text(html, encoding='utf-8')
    logger.info(f"NiiVue viewer saved: {output_path}")
    return html


_VIEWER_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pySegQC NiiVue Viewer</title>
<style>
  :root {
    --bg: #161618; --card: #1e1e22; --fg: #fbfbfb; --muted: #a0a0a8;
    --border: #2a2a30; --accent: #6366f1; --radius: 10px;
    --pass: #22c55e; --review: #eab308; --fail: #ef4444;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,sans-serif; }
  .layout { display:flex; height:100vh; }
  .sidebar {
    width:280px; min-width:280px; background:var(--card);
    border-right:1px solid var(--border); padding:1.25rem;
    display:flex; flex-direction:column; gap:1rem; overflow-y:auto;
  }
  .sidebar h1 { font-size:1.1rem; font-weight:700; color:var(--accent); }
  .sidebar h2 { font-size:0.95rem; font-weight:600; margin-top:0.5rem; }
  .badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:999px;
    font-size:0.75rem; font-weight:700; text-transform:uppercase;
  }
  .badge-pass { background:var(--pass); color:#fff; }
  .badge-review { background:var(--review); color:#000; }
  .badge-fail { background:var(--fail); color:#fff; }
  .meta-row { font-size:0.85rem; color:var(--muted); padding:0.15rem 0; }
  .meta-label { font-weight:600; color:var(--fg); }
  .wl-section { display:flex; flex-direction:column; gap:0.4rem; }
  .wl-section label { font-size:0.78rem; font-weight:600; color:var(--muted); }
  .wl-presets { display:flex; gap:0.35rem; flex-wrap:wrap; }
  .wl-presets button {
    padding:0.3rem 0.55rem; border:1px solid var(--border); border-radius:var(--radius);
    background:var(--card); color:var(--fg); font-size:0.72rem; font-weight:600;
    cursor:pointer; transition:background 0.15s;
  }
  .wl-presets button:hover { background:var(--border); }
  .wl-presets button.active { background:var(--accent); border-color:var(--accent); }
  .wl-inputs { display:flex; gap:0.5rem; align-items:center; }
  .wl-inputs label { font-size:0.75rem; color:var(--muted); min-width:1.5em; }
  .wl-inputs input {
    width:60px; padding:0.25rem 0.4rem; border:1px solid var(--border); border-radius:6px;
    background:var(--bg); color:var(--fg); font-size:0.8rem; text-align:center;
  }
  .nav-btns { display:flex; gap:0.5rem; margin-top:auto; }
  .nav-btns button {
    flex:1; padding:0.5rem; border:1px solid var(--border); border-radius:var(--radius);
    background:var(--card); color:var(--fg); font-weight:600; cursor:pointer;
    transition:background 0.15s;
  }
  .nav-btns button:hover { background:var(--border); }
  .nav-btns button:disabled { opacity:0.3; cursor:default; }
  .canvas-wrap { flex:1; display:flex; align-items:stretch; justify-content:center; background:#000; }
  canvas { width:100% !important; height:100% !important; }
  .help-banner {
    background:#2a2a30; border-radius:var(--radius); padding:0.75rem;
    font-size:0.78rem; color:var(--muted); line-height:1.4;
  }
  .help-banner code { background:#333; padding:0.1em 0.3em; border-radius:4px; font-size:0.85em; }
  .case-counter { font-size:0.8rem; color:var(--muted); }
  #status-msg { font-size:0.8rem; color:var(--muted); min-height:1.2em; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>pySegQC Viewer</h1>
    <div class="case-counter">Case <span id="case-num">1</span> of <span id="case-total">?</span></div>
    <h2 id="case-id">Loading...</h2>
    <div><span id="verdict-badge" class="badge">-</span></div>
    <div class="meta-row" id="meta-structure-row" style="display:none;">
      <span class="meta-label">Structure:</span> <span id="meta-structure">-</span>
    </div>
    <div class="meta-row"><span class="meta-label">Cluster:</span> <span id="meta-cluster">-</span></div>
    <div class="meta-row"><span class="meta-label">Risk Score:</span> <span id="meta-risk">-</span></div>
    <div class="wl-section">
      <label>Window / Level</label>
      <div class="wl-presets">
        <button onclick="setWL(40,80)" id="wl-brain">Brain</button>
        <button onclick="setWL(400,2000)" id="wl-bone">Bone</button>
        <button onclick="setWL(-600,1500)" id="wl-lung">Lung</button>
        <button onclick="setWL(50,350)" id="wl-pelvis">Pelvis</button>
      </div>
      <div class="wl-inputs">
        <label>L</label><input id="wl-level" type="number" value="50">
        <label>W</label><input id="wl-width" type="number" value="350">
        <button class="wl-presets" style="margin:0;" onclick="applyCustomWL()">Apply</button>
      </div>
    </div>
    <div class="wl-section">
      <label>Zoom</label>
      <div class="wl-presets">
        <button onclick="zoomOut()">&#x2212;</button>
        <button onclick="resetZoom()">1:1</button>
        <button onclick="zoomIn()">+</button>
      </div>
    </div>
    <div id="status-msg"></div>
    <div class="help-banner">
      <strong>How to use:</strong> Serve from filesystem root, then open the viewer URL:<br>
      <code>cd / &amp;&amp; python -m http.server 8080</code><br>
      Scroll to navigate slices. Use +/&#x2212; to zoom.
    </div>
    <div class="nav-btns">
      <button id="btn-prev" onclick="navigate(-1)">&#x25C0; Prev</button>
      <button id="btn-next" onclick="navigate(1)">Next &#x25B6;</button>
    </div>
  </div>
  <div class="canvas-wrap">
    <canvas id="gl-canvas"></canvas>
  </div>
</div>

<script type="module">
import { Niivue } from "https://unpkg.com/@niivue/niivue@0.44.0/dist/index.js";

// Case data — inlined at generation time, with JSON sidecar as fallback.
const INLINE_DATA = __INLINE_VIEWER_DATA__;

let viewerData = null;
let cases = [];
let currentIdx = 0;
let nv = null;

async function init() {
  const statusEl = document.getElementById('status-msg');

  // 1. Use inlined data if available, else fall back to viewer_data.json fetch
  if (INLINE_DATA && INLINE_DATA.cases) {
    viewerData = INLINE_DATA;
    cases = INLINE_DATA.cases;
  } else {
    try {
      const resp = await fetch('./viewer_data.json');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      viewerData = await resp.json();
      cases = viewerData.cases || [];
    } catch (e) {
      statusEl.textContent =
        'Could not load case data. Serve this directory: python -m http.server 8080';
      return;
    }
  }

  if (cases.length === 0) {
    statusEl.textContent = 'No cases found.';
    return;
  }

  document.getElementById('case-total').textContent = cases.length;

  // Show structure name if available
  if (viewerData.structure_name) {
    document.getElementById('meta-structure').textContent = viewerData.structure_name;
    document.getElementById('meta-structure-row').style.display = '';
  }

  // Parse starting case from URL hash: viewer.html#case=3
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const startCase = parseInt(params.get('case') || '0');
  currentIdx = cases.findIndex(c => c.index === startCase);
  if (currentIdx < 0) currentIdx = 0;

  // Initialize NiiVue — multiplanar (axial + coronal + sagittal)
  nv = new Niivue({
    backColor: [0.08, 0.08, 0.1, 1],
    show3Dcrosshair: false,
    crosshairWidth: 0,
    multiplanarForceRender: true,
    sliceType: 3,
  });
  nv.attachToCanvas(document.getElementById('gl-canvas'));

  await loadCase(currentIdx);
}

function toVolumeUrl(absPath) {
  if (!absPath) return null;
  if (absPath.startsWith('http://') || absPath.startsWith('https://')) return absPath;
  return absPath;
}

async function loadCase(idx) {
  const c = cases[idx];
  currentIdx = idx;

  // Update sidebar
  document.getElementById('case-id').textContent = c.case_id || ('Case ' + c.index);
  document.getElementById('case-num').textContent = idx + 1;
  document.getElementById('meta-cluster').textContent = c.cluster;
  document.getElementById('meta-risk').textContent = c.risk_score;

  const badge = document.getElementById('verdict-badge');
  const v = (c.verdict || 'pass').toLowerCase();
  badge.textContent = v.toUpperCase();
  badge.className = 'badge badge-' + v;

  document.getElementById('btn-prev').disabled = (idx <= 0);
  document.getElementById('btn-next').disabled = (idx >= cases.length - 1);

  // Update URL hash
  window.location.hash = 'case=' + c.index;

  // Load volumes
  const statusEl = document.getElementById('status-msg');
  const imgUrl = toVolumeUrl(c.image_path);
  if (!imgUrl) {
    statusEl.textContent = 'No image path for this case';
    return;
  }

  statusEl.textContent = 'Loading volumes...';
  try {
    const volumes = [{ url: imgUrl, colormap: 'gray' }];
    const maskUrl = toVolumeUrl(c.mask_path);
    if (maskUrl) {
      volumes.push({
        url: maskUrl,
        colormap: 'actc',
        opacity: 1.0,
      });
    }
    await nv.loadVolumes(volumes);
    // Zero out non-target voxels so only the reviewed structure is visible
    const label = viewerData && viewerData.structure_label;
    if (label != null && nv.volumes.length > 1) {
      const img = nv.volumes[1].img;
      for (let i = 0; i < img.length; i++) {
        if (Math.round(img[i]) !== label) img[i] = 0;
      }
    }
    // Convert solid mask to contour (keep only boundary voxels)
    if (nv.volumes.length > 1) {
      const vol = nv.volumes[1];
      const img = vol.img;
      const nx = vol.hdr.dims[1], ny = vol.hdr.dims[2], nz = vol.hdr.dims[3];
      if (nx * ny * nz === img.length) {
        const src = img.slice();
        for (let z = 0; z < nz; z++) {
          for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
              const idx = x + y * nx + z * nx * ny;
              if (src[idx] === 0) continue;
              const v = src[idx];
              const interior =
                (x > 0 && src[idx-1] === v) &&
                (x < nx-1 && src[idx+1] === v) &&
                (y > 0 && src[idx-nx] === v) &&
                (y < ny-1 && src[idx+nx] === v) &&
                (z > 0 && src[idx-nx*ny] === v) &&
                (z < nz-1 && src[idx+nx*ny] === v);
              if (interior) img[idx] = 0;
            }
          }
        }
      }
      nv.updateGLVolume();
    }
    // Preserve W/L across case navigation
    const wlL = parseFloat(document.getElementById('wl-level').value);
    const wlW = parseFloat(document.getElementById('wl-width').value);
    if (!isNaN(wlL) && !isNaN(wlW)) setWL(wlL, wlW);
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = 'Load failed — try: cd / && python -m http.server 8080';
  }
}

window.navigate = function(delta) {
  const next = currentIdx + delta;
  if (next >= 0 && next < cases.length) {
    loadCase(next);
  }
};

window.setWL = function(level, width) {
  if (!nv || nv.volumes.length === 0) return;
  const vol = nv.volumes[0];
  vol.cal_min = level - width / 2;
  vol.cal_max = level + width / 2;
  nv.updateGLVolume();
  document.getElementById('wl-level').value = level;
  document.getElementById('wl-width').value = width;
  document.querySelectorAll('.wl-presets button').forEach(b => b.classList.remove('active'));
};

window.applyCustomWL = function() {
  const level = parseFloat(document.getElementById('wl-level').value);
  const width = parseFloat(document.getElementById('wl-width').value);
  if (!isNaN(level) && !isNaN(width)) setWL(level, width);
};

window.zoomIn = function() {
  if (!nv) return;
  nv.scene.volScaleMultiplier = (nv.scene.volScaleMultiplier || 1) * 1.4;
  nv.drawScene();
};
window.zoomOut = function() {
  if (!nv) return;
  nv.scene.volScaleMultiplier = Math.max(0.5, (nv.scene.volScaleMultiplier || 1) / 1.4);
  nv.drawScene();
};
window.resetZoom = function() {
  if (!nv) return;
  nv.scene.volScaleMultiplier = 1.0;
  nv.drawScene();
};

init();
</script>
</body>
</html>
'''
