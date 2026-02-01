"""
NiiVue-based NIfTI viewer page generation.

Generates a standalone HTML viewer page and companion JSON sidecar
for interactive orthogonal viewing of NIfTI images with mask overlays.
The viewer uses NiiVue (WebGL2) for proper voxel spacing and orientation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import get_case_id as _get_case_id

logger = logging.getLogger(__name__)


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
    output_dir = str(Path(output_path).parent.resolve())
    cases = []
    for i in range(len(metadata_df)):
        case = {
            'index': i,
            'case_id': _get_case_id(metadata_df, i),
            'cluster': int(cluster_labels[i]),
            'verdict': str(qa_results['verdicts'][i]),
            'risk_score': round(float(qa_results['qa_risk_scores'][i]), 4),
        }
        # Add file paths relative to output directory (for HTTP serving)
        if 'Image_Path' in metadata_df.columns:
            val = metadata_df.iloc[i].get('Image_Path')
            if val is not None and pd.notna(val):
                case['image_path'] = os.path.relpath(
                    str(Path(val).resolve()), str(output_dir)
                )
        if 'Mask_Path' in metadata_df.columns:
            val = metadata_df.iloc[i].get('Mask_Path')
            if val is not None and pd.notna(val):
                case['mask_path'] = os.path.relpath(
                    str(Path(val).resolve()), str(output_dir)
                )
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
    output_path = Path(output_path)
    resolved_dir = output_path.parent.resolve()
    serve_dir = resolved_dir.parent  # serve from parent so relative paths resolve
    viewer_subpath = resolved_dir.name + '/viewer.html'
    html = _VIEWER_TEMPLATE.replace('__INLINE_VIEWER_DATA__', inline_json)
    html = html.replace('__SERVE_DIR__', str(serve_dir))
    html = html.replace('__VIEWER_SUBPATH__', viewer_subpath)
    output_path.write_text(html, encoding='utf-8')
    logger.info(f"NiiVue viewer saved: {output_path}")
    return html


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
# The loading spinner is implemented via a CSS pseudo-element on #status-msg
# when the .loading class is applied, avoiding any innerHTML usage.
# ---------------------------------------------------------------------------

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
  .help-banner kbd {
    background:#333; padding:0.1em 0.35em; border-radius:4px; font-size:0.85em;
    border:1px solid #444; font-family:inherit;
  }
  .case-counter { font-size:0.8rem; color:var(--muted); }
  #status-msg { font-size:0.8rem; color:var(--muted); min-height:1.2em; }
  #status-msg.loading::before {
    content:''; display:inline-block; width:14px; height:14px;
    border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%;
    animation:spin 0.6s linear infinite; vertical-align:middle; margin-right:0.4rem;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
  @media (max-width: 700px) {
    .layout { flex-direction:column; }
    .sidebar { width:100%; min-width:0; max-height:40vh; border-right:none; border-bottom:1px solid var(--border); }
  }
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
        <button onclick="setWL(30,400)" id="wl-soft-tissue">Soft Tissue</button>
      </div>
      <div class="wl-inputs">
        <label>L</label><input id="wl-level" type="number" value="30">
        <label>W</label><input id="wl-width" type="number" value="400">
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
      <strong>Keyboard:</strong> <kbd>&#x2190;</kbd><kbd>&#x2192;</kbd> navigate cases.
      <kbd>1</kbd>&#x2013;<kbd>4</kbd> W/L presets. Scroll to change slices.<br>
      <strong>Serve:</strong> <code>cd __SERVE_DIR__ &amp;&amp; python -m http.server 8080</code>
      then open <code>http://localhost:8080/__VIEWER_SUBPATH__</code>
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
const INLINE_DATA = __INLINE_VIEWER_DATA__;

let viewerData = null;
let cases = [];
let currentIdx = 0;
let nv = null;

function setStatus(text, loading) {
  const el = document.getElementById('status-msg');
  el.textContent = text;
  el.classList.toggle('loading', !!loading);
}

async function init() {
  // Load NiiVue from CDN with fallback
  let Niivue;
  try {
    ({ Niivue } = await import("https://cdn.jsdelivr.net/npm/@niivue/niivue@0.65.0/dist/index.js"));
  } catch {
    try {
      ({ Niivue } = await import("https://unpkg.com/@niivue/niivue@0.65.0/dist/index.js"));
    } catch {
      setStatus('Failed to load NiiVue library. Check your internet connection.');
      return;
    }
  }

  // Use inlined data if available, else fall back to viewer_data.json fetch
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
      setStatus('Could not load case data. Serve this directory: python -m http.server 8080');
      return;
    }
  }

  if (cases.length === 0) {
    setStatus('No cases found.');
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

  // Initialize NiiVue — multiplanar 2x2 grid (axial + coronal + sagittal + render)
  nv = new Niivue({
    backColor: [0.08, 0.08, 0.1, 1],
    show3Dcrosshair: false,
    crosshairWidth: 0,
    multiplanarForceRender: true,
    multiplanarLayout: 2,
    yoke3Dto2DZoom: true,
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
  const imgUrl = toVolumeUrl(c.image_path);
  if (!imgUrl) {
    setStatus('No image path for this case');
    return;
  }

  setStatus('Loading volumes\u2026', true);
  try {
    const volumes = [{ url: imgUrl, colormap: 'gray' }];
    const maskUrl = toVolumeUrl(c.mask_path);
    if (maskUrl) {
      volumes.push({
        url: maskUrl,
        colormap: 'red',
        opacity: 1.0,
      });
    }
    await nv.loadVolumes(volumes);
    // Binarize mask to target structure, then extract 3D boundary (contour)
    const label = viewerData && viewerData.structure_label;
    if (nv.volumes.length > 1) {
      const img = nv.volumes[1].img;
      const hdr = nv.volumes[1].hdr;
      const nx = hdr.dims[1], ny = hdr.dims[2], nz = hdr.dims[3];
      const nxy = nx * ny;
      // Binarize: keep only the target label
      for (let i = 0; i < img.length; i++) {
        img[i] = (label != null && Math.round(img[i]) === label) ? 1 : 0;
      }
      // 3D boundary detection: keep voxels with at least one zero face-neighbor
      const keep = new Uint8Array(img.length);
      for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            const i = x + y * nx + z * nxy;
            if (img[i] === 0) continue;
            if (x === 0 || x === nx-1 || y === 0 || y === ny-1 || z === 0 || z === nz-1 ||
                img[i-1] === 0 || img[i+1] === 0 ||
                img[i-nx] === 0 || img[i+nx] === 0 ||
                img[i-nxy] === 0 || img[i+nxy] === 0) {
              keep[i] = 1;
            }
          }
        }
      }
      for (let i = 0; i < img.length; i++) img[i] = keep[i];
      // Fix colormap range so value 1 maps to full brightness
      nv.volumes[1].cal_min = 0;
      nv.volumes[1].cal_max = 1;
    }
    nv.updateGLVolume();
    // Preserve W/L across case navigation
    const wlL = parseFloat(document.getElementById('wl-level').value);
    const wlW = parseFloat(document.getElementById('wl-width').value);
    if (!isNaN(wlL) && !isNaN(wlW)) setWL(wlL, wlW);
    setStatus('');
  } catch (e) {
    setStatus('Load failed \u2014 serve this directory with: python -m http.server 8080');
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

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'ArrowLeft':  navigate(-1); break;
    case 'ArrowRight': navigate(1); break;
    case '1': setWL(40, 80); break;
    case '2': setWL(400, 2000); break;
    case '3': setWL(-600, 1500); break;
    case '4': setWL(30, 400); break;
  }
});

try { await init(); } catch (e) {
  setStatus('Viewer initialization failed: ' + e.message);
}
</script>
</body>
</html>
'''
