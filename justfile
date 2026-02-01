# pySegQC development task runner
# Run `just --list` to see all available recipes

# Default: show available recipes
default:
    @just --list

# ─── Quality & CI ──────────────────────────────────────────

# Run all pre-deploy checks (lint, format check, type check, tests)
pre-deploy: lint typecheck test
    @echo "\n✓ All pre-deploy checks passed"

# Lint with ruff (fast Python linter)
lint:
    ruff check src/ tests/

# Auto-fix lint issues
lint-fix:
    ruff check src/ tests/ --fix

# Check formatting (ruff format in check mode)
fmt-check:
    ruff format src/ tests/ --check

# Auto-format code
fmt:
    ruff format src/ tests/

# Type check with mypy
typecheck:
    mypy src/

# ─── Testing ───────────────────────────────────────────────

# Run unit tests (excludes slow and e2e)
test:
    python -m pytest tests/ -x

# Run unit tests with verbose output
test-v:
    python -m pytest tests/ -x -v

# Run e2e browser tests (requires playwright)
test-e2e:
    python -m pytest tests/test_viewer_e2e.py -m e2e -v

# Run all tests including slow NIfTI tests
test-all:
    python -m pytest tests/ -m 'not e2e' --timeout=600

# Run tests + e2e (everything)
test-everything:
    python -m pytest tests/ -m 'not slow' -v
    python -m pytest tests/test_viewer_e2e.py -m e2e -v

# Run a single test file
test-file file:
    python -m pytest {{file}} -v

# ─── Scratch / Test Data Runs ──────────────────────────────

# Directory where test data lives and results are written
scratch_dir := "scratch"
serve_port := "8080"

# Clear all generated results from scratch (preserves raw data)
scratch-clean:
    rm -rf {{scratch_dir}}/*_clustering_results
    rm -rf {{scratch_dir}}/*_results
    rm -rf {{scratch_dir}}/viewer_test_results
    rm -f  {{scratch_dir}}/*.xlsx
    rm -f  {{scratch_dir}}/.DS_Store
    @echo "Cleaned scratch results (raw data preserved in test_data/, train/, test/)"

# Clear everything in scratch including raw data
scratch-nuke:
    rm -rf {{scratch_dir}}/*
    @echo "Nuked scratch/ completely"

# Extract features from train data
scratch-extract:
    pysegqc extract {{scratch_dir}}/train/ \
        --image-dir image --mask-dir mask \
        --output {{scratch_dir}}/train_features.xlsx \
        --n-jobs 4

# Run full analysis on train features (per-structure mode, volume-independent)
scratch-analyze: scratch-extract
    pysegqc analyze {{scratch_dir}}/train_features.xlsx \
        --auto-k --volume-independent \
        --select-training-cases
    @just auto-serve {{scratch_dir}}/train_features_clustering_results

# Run analysis on existing features (skip extraction)
scratch-analyze-only:
    pysegqc analyze {{scratch_dir}}/train_features.xlsx \
        --auto-k --volume-independent \
        --select-training-cases
    @just auto-serve {{scratch_dir}}/train_features_clustering_results

# Run predict on test data using trained models
scratch-predict:
    pysegqc predict {{scratch_dir}}/test/ \
        {{scratch_dir}}/train_features_clustering_results/ \
        --image-dir image --mask-dir mask
    @just auto-serve {{scratch_dir}}/train_features_clustering_results/_nifti_prediction_features_predictions

# Kill any existing server on the serve port
[private]
kill-server:
    -lsof -ti:{{serve_port}} | xargs kill 2>/dev/null; true

# Start background HTTP server and open dashboard in browser
[private]
auto-serve results_dir:
    #!/usr/bin/env bash
    just kill-server
    echo -e "\n🌐 Starting viewer server on port {{serve_port}}..."
    # Serve from scratch/ so relative NIfTI paths (../../) resolve for nested results
    serve_root="{{scratch_dir}}"
    subpath=$(python3 -c "import os; print(os.path.relpath('{{results_dir}}', '${serve_root}'))")
    nohup python -m http.server {{serve_port}} --directory "${serve_root}" > /dev/null 2>&1 &
    sleep 0.5
    base="http://localhost:{{serve_port}}/${subpath}"
    if [ -f "{{results_dir}}/analysis_dashboard.html" ]; then
      page="analysis_dashboard.html"
    elif [ -f "{{results_dir}}/prediction_dashboard.html" ]; then
      page="prediction_dashboard.html"
    else
      page="viewer.html"
    fi
    echo "   Dashboard: ${base}/${page}"
    open "${base}/${page}" 2>/dev/null || true

# Serve results directory and open dashboard (foreground, default: latest scratch results)
serve results_dir=(scratch_dir + "/train_features_clustering_results"):
    #!/usr/bin/env bash
    just kill-server
    # Serve from scratch/ so relative NIfTI paths (../../) resolve for nested results
    serve_root="{{scratch_dir}}"
    subpath=$(python3 -c "import os; print(os.path.relpath('{{results_dir}}', '${serve_root}'))")
    base="http://localhost:{{serve_port}}/${subpath}"
    if [ -f "{{results_dir}}/analysis_dashboard.html" ]; then
      page="analysis_dashboard.html"
    elif [ -f "{{results_dir}}/prediction_dashboard.html" ]; then
      page="prediction_dashboard.html"
    else
      page="viewer.html"
    fi
    echo "Serving from ${serve_root} so NIfTI relative paths resolve"
    echo "Dashboard: ${base}/${page}"
    open "${base}/${page}" 2>/dev/null || true
    python -m http.server {{serve_port}} --directory "${serve_root}"

# Stop the background viewer server
serve-stop:
    @just kill-server
    @echo "✓ Server stopped"

# Full scratch pipeline: clean → extract → analyze → serve
scratch-run: scratch-clean scratch-analyze
    @echo "\n✓ Full scratch pipeline complete (viewer server running on port {{serve_port}})"

# ─── Setup ─────────────────────────────────────────────────

# Install package in editable mode with dev dependencies
install:
    DISABLE_CUDA_EXTENSIONS=1 pip install -e ".[dev]"

# Install + playwright browser
install-all: install
    pip install pytest-playwright
    playwright install chromium

# Show current package version
version:
    python -c "import pysegqc; print(pysegqc.__version__)"
