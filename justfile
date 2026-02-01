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

# Run full analysis on train features (concat mode, volume-independent)
scratch-analyze: scratch-extract
    pysegqc analyze {{scratch_dir}}/train_features.xlsx \
        --mode concat --auto-k --volume-independent \
        --select-training-cases

# Run analysis on existing features (skip extraction)
scratch-analyze-only:
    pysegqc analyze {{scratch_dir}}/train_features.xlsx \
        --mode concat --auto-k --volume-independent \
        --select-training-cases

# Run predict on test data using trained models
scratch-predict:
    pysegqc predict {{scratch_dir}}/test/ \
        {{scratch_dir}}/train_features_clustering_results/ \
        --image-dir image --mask-dir mask

# Full scratch pipeline: clean → extract → analyze
scratch-run: scratch-clean scratch-analyze
    @echo "\n✓ Full scratch pipeline complete"

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
