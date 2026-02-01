"""
Playwright e2e tests for the NiiVue viewer page.

These tests verify DOM-level behaviour (sidebar, navigation, W/L controls,
keyboard shortcuts, responsive layout) using synthetic inline data.
NiiVue/WebGL volume rendering may not work in headless Chromium, so tests
assert on sidebar DOM state rather than canvas pixels.

Run with:
    pytest tests/test_viewer_e2e.py -m e2e
"""

import http.server
import threading

import pytest

from pysegqc.viewer import generate_viewer_html

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_VIEWER_DATA = {
    "structure_name": "Brainstem",
    "structure_label": 2,
    "cases": [
        {
            "index": 0,
            "case_id": "PAT001",
            "cluster": 0,
            "verdict": "pass",
            "risk_score": 0.05,
            "image_path": "/fake/pat001.nii.gz",
            "mask_path": "/fake/pat001_mask.nii.gz",
        },
        {
            "index": 1,
            "case_id": "PAT002",
            "cluster": 1,
            "verdict": "review",
            "risk_score": 0.45,
            "image_path": "/fake/pat002.nii.gz",
            "mask_path": "/fake/pat002_mask.nii.gz",
        },
        {
            "index": 2,
            "case_id": "PAT003",
            "cluster": 2,
            "verdict": "fail",
            "risk_score": 0.92,
            "image_path": "/fake/pat003.nii.gz",
            "mask_path": "/fake/pat003_mask.nii.gz",
        },
    ],
}


@pytest.fixture(scope="module")
def viewer_server(tmp_path_factory):
    """Generate viewer HTML and serve it via HTTP on a random port."""
    tmp = tmp_path_factory.mktemp("viewer")
    html_path = tmp / "viewer.html"
    generate_viewer_html(html_path, viewer_data=SYNTHETIC_VIEWER_DATA)

    # Serve the directory
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    httpd.directory = str(tmp)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp), **kw)

        def log_message(self, *a):
            pass  # silence request logs during tests

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}/viewer.html"

    httpd.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_viewer_loads(page, viewer_server):
    """Page loads with correct title and sidebar visible."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    assert page.title() == "pySegQC NiiVue Viewer"
    assert page.locator(".sidebar").is_visible()


@pytest.mark.e2e
def test_case_counter_display(page, viewer_server):
    """Shows 'Case 1 of 3' and first case ID after init."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    # Wait for JS init to populate the counter
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    assert page.locator("#case-num").text_content() == "1"
    assert page.locator("#case-total").text_content() == "3"
    assert page.locator("#case-id").text_content() == "PAT001"


@pytest.mark.e2e
def test_verdict_badge_styling(page, viewer_server):
    """Badge shows correct verdict text and CSS class for first case (pass)."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    badge = page.locator("#verdict-badge")
    assert badge.text_content() == "PASS"
    assert "badge-pass" in badge.get_attribute("class")


@pytest.mark.e2e
def test_structure_name_displayed(page, viewer_server):
    """Structure name 'Brainstem' is shown when provided in viewer data."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    row = page.locator("#meta-structure-row")
    assert row.is_visible()
    assert page.locator("#meta-structure").text_content() == "Brainstem"


@pytest.mark.e2e
def test_next_prev_navigation(page, viewer_server):
    """Click Next -> case 2, click Prev -> back to case 1."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    # Navigate forward
    page.locator("#btn-next").click()
    page.wait_for_function("document.getElementById('case-num').textContent === '2'", timeout=5_000)
    assert page.locator("#case-id").text_content() == "PAT002"
    assert page.locator("#verdict-badge").text_content() == "REVIEW"

    # Navigate back
    page.locator("#btn-prev").click()
    page.wait_for_function("document.getElementById('case-num').textContent === '1'", timeout=5_000)
    assert page.locator("#case-id").text_content() == "PAT001"


@pytest.mark.e2e
def test_keyboard_navigation(page, viewer_server):
    """ArrowRight advances case, ArrowLeft goes back."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    # Arrow right -> case 2
    page.keyboard.press("ArrowRight")
    page.wait_for_function("document.getElementById('case-num').textContent === '2'", timeout=5_000)
    assert page.locator("#case-id").text_content() == "PAT002"

    # Arrow left -> case 1
    page.keyboard.press("ArrowLeft")
    page.wait_for_function("document.getElementById('case-num').textContent === '1'", timeout=5_000)
    assert page.locator("#case-id").text_content() == "PAT001"


@pytest.mark.e2e
def test_wl_preset_buttons(page, viewer_server):
    """Click 'Soft Tissue' -> input fields update to L=50, W=350."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    page.locator("#wl-soft-tissue").click()
    assert page.locator("#wl-level").input_value() == "50"
    assert page.locator("#wl-width").input_value() == "350"


@pytest.mark.e2e
def test_custom_wl_input(page, viewer_server):
    """Type custom L/W values, click Apply, verify inputs persist."""
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    page.locator("#wl-level").fill("100")
    page.locator("#wl-width").fill("500")
    page.locator("button", has_text="Apply").click()
    assert page.locator("#wl-level").input_value() == "100"
    assert page.locator("#wl-width").input_value() == "500"


@pytest.mark.e2e
def test_url_hash_navigation(page, viewer_server):
    """Navigate to viewer.html#case=2 -> loads case at index 2."""
    page.goto(viewer_server + "#case=2")
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    # Case index 2 is the third case (PAT003)
    assert page.locator("#case-id").text_content() == "PAT003"
    assert page.locator("#case-num").text_content() == "3"


@pytest.mark.e2e
def test_responsive_layout(page, viewer_server):
    """Resize viewport to 600px wide -> sidebar stacks above canvas."""
    page.set_viewport_size({"width": 600, "height": 800})
    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")

    sidebar = page.locator(".sidebar")
    assert sidebar.is_visible()
    # In responsive mode, sidebar should have no right border (flex-direction: column)
    # and the layout should stack vertically. Verify sidebar width matches viewport.
    box = sidebar.bounding_box()
    assert box is not None
    assert box["width"] >= 580  # should be ~full viewport width (600px minus minor margins)


@pytest.mark.e2e
def test_zoom_buttons_no_errors(page, viewer_server):
    """Click zoom in/out/reset buttons -> no JS errors on page."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(viewer_server)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_function("document.getElementById('case-total').textContent !== '?'", timeout=10_000)

    # Click zoom buttons (they may not visually work without WebGL, but should not throw)
    page.locator("button", has_text="+").click()
    page.locator("button", has_text="1:1").click()
    page.locator("button", has_text="\u2212").click()

    # Filter out NiiVue/WebGL errors (expected in headless mode)
    real_errors = [e for e in errors if "WebGL" not in e and "niivue" not in e.lower()]
    assert len(real_errors) == 0, f"Unexpected JS errors: {real_errors}"
