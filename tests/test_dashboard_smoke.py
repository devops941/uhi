"""Dashboard smoke tests.

The spec asks for a smoke test that the dashboard imports and runs. We go a
step further and drive every page through Streamlit's AppTest so a broken page
fails CI instead of failing silently in the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "app" / "dashboard.py"

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(page: str, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(str(DASHBOARD), default_timeout=timeout)
    at.run()
    # the sidebar radio picks the page; index 0 is Overview
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.fixture(scope="module")
def dashboard_artefacts():
    """Make sure the default-location pipeline artefacts exist for the app."""
    from src.config import load_config
    from src.pipeline import run_pipeline

    cfg = load_config()
    if not (cfg.path_for("outputs") / "report.md").exists():
        run_pipeline(cfg, source="synthetic")
    return cfg


def test_dashboard_module_imports():
    import importlib.util

    spec = importlib.util.spec_from_file_location("uhi_dashboard", DASHBOARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules["uhi_dashboard"] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "page_overview")
    assert hasattr(module, "page_priority")


@pytest.mark.parametrize(
    "page",
    [
        "Overview",
        "Layers",
        "Visual Gallery",
        "Relationships",
        "Interpolation",
        "Built-up vs Green",
        "Priority",
        "Upload Data",
        "Download",
    ],
)
def test_every_page_renders_without_error(page, dashboard_artefacts):
    at = _run(page)
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
