"""Tests for storage, reporting, figures and the end-to-end pipeline."""

from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest

from src.data import db as db_mod


# ----------------------------------------------------------------------
# database
# ----------------------------------------------------------------------
def test_init_db_creates_every_table(fast_config):
    engine = db_mod.init_db(fast_config)
    names = set(db_mod.table_names(engine))
    for table in [
        "wards",
        "samples",
        "sample_features",
        "grid_cells",
        "correlation_results",
        "regression_results",
        "model_summary",
        "idw_cv_results",
        "ward_summary",
        "group_comparison",
    ]:
        assert table in names, table


def test_init_db_is_idempotent(fast_config):
    db_mod.init_db(fast_config)
    engine = db_mod.init_db(fast_config)  # must not raise
    assert "samples" in db_mod.table_names(engine)


def test_write_and_read_round_trip(fast_config):
    engine = db_mod.init_db(fast_config)
    df = pd.DataFrame({"metric": ["a", "b"], "value": [1.0, 2.0]})
    db_mod.write_table(engine, "model_summary", df)
    back = db_mod.read_table(engine, "model_summary")
    assert back["metric"].tolist() == ["a", "b"]


def test_write_table_ignores_empty_frames(fast_config):
    engine = db_mod.init_db(fast_config)
    db_mod.write_table(engine, "model_summary", pd.DataFrame())
    assert "model_summary" in db_mod.table_names(engine)


def test_summarise_db_counts_rows(fast_config):
    engine = db_mod.init_db(fast_config)
    db_mod.write_table(engine, "model_summary", pd.DataFrame({"metric": ["x"], "value": [1.0]}))
    counts = db_mod.summarise_db(engine)
    assert counts["model_summary"] == 1


def test_gpkg_layers_lists_written_layers(fast_config, prepared):
    import geopandas as gpd

    db_mod.write_layers(fast_config, {"boundary": prepared.boundary.to_crs("EPSG:4326")})
    layers = db_mod.gpkg_layers(fast_config.path_for("geopackage"))
    assert "boundary" in layers
    back = db_mod.read_layer(fast_config.path_for("geopackage"), "boundary")
    assert isinstance(back, gpd.GeoDataFrame)
    assert len(back) == 1


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_dotted_lookup(base_config):
    assert base_config.get("synthetic.n_samples") == 200
    assert base_config.get("does.not.exist", "fallback") == "fallback"
    assert "project.city" in base_config


def test_config_with_overrides_is_isolated(base_config):
    cfg = base_config.with_overrides({"synthetic.n_samples": 5})
    assert cfg.get("synthetic.n_samples") == 5
    assert base_config.get("synthetic.n_samples") == 200


def test_config_missing_file_raises():
    from src.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def test_report_renders_with_key_sections(fast_config):
    from src.report import generate as report_mod

    results = {
        "source": "synthetic",
        "crs_metric": "EPSG:32643",
        "n_samples": 10,
        "n_grid_cells": 4,
        "grid_cell_m": 100.0,
        "mean_temp": 33.0,
        "n_hotspots": 2,
        "n_coldspots": 1,
        "hotspot_pct": 50.0,
        "selected_power": 2.0,
        "idw_rmse": 1.1,
        "correlations": pd.DataFrame(
            {
                "variable": ["ndvi_mean"],
                "pearson_r": [-0.8],
                "pearson_p": [0.0],
                "spearman_rho": [-0.8],
                "spearman_p": [0.0],
                "n": [10],
            }
        ),
        "regression": pd.DataFrame(
            {
                "term": ["const", "ndvi_mean"],
                "coef": [35.0, -3.0],
                "std_err": [1.0, 0.5],
                "p_value": [0.0, 0.01],
                "ci_low": [33.0, -4.0],
                "ci_high": [37.0, -2.0],
                "vif": [float("nan"), 2.0],
            }
        ),
        "standardised": pd.DataFrame(
            {"term": ["ndvi_mean"], "std_coef": [-0.5], "p_value": [0.01]}
        ),
        "model_summary": {
            "r2": 0.7,
            "adj_r2": 0.68,
            "max_vif": 2.0,
            "moran_i_residuals": 0.2,
            "ndvi_uplift": 0.1,
            "cooling_estimate_c": 0.3,
        },
        "group_comparison": pd.DataFrame(
            {
                "group_a": ["built_up"],
                "group_b": ["green"],
                "mean_a": [38.0],
                "mean_b": [34.0],
                "diff": [4.0],
                "welch_p": [0.001],
                "mannwhitney_p": [0.002],
                "cohens_d": [2.0],
            }
        ),
        "cv_table": pd.DataFrame(
            {
                "power": [1.0, 2.0],
                "rmse": [1.2, 1.1],
                "mae": [0.9, 0.8],
                "r2": [0.5, 0.6],
                "is_selected": [0, 1],
            }
        ),
        "uhi_stats": {
            "rural_baseline_c": 30.0,
            "max_uhi_c": 5.0,
            "mean_uhi_c": 1.0,
            "area_above_2c_km2": 3.0,
            "pct_above_2c": 12.0,
            "area_above_4c_km2": 0.5,
            "pct_above_4c": 2.0,
        },
        "ward_summary": pd.DataFrame(
            {"ward_id": ["W01"], "mean_temp": [36.0], "priority_score": [0.7], "priority_rank": [1]}
        ),
        "top_cells": pd.DataFrame({"cell_id": [1], "temp_idw": [38.0], "priority_score": [0.9]}),
        "priority_weights": {"temp": 0.4},
        "figures": ["temperature_idw.png"],
    }
    md = report_mod.render_markdown(fast_config, results)
    for section in [
        "Headline results",
        "Correlation",
        "OLS regression",
        "UHI intensity",
        "Tree-planting priority",
        "Future work",
    ]:
        assert section in md
    assert "Coimbatore" in md


def test_report_writes_file(fast_config, tmp_path):
    from src.report import generate as report_mod

    results = {"source": "synthetic", "correlations": pd.DataFrame(), "regression": pd.DataFrame()}
    out = report_mod.write_report(fast_config, results, out_path=tmp_path / "r.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("#")


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def test_static_figures_render(tmp_path):
    import numpy as np

    from src.viz import maps as viz

    arr = np.random.default_rng(0).uniform(25, 40, size=(20, 20))
    p = viz.plot_surface(arr, "t", tmp_path / "s.png")
    assert p.exists() and p.stat().st_size > 0

    joined = pd.DataFrame(
        {
            "temp_c": np.random.default_rng(1).uniform(25, 40, 60),
            "ndvi_mean": np.random.default_rng(2).uniform(0, 1, 60),
            "builtup_frac": np.random.default_rng(3).uniform(0, 1, 60),
            "dist_water_m": np.random.default_rng(4).uniform(0, 2000, 60),
        }
    )
    assert viz.plot_scatter_grid(joined, tmp_path / "sc.png").exists()

    cv = pd.DataFrame(
        {
            "power": [1.0, 2.0, 3.0],
            "rmse": [1.2, 1.1, 1.3],
            "mae": [0.9, 0.8, 1.0],
            "is_selected": [0, 1, 0],
        }
    )
    assert viz.plot_idw_cv(cv, tmp_path / "cv.png").exists()


def test_folium_map_builds(prepared):
    from src.viz import maps as viz

    grid = prepared.grid.copy()
    grid["temp_idw"] = 30.0
    grid["uhi_intensity"] = 1.0
    grid["is_hotspot"] = 0
    grid["is_coldspot"] = 0
    grid["ndvi_mean"] = 0.5
    grid["builtup_frac"] = 0.3
    grid["priority_score"] = 0.5
    grid["landcover_class"] = "tree_cover"
    fmap = viz.build_folium_map(
        grid,
        samples=prepared.samples,
        boundary=prepared.boundary,
        layers=["temperature", "ndvi", "builtup", "hotspots", "priority"],
    )
    html = fmap._repr_html_()
    assert "folium" in html.lower()
    assert len(html) > 500


# ----------------------------------------------------------------------
# pipeline smoke test
# ----------------------------------------------------------------------
def test_full_pipeline_runs_on_synthetic(fast_config):
    from src.pipeline import run_pipeline

    results = run_pipeline(fast_config, source="synthetic")
    assert results["n_samples"] > 0
    assert results["n_grid_cells"] > 0
    assert results["model_summary"]["r2"] > 0.4
    assert results["selected_power"] in fast_config.get("idw.power_candidates")
    assert results["report_path"].exists()

    # artefacts must exist
    out = fast_config.path_for("outputs")
    for name in [
        "temperature_idw.tif",
        "uhi_intensity.tif",
        "hotspots.tif",
        "priority_score.tif",
        "report.md",
    ]:
        assert (out / name).exists(), name
    assert fast_config.path_for("geopackage").exists()

    engine = db_mod.get_engine(fast_config)
    counts = db_mod.summarise_db(engine)
    assert counts["grid_cells"] > 0
    assert counts["samples"] > 0
    assert counts["correlation_results"] >= 3


def test_pipeline_cli_help_runs(capsys):
    from src.pipeline import main

    with pytest.raises(SystemExit):
        main(["--help"])


def test_dashboard_module_imports():
    """Smoke test: the dashboard module must import without a Streamlit runtime."""
    sys.modules.pop("app.dashboard", None)
    mod = importlib.import_module("app.dashboard")
    assert hasattr(mod, "main")
    assert hasattr(mod, "page_overview")


def test_dashboard_theme_helpers_import():
    mod = importlib.import_module("app.theme")
    assert hasattr(mod, "inject_theme")
    assert mod.VIOLET.startswith("#")
