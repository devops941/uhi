"""Tests for UHI intensity, Gi* hotspots and priority scoring."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src.analysis import priority, uhi


# ----------------------------------------------------------------------
# a small hand-made grid: hot built-up core, warm mid-ring, cool green park
# ----------------------------------------------------------------------
def _hand_grid(n: int = 6, cell: float = 100.0):
    polys, temps, built, ndvi, lc = [], [], [], [], []
    centre = (n - 1) / 2
    park = {(0, 0), (0, 1), (1, 0), (1, 1)}  # a cold green corner
    for i in range(n):
        for j in range(n):
            polys.append(box(j * cell, i * cell, (j + 1) * cell, (i + 1) * cell))
            dist = max(abs(i - centre), abs(j - centre))
            if (i, j) in park:
                temps.append(27.0)
                built.append(0.02)
                ndvi.append(0.8)
                lc.append("tree_cover")
            elif dist <= 1:
                temps.append(40.0)
                built.append(0.9)
                ndvi.append(0.1)
                lc.append("built_up")
            else:
                temps.append(34.0)
                built.append(0.2)
                ndvi.append(0.4)
                lc.append("grassland")
    grid = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(n * n),
            "centroid_x": [p.centroid.x for p in polys],
            "centroid_y": [p.centroid.y for p in polys],
            "temp_idw": temps,
            "builtup_frac": built,
            "ndvi_mean": ndvi,
            "landcover_class": lc,
            "ward_id": ["W1"] * (n * n),
        },
        geometry=polys,
        crs="EPSG:32643",
    )
    return grid


# ----------------------------------------------------------------------
# UHI intensity
# ----------------------------------------------------------------------
def test_uhi_baseline_is_mean_of_rural_cells(fast_config):
    grid = _hand_grid()
    res = uhi.compute_uhi(grid, fast_config)
    assert res.stats["rural_baseline_c"] < grid["temp_idw"].mean()
    assert res.stats["max_uhi_c"] > 0
    assert len(res.rural_reference_cells) > 0


def test_uhi_intensity_is_zero_for_baseline(fast_config):
    grid = _hand_grid()
    res = uhi.compute_uhi(grid, fast_config)
    rural = res.grid.iloc[res.rural_reference_cells]
    baseline = res.stats["rural_baseline_c"]
    np.testing.assert_allclose(rural["uhi_intensity"] + baseline, rural["temp_idw"], atol=1e-9)


def test_uhi_threshold_areas_are_reported(fast_config):
    grid = _hand_grid()
    res = uhi.compute_uhi(grid, fast_config)
    assert "area_above_2c_km2" in res.stats
    assert "pct_above_2c" in res.stats
    assert 0 <= res.stats["pct_above_2c"] <= 100


# ----------------------------------------------------------------------
# Gi* hotspots
# ----------------------------------------------------------------------
def test_hotspots_land_on_the_hot_core(fast_config):
    grid = _hand_grid(n=8)
    out = uhi.run_uhi_hotspots(grid, fast_config)
    hot = out[out["is_hotspot"] == 1]
    assert len(hot) > 0
    # hotspots must be much hotter than the city average and greener nowhere near
    assert hot["temp_idw"].mean() > out["temp_idw"].mean() + 1.0
    assert hot["ndvi_mean"].mean() < out["ndvi_mean"].mean()


def test_coldspots_land_on_the_cool_fringe(fast_config):
    grid = _hand_grid(n=8)
    out = uhi.run_uhi_hotspots(grid, fast_config)
    cold = out[out["is_coldspot"] == 1]
    assert len(cold) > 0
    assert cold["temp_idw"].mean() < out["temp_idw"].mean() - 1.0


def test_hotspot_columns_are_created_even_for_tiny_grids(fast_config):
    grid = _hand_grid(n=2)
    out = uhi.detect_hotspots(grid, fast_config)
    assert {"gi_zscore", "gi_pvalue", "is_hotspot", "is_coldspot"} <= set(out.columns)
    assert (out["is_hotspot"] == 0).all()


def test_knn_weights_are_supported(fast_config):
    cfg = fast_config.with_overrides({"hotspots.weights": "knn"})
    grid = _hand_grid(n=6)
    w = uhi.build_weights(grid, cfg)
    assert w.n == len(grid)


# ----------------------------------------------------------------------
# priority scoring
# ----------------------------------------------------------------------
def test_minmax_scales_to_unit_interval():
    out = priority.minmax(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_minmax_handles_constant_array():
    out = priority.minmax(np.array([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5])


def test_priority_ranks_hot_bare_built_cells_first(fast_config):
    grid = _hand_grid(n=6)
    grid = uhi.run_uhi_hotspots(grid, fast_config)
    res = priority.score_priority(fast_config, grid)
    top = res.top_cells.iloc[0]
    assert top["temp_idw"] >= grid["temp_idw"].mean()
    assert top["builtup_frac"] >= grid["builtup_frac"].mean()
    assert res.grid["priority_rank"].min() == 1
    assert res.grid["priority_score"].between(0, 1).all()


def test_priority_weights_are_normalised(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(), fast_config)
    res = priority.score_priority(fast_config, grid)
    assert sum(res.weights.values()) == pytest.approx(1.0)


def test_priority_respects_custom_weights(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(), fast_config)
    only_ndvi = priority.score_priority(
        fast_config, grid, weights={"temp": 0, "low_ndvi": 1, "builtup": 0, "pop_proxy": 0}
    )
    corr = np.corrcoef(only_ndvi.grid["priority_score"], only_ndvi.grid["ndvi_mean"])[0, 1]
    assert corr < 0  # scoring low NDVI alone must rank bare cells higher


def test_priority_is_deterministic(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(), fast_config)
    a = priority.score_priority(fast_config, grid)
    b = priority.score_priority(fast_config, grid)
    np.testing.assert_allclose(a.grid["priority_score"], b.grid["priority_score"])


def test_priority_rejects_zero_weights(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(), fast_config)
    with pytest.raises(ValueError, match="positive number"):
        priority.score_priority(
            fast_config,
            grid,
            weights={"temp": 0, "low_ndvi": 0, "builtup": 0, "pop_proxy": 0},
        )


def test_ward_summary_has_expected_columns(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(n=8), fast_config)
    res = priority.score_priority(fast_config, grid)
    ws = res.ward_summary
    assert len(ws) == 1
    assert {
        "ward_id",
        "mean_temp",
        "mean_uhi",
        "mean_ndvi",
        "mean_builtup",
        "hotspot_area_pct",
        "priority_score",
        "priority_rank",
    } <= set(ws.columns)
    assert ws.iloc[0]["priority_rank"] == 1


def test_ward_summary_handles_missing_wards(fast_config):
    grid = _hand_grid()
    grid["ward_id"] = None
    out = uhi.run_uhi_hotspots(grid, fast_config)
    res = priority.score_priority(fast_config, out)
    assert res.ward_summary.empty


def test_cooling_estimate_is_derived_from_the_coefficient(fast_config):
    grid = uhi.run_uhi_hotspots(_hand_grid(), fast_config)
    res = priority.score_priority(fast_config, grid, ndvi_coef=-3.0)
    assert res.cooling_estimate["estimated_cooling_c"] == pytest.approx(0.3)
    assert "not a guarantee" in res.cooling_estimate["note"]


def test_priority_end_to_end_on_synthetic_city(prepared, synthetic_inputs, fast_config):
    from src.analysis import idw as idw_mod
    from src.features import extract

    grid = extract.extract_grid_features(
        fast_config,
        prepared.grid,
        prepared.buildings,
        synthetic_inputs.landcover_polygons,
        synthetic_inputs.ndvi_path,
        synthetic_inputs.landcover_path,
        prepared.crs,
    )
    xy = np.column_stack([prepared.samples.geometry.x, prepared.samples.geometry.y])
    z = prepared.samples["temp_c"].to_numpy(dtype=float)
    idw_res = idw_mod.interpolate_to_grid(
        fast_config, xy, z, grid, prepared.crs, prepared.grid_res_m
    )
    xs = np.sort(grid["centroid_x"].unique())
    ys = np.sort(grid["centroid_y"].unique())
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid["temp_idw"] = [
        idw_res.surface[yi[cy], xi[cx]]
        for cx, cy in zip(grid["centroid_x"], grid["centroid_y"], strict=False)
    ]

    out = uhi.run_uhi_hotspots(grid, fast_config)
    res = priority.score_priority(fast_config, out)

    # the top-priority cells must be hot, bare and built-up relative to the city
    top = res.grid.nsmallest(20, "priority_rank")
    assert top["temp_idw"].mean() > res.grid["temp_idw"].mean()
    assert top["builtup_frac"].mean() > res.grid["builtup_frac"].mean()
    assert top["ndvi_mean"].mean() < res.grid["ndvi_mean"].mean()
    # hotspots must not be concentrated on the lake or the parks
    hot = res.grid[res.grid["is_hotspot"] == 1]
    if len(hot):
        assert (hot["landcover_class"] == "water").mean() < 0.35
