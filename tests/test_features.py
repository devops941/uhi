"""Tests for feature extraction, including hand-checked tiny rasters."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.features import extract


# ----------------------------------------------------------------------
# hand-checked raster cases
# ----------------------------------------------------------------------
def test_zonal_mean_on_tiny_raster_matches_hand_calculation(tiny_raster, fast_config):
    path, values, transform = tiny_raster
    # a point at the centre of the middle cell (row 1, col 1) -> value 0.5
    centre_x = transform.c + 1.5 * 100
    centre_y = transform.f - 1.5 * 100
    points = gpd.GeoDataFrame(
        {"sample_id": ["P1"]}, geometry=[Point(centre_x, centre_y)], crs="EPSG:32643"
    )
    # tiny buffer so the mean is dominated by the centre cell
    out = extract.buffer_ndvi_zonal(points, path, "EPSG:32643", buffer_m=30.0)
    # the 3x3 mean is 0.5, and the centre cell alone is 0.5
    assert out[0] == pytest.approx(0.5, abs=1e-6)


def test_zonal_mean_full_extent_is_overall_mean(tiny_raster):
    path, values, transform = tiny_raster
    centre_x = transform.c + 1.5 * 100
    centre_y = transform.f - 1.5 * 100
    points = gpd.GeoDataFrame(
        {"sample_id": ["P1"]}, geometry=[Point(centre_x, centre_y)], crs="EPSG:32643"
    )
    out = extract.buffer_ndvi_zonal(points, path, "EPSG:32643", buffer_m=400.0)
    assert out[0] == pytest.approx(values.mean(), abs=1e-6)


def test_dominant_class_under_points_picks_correct_cell(tiny_raster, tmp_path):
    import rasterio

    path, values, transform = tiny_raster
    classes = np.array([[10, 10, 20], [10, 20, 20], [30, 30, 30]], dtype="int16")
    class_path = tmp_path / "tiny_lc.tif"
    with rasterio.open(
        class_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="int16",
        crs="EPSG:32643",
        transform=transform,
        nodata=-1,
    ) as dst:
        dst.write(classes, 1)

    # points at the centre of each corner cell
    xs = [transform.c + 0.5 * 100, transform.c + 2.5 * 100]
    ys = [transform.f - 0.5 * 100, transform.f - 2.5 * 100]
    points = gpd.GeoDataFrame(
        {"sample_id": ["a", "b", "c", "d"]},
        geometry=[
            Point(xs[0], ys[0]),
            Point(xs[1], ys[0]),
            Point(xs[0], ys[1]),
            Point(xs[1], ys[1]),
        ],
        crs="EPSG:32643",
    )
    out = extract.dominant_class_under_points(points, class_path, "EPSG:32643")
    assert out.tolist() == [10.0, 20.0, 30.0, 30.0]


def test_dominant_class_requires_metric_raster(tiny_raster):
    path, _, _ = tiny_raster
    points = gpd.GeoDataFrame(geometry=[Point(500050, 999950)], crs="EPSG:32643")
    with pytest.raises(ValueError, match="must already be in"):
        extract.dominant_class_under_points(points, path, "EPSG:3857")


# ----------------------------------------------------------------------
# built-up fraction
# ----------------------------------------------------------------------
def test_builtup_fraction_is_one_for_fully_covered_cell():
    cell = box(0, 0, 100, 100)
    grid = gpd.GeoDataFrame({"cell_id": [0]}, geometry=[cell], crs="EPSG:32643")
    buildings = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32643")
    frac = extract.grid_builtup_fraction(grid, buildings, "EPSG:32643")
    assert frac[0] == pytest.approx(1.0, abs=1e-6)


def test_builtup_fraction_is_half_for_half_covered_cell():
    grid = gpd.GeoDataFrame({"cell_id": [0]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32643")
    buildings = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100)], crs="EPSG:32643")
    frac = extract.grid_builtup_fraction(grid, buildings, "EPSG:32643")
    assert frac[0] == pytest.approx(0.5, abs=1e-6)


def test_builtup_fraction_zero_without_buildings():
    grid = gpd.GeoDataFrame({"cell_id": [0]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32643")
    frac = extract.grid_builtup_fraction(grid, None, "EPSG:32643")
    assert frac[0] == 0.0


def test_builtup_fraction_never_exceeds_one():
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:32643",
    )
    # overlapping buildings that spill outside the cells
    buildings = gpd.GeoDataFrame(
        geometry=[box(-50, -50, 250, 150), box(0, 0, 120, 100)], crs="EPSG:32643"
    )
    frac = extract.grid_builtup_fraction(grid, buildings, "EPSG:32643")
    assert (frac >= 0).all() and (frac <= 1.0).all()


def test_buffer_builtup_fraction_between_zero_and_one(prepared):
    frac = extract.buffer_builtup_fraction(
        prepared.samples, prepared.buildings, prepared.crs, 100.0
    )
    assert len(frac) == len(prepared.samples)
    assert np.isfinite(frac).all()
    assert (frac >= 0).all() and (frac <= 1.0).all()


# ----------------------------------------------------------------------
# distance to layer
# ----------------------------------------------------------------------
def test_distance_to_layer_is_zero_inside():
    points = gpd.GeoDataFrame(geometry=[Point(50, 50)], crs="EPSG:32643")
    layer = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32643")
    d = extract.distance_to_layer(points, layer, "EPSG:32643")
    assert d[0] == pytest.approx(0.0)


def test_distance_to_layer_measures_gap():
    points = gpd.GeoDataFrame(geometry=[Point(200, 50)], crs="EPSG:32643")
    layer = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32643")
    d = extract.distance_to_layer(points, layer, "EPSG:32643")
    assert d[0] == pytest.approx(100.0)


def test_distance_to_layer_returns_nan_without_layer():
    points = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32643")
    d = extract.distance_to_layer(points, None, "EPSG:32643")
    assert np.isnan(d[0])


# ----------------------------------------------------------------------
# full extraction
# ----------------------------------------------------------------------
def test_sample_features_have_no_nan(prepared, synthetic_inputs, fast_config):
    feats = extract.extract_sample_features(
        fast_config,
        prepared.samples,
        prepared.buildings,
        synthetic_inputs.landcover_polygons,
        synthetic_inputs.ndvi_path,
        synthetic_inputs.landcover_path,
        prepared.crs,
    )
    assert len(feats) == len(prepared.samples)
    assert feats["ndvi_mean"].notna().all()
    assert feats["builtup_frac"].notna().all()
    assert feats["dist_water_m"].notna().all()
    assert set(feats.columns) == {
        "sample_id",
        "ndvi_mean",
        "builtup_frac",
        "landcover_class",
        "dist_water_m",
    }


def test_grid_features_have_no_nan(prepared, synthetic_inputs, fast_config):
    grid = extract.extract_grid_features(
        fast_config,
        prepared.grid,
        prepared.buildings,
        synthetic_inputs.landcover_polygons,
        synthetic_inputs.ndvi_path,
        synthetic_inputs.landcover_path,
        prepared.crs,
    )
    assert len(grid) == len(prepared.grid)
    assert grid["ndvi_mean"].notna().all()
    assert grid["builtup_frac"].notna().all()
    assert (grid["builtup_frac"] >= 0).all() and (grid["builtup_frac"] <= 1).all()
    assert grid["landcover_class"].ne("unknown").all()


def test_code_to_name_handles_str_and_int_keys():
    assert extract.code_to_name(3, {3: "built_up"}, {}) == "built_up"
    assert extract.code_to_name(3, {"3": "built_up"}, {}) == "built_up"
    assert extract.code_to_name(99, {3: "built_up"}, {}) == "class_99"
    assert extract.code_to_name(np.nan, {}, {}) == "unknown"
