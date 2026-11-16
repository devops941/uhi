"""Tests for the data layer: validation, CRS handling, cleaning, grid."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.data import ingest, preprocess
from src.data.ingest import ValidationError
from src.data.preprocess import CRSError


# ----------------------------------------------------------------------
# schema validation
# ----------------------------------------------------------------------
def _valid_samples(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(n)],
            "lat": 11.0 + rng.normal(0, 0.002, n),
            "lon": 76.9 + rng.normal(0, 0.002, n),
            "temp_c": 30 + rng.normal(0, 2, n),
            "datetime": ["2023-04-14T10:30:00"] * n,
        }
    )


def test_validate_samples_accepts_good_input(base_config):
    out = ingest.validate_samples(_valid_samples(), base_config)
    assert len(out) == 6
    assert pd.api.types.is_datetime64_any_dtype(out["datetime"])


def test_validate_samples_rejects_missing_column(base_config):
    df = _valid_samples().drop(columns=["temp_c"])
    with pytest.raises(ValidationError, match="missing required column"):
        ingest.validate_samples(df, base_config)


def test_validate_samples_rejects_out_of_range_lat(base_config):
    df = _valid_samples()
    df.loc[0, "lat"] = 120.0
    with pytest.raises(ValidationError, match="latitude"):
        ingest.validate_samples(df, base_config)


def test_validate_samples_drops_impossible_temperatures(base_config):
    df = _valid_samples()
    df.loc[0, "temp_c"] = 999.0
    df.loc[1, "temp_c"] = -40.0
    out = ingest.validate_samples(df, base_config)
    assert len(out) == 4
    assert out.attrs["rejected_out_of_range"] == 2


def test_validate_samples_rejects_all_bad_temps(base_config):
    df = _valid_samples()
    df["temp_c"] = 900.0
    with pytest.raises(ValidationError, match="no rows left"):
        ingest.validate_samples(df, base_config)


def test_validate_samples_rejects_non_numeric(base_config):
    df = _valid_samples()
    df["temp_c"] = df["temp_c"].astype(object)
    df.loc[0, "temp_c"] = "hot"
    with pytest.raises(ValidationError, match="non-numeric"):
        ingest.validate_samples(df, base_config)


# ----------------------------------------------------------------------
# CRS rules
# ----------------------------------------------------------------------
def test_guard_metric_raises_on_geographic():
    gdf = gpd.GeoDataFrame(geometry=[Point(76.9, 11.0)], crs="EPSG:4326")
    with pytest.raises(CRSError, match="geographic"):
        preprocess.guard_metric(gdf, "samples")


def test_guard_metric_allows_projected():
    gdf = gpd.GeoDataFrame(geometry=[Point(500000, 1000000)], crs="EPSG:32643")
    preprocess.guard_metric(gdf, "samples")  # must not raise


def test_ensure_projected_reprojects_geographic():
    gdf = gpd.GeoDataFrame(geometry=[Point(76.9, 11.0)], crs="EPSG:4326")
    out = preprocess.ensure_projected(gdf, "x")
    assert out.crs.is_projected


def test_ensure_projected_requires_crs():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    with pytest.raises(CRSError, match="CRS is missing"):
        preprocess.ensure_projected(gdf, "x")


# ----------------------------------------------------------------------
# cleaning
# ----------------------------------------------------------------------
def test_clean_samples_drops_duplicates_and_outside_points(base_config):
    df = _valid_samples(8)
    # duplicate a coordinate
    df.loc[1, "lat"] = df.loc[0, "lat"]
    df.loc[1, "lon"] = df.loc[0, "lon"]
    # push one point far outside the boundary
    df.loc[2, "lat"] = 12.5
    df.loc[2, "lon"] = 78.5

    boundary = gpd.GeoDataFrame(geometry=[box(76.85, 10.95, 77.0, 11.05)], crs="EPSG:4326")
    boundary_m = preprocess.ensure_projected(boundary, "boundary")
    out = preprocess.clean_samples(df, boundary_m, boundary_m.crs, base_config)

    assert out.crs.is_projected
    assert len(out) <= 7
    assert out.attrs["cleaning_report"]["n_duplicates_removed"] == 1
    assert out.attrs["cleaning_report"]["n_outside_boundary_removed"] >= 1
    assert "is_outlier" in out.columns


def test_clean_samples_flags_local_outlier(base_config):
    df = _valid_samples(30)
    df["temp_c"] = 32.0
    df.loc[5, "temp_c"] = 58.0  # a lone spike, inside the plausible range
    boundary = gpd.GeoDataFrame(geometry=[box(76.8, 10.9, 77.1, 11.1)], crs="EPSG:4326")
    boundary_m = preprocess.ensure_projected(boundary, "boundary")
    out = preprocess.clean_samples(df, boundary_m, boundary_m.crs, base_config)
    flagged = out.loc[out["sample_id"] == "S5", "is_outlier"]
    assert not flagged.empty and int(flagged.iloc[0]) == 1


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
def test_build_grid_covers_boundary(base_config):
    boundary = gpd.GeoDataFrame(geometry=[box(500000, 1000000, 500500, 1000500)], crs="EPSG:32643")
    grid = preprocess.build_grid(boundary, "EPSG:32643", cell_m=100.0)
    # 500 m x 500 m at 100 m -> 25 cells
    assert len(grid) == 25
    assert grid["cell_id"].is_unique
    assert abs(grid.geometry.area.iloc[0] - 10000.0) < 1e-6


def test_build_grid_refuses_degrees(base_config):
    boundary = gpd.GeoDataFrame(geometry=[box(76.9, 11.0, 77.0, 11.1)], crs="EPSG:4326")
    with pytest.raises(CRSError):
        preprocess.build_grid(boundary, "EPSG:4326", cell_m=100.0)


def test_assign_wards_attaches_ids():
    grid = gpd.GeoDataFrame(
        {
            "cell_id": [0, 1],
            "centroid_x": [500050.0, 500150.0],
            "centroid_y": [1000050.0, 1000050.0],
        },
        geometry=[box(500000, 1000000, 500100, 1000100), box(500100, 1000000, 500200, 1000100)],
        crs="EPSG:32643",
    )
    wards = gpd.GeoDataFrame(
        {"ward_id": ["W01", "W02"], "ward_name": ["a", "b"]},
        geometry=[box(500000, 1000000, 500100, 1000100), box(500100, 1000000, 500200, 1000100)],
        crs="EPSG:32643",
    )
    out = preprocess.assign_wards(grid, wards)
    assert out["ward_id"].tolist() == ["W01", "W02"]


def test_preprocess_full_report(prepared, fast_config):
    rep = prepared.report
    assert rep["crs_metric"] == "EPSG:32643"
    assert rep["n_grid_cells"] > 0
    assert prepared.n_samples > 0
    assert prepared.samples.crs.is_projected


# ----------------------------------------------------------------------
# synthetic generator contract
# ----------------------------------------------------------------------
def test_synthetic_generator_writes_all_layers(synthetic_inputs, fast_config):
    raw = fast_config.path_for("data_raw")
    for name in [
        "samples.csv",
        "ndvi.tif",
        "landcover.tif",
        "buildings.geojson",
        "boundary.geojson",
        "wards.geojson",
        "synthetic_truth.json",
    ]:
        assert (raw / name).exists(), name


def test_synthetic_samples_within_boundary(prepared):
    boundary = prepared.boundary.geometry.iloc[0]
    assert prepared.samples.within(boundary).all()
