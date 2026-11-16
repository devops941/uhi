"""Data access for the dashboard.

Everything is read from the artefacts the pipeline already wrote (SQLite,
GeoPackage, GeoTIFF, Parquet), with Streamlit caching so page switches are
instant. If the artefacts are missing the loader falls back to running the
pipeline once.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st


class RasterWgs(TypedDict):
    array: np.ndarray
    extent: tuple[float, float, float, float]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import db as db_mod  # noqa: E402


@st.cache_resource(show_spinner="Loading pipeline outputs...")
def get_config():
    return load_config()


@st.cache_resource(show_spinner="Connecting to the results database...")
def get_engine():
    cfg = get_config()
    db_path = cfg.path_for("sqlite_db")
    if not db_path.exists():
        _run_pipeline()
    return db_mod.get_engine(cfg)


def _run_pipeline() -> None:
    from src.pipeline import run_pipeline

    cfg = get_config()
    with st.spinner("First run: building the whole project (synthetic city)... "):
        run_pipeline(cfg, source="synthetic")


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    return db_mod.read_table(get_engine(), name)


@st.cache_data(show_spinner="Loading the analysis grid...")
def load_grid() -> gpd.GeoDataFrame:
    cfg = get_config()
    gpkg = cfg.path_for("geopackage")
    if gpkg.exists() and "grid" in db_mod.gpkg_layers(gpkg):
        return db_mod.read_layer(gpkg, "grid")
    # fall back to the precomputed dashboard payload
    parquet = cfg.path_for("outputs") / "dashboard" / "grid.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        if "lat" in df.columns and "lon" in df.columns:
            return gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
            )
        return gpd.GeoDataFrame(df)
    _run_pipeline()
    return db_mod.read_layer(gpkg, "grid")


@st.cache_data(show_spinner=False)
def load_grid_flat() -> pd.DataFrame:
    """Grid rows with lat/lon columns - the dashboard payload."""
    parquet = get_config().path_for("outputs") / "dashboard" / "grid.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    grid = load_grid()
    out = grid.copy()
    out["lon"] = out.geometry.centroid.x
    out["lat"] = out.geometry.centroid.y
    return pd.DataFrame(out.drop(columns="geometry"))


@st.cache_data(show_spinner=False)
def load_samples() -> gpd.GeoDataFrame:
    cfg = get_config()
    gpkg = cfg.path_for("geopackage")
    if gpkg.exists() and "samples" in db_mod.gpkg_layers(gpkg):
        return db_mod.read_layer(gpkg, "samples")
    return gpd.GeoDataFrame()


@st.cache_data(show_spinner=False)
def load_boundary() -> gpd.GeoDataFrame:
    cfg = get_config()
    gpkg = cfg.path_for("geopackage")
    if gpkg.exists() and "boundary" in db_mod.gpkg_layers(gpkg):
        return db_mod.read_layer(gpkg, "boundary")
    return gpd.GeoDataFrame()


@st.cache_data(show_spinner=False)
def load_wards() -> gpd.GeoDataFrame:
    cfg = get_config()
    gpkg = cfg.path_for("geopackage")
    if gpkg.exists() and "wards" in db_mod.gpkg_layers(gpkg):
        return db_mod.read_layer(gpkg, "wards")
    return gpd.GeoDataFrame()


@st.cache_data(show_spinner=False)
def load_raster(name: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import rasterio

    path = get_config().path_for("outputs") / name
    if not path.exists():
        return np.zeros((1, 1), dtype="float32"), (0, 1, 0, 1)
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).filled(np.nan).astype("float32")
        b = src.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


@st.cache_data(show_spinner=False)
def load_geotiff_as_wgs(name: str) -> RasterWgs | None:
    """Reprojected array plus lon/lat extents, for Plotly overlays."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    path = get_config().path_for("outputs") / name
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        dst_crs = CRS.from_string("EPSG:4326")
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        bounds = rasterio.transform.array_bounds(height, width, transform)
        left, bottom, right, top = bounds
    return {"array": dst, "extent": (left, right, bottom, top)}


def artefacts_ready() -> bool:
    cfg = get_config()
    return cfg.path_for("sqlite_db").exists() and (cfg.path_for("outputs") / "report.md").exists()


def dataset_summary() -> dict:
    cfg = get_config()
    grid = load_grid_flat()
    return {
        "n_cells": int(len(grid)),
        "n_samples": int(len(load_samples())),
        "cell_m": float(cfg.get("synthetic.grid_size_m", 100.0)),
        "city": cfg.get("project.city"),
        "crs": cfg.get("project.crs_raw"),
    }


def get_figures_dir() -> Path:
    cfg = get_config()
    return cfg.path_for("figures", create=True)


def list_figures() -> dict[str, Path]:
    fig_dir = get_figures_dir()
    return {p.name: p for p in sorted(fig_dir.glob("*.png"))}


def save_uploaded_file(uploaded_file, target_filename: str) -> Path:
    cfg = get_config()
    raw_dir = cfg.path_for("data_raw", create=True)
    target = raw_dir / target_filename
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def recompute_pipeline_with_custom_data(source: str = "synthetic", skip_generate: bool = True):
    from src.pipeline import run_pipeline

    cfg = get_config()
    st.cache_data.clear()
    st.cache_resource.clear()
    return run_pipeline(cfg, source=source, skip_generate=skip_generate)

