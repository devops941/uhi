"""Preprocessing: reproject, clip, clean and build the 100 m analysis grid.

CRS rule (spec Section 4): raw data is stored in EPSG:4326 and *everything* is
reprojected to a metric projected CRS before any distance, buffer, area or
interpolation operation. Interpolating in degrees is never allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, box


class CRSError(ValueError):
    """Raised when a distance/buffer operation is attempted in degrees."""


@dataclass
class PreparedData:
    """Output of the preprocessing stage - everything in the metric CRS."""

    crs: str
    samples: gpd.GeoDataFrame  # cleaned samples + is_outlier flag
    boundary: gpd.GeoDataFrame  # single dissolved polygon
    wards: gpd.GeoDataFrame | None
    buildings: gpd.GeoDataFrame | None
    grid: gpd.GeoDataFrame  # 100 m x 100 m analysis grid
    grid_res_m: float
    report: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))


# ----------------------------------------------------------------------
# CRS helpers
# ----------------------------------------------------------------------
def ensure_projected(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Reproject to a metric CRS, raising if the input has no CRS."""
    if gdf.crs is None:
        raise CRSError(f"{name}: cannot reproject - CRS is missing")
    if gdf.crs.is_projected:
        return gdf
    utm = gdf.estimate_utm_crs()
    if utm is None:  # pragma: no cover - defensive
        raise CRSError(f"{name}: could not estimate a UTM CRS")
    return gdf.to_crs(utm)


def guard_metric(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Raise if a distance/buffer operation would run in degrees."""
    if gdf.crs is None:
        raise CRSError(f"{name}: CRS is required for distance operations")
    if gdf.crs.is_geographic:
        raise CRSError(
            f"{name}: distance/buffer/area operations are not allowed on geographic "
            f"(degree) coordinates ({gdf.crs.to_string()}); reproject to a metric CRS first"
        )


def resolve_metric_crs(cfg, gdf: gpd.GeoDataFrame) -> str:
    override = cfg.get("project.crs_projected")
    if override:
        return override
    utm = gdf.estimate_utm_crs()
    return utm.to_string() if utm is not None else "EPSG:3857"


# ----------------------------------------------------------------------
# cleaning
# ----------------------------------------------------------------------
def clean_samples(
    samples: pd.DataFrame,
    boundary_m: gpd.GeoDataFrame,
    crs: str,
    cfg,
) -> gpd.GeoDataFrame:
    """Drop duplicates, range violations, local outliers and out-of-bound points.

    Returns a GeoDataFrame in the metric CRS with an ``is_outlier`` column.
    """
    df = samples.copy()

    # --- duplicate coordinates -------------------------------------------
    if bool(cfg.get("cleaning.drop_duplicate_coords", True)):
        before = len(df)
        df = df.drop_duplicates(subset=["lat", "lon"]).copy()
        n_dupes = before - len(df)
    else:
        n_dupes = 0

    # --- plausible temperature range -------------------------------------
    tmin = float(cfg.get("cleaning.temp_min_c", 10.0))
    tmax = float(cfg.get("cleaning.temp_max_c", 60.0))
    before = len(df)
    df = df[(df["temp_c"] >= tmin) & (df["temp_c"] <= tmax)].copy()
    n_range = before - len(df)

    if df.empty:
        raise ValueError("no samples left after duplicate/range cleaning")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"], strict=False)],
        crs="EPSG:4326",
    ).to_crs(crs)

    # --- local-median outliers (k nearest neighbours) --------------------
    k = int(cfg.get("cleaning.outlier_k_neighbours", 8))
    sd_threshold = float(cfg.get("cleaning.outlier_sd_threshold", 3.0))
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    temp = gdf["temp_c"].to_numpy(dtype=float)
    is_outlier = np.zeros(len(gdf), dtype=int)
    if len(gdf) > k + 1:
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)
        neighbours = idx[:, 1:]  # exclude the point itself
        local_median = np.median(temp[neighbours], axis=1)
        residual = temp - local_median
        mad = np.median(np.abs(residual - np.median(residual)))
        robust_sd = 1.4826 * mad if mad > 0 else residual.std(ddof=0)
        if robust_sd > 0:
            is_outlier = (np.abs(residual) > sd_threshold * robust_sd).astype(int)
    gdf["is_outlier"] = is_outlier

    # --- keep only points inside the city boundary ------------------------
    inside = gdf.within(boundary_m.geometry.iloc[0])
    n_outside = int((~inside).sum())
    gdf = gdf[inside].copy()

    gdf = gdf.reset_index(drop=True)
    gdf.attrs["cleaning_report"] = {
        "n_duplicates_removed": int(n_dupes),
        "n_out_of_range_removed": int(n_range),
        "n_outside_boundary_removed": n_outside,
        "n_local_outliers_flagged": int(gdf["is_outlier"].sum()),
        "n_final": int(len(gdf)),
    }
    return gdf


# ----------------------------------------------------------------------
# analysis grid
# ----------------------------------------------------------------------
def build_grid(boundary_m: gpd.GeoDataFrame, crs: str, cell_m: float = 100.0) -> gpd.GeoDataFrame:
    """Build a regular cell_m x cell_m grid covering the boundary bounding box."""
    guard_metric(boundary_m, "boundary")
    minx, miny, maxx, maxy = boundary_m.total_bounds
    xs = np.arange(minx, maxx + cell_m, cell_m)
    ys = np.arange(miny, maxy + cell_m, cell_m)

    polys = [box(x, y, x + cell_m, y + cell_m) for y in ys[:-1] for x in xs[:-1]]
    grid = gpd.GeoDataFrame(geometry=polys, crs=crs)
    grid["cell_id"] = np.arange(len(grid))
    grid["centroid_x"] = grid.geometry.centroid.x
    grid["centroid_y"] = grid.geometry.centroid.y

    # keep only cells whose centroid falls inside the boundary
    inside = grid.geometry.representative_point().within(boundary_m.geometry.iloc[0])
    grid = grid[inside].reset_index(drop=True)
    grid["cell_id"] = np.arange(len(grid))
    grid["area_m2"] = grid.geometry.area
    return grid[["cell_id", "centroid_x", "centroid_y", "area_m2", "geometry"]]


def assign_wards(grid: gpd.GeoDataFrame, wards_m: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Attach ward_id to every grid cell.

    Joins on each cell's representative point so a cell that merely touches a
    neighbouring ward boundary is not double-assigned; the first match wins.
    """
    out = grid.copy()
    if wards_m is None or wards_m.empty:
        out["ward_id"] = None
        return out

    points = gpd.GeoDataFrame(
        {"cell_id": out["cell_id"].to_numpy()},
        geometry=out.geometry.representative_point(),
        crs=out.crs,
    )
    joined = gpd.sjoin(points, wards_m[["ward_id", "geometry"]], how="left", predicate="within")
    joined = joined[~joined["cell_id"].duplicated(keep="first")]
    lookup = joined.set_index("cell_id")["ward_id"]
    out["ward_id"] = out["cell_id"].map(lookup).to_numpy()
    return out


def ward_area_km2(wards_m: gpd.GeoDataFrame) -> pd.DataFrame:
    guard_metric(wards_m, "wards")
    return pd.DataFrame(
        {
            "ward_id": wards_m["ward_id"].to_numpy(),
            "ward_name": wards_m["ward_name"].to_numpy(),
            "area_km2": (wards_m.geometry.area / 1e6).to_numpy(),
        }
    )


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def preprocess(cfg, raw_inputs) -> PreparedData:
    """Reproject, clip, clean and build the grid. This is the Phase-2 handover."""
    boundary_m = ensure_projected(raw_inputs.boundary, "boundary")
    boundary_m = boundary_m.dissolve().reset_index(drop=True)
    crs = resolve_metric_crs(cfg, boundary_m)
    boundary_m = boundary_m.to_crs(crs)
    guard_metric(boundary_m, "boundary")

    wards_m = None
    if raw_inputs.wards is not None:
        wards_m = ensure_projected(raw_inputs.wards, "wards").to_crs(crs)
        wards_m = gpd.clip(wards_m, boundary_m)

    buildings_m = None
    if raw_inputs.buildings is not None and not raw_inputs.buildings.empty:
        buildings_m = ensure_projected(raw_inputs.buildings, "buildings").to_crs(crs)
        buildings_m = gpd.clip(buildings_m, boundary_m)

    samples = clean_samples(raw_inputs.samples, boundary_m, crs, cfg)
    cleaning_report = dict(samples.attrs.get("cleaning_report", {}))

    cell_m = float(cfg.get("synthetic.grid_size_m", 100.0))
    grid = build_grid(boundary_m, crs, cell_m)
    grid = assign_wards(grid, wards_m)

    report = {
        "crs_metric": crs,
        "n_grid_cells": int(len(grid)),
        "grid_cell_m": cell_m,
        **cleaning_report,
    }
    return PreparedData(
        crs=crs,
        samples=samples,
        boundary=boundary_m,
        wards=wards_m,
        buildings=buildings_m,
        grid=grid,
        grid_res_m=cell_m,
        report=report,
    )
