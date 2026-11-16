"""Load and validate raw inputs against the schemas in the spec (Section 4).

Validation is deliberately strict and fails loudly: a wrong CRS, a missing
column or an out-of-range temperature must stop the pipeline rather than
silently produce a meaningless heat map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

REQUIRED_SAMPLE_COLUMNS = ["sample_id", "lat", "lon", "temp_c", "datetime"]
REQUIRED_BOUNDARY_COLUMNS = ["geometry"]
REQUIRED_WARD_COLUMNS = ["ward_id", "ward_name", "geometry"]


class ValidationError(ValueError):
    """Raised when an input fails a schema or sanity check."""


@dataclass
class RawInputs:
    """Validated raw inputs, still in EPSG:4326."""

    samples: pd.DataFrame
    boundary: gpd.GeoDataFrame
    wards: gpd.GeoDataFrame | None = None
    buildings: gpd.GeoDataFrame | None = None
    landcover_polygons: gpd.GeoDataFrame | None = None
    ndvi_path: Path | None = None
    landcover_path: Path | None = None
    report: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def _check_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    if gdf.crs is None:
        raise ValidationError(f"{name}: missing CRS")
    if gdf.crs.is_geographic:
        # raw vector data is allowed to be geographic - it is reprojected later
        return
    raise ValidationError(
        f"{name}: raw inputs must be stored in EPSG:4326 (got {gdf.crs.to_string()})"
    )


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"{name}: missing required column(s): {missing}")


def validate_samples(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Validate the sample table and coerce dtypes."""
    _check_columns(df, REQUIRED_SAMPLE_COLUMNS, "samples")

    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")

    bad = out[["lat", "lon", "temp_c"]].isna().any(axis=1)
    if bad.any():
        raise ValidationError(f"samples: {int(bad.sum())} row(s) have non-numeric lat/lon/temp")

    if out["lat"].abs().max() > 90:
        raise ValidationError("samples: latitude outside [-90, 90]")
    if out["lon"].abs().max() > 180:
        raise ValidationError("samples: longitude outside [-180, 180]")

    temp_min = float(cfg.get("cleaning.temp_min_c", 10.0))
    temp_max = float(cfg.get("cleaning.temp_max_c", 60.0))
    n_before = len(out)
    out = out[(out["temp_c"] >= temp_min) & (out["temp_c"] <= temp_max)].copy()
    out.attrs["rejected_out_of_range"] = n_before - len(out)

    if out.empty:
        raise ValidationError("samples: no rows left after the temperature range check")

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out.reset_index(drop=True)


def validate_raster(
    path: str | Path,
    name: str,
    expect_range: tuple[float, float] | None = None,
    integer: bool = False,
) -> dict:
    """Validate a GeoTIFF: single band, has a CRS and sane value ranges."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"{name}: file not found: {p}")
    with rasterio.open(p) as src:
        if src.crs is None:
            raise ValidationError(f"{name}: raster has no CRS")
        if src.count < 1:
            raise ValidationError(f"{name}: raster has no bands")
        band = src.read(1, masked=True)
        info = {
            "path": str(p),
            "crs": src.crs.to_string(),
            "shape": (src.height, src.width),
            "res": src.res,
            "dtype": str(src.dtypes[0]),
            "min": float(band.min()) if band.count() else None,
            "max": float(band.max()) if band.count() else None,
        }
    if integer and not np.issubdtype(np.dtype(info["dtype"]), np.integer):
        raise ValidationError(f"{name}: expected an integer raster, got {info['dtype']}")
    if expect_range is not None and info["min"] is not None:
        lo, hi = expect_range
        if info["min"] < lo or info["max"] > hi:
            raise ValidationError(
                f"{name}: values outside {expect_range} (min={info['min']:.3f}, "
                f"max={info['max']:.3f})"
            )
    return info


def validate_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _check_columns(gdf, REQUIRED_BOUNDARY_COLUMNS, "boundary")
    _check_crs(gdf, "boundary")
    if gdf.empty:
        raise ValidationError("boundary: empty layer")
    if (~gdf.geometry.is_valid).any():
        raise ValidationError("boundary: invalid geometry")
    if gdf.geometry.isna().any():
        raise ValidationError("boundary: null geometry")
    return gdf


def validate_wards(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _check_columns(gdf, REQUIRED_WARD_COLUMNS, "wards")
    _check_crs(gdf, "wards")
    if gdf["ward_id"].duplicated().any():
        raise ValidationError("wards: duplicated ward_id")
    return gdf


def validate_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _check_columns(gdf, ["geometry"], "buildings")
    _check_crs(gdf, "buildings")
    return gdf


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_inputs(cfg, raw_dir: str | Path | None = None) -> RawInputs:
    """Load and validate every raw input, following the file-name schemas."""
    raw = Path(raw_dir) if raw_dir else cfg.path_for("data_raw")

    samples_path = raw / "samples.csv"
    if not samples_path.exists():
        raise ValidationError(
            f"samples.csv not found in {raw}. Run the synthetic generator first "
            "(python -m src.pipeline --source synthetic)."
        )
    samples = validate_samples(pd.read_csv(samples_path), cfg)

    boundary = validate_boundary(gpd.read_file(raw / "boundary.geojson"))

    wards_path = raw / "wards.geojson"
    wards = validate_wards(gpd.read_file(wards_path)) if wards_path.exists() else None

    buildings_path = raw / "buildings.geojson"
    buildings = (
        validate_buildings(gpd.read_file(buildings_path)) if buildings_path.exists() else None
    )

    lc_poly_path = raw / "landcover.geojson"
    lc_polys = gpd.read_file(lc_poly_path) if lc_poly_path.exists() else None

    ndvi_path = raw / "ndvi.tif"
    lc_path = raw / "landcover.tif"
    report = {}
    if ndvi_path.exists():
        report["ndvi"] = validate_raster(ndvi_path, "ndvi", expect_range=(-1.0, 1.0))
    if lc_path.exists():
        report["landcover"] = validate_raster(lc_path, "landcover", integer=True)

    report["n_samples"] = int(len(samples))
    report["rejected_out_of_range"] = int(samples.attrs.get("rejected_out_of_range", 0))

    return RawInputs(
        samples=samples,
        boundary=boundary,
        wards=wards,
        buildings=buildings,
        landcover_polygons=lc_polys,
        ndvi_path=ndvi_path if ndvi_path.exists() else None,
        landcover_path=lc_path if lc_path.exists() else None,
        report=report,
    )
