"""Phase 3 - feature extraction.

For every temperature sample the pipeline attaches:

    ndvi_mean       mean NDVI inside a 100 m circular buffer (zonal statistics)
    builtup_frac    area(buildings intersected with the buffer) / buffer area
    landcover_class dominant land-cover class under the point
    dist_water_m    distance from the point to the nearest water polygon

The same routine fills the 100 m analysis grid with NDVI, built-up fraction and
the dominant land-cover class so the interpolation output can be joined to
covariates.

Everything here runs in the metric CRS; the raster is reprojected on the fly to
match the vector CRS, so the buffers are genuinely 100 m.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import Resampling, reproject
from rasterstats import zonal_stats

from src.data.preprocess import guard_metric


def raster_to_metric(
    path,
    crs: str,
    cache_dir=None,
    resampling: str = "bilinear",
) -> Path:
    """Return a path to a copy of ``path`` reprojected into ``crs``.

    Zonal statistics and rasterised masks need the raster and the vector layer
    in the same CRS. Reprojecting once and caching the result keeps the
    per-feature loops fast.
    """
    from pathlib import Path

    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    src_path = Path(path)
    if cache_dir is None:
        cache_dir = src_path.parent.parent / "processed"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{src_path.stem}_{crs.replace(':', '')}.tif"
    dst_path = cache_dir / tag

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"raster {src_path} has no CRS")
        is_int = np.issubdtype(np.dtype(src.dtypes[0]), np.integer)
        # integer class rasters keep their code values; float rasters use NaN
        if is_int:
            out_dtype = "int16"
            out_nodata = int(src.nodata) if src.nodata is not None else -1
        else:
            out_dtype = "float32"
            out_nodata = np.nan

        if src.crs.to_string() == crs and dst_path.exists():
            return dst_path

        if src.crs.to_string() == crs:
            data = src.read(1)
            profile = src.profile.copy()
            profile.update(dtype=out_dtype, nodata=out_nodata)
        else:
            dst_crs = rasterio.crs.CRS.from_string(crs)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            data = np.full((height, width), out_nodata, dtype=out_dtype)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                src_nodata=src.nodata,
                dst_nodata=out_nodata,
                resampling=Resampling.nearest if is_int else getattr(Resampling, resampling),
            )
            profile = src.profile.copy()
            profile.update(
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                dtype=out_dtype,
                nodata=out_nodata,
            )
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data.astype(out_dtype), 1)
    return dst_path


# ----------------------------------------------------------------------
# raster helpers
# ----------------------------------------------------------------------
def reproject_raster_to(path, crs: str) -> tuple[np.ndarray, dict]:
    """Read a raster and reproject it onto the vector CRS.

    Returns ``(array, profile)`` where profile holds transform, crs, nodata.
    """
    import rasterio

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"raster {path} has no CRS")
        if src.crs.to_string() == crs:
            array = src.read(1)
            return array, {
                "transform": src.transform,
                "crs": src.crs,
                "nodata": src.nodata,
            }

        dst_crs = rasterio.crs.CRS.from_string(crs)
        transform, width, height = rasterio.warp.calculate_default_transform(
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
        return dst, {"transform": transform, "crs": dst_crs, "nodata": np.nan}


def dominant_class_under_points(
    points: gpd.GeoDataFrame, class_raster_path, crs: str
) -> np.ndarray:
    """Sample the land-cover class code at each point location (raster in ``crs``)."""
    import rasterio

    with rasterio.open(class_raster_path) as src:
        if src.crs is None or src.crs.to_string() != crs:
            raise ValueError(
                f"land-cover raster must already be in {crs} "
                f"(got {None if src.crs is None else src.crs.to_string()}); "
                "call raster_to_metric first"
            )
        array = src.read(1)
        transform = src.transform
    rows, cols = rasterio_rowcol(transform, points.geometry.x, points.geometry.y)
    valid = (rows >= 0) & (rows < array.shape[0]) & (cols >= 0) & (cols < array.shape[1])
    out = np.full(len(points), np.nan)
    out[valid] = array[rows[valid], cols[valid]]
    return out


def rasterio_rowcol(transform, xs, ys) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised world -> (row, col) using an affine transform."""
    inv = ~transform
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    return np.floor(np.asarray(rows)).astype(int), np.floor(np.asarray(cols)).astype(int)


def code_to_name(code, *lookups) -> str:
    """Map a raster class code to a name, tolerating str/int YAML keys."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "unknown"
    c = int(code)
    for lookup in lookups:
        if not lookup:
            continue
        for key in (c, str(c)):
            if key in lookup:
                return str(lookup[key])
    return f"class_{c}"


def buffer_ndvi_zonal(points: gpd.GeoDataFrame, ndvi_path, crs: str, buffer_m: float) -> np.ndarray:
    """Mean NDVI inside a circular buffer around each point."""
    buffers = points.copy()
    buffers["geometry"] = points.geometry.buffer(buffer_m)
    stats = zonal_stats(
        buffers.geometry,
        str(ndvi_path),
        stats=["mean"],
        geojson_out=False,
        all_touched=False,
    )
    return np.array([s["mean"] if s["mean"] is not None else np.nan for s in stats])


def buffer_builtup_fraction(
    points: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    crs: str,
    buffer_m: float,
) -> np.ndarray:
    """Built-up fraction = area(buildings inside buffer) / buffer area.

    Buildings are rasterised onto a fine local grid for speed, then the fraction
    is computed per buffer with a single pass.
    """
    n = len(points)
    if buildings is None or buildings.empty:
        return np.zeros(n, dtype=float)

    guard_metric(points, "samples")
    guard_metric(buildings, "buildings")

    buffers = points.geometry.buffer(buffer_m)
    # local raster covering the buffers bounding box at 5 m resolution
    minx, miny, maxx, maxy = buffers.total_bounds
    res = 5.0
    width = max(int(np.ceil((maxx - minx) / res)), 1)
    height = max(int(np.ceil((maxy - miny) / res)), 1)
    transform = __import__("affine").Affine.translation(minx, maxy) * __import__(
        "affine"
    ).Affine.scale(res, -res)

    building_mask = rasterize(
        ((geom, 1) for geom in buildings.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    fractions = np.zeros(n, dtype=float)
    for i, geom in enumerate(buffers):
        mask = geometry_mask([geom], out_shape=(height, width), transform=transform, invert=True)
        total = mask.sum()
        if total == 0:
            fractions[i] = 0.0
            continue
        fractions[i] = float((mask & (building_mask == 1)).sum()) / float(total)
    return fractions


def distance_to_layer(
    points: gpd.GeoDataFrame, layer: gpd.GeoDataFrame | None, crs: str
) -> np.ndarray:
    """Distance (metres) from each point to the nearest geometry of ``layer``."""
    if layer is None or layer.empty:
        return np.full(len(points), np.nan)
    guard_metric(points, "samples")
    guard_metric(layer, "layer")
    union = layer.geometry.union_all()
    return points.geometry.distance(union).to_numpy(dtype=float)


def water_layer(landcover_polygons: gpd.GeoDataFrame | None, cfg) -> gpd.GeoDataFrame | None:
    """Pick the water polygons out of the land-cover layer using config codes."""
    if landcover_polygons is None or landcover_polygons.empty:
        return None
    codes = cfg.get("landcover.classes", {}) or {}
    water_codes = [int(k) for k, v in codes.items() if str(v) == "water"]
    if "class_name" in landcover_polygons.columns:
        water = landcover_polygons[landcover_polygons["class_name"] == "water"]
    else:
        water = landcover_polygons[landcover_polygons["class_code"].isin(water_codes)]
    return water if not water.empty else None


# ----------------------------------------------------------------------
# sample features
# ----------------------------------------------------------------------
def extract_sample_features(
    cfg,
    samples: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    landcover_polygons: gpd.GeoDataFrame | None,
    ndvi_path,
    landcover_raster_path,
    crs: str,
    cache_dir=None,
) -> pd.DataFrame:
    """Build the ``sample_features`` table (Section 6, step 3)."""
    buffer_m = float(cfg.get("features.buffer_m", 100.0))
    lookup = cfg.get("landcover.synthetic_classes", {}) or {}
    lc_lookup = cfg.get("landcover.classes", {}) or {}

    if ndvi_path is not None:
        ndvi_metric = raster_to_metric(ndvi_path, crs, cache_dir)
        ndvi_mean = buffer_ndvi_zonal(samples, ndvi_metric, crs, buffer_m)
    else:
        ndvi_mean = np.full(len(samples), np.nan)

    builtup = buffer_builtup_fraction(samples, buildings, crs, buffer_m)

    if landcover_raster_path is not None:
        lc_metric = raster_to_metric(landcover_raster_path, crs, cache_dir)
        codes = dominant_class_under_points(samples, lc_metric, crs)
    else:
        codes = np.full(len(samples), np.nan)

    landcover_class = [code_to_name(c, lookup, lc_lookup) for c in codes]

    water = water_layer(landcover_polygons, cfg)
    if water is not None:
        water_m = water.to_crs(crs)
        dist_water = distance_to_layer(samples, water_m, crs)
    else:
        dist_water = np.full(len(samples), np.nan)

    return pd.DataFrame(
        {
            "sample_id": samples["sample_id"].to_numpy(),
            "ndvi_mean": ndvi_mean,
            "builtup_frac": builtup,
            "landcover_class": landcover_class,
            "dist_water_m": dist_water,
        }
    )


# ----------------------------------------------------------------------
# grid covariates
# ----------------------------------------------------------------------
def extract_grid_features(
    cfg,
    grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    landcover_polygons: gpd.GeoDataFrame | None,
    ndvi_path,
    landcover_raster_path,
    crs: str,
    cache_dir=None,
) -> gpd.GeoDataFrame:
    """Attach NDVI, built-up fraction, land cover and distance-to-water to the grid."""
    out = grid.copy()
    lookup = cfg.get("landcover.synthetic_classes", {}) or {}
    lc_lookup = cfg.get("landcover.classes", {}) or {}

    # NDVI: mean inside each 100 m cell (polygon zonal statistics)
    if ndvi_path is not None:
        ndvi_metric = raster_to_metric(ndvi_path, crs, cache_dir)
        stats = zonal_stats(
            out.geometry,
            str(ndvi_metric),
            stats=["mean"],
            all_touched=False,
        )
        out["ndvi_mean"] = [s["mean"] for s in stats]
    else:
        out["ndvi_mean"] = np.nan

    out["builtup_frac"] = grid_builtup_fraction(out, buildings, crs)

    if landcover_raster_path is not None:
        lc_metric = raster_to_metric(landcover_raster_path, crs, cache_dir)
        codes = dominant_class_under_points(
            gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(out["centroid_x"], out["centroid_y"]), crs=crs
            ),
            lc_metric,
            crs,
        )
    else:
        codes = np.full(len(out), np.nan)

    out["landcover_class"] = [code_to_name(c, lookup, lc_lookup) for c in codes]

    water = water_layer(landcover_polygons, cfg)
    if water is not None:
        water_m = water.to_crs(crs)
        union = water_m.geometry.union_all()
        out["dist_water_m"] = out.geometry.centroid.distance(union).to_numpy()
    else:
        out["dist_water_m"] = np.nan

    # interpolate any remaining gaps so the grid has no NaN covariates
    for col in ["ndvi_mean", "builtup_frac"]:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    out["dist_water_m"] = out["dist_water_m"].fillna(out["dist_water_m"].max())
    return out


def grid_builtup_fraction(
    grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None, crs: str
) -> np.ndarray:
    """Exact area overlay: area(buildings ∩ cell) / area(cell)."""
    n = len(grid)
    if buildings is None or buildings.empty:
        return np.zeros(n, dtype=float)

    guard_metric(grid, "grid")
    guard_metric(buildings, "buildings")

    inter = gpd.overlay(
        grid[["cell_id", "geometry"]],
        buildings[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    if inter.empty:
        return np.zeros(n, dtype=float)
    inter["area"] = inter.geometry.area
    areas = inter.groupby("cell_id")["area"].sum()
    cell_area = grid.set_index("cell_id").geometry.area
    frac = (areas / cell_area).reindex(grid["cell_id"].to_numpy()).fillna(0.0)
    return np.clip(frac.to_numpy(dtype=float), 0.0, 1.0)
