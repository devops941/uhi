"""Synthetic city generator - the mandatory download-free fallback.

Produces a 5 km x 5 km study area with an urban core, parks, a lake and a rural
fringe, together with all input layers the pipeline needs:

    samples.csv        lat, lon, temp_c, datetime          (EPSG:4326)
    ndvi.tif           single band, -1..1                  (EPSG:4326)
    landcover.tif      integer class codes                 (EPSG:4326)
    buildings.geojson  building footprints                 (EPSG:4326)
    boundary.geojson   city boundary                       (EPSG:4326)
    wards.geojson      ward polygons                       (EPSG:4326)

Temperature follows the spec's generative model::

    T = base + a * builtup - b * ndvi - c * water_proximity + spatial_noise

The true coefficients are written to ``synthetic_truth.json`` so the
synthetic-truth regression test can check that the pipeline recovers them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

# Coimbatore, Tamil Nadu - UTM zone 43N
CITY_LAT = 11.0168
CITY_LON = 76.9558
METRIC_CRS = "EPSG:32643"


@dataclass
class SyntheticCity:
    """Container for the generated layers (all in the metric CRS)."""

    crs: str
    transform: Affine
    size_m: float
    cell_m: float
    ndvi: np.ndarray
    builtup: np.ndarray
    water_prox: np.ndarray
    landcover: np.ndarray
    temp_field: np.ndarray
    boundary: Polygon = None
    wards: gpd.GeoDataFrame = None
    buildings: gpd.GeoDataFrame = None
    samples: pd.DataFrame = None
    truth: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ndvi.shape


def _centre_xy() -> tuple[float, float]:
    tr = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    return tr.transform(CITY_LON, CITY_LAT)


def _correlated_noise(rng: np.random.Generator, shape, length_cells: float) -> np.ndarray:
    """Gaussian random field with an approximate correlation length.

    The field is centred and scaled so it has mean 0 and standard deviation 1,
    which keeps the synthetic base temperature exactly on its configured value.
    """
    white = rng.normal(size=shape)
    sigma = max(length_cells / 2.355, 0.5)
    smooth = gaussian_filter(white, sigma=sigma, mode="reflect")
    smooth = smooth - smooth.mean()
    std = smooth.std()
    return smooth / std if std > 0 else smooth


def _landscape(cfg, rng, cell: float | None = None) -> SyntheticCity:
    size = float(cfg.get("synthetic.size_m", 5000.0))
    if cell is None:
        cell = float(cfg.get("synthetic.raster_res_m", 25.0))
    n = int(round(size / cell))
    cx, cy = _centre_xy()

    # cell-centre coordinates in metric space
    xs = cx - size / 2 + cell * (np.arange(n) + 0.5)
    ys = cy + size / 2 - cell * (np.arange(n) + 0.5)
    xx, yy = np.meshgrid(xs, ys)

    core_r = float(cfg.get("synthetic.urban_core_radius_m", 1500.0))
    # feature sizes are expressed as fractions of the domain so a smaller test
    # city keeps the same layout instead of the lake swallowing the core
    s = size / 5000.0
    # urban core slightly off-centre so the map is not perfectly symmetric
    core_x, core_y = cx + 0.10 * size, cy - 0.05 * size
    d_core = np.hypot(xx - core_x, yy - core_y)

    # ---- water: an elliptical lake plus a narrow river -------------------
    lake_x, lake_y = cx - 0.30 * size, cy + 0.26 * size
    lake = (((xx - lake_x) / (520.0 * s)) ** 2 + ((yy - lake_y) / (360.0 * s)) ** 2) <= 1.0
    river_amp = 90 * s
    river_half = 60 * s
    river_path = (yy < cy + 0.02 * size + river_amp * np.sin((xx - cx) / (420.0 * s))) & (
        yy > cy + 0.02 * size - river_amp * np.sin((xx - cx) / (420.0 * s)) - river_half
    )
    river_path &= (xx > cx - 0.45 * size) & (xx < cx + 0.45 * size)
    water = lake | river_path

    # ---- parks -----------------------------------------------------------
    parks = np.zeros_like(water)
    for px, py, pr in [
        (cx - 0.16 * size, cy - 0.20 * size, 330.0 * s),
        (cx + 0.26 * size, cy + 0.14 * size, 260.0 * s),
        (cx + 0.02 * size, cy - 0.34 * size, 210.0 * s),
    ]:
        parks |= np.hypot(xx - px, yy - py) <= pr

    # ---- built-up fraction ----------------------------------------------
    builtup = np.exp(-((d_core / (0.62 * core_r)) ** 2))
    # smooth texture: the correlation length must be much larger than the
    # 100 m feature buffer, otherwise the buffer mean and the point value
    # disagree and the regression sees noise instead of signal
    texture_len = float(cfg.get("synthetic.texture_length_m", 700.0))
    texture = _correlated_noise(rng, (n, n), texture_len / cell)
    builtup = builtup + 0.08 * texture
    # secondary satellite settlements in the rural fringe
    for sx, sy in [
        (cx - 0.40 * size, cy - 0.36 * size),
        (cx + 0.40 * size, cy - 0.30 * size),
        (cx - 0.36 * size, cy + 0.38 * size),
    ]:
        builtup += 0.55 * np.exp(-(((xx - sx) / (420.0 * s)) ** 2 + ((yy - sy) / (420.0 * s)) ** 2))
    builtup = np.where(water, 0.0, builtup)
    builtup = np.where(parks, builtup * 0.10, builtup)
    builtup = np.clip(builtup, 0.0, 1.0)

    # ---- NDVI ------------------------------------------------------------
    # partly driven by built-up, partly independent vegetation variation so the
    # two predictors are correlated but not collinear (keeps VIF sane)
    ndvi = 0.78 - 0.62 * builtup
    ndvi = ndvi + 0.10 * _correlated_noise(rng, (n, n), texture_len / cell)
    ndvi = np.where(parks, np.maximum(ndvi, 0.62), ndvi)
    ndvi = np.where(water, -0.35, ndvi)
    ndvi = np.clip(ndvi, -0.6, 0.95)

    # ---- water proximity (1 at the water edge, decaying outwards) --------
    d_water = np.abs(_distance_to_mask(water, cell))
    water_prox = np.exp(-d_water / 700.0)

    # ---- land cover classes ---------------------------------------------
    landcover = np.full((n, n), 1, dtype=np.int16)  # grassland default
    landcover = np.where(builtup > 0.45, 3, landcover)  # built-up
    landcover = np.where((builtup <= 0.45) & (ndvi > 0.55), 0, landcover)  # tree cover
    landcover = np.where((builtup <= 0.15) & (ndvi <= 0.45), 2, landcover)  # cropland
    landcover = np.where((builtup < 0.05) & (ndvi < 0.20), 5, landcover)  # bare
    landcover = np.where(water, 4, landcover)  # water
    landcover = np.where(parks, 0, landcover)  # parks are tree cover

    # ---- temperature truth field ----------------------------------------
    base = float(cfg.get("synthetic.base_temp_c", 29.0))
    a = float(cfg.get("synthetic.coef_builtup", 7.0))
    b = float(cfg.get("synthetic.coef_ndvi", 5.0))
    c = float(cfg.get("synthetic.coef_water", 3.0))
    noise_sd = float(cfg.get("synthetic.noise_sd", 0.6))
    corr_len = float(cfg.get("synthetic.noise_correlation_length_m", 800.0))

    # water cools; the lake itself is coldest. The coefficients are stored with
    # their signs (built-up positive, NDVI and water proximity negative), so the
    # temperature is simply the weighted sum - no extra sign is applied here.
    temp = base + a * builtup + b * np.clip(ndvi, 0, 1) + c * water_prox
    temp = np.where(water, base + c, temp)
    temp += noise_sd * _correlated_noise(rng, (n, n), corr_len / cell)

    transform = Affine.translation(cx - size / 2, cy + size / 2) * Affine.scale(cell, -cell)
    boundary = box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)

    truth = {
        "base_temp_c": base,
        "coef_builtup": a,
        "coef_ndvi": b,
        "coef_water": c,
        "noise_sd": noise_sd,
        "crs_metric": METRIC_CRS,
        "raster_res_m": cell,
        "size_m": size,
    }
    return SyntheticCity(
        crs=METRIC_CRS,
        transform=transform,
        size_m=size,
        cell_m=cell,
        ndvi=ndvi,
        builtup=builtup,
        water_prox=water_prox,
        landcover=landcover,
        temp_field=temp,
        boundary=boundary,
        truth=truth,
    )


def _distance_to_mask(mask: np.ndarray, cell_m: float) -> np.ndarray:
    """Approximate Euclidean distance (metres) from every cell to the mask."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~mask) * cell_m


# ----------------------------------------------------------------------
# vector layers
# ----------------------------------------------------------------------
def _build_wards(city: SyntheticCity, rng) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = city.boundary.bounds
    n_side = 4
    step = (maxx - minx) / n_side
    jitter = 0.30 * step
    centres = []
    for i in range(n_side):
        for j in range(n_side):
            px = minx + (i + 0.5) * step + rng.uniform(-jitter, jitter)
            py = miny + (j + 0.5) * step + rng.uniform(-jitter, jitter)
            centres.append((px, py))
    centres.append(((minx + maxx) / 2, (miny + maxy) / 2))

    # add far-away points so the outer Voronoi cells are bounded
    pad = 4 * (maxx - minx)
    frame = [
        (minx - pad, miny - pad),
        (minx - pad, maxy + pad),
        (maxx + pad, miny - pad),
        (maxx + pad, maxy + pad),
    ]
    vor = Voronoi(np.array(centres + frame))

    polys, names = [], []
    for k in range(len(centres)):
        region = vor.regions[vor.point_region[k]]
        if not region or -1 in region:
            continue
        poly = Polygon([vor.vertices[i] for i in region]).intersection(city.boundary)
        if poly.is_empty or poly.area <= 0:
            continue
        polys.append(poly)
        names.append(f"Ward {len(polys):02d}")

    wards = gpd.GeoDataFrame(
        {"ward_name": names},
        geometry=polys,
        crs=city.crs,
    )
    wards["ward_id"] = [f"W{idx + 1:02d}" for idx in range(len(wards))]
    wards["area_km2"] = wards.geometry.area / 1e6
    return wards[["ward_id", "ward_name", "area_km2", "geometry"]]


def _build_buildings(city: SyntheticCity, rng) -> gpd.GeoDataFrame:
    """Place footprints whose coverage tracks the built-up fraction field.

    Each cell is divided into a 4x4 grid of micro-cells and ``round(frac*16)``
    of them are filled with a building. Coverage therefore matches the
    generating field to within 1/16, and footprints never overlap. That makes
    the overlay-derived ``builtup_frac`` feature a faithful measurement of the
    truth, so the regression recovers the generating coefficient instead of a
    shrunken (or sign-flipped) version of it.
    """
    rows, cols = np.where(city.builtup > 0.02)
    if len(rows) == 0:
        return gpd.GeoDataFrame({"building_id": [], "geometry": []}, crs=city.crs)

    micro = 4
    micro_side = city.cell_m / micro
    polys, ids = [], []
    bid = 0
    for r, c in zip(rows, cols, strict=False):
        frac = float(city.builtup[r, c])
        n_build = int(round(frac * micro * micro))
        if n_build <= 0:
            continue
        # deterministic, evenly spread selection of micro-cells
        slots = [(i, j) for i in range(micro) for j in range(micro)]
        step = len(slots) / n_build
        chosen = [slots[int(k * step)] for k in range(n_build)]

        minx, maxy = city.transform * (c, r)
        for mi, mj in chosen:
            bx = minx + mj * micro_side
            by = maxy - (mi + 1) * micro_side
            polys.append(box(bx, by, bx + micro_side, by + micro_side))
            ids.append(f"B{bid:05d}")
            bid += 1
    return gpd.GeoDataFrame({"building_id": ids}, geometry=polys, crs=city.crs)


# ----------------------------------------------------------------------
# sample points
# ----------------------------------------------------------------------
def _build_samples(city: SyntheticCity, cfg, rng) -> pd.DataFrame:
    n_samples = int(cfg.get("synthetic.n_samples", 200))
    cell = city.cell_m
    cx, cy = _centre_xy()

    # stratified design: over-sample the urban core so the regression has
    # contrast in built-up fraction, then a rural fringe sample.
    urban_share = 0.55
    n_urban = int(round(n_samples * urban_share))
    n_rural = n_samples - n_urban

    def draw(mask_rows, mask_cols, count):
        if len(mask_rows) == 0:
            return np.empty((0, 2), dtype=int)
        idx = rng.choice(len(mask_rows), size=count, replace=count > len(mask_rows))
        return np.column_stack([mask_rows[idx], mask_cols[idx]])

    ur, uc = np.where(city.builtup > 0.35)
    rr, rc = np.where(city.builtup <= 0.35)

    picks = np.vstack([draw(ur, uc, n_urban), draw(rr, rc, n_rural)])
    # random position inside the chosen cell
    jitter = (rng.random(picks.shape) - 0.5) * cell * 0.9
    cols = picks[:, 1]
    rows = picks[:, 0]
    px = city.transform.c + (cols + 0.5) * cell + jitter[:, 0]
    py = city.transform.f + (rows + 0.5) * (-cell) + jitter[:, 1]

    to_wgs = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(px, py)

    # temperature: truth field value at the sampled cell + local noise
    noise_sd = float(cfg.get("synthetic.noise_sd", 0.6))
    temp = city.temp_field[rows, cols] + rng.normal(0, noise_sd * 0.8, size=len(rows))

    datetimes = pd.to_datetime("2023-04-14 10:30:00") + pd.to_timedelta(
        rng.integers(0, 180, size=len(rows)), unit="m"
    )

    df = pd.DataFrame(
        {
            "sample_id": [f"S{i:04d}" for i in range(len(rows))],
            "lat": lat,
            "lon": lon,
            "temp_c": temp,
            "datetime": datetimes.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    # a few deliberate data-quality problems so the cleaning step has work to do
    if len(df) > 20:
        df.loc[df.index[3], "temp_c"] = 999.0  # out of plausible range
        df.loc[df.index[7], "temp_c"] = -5.0  # out of plausible range
        df.loc[df.index[11], "lat"] = df.loc[df.index[10], "lat"]
        df.loc[df.index[11], "lon"] = df.loc[df.index[10], "lon"]  # duplicate coords
    return df


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def _write_raster_4326(
    path: Path, array: np.ndarray, transform: Affine, crs: str, dtype: str, nodata: float
) -> None:
    """Reproject a metric array to EPSG:4326 and write it as a GeoTIFF."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(
        path.with_suffix(".tmp.tif"),
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as src:
        src.write(array.astype(dtype), 1)
        dst_crs = "EPSG:4326"
        dst_transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=dst_transform, width=width, height=height)
        with rasterio.open(path, "w", **profile) as dst:
            resampling = Resampling.nearest if dtype.startswith("int") else Resampling.bilinear
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )
    path.with_suffix(".tmp.tif").unlink(missing_ok=True)


def generate_synthetic_city(cfg, out_dir: str | Path | None = None) -> dict:
    """Generate the synthetic city and write every raw layer to disk."""
    raw_dir = Path(out_dir) if out_dir else cfg.path_for("data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("synthetic.seed", 42))
    rng = np.random.default_rng(seed)

    city = _landscape(cfg, rng)
    city.wards = _build_wards(city, rng)
    city.buildings = _build_buildings(city, rng)
    city.samples = _build_samples(city, cfg, rng)

    to_wgs = Transformer.from_crs(city.crs, "EPSG:4326", always_xy=True)

    # ---- rasters (raw copies are EPSG:4326 per the CRS rule) -------------
    ndvi_path = raw_dir / "ndvi.tif"
    lc_path = raw_dir / "landcover.tif"
    _write_raster_4326(ndvi_path, city.ndvi, city.transform, city.crs, "float32", -9999.0)
    _write_raster_4326(lc_path, city.landcover, city.transform, city.crs, "int16", -1)

    # ---- vector layers ---------------------------------------------------
    boundary = gpd.GeoDataFrame(
        {"name": [cfg.get("project.city", "Synthetic City")]},
        geometry=[city.boundary],
        crs=city.crs,
    ).to_crs("EPSG:4326")
    boundary_path = raw_dir / "boundary.geojson"
    boundary.to_file(boundary_path, driver="GeoJSON")

    wards_path = raw_dir / "wards.geojson"
    city.wards.to_crs("EPSG:4326").to_file(wards_path, driver="GeoJSON")

    buildings_path = raw_dir / "buildings.geojson"
    city.buildings.to_crs("EPSG:4326").to_file(buildings_path, driver="GeoJSON")

    # land cover polygons for map display
    lc_poly_path = raw_dir / "landcover.geojson"
    _landcover_polygons(city, cfg).to_file(lc_poly_path, driver="GeoJSON")

    samples_path = raw_dir / "samples.csv"
    city.samples.to_csv(samples_path, index=False)

    truth_path = raw_dir / "synthetic_truth.json"
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(city.truth, fh, indent=2)

    return {
        "city": city,
        "samples_csv": samples_path,
        "ndvi_tif": ndvi_path,
        "landcover_tif": lc_path,
        "landcover_geojson": lc_poly_path,
        "buildings_geojson": buildings_path,
        "boundary_geojson": boundary_path,
        "wards_geojson": wards_path,
        "truth_json": truth_path,
        "crs_metric": city.crs,
        "_to_wgs": to_wgs,
    }


def _landcover_polygons(city: SyntheticCity, cfg) -> gpd.GeoDataFrame:
    """Vectorise the land-cover raster into polygons (for the map layers)."""
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape

    lookup = cfg.get("landcover.synthetic_classes", {})
    records = []
    for geom, value in rio_shapes(city.landcover.astype("int32"), transform=city.transform):
        code = int(value)
        if code < 0:
            continue
        records.append(
            {
                "class_code": code,
                "class_name": lookup.get(str(code), lookup.get(code, f"class_{code}")),
                "geometry": shapely_shape(geom),
            }
        )
    gdf = gpd.GeoDataFrame(records, crs=city.crs)
    return gdf.dissolve(by=["class_code", "class_name"]).reset_index().to_crs("EPSG:4326")
