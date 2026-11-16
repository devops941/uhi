"""SQLite (tabular results) + GeoPackage (spatial layers) storage helpers.

Schema follows Section 8 of the project spec. Tables are created from a single
DDL script so that ``init_db`` is idempotent.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ----------------------------------------------------------------------
# DDL - Section 8 of the spec
# ----------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wards (
    ward_id     TEXT PRIMARY KEY,
    ward_name   TEXT,
    area_km2    REAL
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id   TEXT PRIMARY KEY,
    lat         REAL,
    lon         REAL,
    x_m         REAL,
    y_m         REAL,
    temp_c      REAL,
    datetime    TEXT,
    ward_id     TEXT REFERENCES wards(ward_id),
    is_outlier  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sample_features (
    sample_id       TEXT PRIMARY KEY REFERENCES samples(sample_id),
    ndvi_mean       REAL,
    builtup_frac    REAL,
    landcover_class TEXT,
    dist_water_m    REAL
);

CREATE TABLE IF NOT EXISTS grid_cells (
    cell_id         INTEGER PRIMARY KEY,
    ward_id         TEXT REFERENCES wards(ward_id),
    centroid_x      REAL,
    centroid_y      REAL,
    ndvi_mean       REAL,
    builtup_frac    REAL,
    landcover_class TEXT,
    temp_idw        REAL,
    uhi_intensity   REAL,
    gi_zscore       REAL,
    gi_pvalue       REAL,
    is_hotspot      INTEGER,
    priority_score  REAL,
    priority_rank   INTEGER
);

CREATE TABLE IF NOT EXISTS correlation_results (
    variable        TEXT PRIMARY KEY,
    pearson_r       REAL,
    pearson_p       REAL,
    spearman_rho    REAL,
    spearman_p      REAL,
    n               INTEGER
);

CREATE TABLE IF NOT EXISTS regression_results (
    term        TEXT PRIMARY KEY,
    coef        REAL,
    std_err     REAL,
    p_value     REAL,
    ci_low      REAL,
    ci_high     REAL,
    vif         REAL
);

CREATE TABLE IF NOT EXISTS model_summary (
    metric  TEXT PRIMARY KEY,
    value   REAL
);

CREATE TABLE IF NOT EXISTS idw_cv_results (
    power       REAL PRIMARY KEY,
    rmse        REAL,
    mae         REAL,
    r2          REAL,
    is_selected INTEGER
);

CREATE TABLE IF NOT EXISTS ward_summary (
    ward_id           TEXT PRIMARY KEY REFERENCES wards(ward_id),
    mean_temp         REAL,
    mean_uhi          REAL,
    mean_ndvi         REAL,
    mean_builtup      REAL,
    hotspot_area_pct  REAL,
    priority_score    REAL,
    priority_rank     INTEGER
);

CREATE TABLE IF NOT EXISTS group_comparison (
    group_a        TEXT,
    group_b        TEXT,
    mean_a         REAL,
    mean_b         REAL,
    diff           REAL,
    welch_p        REAL,
    mannwhitney_p  REAL,
    cohens_d       REAL
);

-- extension from spec Section 6: IDW vs Ordinary Kriging on the same LOO split
CREATE TABLE IF NOT EXISTS interpolation_comparison (
    method     TEXT PRIMARY KEY,
    parameter  TEXT,
    rmse       REAL,
    mae        REAL,
    r2         REAL,
    is_best    INTEGER
);
"""

SPATIAL_LAYERS = ["boundary", "wards", "samples", "grid", "buildings", "hotspots"]


def init_db(cfg) -> Engine:
    """Create the SQLite database (idempotent) and return a SQLAlchemy engine."""
    db_path = cfg.path_for("sqlite_db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for statement in filter(None, (s.strip() for s in SCHEMA_SQL.split(";"))):
            conn.execute(text(statement))
    return engine


def get_engine(cfg) -> Engine:
    db_path = cfg.path_for("sqlite_db")
    if not db_path.exists():
        return init_db(cfg)
    return create_engine(f"sqlite:///{db_path}")


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def write_table(engine: Engine, table: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
    if df is None or df.empty:
        return
    df.to_sql(table, engine, if_exists=if_exists, index=False)


def write_geodataframe(gdf: gpd.GeoDataFrame, gpkg_path: str | Path, layer: str) -> None:
    """Write (or overwrite) a spatial layer inside the GeoPackage."""
    if gdf is None or gdf.empty:
        return
    p = Path(gpkg_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(p, layer=layer, driver="GPKG")


def write_layers(cfg, layers: dict[str, gpd.GeoDataFrame]) -> Path:
    """Write the standard set of spatial layers to the GeoPackage."""
    gpkg = cfg.path_for("geopackage")
    for name, gdf in layers.items():
        write_geodataframe(gdf, gpkg, name)
    return gpkg


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
def read_table(engine: Engine, table: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(f'SELECT * FROM "{table}"'), conn)


def table_names(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()
    return [r[0] for r in rows]


def gpkg_layers(gpkg_path: str | Path) -> list[str]:
    """List the feature layers inside a GeoPackage.

    Uses the GDAL bindings that ship with rasterio/pyogrio rather than fiona,
    which is not a required dependency of this project.
    """
    p = Path(gpkg_path)
    if not p.exists():
        return []
    try:
        from pyogrio import list_layers

        return [row[0] for row in list_layers(p)]
    except ImportError:  # pragma: no cover - pyogrio is a geopandas dependency
        from osgeo import ogr

        ds = ogr.Open(str(p))
        if ds is None:
            return []
        names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
        ds = None
        return names


def read_layer(gpkg_path: str | Path, layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(gpkg_path, layer=layer)


def summarise_db(engine: Engine) -> dict[str, Any]:
    """Row counts for every table - handy for the pipeline log and the dashboard."""
    out: dict[str, Any] = {}
    with engine.connect() as conn:
        for name in table_names(engine):
            try:
                out[name] = int(conn.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar())
            except Exception:  # pragma: no cover - defensive
                out[name] = None
    return out


def fetch_all(engine: Engine, tables: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {t: read_table(engine, t) for t in tables}
