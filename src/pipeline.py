"""End-to-end pipeline CLI.

Usage::

    python -m src.pipeline --source synthetic --config config.yaml
    python -m src.pipeline --source real
    python -m src.pipeline --source synthetic --skip-generate
    python -m src.pipeline --source synthetic --report-pdf

Runs every phase in order (spec Section 10): data layer -> features ->
statistics -> interpolation -> UHI/hotspots -> priority -> outputs -> report,
writing results into SQLite, a GeoPackage, GeoTIFFs and figures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.analysis import idw as idw_mod
from src.analysis import interpolation_compare as interp_mod
from src.analysis import priority as priority_mod
from src.analysis import stats as stats_mod
from src.analysis import uhi as uhi_mod
from src.config import load_config
from src.data import db as db_mod
from src.data import ingest, preprocess
from src.data.synthetic import generate_synthetic_city
from src.features import extract
from src.report import generate as report_mod
from src.viz import maps as viz_mod

RESULT_TABLES = [
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
]


def log(msg: str) -> None:
    print(f"[uhi] {msg}", flush=True)


# ----------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------
def phase_data(cfg, source: str, skip_generate: bool):
    if source == "synthetic" and not skip_generate:
        log("Phase 1-2 | generating synthetic city (no downloads)")
        generate_synthetic_city(cfg)

    log("Phase 2 | ingesting and validating raw inputs")
    raw = ingest.load_inputs(cfg)
    log(
        f"  samples={raw.report.get('n_samples')} "
        f"rejected_out_of_range={raw.report.get('rejected_out_of_range')}"
    )

    log("Phase 2 | preprocessing (reproject, clip, clean, build grid)")
    prep = preprocess.preprocess(cfg, raw)
    for key, value in prep.report.items():
        log(f"  {key}={value}")

    engine = db_mod.init_db(cfg)
    log(f"Phase 2 | SQLite initialised at {cfg.path_for('sqlite_db')}")
    return raw, prep, engine


def phase_features(cfg, raw, prep, engine):
    log("Phase 3 | extracting sample features (buffers, zonal stats, built-up)")
    feats = extract.extract_sample_features(
        cfg,
        prep.samples,
        prep.buildings,
        raw.landcover_polygons,
        raw.ndvi_path,
        raw.landcover_path,
        prep.crs,
    )

    log("Phase 3 | extracting grid covariates")
    grid = extract.extract_grid_features(
        cfg,
        prep.grid,
        prep.buildings,
        raw.landcover_polygons,
        raw.ndvi_path,
        raw.landcover_path,
        prep.crs,
    )
    return feats, grid


def phase_store(cfg, prep, feats, engine):
    log("Phase 2/3 | writing tables to SQLite and layers to GeoPackage")

    wards_tbl = (
        preprocess.ward_area_km2(prep.wards)
        if prep.wards is not None
        else pd.DataFrame(columns=["ward_id", "ward_name", "area_km2"])
    )
    db_mod.write_table(engine, "wards", wards_tbl)

    s = prep.samples.drop(columns=["geometry"]).copy()
    s["x_m"] = prep.samples.geometry.x.to_numpy()
    s["y_m"] = prep.samples.geometry.y.to_numpy()
    s["datetime"] = pd.to_datetime(s["datetime"]).astype(str)
    if "ward_id" not in s.columns:
        s["ward_id"] = None
    db_mod.write_table(
        engine,
        "samples",
        s[["sample_id", "lat", "lon", "x_m", "y_m", "temp_c", "datetime", "ward_id", "is_outlier"]],
    )
    db_mod.write_table(engine, "sample_features", feats)

    layers = {
        "boundary": prep.boundary.to_crs("EPSG:4326"),
        "samples": prep.samples.to_crs("EPSG:4326"),
    }
    if prep.wards is not None:
        layers["wards"] = prep.wards.to_crs("EPSG:4326")
    if prep.buildings is not None:
        layers["buildings"] = prep.buildings.to_crs("EPSG:4326")
    db_mod.write_layers(cfg, layers)
    log(f"  GeoPackage layers: {', '.join(layers)}")


def phase_stats(cfg, prep, feats, engine):
    log("Phase 4 | correlations, OLS regression, VIF, Moran's I, group tests")
    res = stats_mod.run_stats(cfg, prep.samples, feats)
    db_mod.write_table(engine, "correlation_results", res.correlations)
    db_mod.write_table(engine, "regression_results", res.regression)
    summary_df = pd.DataFrame(
        [
            {"metric": k, "value": float(v)}
            for k, v in res.model_summary.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        ]
    )
    db_mod.write_table(engine, "model_summary", summary_df)
    db_mod.write_table(engine, "group_comparison", res.group_comparison)
    log(
        f"  R2={res.model_summary['r2']:.3f} adj_R2={res.model_summary['adj_r2']:.3f} "
        f"max_VIF={res.model_summary['max_vif']:.2f} "
        f"Moran_I={res.model_summary['moran_i_residuals']:.3f}"
    )
    return res


def phase_interpolation(cfg, prep, engine):
    log("Phase 5 | IDW with leave-one-out CV, then IDW vs kriging comparison")
    xy = np.column_stack([prep.samples.geometry.x, prep.samples.geometry.y])
    z = prep.samples["temp_c"].to_numpy(dtype=float)
    res = idw_mod.interpolate_to_grid(cfg, xy, z, prep.grid, prep.crs, prep.grid_res_m)
    db_mod.write_table(engine, "idw_cv_results", res.cv_table)
    log(
        f"  selected power={res.selected_power} "
        f"LOO RMSE={res.loo_metrics['rmse']:.3f} R2={res.loo_metrics['r2']:.3f}"
    )

    comparison = interp_mod.compare_methods(
        cfg,
        xy,
        z,
        prep.grid,
        prep.crs,
        prep.grid_res_m,
        variogram_model=str(cfg.get("interpolation.variogram_model", "spherical")),
    )
    db_mod.write_table(engine, "interpolation_comparison", comparison.metrics)
    for _, row in comparison.metrics.iterrows():
        log(
            f"  {row['method']:<8} RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} "
            f"R2={row['r2']:.3f}"
        )
    log(f"  better method on LOO: {comparison.best_method}")
    return res, comparison


def attach_idw_to_grid(grid: gpd.GeoDataFrame, idw_res, cell_m: float) -> gpd.GeoDataFrame:
    """Map the IDW surface back onto grid cells using the raster transform."""
    xs = np.sort(grid["centroid_x"].unique())
    ys = np.sort(grid["centroid_y"].unique())
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    values = np.array(
        [
            idw_res.surface[yi[cy], xi[cx]]
            for cx, cy in zip(grid["centroid_x"], grid["centroid_y"], strict=False)
        ],
        dtype=float,
    )
    out = grid.copy()
    out["temp_idw"] = values
    return out


def phase_uhi(cfg, grid, engine):
    log("Phase 6 | UHI intensity and Getis-Ord Gi* hotspots")
    out = uhi_mod.run_uhi_hotspots(grid, cfg)
    stats = out.attrs.get("uhi_stats", {})
    log(
        f"  baseline={stats.get('rural_baseline_c'):.2f} C "
        f"max UHI={stats.get('max_uhi_c'):.2f} C "
        f"hotspots={int(out['is_hotspot'].sum())} "
        f"coldspots={int(out['is_coldspot'].sum())}"
    )
    return out, stats


def phase_priority(cfg, grid, ndvi_coef, engine):
    log("Phase 7 | tree-planting priority score")
    res = priority_mod.score_priority(cfg, grid, ndvi_coef=ndvi_coef)
    db_mod.write_table(engine, "ward_summary", res.ward_summary)
    log("  top wards: " + ", ".join(res.ward_summary.head(3)["ward_id"].astype(str).tolist()))
    if res.cooling_estimate:
        log(
            f"  estimated cooling for +{res.cooling_estimate['ndvi_uplift']} NDVI = "
            f"{res.cooling_estimate['estimated_cooling_c']:.3f} C"
        )
    return res


def phase_outputs(cfg, prep, idw_res, grid, engine, raw):
    log("Phase 8 | writing GeoTIFFs, GeoPackage layers and figures")
    out_dir = cfg.path_for("outputs", create=True)
    fig_dir = cfg.path_for("figures", create=True)

    # --- GeoTIFFs -------------------------------------------------------
    idw_mod.write_geotiff(
        out_dir / "temperature_idw.tif", idw_res.surface, idw_res.transform, prep.crs
    )
    xs = np.sort(grid["centroid_x"].unique())
    ys = np.sort(grid["centroid_y"].unique())
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}

    def grid_to_array(col):
        arr = np.full((len(ys), len(xs)), np.nan, dtype="float32")
        for cx, cy, val in zip(grid["centroid_x"], grid["centroid_y"], grid[col], strict=False):
            arr[yi[cy], xi[cx]] = val
        return arr

    transform = idw_res.transform
    idw_mod.write_geotiff(
        out_dir / "uhi_intensity.tif", grid_to_array("uhi_intensity"), transform, prep.crs
    )
    idw_mod.write_geotiff(
        out_dir / "hotspots.tif", grid_to_array("is_hotspot").astype("float32"), transform, prep.crs
    )
    idw_mod.write_geotiff(
        out_dir / "priority_score.tif", grid_to_array("priority_score"), transform, prep.crs
    )

    # --- GeoPackage grid + hotspots -------------------------------------
    grid_wgs = grid.to_crs("EPSG:4326")
    db_mod.write_geodataframe(grid_wgs, cfg.path_for("geopackage"), "grid")
    hotspots = grid_wgs[grid_wgs["is_hotspot"] == 1]
    if not hotspots.empty:
        db_mod.write_geodataframe(hotspots, cfg.path_for("geopackage"), "hotspots")

    # --- figures --------------------------------------------------------
    extent = (
        xs[0] - prep.grid_res_m / 2,
        xs[-1] + prep.grid_res_m / 2,
        ys[0] - prep.grid_res_m / 2,
        ys[-1] + prep.grid_res_m / 2,
    )
    figures = []
    figures.append(
        viz_mod.plot_surface(
            idw_res.surface,
            "Interpolated temperature (IDW)",
            fig_dir / "temperature_idw.png",
            "inferno",
            extent=extent,
            label="Temperature (C)",
        )
    )
    figures.append(
        viz_mod.plot_surface(
            grid_to_array("uhi_intensity"),
            "UHI intensity",
            fig_dir / "uhi_intensity.png",
            "coolwarm",
            extent=extent,
            label="UHI (C)",
        )
    )
    figures.append(
        viz_mod.plot_surface(
            grid_to_array("ndvi_mean"),
            "Mean NDVI",
            fig_dir / "ndvi.png",
            "YlGn",
            extent=extent,
            label="NDVI",
        )
    )
    figures.append(
        viz_mod.plot_surface(
            grid_to_array("builtup_frac"),
            "Built-up fraction",
            fig_dir / "builtup.png",
            "YlOrBr",
            extent=extent,
            label="Built-up fraction",
        )
    )
    figures.append(viz_mod.plot_hotspots(grid, fig_dir / "hotspots.png"))
    figures.append(viz_mod.plot_idw_cv(idw_res.cv_table, fig_dir / "idw_cv.png"))
    if idw_res.loo_predictions is not None:
        figures.append(
            viz_mod.plot_observed_predicted(
                prep.samples["temp_c"].to_numpy(dtype=float),
                idw_res.loo_predictions,
                fig_dir / "observed_vs_predicted.png",
            )
        )
    figures.append(viz_mod.plot_group_boxplot(grid, fig_dir / "builtup_vs_green.png"))
    figures.append(
        viz_mod.plot_ward_ranking(
            pd.read_sql_table("ward_summary", engine), fig_dir / "ward_ranking.png"
        )
    )
    return [str(Path(f).name) for f in figures], out_dir


def phase_dashboard_assets(cfg, prep, grid):
    log("Phase 8 | precomputing dashboard payload")
    out_dir = cfg.path_for("outputs", create=True)
    dash_dir = out_dir / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)

    keep = [
        "cell_id",
        "ward_id",
        "centroid_x",
        "centroid_y",
        "ndvi_mean",
        "builtup_frac",
        "landcover_class",
        "dist_water_m",
        "temp_idw",
        "uhi_intensity",
        "gi_zscore",
        "gi_pvalue",
        "is_hotspot",
        "is_coldspot",
        "priority_score",
        "priority_rank",
    ]
    keep = [c for c in keep if c in grid.columns]
    grid_wgs = grid.to_crs("EPSG:4326")
    grid_wgs["lon"] = grid_wgs.geometry.centroid.x
    grid_wgs["lat"] = grid_wgs.geometry.centroid.y
    grid_wgs[keep + ["lat", "lon"]].to_parquet(dash_dir / "grid.parquet", index=False)
    prep.samples.to_crs("EPSG:4326").drop(columns="geometry").to_csv(
        dash_dir / "samples.csv", index=False
    )
    log(f"  dashboard assets -> {dash_dir}")


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def run_pipeline(
    cfg, source: str = "synthetic", skip_generate: bool = False, report_pdf: bool = False
) -> dict:
    started = time.time()
    raw, prep, engine = phase_data(cfg, source, skip_generate)
    feats, grid = phase_features(cfg, raw, prep, engine)
    phase_store(cfg, prep, feats, engine)

    stats_res = phase_stats(cfg, prep, feats, engine)
    idw_res, interp_res = phase_interpolation(cfg, prep, engine)
    grid = attach_idw_to_grid(grid, idw_res, prep.grid_res_m)

    # persist grid covariates before the UHI/priority columns land on them
    grid, uhi_stats = phase_uhi(cfg, grid, engine)
    prio = phase_priority(cfg, grid, stats_res.model_summary.get("ndvi_coef"), engine)
    grid = prio.grid

    figures, out_dir = phase_outputs(cfg, prep, idw_res, grid, engine, raw)

    grid_cols = [
        "cell_id",
        "ward_id",
        "centroid_x",
        "centroid_y",
        "ndvi_mean",
        "builtup_frac",
        "landcover_class",
        "temp_idw",
        "uhi_intensity",
        "gi_zscore",
        "gi_pvalue",
        "is_hotspot",
        "priority_score",
        "priority_rank",
    ]
    grid_cols = [c for c in grid_cols if c in grid.columns]
    db_mod.write_table(engine, "grid_cells", grid.drop(columns="geometry")[grid_cols])

    phase_dashboard_assets(cfg, prep, grid)

    mean_temp = float(np.nanmean(grid["temp_idw"]))
    n_hot = int(grid["is_hotspot"].sum())
    results = {
        "source": source,
        "crs_metric": prep.crs,
        "n_samples": prep.n_samples,
        "n_grid_cells": int(len(grid)),
        "grid_cell_m": prep.grid_res_m,
        "mean_temp": mean_temp,
        "n_hotspots": n_hot,
        "n_coldspots": int(grid["is_coldspot"].sum()),
        "hotspot_pct": 100.0 * n_hot / max(len(grid), 1),
        "selected_power": idw_res.selected_power,
        "idw_rmse": idw_res.loo_metrics.get("rmse"),
        "cv_table": idw_res.cv_table,
        "interpolation_comparison": interp_res.metrics,
        "best_interpolation": interp_res.best_method,
        "variogram": interp_res.variogram,
        "correlations": stats_res.correlations,
        "regression": stats_res.regression,
        "standardised": stats_res.standardised,
        "model_summary": stats_res.model_summary,
        "group_comparison": stats_res.group_comparison,
        "uhi_stats": uhi_stats,
        "ward_summary": prio.ward_summary,
        "top_cells": prio.top_cells,
        "priority_weights": prio.weights,
        "cooling_estimate": prio.cooling_estimate,
        "figures": figures,
        "grid": grid,
        "prep": prep,
        "idw": idw_res,
        "raw": raw,
        "engine": engine,
        "elapsed_s": round(time.time() - started, 1),
    }

    log("Phase 9 | rendering auto report")
    report_path = report_mod.write_report(cfg, results, as_pdf=report_pdf)
    results["report_path"] = report_path

    log(f"Done in {results['elapsed_s']}s -> report: {report_path}")
    log(f"DB tables: {db_mod.summarise_db(engine)}")
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Urban Heat Island mapping pipeline")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse existing raw files instead of regenerating the city",
    )
    parser.add_argument(
        "--report-pdf", action="store_true", help="also render the report to PDF (needs weasyprint)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    try:
        run_pipeline(
            cfg, source=args.source, skip_generate=args.skip_generate, report_pdf=args.report_pdf
        )
    except Exception as exc:  # surface a clean message rather than a traceback
        log(f"PIPELINE FAILED: {exc.__class__.__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
