"""Phase 6 - UHI intensity and Getis-Ord Gi* hotspot detection.

UHI intensity is the grid-cell temperature minus the mean temperature of the
rural reference cells (low built-up fraction, out near the city edge). Hotspots
come from ``esda.G_Local`` on the cell temperatures, using queen contiguity
weights by default; a cell is a hotspot when its Gi* p-value clears the
configured confidence and its z-score is positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np


@dataclass
class UHIResult:
    grid: gpd.GeoDataFrame
    rural_reference_cells: np.ndarray
    stats: dict = field(default_factory=dict)


def rural_reference_mask(
    grid: gpd.GeoDataFrame, cfg, centre_xy: tuple[float, float] | None = None
) -> np.ndarray:
    """Cells that qualify as the rural reference for the UHI baseline.

    Requires a low built-up fraction *and* a position near the city outskirts,
    so the core cannot drag the baseline upwards.
    """
    thr = float(cfg.get("uhi.rural_builtup_threshold", 0.10))
    min_dist = float(cfg.get("uhi.rural_min_dist_m", 1500.0))

    if centre_xy is None:
        minx, miny, maxx, maxy = grid.total_bounds
        centre_xy = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

    d_centre = np.hypot(
        grid["centroid_x"].to_numpy() - centre_xy[0],
        grid["centroid_y"].to_numpy() - centre_xy[1],
    )
    mask = (grid["builtup_frac"].to_numpy() <= thr) & (d_centre >= min_dist)
    if mask.sum() < 5:
        # relax the distance requirement rather than fail
        mask = grid["builtup_frac"].to_numpy() <= thr
    if mask.sum() < 5:
        # last resort: the coolest decile of cells
        temps = grid["temp_idw"].to_numpy()
        mask = temps <= np.nanpercentile(temps, 10)
    return mask


def compute_uhi(grid: gpd.GeoDataFrame, cfg) -> UHIResult:
    """Add ``uhi_intensity`` and report the headline UHI statistics."""
    out = grid.copy()
    mask = rural_reference_mask(out, cfg)
    baseline = float(np.nanmean(out.loc[mask, "temp_idw"]))
    out["uhi_intensity"] = out["temp_idw"] - baseline

    thresholds = list(cfg.get("uhi.thresholds_c", [2.0, 4.0]))
    cell_area_km2 = float(out.geometry.area.median()) / 1e6 if len(out) else 0.0
    total_area = float(out.geometry.area.sum()) / 1e6

    stats = {
        "rural_baseline_c": baseline,
        "n_rural_cells": int(mask.sum()),
        "max_uhi_c": float(np.nanmax(out["uhi_intensity"])),
        "mean_uhi_c": float(np.nanmean(out["uhi_intensity"])),
        "min_uhi_c": float(np.nanmin(out["uhi_intensity"])),
        "total_area_km2": total_area,
        "cell_area_km2": cell_area_km2,
    }
    for t in thresholds:
        above = out["uhi_intensity"] >= float(t)
        stats[f"area_above_{t:g}c_km2"] = float(out.loc[above].geometry.area.sum()) / 1e6
        stats[f"pct_above_{t:g}c"] = (
            float(out.loc[above].geometry.area.sum() / out.geometry.area.sum() * 100)
            if total_area > 0
            else 0.0
        )
    return UHIResult(grid=out, rural_reference_cells=np.where(mask)[0], stats=stats)


# ----------------------------------------------------------------------
# Getis-Ord Gi*
# ----------------------------------------------------------------------
def build_weights(grid: gpd.GeoDataFrame, cfg):
    """Spatial weights for the grid: queen contiguity or k-nearest neighbours."""
    kind = str(cfg.get("hotspots.weights", "queen")).lower()
    if kind == "knn":
        from libpysal.weights import KNN

        coords = np.column_stack([grid["centroid_x"], grid["centroid_y"]])
        w = KNN(coords, k=int(cfg.get("hotspots.knn_k", 8)))
    else:
        from libpysal.weights import Queen

        w = Queen.from_dataframe(grid, use_index=False)
    w.transform = "r"
    return w


def detect_hotspots(grid: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    """Getis-Ord Gi* on ``temp_idw``; adds z-score, p-value and hotspot flags."""
    from esda import G_Local

    out = grid.copy()
    out["gi_zscore"] = np.nan
    out["gi_pvalue"] = np.nan
    out["is_hotspot"] = 0
    out["is_coldspot"] = 0

    y = out["temp_idw"].to_numpy(dtype=float)
    if len(out) < 5 or not np.isfinite(y).all():
        return out

    w = build_weights(out, cfg)
    permutations = int(cfg.get("hotspots.permutations", 999))
    confidence = float(cfg.get("hotspots.confidence", 0.95))
    alpha = 1.0 - confidence

    g = G_Local(y, w, transform="R", permutations=permutations, star=True, seed=12345)
    zs = np.asarray(g.Zs, dtype=float)
    p = np.asarray(g.p_sim, dtype=float)

    out["gi_zscore"] = zs
    out["gi_pvalue"] = p
    out["is_hotspot"] = ((p <= alpha) & (zs > 0)).astype(int)
    out["is_coldspot"] = ((p <= alpha) & (zs < 0)).astype(int)
    return out


def run_uhi_hotspots(grid: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    """UHI intensity then Gi* hotspots, in the order the spec lists them."""
    uhi = compute_uhi(grid, cfg)
    out = detect_hotspots(uhi.grid, cfg)
    # keep attrs JSON-safe: pandas stores them in Parquet/GPKG metadata
    out.attrs["uhi_stats"] = {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in uhi.stats.items()
    }
    out.attrs["n_rural_cells"] = int(len(uhi.rural_reference_cells))
    return out
