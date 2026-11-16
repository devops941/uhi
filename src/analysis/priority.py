"""Phase 7 - tree-planting priority scoring.

Per the spec (Section 6, step 10)::

    priority = 0.4*norm(temp) + 0.3*(1 - norm(NDVI))
             + 0.2*norm(builtup_frac) + 0.1*norm(pop_proxy)

where ``norm`` is min-max scaling, the weights are configurable, and the
population proxy is building density. Outputs the top wards and top cells, plus
an estimated cooling for a +0.1 NDVI uplift taken from the regression
coefficient (an estimate, not a guarantee).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"temp": 0.4, "low_ndvi": 0.3, "builtup": 0.2, "pop_proxy": 0.1}


def minmax(values: np.ndarray) -> np.ndarray:
    """Min-max scale to 0..1; a constant array maps to 0.5."""
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def _population_proxy(grid: gpd.GeoDataFrame) -> np.ndarray:
    """Building density is the population proxy (see config priority.weights)."""
    if "building_density" in grid.columns:
        return grid["building_density"].to_numpy(dtype=float)
    if "pop_proxy" in grid.columns:
        return grid["pop_proxy"].to_numpy(dtype=float)
    return grid["builtup_frac"].to_numpy(dtype=float)


def normalised_components(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """The four normalised components of the priority score."""
    temp = minmax(grid["temp_idw"].to_numpy(dtype=float))
    low_ndvi = 1.0 - minmax(grid["ndvi_mean"].to_numpy(dtype=float))
    builtup = minmax(grid["builtup_frac"].to_numpy(dtype=float))
    pop = minmax(_population_proxy(grid))
    return pd.DataFrame(
        {
            "n_temp": temp,
            "n_low_ndvi": low_ndvi,
            "n_builtup": builtup,
            "n_pop_proxy": pop,
        },
        index=grid.index,
    )


@dataclass
class PriorityResult:
    grid: gpd.GeoDataFrame
    ward_summary: pd.DataFrame
    top_cells: pd.DataFrame
    weights: dict
    cooling_estimate: dict = field(default_factory=dict)


def score_priority(
    cfg,
    grid: gpd.GeoDataFrame,
    ndvi_coef: float | None = None,
    weights: dict | None = None,
) -> PriorityResult:
    """Compute the priority score, ranks, ward summary and top cells."""
    weights = weights or cfg.get("priority.weights", DEFAULT_WEIGHTS) or DEFAULT_WEIGHTS
    weights = {k: float(v) for k, v in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("priority weights must sum to a positive number")
    weights = {k: v / total for k, v in weights.items()}

    out = grid.copy()
    comp = normalised_components(out)
    score = (
        weights.get("temp", 0.0) * comp["n_temp"]
        + weights.get("low_ndvi", 0.0) * comp["n_low_ndvi"]
        + weights.get("builtup", 0.0) * comp["n_builtup"]
        + weights.get("pop_proxy", 0.0) * comp["n_pop_proxy"]
    )
    out["priority_score"] = score.to_numpy()
    out["priority_rank"] = out["priority_score"].rank(ascending=False, method="first").astype(int)

    top_n_cells = int(cfg.get("priority.top_cells", 50))
    cell_cols = [
        "cell_id",
        "ward_id",
        "centroid_x",
        "centroid_y",
        "temp_idw",
        "uhi_intensity",
        "ndvi_mean",
        "builtup_frac",
        "landcover_class",
        "is_hotspot",
        "priority_score",
        "priority_rank",
    ]
    cell_cols = [c for c in cell_cols if c in out.columns]
    top_cells = out.sort_values("priority_rank").head(top_n_cells)[cell_cols].reset_index(drop=True)

    ward_summary = summarise_wards(out, cfg)

    uplift = float(cfg.get("priority.ndvi_uplift", 0.1))
    cooling = {}
    if ndvi_coef is not None and np.isfinite(ndvi_coef):
        cooling = {
            "ndvi_coef_c_per_unit": float(ndvi_coef),
            "ndvi_uplift": uplift,
            "estimated_cooling_c": float(abs(ndvi_coef) * uplift),
            "note": (
                "Estimate from the OLS NDVI coefficient; treat as an indicative "
                "effect size, not a guarantee."
            ),
        }

    return PriorityResult(
        grid=out,
        ward_summary=ward_summary,
        top_cells=top_cells,
        weights=weights,
        cooling_estimate=cooling,
    )


def summarise_wards(grid: gpd.GeoDataFrame, cfg) -> pd.DataFrame:
    """Per-ward means plus a priority rank (wards with no cells are dropped)."""
    if "ward_id" not in grid.columns or grid["ward_id"].isna().all():
        return pd.DataFrame(
            columns=[
                "ward_id",
                "mean_temp",
                "mean_uhi",
                "mean_ndvi",
                "mean_builtup",
                "hotspot_area_pct",
                "priority_score",
                "priority_rank",
            ]
        )

    work = grid.dropna(subset=["ward_id"]).copy()
    work["_hot_area"] = work["is_hotspot"].astype(float) * work.geometry.area
    work["_area"] = work.geometry.area

    agg = (
        work.groupby("ward_id")
        .apply(
            lambda g: pd.Series(
                {
                    "mean_temp": np.nanmean(g["temp_idw"]),
                    "mean_uhi": np.nanmean(g["uhi_intensity"]),
                    "mean_ndvi": np.nanmean(g["ndvi_mean"]),
                    "mean_builtup": np.nanmean(g["builtup_frac"]),
                    "hotspot_area_pct": (
                        100.0 * g["_hot_area"].sum() / g["_area"].sum()
                        if g["_area"].sum() > 0
                        else 0.0
                    ),
                    "priority_score": float(g["priority_score"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    agg["priority_rank"] = agg["priority_score"].rank(ascending=False, method="first").astype(int)
    top_n = int(cfg.get("priority.top_wards", 10))
    agg = agg.sort_values("priority_rank").head(max(top_n, len(agg))).reset_index(drop=True)
    return agg
