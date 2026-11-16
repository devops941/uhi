"""Static (matplotlib) and interactive (Folium) maps and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# a consistent, colour-blind-safe palette reused by the dashboard
PALETTE = {
    "temp": "inferno",
    "uhi": "coolwarm",
    "ndvi": "YlGn",
    "builtup": "YlOrBr",
    "hot": "#e11d48",
    "cold": "#2563eb",
    "green": "#16a34a",
    "built": "#dc2626",
    "accent": "#7c3aed",
}


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------
# raster / surface figures
# ----------------------------------------------------------------------
def plot_surface(
    array: np.ndarray,
    title: str,
    out_path,
    cmap: str = "inferno",
    extent=None,
    label: str = "",
    figsize=(7, 6),
    dpi: int = 130,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(array, cmap=cmap, extent=extent, origin="upper")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if label:
        cb.set_label(label)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ndvi_builtup(
    ndvi: np.ndarray, builtup: np.ndarray, out_path, figsize=(12, 5.5), dpi: int = 130
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    im0 = axes[0].imshow(ndvi, cmap=PALETTE["ndvi"], vmin=-0.5, vmax=1.0)
    axes[0].set_title("NDVI", fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    im1 = axes[1].imshow(builtup, cmap=PALETTE["builtup"], vmin=0, vmax=1)
    axes[1].set_title("Built-up fraction", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hotspots(
    grid,
    out_path,
    value_col: str = "temp_idw",
    figsize=(7.5, 6.5),
    dpi: int = 130,
):
    """Temperature surface with hotspot / coldspot markers on top."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    grid.plot(column=value_col, cmap=PALETTE["temp"], ax=ax, legend=True, markersize=0.5)
    if "is_hotspot" in grid.columns:
        hot = grid[grid["is_hotspot"] == 1]
        cold = grid[grid.get("is_coldspot", 0) == 1]
        if not hot.empty:
            hot.centroid.plot(ax=ax, color=PALETTE["hot"], markersize=4, label="Hotspot")
        if not cold.empty:
            cold.centroid.plot(ax=ax, color=PALETTE["cold"], markersize=4, label="Coldspot")
    ax.set_title("UHI hotspots (Getis-Ord Gi*)", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# relationship figures
# ----------------------------------------------------------------------
def plot_scatter_grid(joined: pd.DataFrame, out_path, target: str = "temp_c", dpi: int = 130):
    predictors = [c for c in ["ndvi_mean", "builtup_frac", "dist_water_m"] if c in joined.columns]
    n = len(predictors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.4), dpi=dpi)
    if n == 1:
        axes = [axes]
    labels = {
        "ndvi_mean": "Mean NDVI",
        "builtup_frac": "Built-up fraction",
        "dist_water_m": "Distance to water (m)",
    }
    for ax, col in zip(axes, predictors, strict=False):
        ax.scatter(joined[col], joined[target], s=16, alpha=0.6, color=PALETTE["accent"])
        sub = joined[[col, target]].dropna()
        if len(sub) > 2 and sub[col].nunique() > 1:
            coef = np.polyfit(sub[col], sub[target], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color="#111827", lw=1.8)
            r = np.corrcoef(sub[col], sub[target])[0, 1]
            ax.set_title(f"{labels.get(col, col)}  (r={r:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(labels.get(col, col))
        ax.set_ylabel("Temperature (C)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_idw_cv(cv_table: pd.DataFrame, out_path, dpi: int = 130):
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=dpi)
    ax.plot(
        cv_table["power"], cv_table["rmse"], marker="o", lw=2, color=PALETTE["accent"], label="RMSE"
    )
    ax.plot(cv_table["power"], cv_table["mae"], marker="s", lw=2, color="#0ea5e9", label="MAE")
    best = cv_table.loc[cv_table["is_selected"] == 1, "power"].iloc[0]
    ax.axvline(best, color=PALETTE["hot"], ls="--", lw=1.6, label=f"Selected p={best:g}")
    ax.set_xlabel("IDW power (p)")
    ax.set_ylabel("Error (C)")
    ax.set_title("IDW power selection by leave-one-out CV", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_observed_predicted(observed: np.ndarray, predicted: np.ndarray, out_path, dpi: int = 130):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=dpi)
    ax.scatter(observed, predicted, s=18, alpha=0.65, color=PALETTE["accent"])
    lo = float(np.nanmin([np.min(observed), np.min(predicted)]))
    hi = float(np.nanmax([np.max(observed), np.max(predicted)]))
    ax.plot([lo, hi], [lo, hi], color="#111827", lw=1.4, ls="--")
    r2 = 1 - np.nansum((predicted - observed) ** 2) / np.nansum(
        (observed - np.nanmean(observed)) ** 2
    )
    ax.set_xlabel("Observed temperature (C)")
    ax.set_ylabel("LOO IDW prediction (C)")
    ax.set_title(f"Observed vs predicted (R^2={r2:.3f})", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_residuals(fitted: pd.DataFrame, out_path, dpi: int = 130):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=dpi)
    axes[0].scatter(fitted["fitted"], fitted["residual"], s=16, alpha=0.6, color="#0ea5e9")
    axes[0].axhline(0, color="#111827", lw=1.2, ls="--")
    axes[0].set_xlabel("Fitted temperature (C)")
    axes[0].set_ylabel("Residual (C)")
    axes[0].set_title("Residuals vs fitted", fontweight="bold")
    axes[1].hist(fitted["residual"].dropna(), bins=24, color=PALETTE["accent"], alpha=0.85)
    axes[1].set_xlabel("Residual (C)")
    axes[1].set_title("Residual distribution", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_boxplot(
    grid,
    out_path,
    temp_col: str = "temp_idw",
    builtup_thr: float = 0.5,
    ndvi_thr: float = 0.4,
    dpi: int = 130,
):
    built = grid.loc[grid["builtup_frac"] > builtup_thr, temp_col].dropna()
    green = grid.loc[grid["ndvi_mean"] > ndvi_thr, temp_col].dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=dpi)
    bp = ax.boxplot(
        [built, green], tick_labels=["Built-up", "Green"], patch_artist=True, widths=0.55
    )
    for patch, colour in zip(bp["boxes"], [PALETTE["built"], PALETTE["green"]], strict=False):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Built-up vs green cell temperature", fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ward_ranking(ward_summary: pd.DataFrame, out_path, top: int = 10, dpi: int = 130):
    df = ward_summary.sort_values("priority_score", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    ax.barh(df["ward_id"][::-1], df["priority_score"][::-1], color=PALETTE["accent"])
    ax.set_xlabel("Priority score")
    ax.set_title(f"Top {len(df)} wards for tree planting", fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(_ensure(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# interactive maps (Folium)
# ----------------------------------------------------------------------
def build_folium_map(
    grid,
    samples=None,
    boundary=None,
    layers: list[str] | None = None,
    centre: tuple[float, float] | None = None,
    zoom: int = 12,
    heat_alpha: float = 0.65,
):
    """Folium map with toggleable temperature / NDVI / built-up / hotspot layers.

    The grid is aggregated to ~250 m blocks before plotting so the payload stays
    small enough for the browser.
    """
    import folium
    from folium.plugins import HeatMap

    layers = layers or ["temperature", "hotspots"]
    grid_wgs = grid.to_crs("EPSG:4326") if not grid.empty and hasattr(grid, "to_crs") else grid
    if not grid_wgs.empty and hasattr(grid_wgs, "geometry"):
        grid_wgs = grid_wgs[~grid_wgs.geometry.is_empty].copy()

    if centre is None:
        if not grid_wgs.empty and hasattr(grid_wgs, "geometry") and not grid_wgs.geometry.is_empty.all():
            union_g = grid_wgs.geometry.union_all()
            if not union_g.is_empty:
                c = union_g.centroid
                if not c.is_empty:
                    centre = (float(c.y), float(c.x))
        if centre is None and boundary is not None and not boundary.empty:
            b_wgs = boundary.to_crs("EPSG:4326")
            if hasattr(b_wgs, "geometry"):
                b_wgs = b_wgs[~b_wgs.geometry.is_empty]
            if not b_wgs.empty:
                union_b = b_wgs.geometry.union_all()
                if not union_b.is_empty:
                    c = union_b.centroid
                    if not c.is_empty:
                        centre = (float(c.y), float(c.x))
        if centre is None:
            centre = (11.0168, 76.9558)

    fmap = folium.Map(
        location=centre, zoom_start=zoom, tiles="CartoDB dark_matter", control_scale=True
    )

    if boundary is not None and not boundary.empty:
        b_clean = boundary.to_crs("EPSG:4326")
        if hasattr(b_clean, "geometry"):
            b_clean = b_clean[~b_clean.geometry.is_empty]
        if not b_clean.empty:
            folium.GeoJson(
                b_clean.__geo_interface__,
                name="City boundary",
                style_function=lambda _: {"color": "#7c3aed", "weight": 2.4, "fill": False},
            ).add_to(fmap)

    if not grid_wgs.empty and hasattr(grid_wgs, "geometry"):
        cell_pts = np.column_stack([grid_wgs.geometry.centroid.y, grid_wgs.geometry.centroid.x])
    else:
        cell_pts = np.empty((0, 2))

    if "temperature" in layers and "temp_idw" in grid_wgs.columns and not grid_wgs.empty:
        vals = grid_wgs["temp_idw"].to_numpy(dtype=float)
        if len(vals) > 0 and np.isfinite(vals).any():
            lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
            norm = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
            heat = [
                [lat, lon, float(w)]
                for (lat, lon), w in zip(cell_pts, norm, strict=False)
                if np.isfinite(w)
            ]
            fg = folium.FeatureGroup(name="Temperature (IDW)", show=True)
            HeatMap(
                heat,
                radius=14,
                blur=18,
                min_opacity=0.25,
                max_opacity=heat_alpha,
                gradient={
                    0.1: "#3b0f70",
                    0.4: "#8c2981",
                    0.6: "#de4968",
                    0.8: "#fe9f6d",
                    1.0: "#fcfdbf",
                },
            ).add_to(fg)
            fg.add_to(fmap)

    if "ndvi" in layers and "ndvi_mean" in grid_wgs.columns and not grid_wgs.empty:
        vals = grid_wgs["ndvi_mean"].to_numpy(dtype=float)
        if len(vals) > 0 and np.isfinite(vals).any():
            lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
            norm = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
            heat = [
                [lat, lon, float(w)]
                for (lat, lon), w in zip(cell_pts, norm, strict=False)
                if np.isfinite(w)
            ]
            fg = folium.FeatureGroup(name="NDVI", show=False)
            HeatMap(
                heat,
                radius=13,
                blur=16,
                min_opacity=0.2,
                gradient={0.1: "#0b3d0b", 0.5: "#41ab5d", 1.0: "#c7f9cc"},
            ).add_to(fg)
            fg.add_to(fmap)

    if "builtup" in layers and "builtup_frac" in grid_wgs.columns and not grid_wgs.empty:
        vals = grid_wgs["builtup_frac"].to_numpy(dtype=float)
        if len(vals) > 0 and np.isfinite(vals).any():
            heat = [
                [lat, lon, float(w)]
                for (lat, lon), w in zip(cell_pts, vals, strict=False)
                if np.isfinite(w)
            ]
            fg = folium.FeatureGroup(name="Built-up fraction", show=False)
            HeatMap(
                heat,
                radius=13,
                blur=16,
                min_opacity=0.2,
                gradient={0.1: "#fff7bc", 0.6: "#f03b20", 1.0: "#800026"},
            ).add_to(fg)
            fg.add_to(fmap)

    if "hotspots" in layers and "is_hotspot" in grid_wgs.columns and not grid_wgs.empty:
        hot = grid_wgs[grid_wgs["is_hotspot"] == 1]
        cold = grid_wgs[grid_wgs.get("is_coldspot", 0) == 1]
        fg = folium.FeatureGroup(name="Hotspots / coldspots", show=False)
        for _, row in hot.iterrows():
            geom = getattr(row, "geometry", None)
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                if not c.is_empty:
                    folium.CircleMarker(
                        [float(c.y), float(c.x)],
                        radius=3,
                        color="#e11d48",
                        fill=True,
                        fill_opacity=0.75,
                        weight=0,
                    ).add_to(fg)
        for _, row in cold.iterrows():
            geom = getattr(row, "geometry", None)
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                if not c.is_empty:
                    folium.CircleMarker(
                        [float(c.y), float(c.x)],
                        radius=3,
                        color="#2563eb",
                        fill=True,
                        fill_opacity=0.75,
                        weight=0,
                    ).add_to(fg)
        fg.add_to(fmap)

    if "priority" in layers and "priority_score" in grid_wgs.columns and not grid_wgs.empty:
        top = grid_wgs.nlargest(60, "priority_score")
        fg = folium.FeatureGroup(name="Top priority cells", show=False)
        for _, row in top.iterrows():
            geom = getattr(row, "geometry", None)
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                if not c.is_empty:
                    folium.CircleMarker(
                        [float(c.y), float(c.x)],
                        radius=4,
                        color="#7c3aed",
                        fill=True,
                        fill_opacity=0.8,
                        weight=0,
                        tooltip=(
                            f"Priority {row['priority_score']:.3f}<br>"
                            f"Temp {row.get('temp_idw', float('nan')):.1f} C<br>"
                            f"NDVI {row.get('ndvi_mean', float('nan')):.2f}"
                        ),
                    ).add_to(fg)
        fg.add_to(fmap)

    if samples is not None and not samples.empty:
        samples_wgs = samples.to_crs("EPSG:4326")
        fg = folium.FeatureGroup(name="Temperature samples", show=False)
        for _, row in samples_wgs.iterrows():
            geom = getattr(row, "geometry", None)
            if geom is not None and not geom.is_empty:
                c = geom.centroid if hasattr(geom, "centroid") else geom
                if not c.is_empty:
                    folium.CircleMarker(
                        [float(c.y), float(c.x)],
                        radius=2.6,
                        color="#fbbf24",
                        fill=True,
                        fill_opacity=0.9,
                        weight=0,
                        tooltip=f"{row.get('sample_id', '')}: {row.get('temp_c', float('nan')):.1f} C",
                    ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
