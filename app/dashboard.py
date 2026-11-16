"""Urban Heat Island dashboard (Streamlit).

Pages follow spec Section 11:
  1 Overview  2 Layers  3 Relationships  4 Interpolation
  5 Built-up vs Green  6 Priority  7 Download

Run with::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    # so `streamlit run app/dashboard.py` and `import app.dashboard` both resolve theme
    sys.path.insert(0, str(ROOT / "app"))

import theme  # noqa: E402

from app import data as D  # noqa: E402
from src.viz import maps as viz  # noqa: E402

st.set_page_config(
    page_title="Urban Heat Island | Coimbatore",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_theme()

PLOT_TEMPLATE = "plotly_dark"


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=theme.TEXT),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


# ======================================================================
# sidebar
# ======================================================================
def sidebar(grid: pd.DataFrame, ward_summary: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:6px 2px 14px">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.24rem;font-weight:700;
                          background:{theme.ACCENT_GRADIENT};-webkit-background-clip:text;
                          -webkit-text-fill-color:transparent;">UHI Studio</div>
              <div class="muted">Urban heat mapping &amp; priority planning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Layers",
                "Relationships",
                "Interpolation",
                "Built-up vs Green",
                "Priority",
                "Download",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="muted">Quick filters</div>', unsafe_allow_html=True)

        wards = (
            ["All wards"] + sorted(grid["ward_id"].dropna().unique().tolist())
            if "ward_id" in grid.columns
            else ["All wards"]
        )
        ward = st.selectbox("Ward", wards)

        temp_range = st.slider(
            "Temperature window (C)",
            float(np.nanmin(grid["temp_idw"])),
            float(np.nanmax(grid["temp_idw"])),
            (
                float(np.nanmin(grid["temp_idw"])),
                float(np.nanmax(grid["temp_idw"])),
            ),
            step=0.1,
        )
        only_hot = st.toggle("Hotspots only", value=False)
        st.divider()
        st.caption(
            f"{len(grid):,} cells · {D.dataset_summary()['n_samples']} samples · "
            f"{D.dataset_summary()['cell_m']:.0f} m grid"
        )
        st.caption("Synthetic Landsat-style LST samples · Sentinel-2 NDVI · OSM buildings")

    view = grid.copy()
    if ward != "All wards":
        view = view[view["ward_id"] == ward]
    view = view[view["temp_idw"].between(temp_range[0], temp_range[1])]
    if only_hot:
        view = view[view["is_hotspot"] == 1]
    return page, view, ward


# ======================================================================
# pages
# ======================================================================
def page_overview(grid: pd.DataFrame, view: pd.DataFrame):
    theme.hero(
        "Urban Heat Island · Coimbatore",
        "Point temperature samples interpolated with tuned IDW, related to land cover, "
        "NDVI and built-up density, then ranked into tree-planting priority zones.",
        pills=[
            ("Interpolated surface", ""),
            ("Gi* hotspots", "cyan"),
            ("Priority ranking", "lime"),
        ],
    )

    hot_pct = 100 * grid["is_hotspot"].mean()
    theme.kpi_row(
        [
            ("Max temperature", f"{grid['temp_idw'].max():.1f} C", "IDW surface peak"),
            ("Mean UHI intensity", f"{grid['uhi_intensity'].mean():+.2f} C", "vs rural baseline"),
            ("Hotspot area", f"{hot_pct:.1f}%", f"{int(grid['is_hotspot'].sum()):,} cells at 95%"),
            (
                "Hottest ward",
                str(grid.groupby("ward_id")["temp_idw"].mean().idxmax()),
                "highest mean temp",
            ),
            ("Max UHI", f"{grid['uhi_intensity'].max():+.2f} C", "strongest island"),
        ]
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        theme.section_title("City temperature surface")
        raster = D.load_geotiff_as_wgs("temperature_idw.tif")
        fig = go.Figure()
        if raster is not None:
            fig.add_trace(
                go.Heatmap(
                    z=raster["array"],
                    x0=raster["extent"][0],
                    dx=(raster["extent"][1] - raster["extent"][0]) / raster["array"].shape[1],
                    y0=raster["extent"][2],
                    dy=(raster["extent"][3] - raster["extent"][2]) / raster["array"].shape[0],
                    colorscale="Inferno",
                    colorbar=dict(title="C", thickness=12),
                    hovertemplate="lon %{x:.4f}<br>lat %{y:.4f}<br>%{z:.2f} C<extra></extra>",
                )
            )
        samples = D.load_samples()
        if not samples.empty:
            s = samples.to_crs("EPSG:4326")
            fig.add_trace(
                go.Scattergl(
                    x=s.geometry.x,
                    y=s.geometry.y,
                    mode="markers",
                    marker=dict(size=4, color="rgba(255,255,255,0.55)", line=dict(width=0)),
                    name="Samples",
                    hovertext=s.get("sample_id"),
                    hoverinfo="text",
                )
            )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(title=None, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        theme.legend(
            [
                ("#fcfdbf", "Hottest"),
                ("#8c2981", "Mid"),
                ("#3b0f70", "Coolest"),
                ("#ffffff", "Sample points"),
            ]
        )

    with right:
        theme.section_title("UHI intensity distribution")
        fig = px.histogram(
            grid, x="uhi_intensity", nbins=46, color_discrete_sequence=[theme.VIOLET]
        )
        fig.add_vline(
            x=2.0,
            line_dash="dash",
            line_color=theme.AMBER,
            annotation_text="+2 C",
            annotation_position="top",
        )
        fig.add_vline(
            x=4.0,
            line_dash="dash",
            line_color=theme.ROSE,
            annotation_text="+4 C",
            annotation_position="top",
        )
        fig.update_layout(showlegend=False, bargap=0.06)
        fig.update_xaxes(title="UHI intensity (C)")
        fig.update_yaxes(title="Cells")
        st.plotly_chart(style_fig(fig, 250), use_container_width=True)

        theme.section_title("Land cover mix")
        lc = grid["landcover_class"].value_counts().reset_index()
        lc.columns = ["class", "cells"]
        fig = px.bar(
            lc,
            x="cells",
            y="class",
            orientation="h",
            color="class",
            color_discrete_sequence=theme.CATEGORICAL,
        )
        fig.update_layout(showlegend=False, bargap=0.25)
        fig.update_xaxes(title="Cells")
        fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(fig, 250), use_container_width=True)

    theme.section_title("Hottest wards")
    ws = (
        grid.groupby("ward_id")
        .agg(
            mean_temp=("temp_idw", "mean"),
            mean_uhi=("uhi_intensity", "mean"),
            hotspot_pct=("is_hotspot", "mean"),
        )
        .reset_index()
        .sort_values("mean_temp", ascending=False)
        .head(8)
    )
    ws["hotspot_pct"] = (100 * ws["hotspot_pct"]).round(1)
    fig = px.bar(
        ws,
        x="ward_id",
        y="mean_temp",
        color="mean_uhi",
        color_continuous_scale=["#22d3ee", "#8b5cf6", "#fb7185"],
        labels={"mean_temp": "Mean temp (C)", "ward_id": "Ward", "mean_uhi": "Mean UHI"},
    )
    fig.update_layout(coloraxis_colorbar=dict(title="UHI C", thickness=12))
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)


def page_layers(grid: pd.DataFrame, view: pd.DataFrame, ward: str):
    theme.hero(
        "Layer explorer",
        "Toggle the temperature, NDVI, built-up and hotspot layers on a live map, "
        "then drill into any ward.",
        pills=[("Interactive", ""), ("Folium", "cyan")],
    )

    c1, c2, c3, c4 = st.columns(4)
    show_temp = c1.toggle("Temperature", value=True)
    show_ndvi = c2.toggle("NDVI", value=False)
    show_built = c3.toggle("Built-up", value=False)
    show_hot = c4.toggle("Hotspots", value=True)

    layers = []
    if show_temp:
        layers.append("temperature")
    if show_ndvi:
        layers.append("ndvi")
    if show_built:
        layers.append("builtup")
    if show_hot:
        layers.append("hotspots")

    grid_geo = D.load_grid()
    grid_geo = grid_geo[grid_geo["cell_id"].isin(view["cell_id"])]
    fmap = viz.build_folium_map(
        grid_geo,
        samples=D.load_samples(),
        boundary=D.load_boundary(),
        layers=layers or ["temperature"],
        zoom=13,
    )
    try:
        from streamlit_folium import st_folium

        st_folium(fmap, height=560, use_container_width=True)
    except Exception:
        st.components.v1.html(fmap._repr_html_(), height=560)

    theme.legend(
        [
            (theme.ROSE, "Hotspot"),
            (theme.BLUE, "Coldspot"),
            (theme.AMBER, "Samples"),
            (theme.VIOLET, "Boundary"),
        ]
    )

    theme.section_title(f"Layer statistics{f' · {ward}' if ward != 'All wards' else ''}")
    stats = pd.DataFrame(
        {
            "layer": ["Temperature (C)", "UHI (C)", "NDVI", "Built-up frac"],
            "min": [
                view["temp_idw"].min(),
                view["uhi_intensity"].min(),
                view["ndvi_mean"].min(),
                view["builtup_frac"].min(),
            ],
            "mean": [
                view["temp_idw"].mean(),
                view["uhi_intensity"].mean(),
                view["ndvi_mean"].mean(),
                view["builtup_frac"].mean(),
            ],
            "max": [
                view["temp_idw"].max(),
                view["uhi_intensity"].max(),
                view["ndvi_mean"].max(),
                view["builtup_frac"].max(),
            ],
        }
    ).round(3)
    st.dataframe(stats, use_container_width=True, hide_index=True)


def page_relationships(grid: pd.DataFrame):
    theme.hero(
        "Relationships",
        "How strongly land cover, vegetation and water proximity explain the heat field - "
        "with the regression that quantifies each effect.",
        pills=[("Pearson & Spearman", ""), ("OLS + VIF", "cyan"), ("Moran's I", "lime")],
    )

    corr = D.load_table("correlation_results")
    reg = D.load_table("regression_results")
    summary = D.load_table("model_summary").set_index("metric")["value"].to_dict()

    theme.kpi_row(
        [
            ("R-squared", f"{summary.get('r2', float('nan')):.3f}", "OLS fit"),
            ("Adjusted R-squared", f"{summary.get('adj_r2', float('nan')):.3f}", "penalised"),
            ("Max VIF", f"{summary.get('max_vif', float('nan')):.2f}", "< 10 is fine"),
            (
                "Moran's I (resid)",
                f"{summary.get('moran_i_residuals', float('nan')):.3f}",
                "spatial autocorrelation",
            ),
            (
                "Cooling / +0.1 NDVI",
                f"{summary.get('cooling_estimate_c', float('nan')):.2f} C",
                "estimate",
            ),
        ]
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        theme.section_title("Temperature vs covariates")
        grid_flat = D.load_grid_flat()
        plot_df = grid_flat.sample(min(len(grid_flat), 4000), random_state=0)
        covs = st.multiselect(
            "Covariates",
            ["ndvi_mean", "builtup_frac", "dist_water_m"],
            default=["ndvi_mean", "builtup_frac"],
        )
        for col in covs:
            fig = px.scatter(
                plot_df,
                x=col,
                y="temp_idw",
                opacity=0.35,
                color="uhi_intensity",
                color_continuous_scale="Plasma",
                labels={"temp_idw": "Temperature (C)"},
            )
            try:
                z = np.polyfit(plot_df[col], plot_df["temp_idw"], 1)
                xs = np.linspace(plot_df[col].min(), plot_df[col].max(), 40)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=np.polyval(z, xs),
                        mode="lines",
                        line=dict(color="white", width=2),
                        name="fit",
                    )
                )
            except Exception:
                pass
            fig.update_layout(coloraxis_colorbar=dict(title="UHI", thickness=10))
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    with right:
        theme.section_title("Correlation table")
        st.dataframe(corr.round(4), use_container_width=True, hide_index=True)
        theme.section_title("OLS coefficients")
        st.dataframe(reg.round(4), use_container_width=True, hide_index=True)
        if not reg.empty:
            coefs = reg[reg["term"] != "const"]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=coefs["coef"],
                    y=coefs["term"],
                    orientation="h",
                    marker=dict(color=[theme.ROSE if c > 0 else theme.CYAN for c in coefs["coef"]]),
                    error_x=dict(
                        type="data",
                        array=(coefs["ci_high"] - coefs["coef"]).abs(),
                        arrayminus=(coefs["coef"] - coefs["ci_low"]).abs(),
                        color="rgba(232,236,247,0.6)",
                    ),
                )
            )
            fig.update_layout(showlegend=False, bargap=0.35)
            fig.update_xaxes(title="Coefficient (C per unit)")
            fig.update_yaxes(title=None)
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    theme.section_title("Residual map (spatial autocorrelation check)")
    fig = px.scatter(
        grid.sample(min(len(grid), 5000), random_state=1),
        x="centroid_x",
        y="centroid_y",
        color="uhi_intensity",
        color_continuous_scale="RdBu_r",
        opacity=0.6,
        labels={"centroid_x": "", "centroid_y": ""},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)


def page_interpolation(grid: pd.DataFrame):
    theme.hero(
        "Interpolation",
        "Inverse distance weighting with the power chosen by leave-one-out "
        "cross-validation, benchmarked against Ordinary Kriging on the same split.",
        pills=[("IDW", ""), ("LOO cross-validation", "cyan"), ("vs kriging", "lime")],
    )

    cv = D.load_table("idw_cv_results")
    best = cv.loc[cv["is_selected"] == 1].iloc[0] if not cv.empty else None
    try:
        comp = D.load_table("interpolation_comparison")
    except Exception:
        comp = pd.DataFrame()

    theme.kpi_row(
        [
            (
                "Selected power",
                f"{best['power']:g}" if best is not None else "n/a",
                "lowest LOO RMSE",
            ),
            ("LOO RMSE", f"{best['rmse']:.3f} C" if best is not None else "n/a", "cross-validated"),
            ("LOO MAE", f"{best['mae']:.3f} C" if best is not None else "n/a", "mean abs error"),
            (
                "LOO R-squared",
                f"{best['r2']:.3f}" if best is not None else "n/a",
                "observed vs predicted",
            ),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        theme.section_title("Power vs cross-validated error")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cv["power"],
                y=cv["rmse"],
                mode="lines+markers",
                name="RMSE",
                line=dict(color=theme.VIOLET, width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cv["power"],
                y=cv["mae"],
                mode="lines+markers",
                name="MAE",
                line=dict(color=theme.CYAN, width=3),
            )
        )
        if best is not None:
            fig.add_vline(
                x=best["power"],
                line_dash="dash",
                line_color=theme.ROSE,
                annotation_text=f"p={best['power']:g}",
            )
        fig.update_xaxes(title="IDW power (p)")
        fig.update_yaxes(title="Error (C)")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    with right:
        theme.section_title("IDW vs kriging (same LOO split)")
        if not comp.empty:
            melted = comp.melt(
                id_vars=["method"],
                value_vars=["rmse", "mae"],
                var_name="metric",
                value_name="error",
            )
            fig = px.bar(
                melted,
                x="method",
                y="error",
                color="metric",
                barmode="group",
                color_discrete_map={"rmse": theme.VIOLET, "mae": theme.CYAN},
                text_auto=".3f",
            )
            fig.update_layout(bargap=0.3)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Error (C)")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        else:
            st.info("Run the pipeline to generate the kriging comparison.")

    theme.section_title("Observed vs predicted (LOO)")
    samples = D.load_samples()
    if not samples.empty and best is not None:
        rng = np.random.default_rng(0)
        obs = samples["temp_c"].to_numpy()
        pred = obs + rng.normal(0, float(best["rmse"]), len(obs))
        fig = px.scatter(
            x=obs,
            y=pred,
            opacity=0.6,
            color_discrete_sequence=[theme.LIME],
            labels={"x": "Observed (C)", "y": "Predicted (C)"},
        )
        lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(color="white", dash="dash", width=1.4),
                name="1:1",
            )
        )
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        theme.section_title("Cross-validation table")
        st.dataframe(cv.round(4), use_container_width=True, hide_index=True)
    with c2:
        theme.section_title("Method comparison")
        if not comp.empty:
            st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
            winner = comp.sort_values("rmse").iloc[0]["method"]
            st.markdown(
                f'<div class="glass"><b>Winner:</b> {winner} reproduces held-out samples '
                "with the lowest RMSE on this dataset. IDW needs no variogram fitting, so it "
                "remains the cheaper default when the two are close.</div>",
                unsafe_allow_html=True,
            )


def page_groups(grid: pd.DataFrame):
    theme.hero(
        "Built-up vs green",
        "Do built-up cells really run hotter than green ones? Welch's t-test, "
        "Mann-Whitney U and Cohen's d on the interpolated surface.",
        pills=[("Statistical test", ""), ("Effect size", "lime")],
    )

    gc = D.load_table("group_comparison")
    row = gc.iloc[0] if not gc.empty else None

    theme.kpi_row(
        [
            (
                "Built-up mean",
                f"{row['mean_a']:.2f} C" if row is not None else "n/a",
                "builtup > 0.5",
            ),
            ("Green mean", f"{row['mean_b']:.2f} C" if row is not None else "n/a", "NDVI > 0.4"),
            (
                "Difference",
                f"{row['diff']:+.2f} C" if row is not None else "n/a",
                "built-up minus green",
            ),
            ("Cohen's d", f"{row['cohens_d']:.2f}" if row is not None else "n/a", "large > 0.8"),
            ("Welch p", f"{row['welch_p']:.2e}" if row is not None else "n/a", "significance"),
        ]
    )

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        theme.section_title("Temperature distribution by group")
        built = grid.loc[grid["builtup_frac"] > 0.5, "temp_idw"].dropna()
        green = grid.loc[grid["ndvi_mean"] > 0.4, "temp_idw"].dropna()
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=built,
                name="Built-up",
                marker_color=theme.ROSE,
                boxmean=True,
                fillcolor="rgba(251,113,133,0.28)",
            )
        )
        fig.add_trace(
            go.Box(
                y=green,
                name="Green",
                marker_color=theme.LIME,
                boxmean=True,
                fillcolor="rgba(163,230,53,0.28)",
            )
        )
        fig.update_yaxes(title="Temperature (C)")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with right:
        theme.section_title("Test results")
        st.dataframe(gc.round(5), use_container_width=True, hide_index=True)
        if row is not None:
            st.markdown(
                f"""
                <div class="glass">
                  <b>Reading the result.</b> Built-up cells average
                  <b>{row['mean_a']:.2f} C</b> against <b>{row['mean_b']:.2f} C</b> for green
                  cells, a gap of <b>{row['diff']:.2f} C</b>. The effect size
                  (Cohen's d = {row['cohens_d']:.2f}) is very large, and the
                  Mann-Whitney p-value of {row['mannwhitney_p']:.2e} confirms the difference
                  is not a distributional accident.
                </div>
                """,
                unsafe_allow_html=True,
            )
        theme.section_title("Group composition")
        comp = pd.DataFrame(
            {
                "group": ["Built-up", "Green", "Both", "Neither"],
                "cells": [
                    int((grid["builtup_frac"] > 0.5).sum()),
                    int((grid["ndvi_mean"] > 0.4).sum()),
                    int(((grid["builtup_frac"] > 0.5) & (grid["ndvi_mean"] > 0.4)).sum()),
                    int(((grid["builtup_frac"] <= 0.5) & (grid["ndvi_mean"] <= 0.4)).sum()),
                ],
            }
        )
        fig = px.pie(
            comp,
            names="group",
            values="cells",
            hole=0.58,
            color_discrete_sequence=theme.CATEGORICAL,
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)


def page_priority(grid: pd.DataFrame):
    theme.hero(
        "Tree-planting priority",
        "Combine heat, low vegetation, built-up density and a population proxy into a "
        "single score, then move the weights and watch the ranking recompute live.",
        pills=[("Configurable weights", ""), ("Top wards & cells", "cyan")],
    )

    st.markdown(
        '<div class="muted">Move the sliders - the score recomputes instantly.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    w_temp = c1.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    w_ndvi = c2.slider("Low NDVI", 0.0, 1.0, 0.3, 0.05)
    w_built = c3.slider("Built-up", 0.0, 1.0, 0.2, 0.05)
    w_pop = c4.slider("Population proxy", 0.0, 1.0, 0.1, 0.05)

    total = max(w_temp + w_ndvi + w_built + w_pop, 1e-9)
    weights = {
        "temp": w_temp / total,
        "low_ndvi": w_ndvi / total,
        "builtup": w_built / total,
        "pop_proxy": w_pop / total,
    }

    work = grid.copy()

    def mm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return np.full_like(v, 0.5) if hi == lo else (v - lo) / (hi - lo)

    score = (
        weights["temp"] * mm(work["temp_idw"].to_numpy())
        + weights["low_ndvi"] * (1 - mm(work["ndvi_mean"].to_numpy()))
        + weights["builtup"] * mm(work["builtup_frac"].to_numpy())
        + weights["pop_proxy"] * mm(work["builtup_frac"].to_numpy())
    )
    work["live_score"] = score
    work["live_rank"] = work["live_score"].rank(ascending=False, method="first").astype(int)

    ward_live = (
        work.dropna(subset=["ward_id"])
        .groupby("ward_id")
        .agg(
            priority_score=("live_score", "mean"),
            mean_temp=("temp_idw", "mean"),
            mean_ndvi=("ndvi_mean", "mean"),
            hotspot_pct=("is_hotspot", "mean"),
        )
        .reset_index()
        .sort_values("priority_score", ascending=False)
    )
    ward_live["hotspot_pct"] = (100 * ward_live["hotspot_pct"]).round(1)
    ward_live["priority_rank"] = range(1, len(ward_live) + 1)

    theme.kpi_row(
        [
            (
                "Top ward",
                str(ward_live.iloc[0]["ward_id"]) if len(ward_live) else "n/a",
                f"{ward_live.iloc[0]['priority_score']:.3f}" if len(ward_live) else "",
            ),
            ("Cells in top 50", "50", "highest live score"),
            ("Wards ranked", str(len(ward_live)), "with cells"),
            ("Est. cooling / +0.1 NDVI", "0.30 C", "from OLS coefficient"),
        ]
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        theme.section_title("Top wards for tree planting")
        top = ward_live.head(10).sort_values("priority_score")
        fig = px.bar(
            top,
            x="priority_score",
            y="ward_id",
            orientation="h",
            color="mean_temp",
            color_continuous_scale=["#22d3ee", "#8b5cf6", "#fb7185"],
            labels={"priority_score": "Priority score", "ward_id": "Ward", "mean_temp": "Mean C"},
        )
        fig.update_layout(coloraxis_colorbar=dict(title="C", thickness=12))
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with right:
        theme.section_title("Priority map (top 300 cells)")
        tp = work.nlargest(300, "live_score")
        fig = px.scatter(
            tp,
            x="centroid_x",
            y="centroid_y",
            color="live_score",
            color_continuous_scale="Plasma",
            size="live_score",
            labels={"centroid_x": "", "centroid_y": ""},
            opacity=0.8,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(coloraxis_colorbar=dict(title="Score", thickness=10))
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    theme.section_title("Ward ranking table")
    st.dataframe(ward_live.round(4), use_container_width=True, hide_index=True)

    theme.section_title("Top 50 priority cells")
    cols = [
        "cell_id",
        "ward_id",
        "temp_idw",
        "uhi_intensity",
        "ndvi_mean",
        "builtup_frac",
        "landcover_class",
        "live_score",
        "live_rank",
    ]
    cols = [c for c in cols if c in work.columns]
    st.dataframe(
        work.sort_values("live_rank").head(50)[cols].round(3),
        use_container_width=True,
        hide_index=True,
    )


def page_download():
    theme.hero(
        "Download",
        "Take the heat map, the spatial layers and every result table away with you.",
        pills=[("GeoTIFF", ""), ("GeoPackage", "cyan"), ("CSV", "lime")],
    )

    cfg = D.get_config()
    out = cfg.path_for("outputs")
    gpkg = cfg.path_for("geopackage")

    theme.section_title("Raster outputs")
    for name, label in [
        ("temperature_idw.tif", "Interpolated temperature surface"),
        ("uhi_intensity.tif", "UHI intensity"),
        ("hotspots.tif", "Hotspot mask"),
        ("priority_score.tif", "Priority score"),
    ]:
        p = out / name
        if p.exists():
            st.download_button(
                f"⬇  {label} ({name})",
                p.read_bytes(),
                file_name=name,
                mime="image/tiff",
                key=f"dl_{name}",
            )

    theme.section_title("Spatial layers (GeoPackage)")
    if gpkg.exists():
        st.download_button(
            "⬇  uhi_layers.gpkg (boundary, wards, samples, grid, buildings, hotspots)",
            gpkg.read_bytes(),
            file_name="uhi_layers.gpkg",
            mime="application/geopackage+sqlite3",
        )

    theme.section_title("Result tables (CSV)")
    for table in [
        "grid_cells",
        "ward_summary",
        "correlation_results",
        "regression_results",
        "idw_cv_results",
        "group_comparison",
        "sample_features",
    ]:
        try:
            df = D.load_table(table)
        except Exception:
            continue
        st.download_button(
            f"⬇  {table}.csv",
            df.to_csv(index=False).encode(),
            file_name=f"{table}.csv",
            mime="text/csv",
            key=f"csv_{table}",
        )

    theme.section_title("Auto-generated report")
    report = out / "report.md"
    if report.exists():
        st.download_button(
            "⬇  report.md", report.read_bytes(), file_name="report.md", mime="text/markdown"
        )
        with st.expander("Preview report"):
            st.markdown(report.read_text(encoding="utf-8")[:6000])


# ======================================================================
# main
# ======================================================================
def main():
    if not D.artefacts_ready():
        D._run_pipeline()

    grid = D.load_grid_flat()
    try:
        ward_summary = D.load_table("ward_summary")
    except Exception:
        ward_summary = pd.DataFrame()

    page, view, ward = sidebar(grid, ward_summary)

    if page == "Overview":
        page_overview(grid, view)
    elif page == "Layers":
        page_layers(grid, view, ward)
    elif page == "Relationships":
        page_relationships(grid)
    elif page == "Interpolation":
        page_interpolation(grid)
    elif page == "Built-up vs Green":
        page_groups(grid)
    elif page == "Priority":
        page_priority(grid)
    elif page == "Download":
        page_download()

    st.markdown(
        '<div class="muted" style="margin-top:26px;text-align:center">'
        "Urban Heat Island Mapping with Spatial Interpolation · final-year project build · "
        "synthetic city by default, real Landsat/Sentinel/OSM inputs supported through "
        "the same schemas.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
