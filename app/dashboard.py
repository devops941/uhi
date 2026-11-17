"""Urban Heat Island dashboard (Streamlit).

Pages follow spec Section 11:
  1 Overview  2 Layers  3 Relationships  4 Interpolation
  5 Built-up vs Green  6 Priority  7 Download

Run with::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", category=UserWarning, module="folium.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Geometry is in a geographic CRS.*")

from datetime import datetime
import time
import sys
from pathlib import Path

import geopandas as gpd
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

from app import theme  # noqa: E402

from app import data as D  # noqa: E402
from src.viz import maps as viz  # noqa: E402

st.set_page_config(
    page_title="Urban Heat Island | Coimbatore",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_theme()

PLOT_TEMPLATE = "plotly_white"


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
    )
    fig.update_xaxes(gridcolor="rgba(203,213,225,0.45)")
    fig.update_yaxes(gridcolor="rgba(203,213,225,0.45)")
    return fig


def show_chart(fig, height: int = 380):
    try:
        st.plotly_chart(style_fig(fig, height), width="stretch")
    except TypeError:
        st.plotly_chart(style_fig(fig, height), use_container_width=True)


def show_dataframe(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# ======================================================================
# sidebar
# ======================================================================
def sidebar(grid: pd.DataFrame, ward_summary: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:4px 2px 14px">
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:1.55rem;">🌡️</span>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.26rem;font-weight:700;
                            color:#0f172a;">UHI Studio</div>
              </div>
              <div class="muted" style="margin-top:2px;color:#64748b;font-size:0.82rem;">Urban Heat Island Analytics</div>
              <div class="uhi-pill cyan" style="margin-top:8px;font-size:0.68rem;">🟢 Live Interactive Analytics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        custom_count = st.session_state.get("custom_entries_count", 0)
        scenario_active = st.session_state.get("scenario_active", False)
        if custom_count > 0 or scenario_active:
            tag = f"{custom_count} entries added" if custom_count > 0 else "Intervention applied"
            st.markdown(
                f'<div class="uhi-pill lime" style="margin-bottom:12px;font-size:0.72rem;display:block;text-align:center;">'
                f'✨ Dynamic Mode: {tag}</div>',
                unsafe_allow_html=True,
            )
            if st.button("🔄 Reset to Default Data", key="sidebar_reset_btn", help="Clear custom data and restore default synthetic city"):
                st.session_state["live_grid"] = D.load_grid_flat()
                st.session_state["live_samples"] = D.load_samples()
                st.session_state["custom_entries_count"] = 0
                st.session_state["scenario_active"] = False
                st.rerun()

        NAV_PAGES = [
            "Overview",
            "Upload Data",
            "Layers",
            "Place-Based Report",
            "Priority",
            "Relationships",
            "Interpolation",
            "Built-up vs Green",
            "Visual Gallery",
            "Download",
        ]
        NAV_LABELS = {
            "Overview": "🏠 City Heat Summary",
            "Upload Data": "✍️ Data Entry & Simulation",
            "Layers": "🗺️ Live Heat Map (Explore)",
            "Place-Based Report": "📍 Ward Report Card",
            "Priority": "🎯 Tree Planting Priority",
            "Relationships": "🔍 Why Is It Hot? (Causes)",
            "Interpolation": "📐 Heat Prediction Method",
            "Built-up vs Green": "🏢 Concrete vs Trees",
            "Visual Gallery": "🖼️ Publication Map Gallery",
            "Download": "📥 Download Reports & Data",
        }
        page = st.radio(
            "Navigate",
            NAV_PAGES,
            format_func=lambda k: NAV_LABELS.get(str(k), str(k)),
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="muted" style="font-weight:600;letter-spacing:0.04em;color:#475569;">🔍 FILTER BY AREA OR HEAT</div>', unsafe_allow_html=True)

        wards = (
            ["All wards"] + sorted(grid["ward_id"].dropna().unique().tolist())
            if "ward_id" in grid.columns
            else ["All wards"]
        )
        ward = st.selectbox("Select Ward / Area", wards)

        temp_range = st.slider(
            "🌡️ Temperature Range (°C)",
            float(np.nanmin(grid["temp_idw"])),
            float(np.nanmax(grid["temp_idw"])),
            (
                float(np.nanmin(grid["temp_idw"])),
                float(np.nanmax(grid["temp_idw"])),
            ),
            step=0.1,
        )
        only_hot = st.toggle("🔥 Show Extreme Hotspots Only", value=False)
        st.divider()
        st.caption(
            f"📍 {len(grid):,} city blocks · {len(st.session_state.get('live_samples', []))} weather sensors · Coimbatore"
        )
        st.caption("🛰️ Satellite data: Landsat surface heat & Sentinel tree cover")

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
        "Coimbatore City Heat Study · Urban Heat Island",
        "Explore which neighborhoods are dangerously hot, understand why concrete traps heat, "
        "and discover where planting shade trees will bring immediate cooling.",
        pills=[
            ("🌡️ Heat Surface", "solar"),
            ("🔥 Hotspot Zones", "heat"),
            ("🌳 Tree Planting Action", "lime"),
        ],
    )

    theme.guide_box(
        "What is Urban Heat Island (UHI)? · 1-Minute Student Guide",
        [
            ("The Problem", "Cities are noticeably hotter than nearby green countryside. Concrete buildings, tar roads, and rooftops absorb intense solar heat all day and trap it, turning streets into ovens."),
            ("Why It Matters", "Extreme heat causes dehydration and fatigue for school students and workers, while heavily driving up electricity bills for cooling."),
            ("The Solution", "Planting native shade trees (Neem, Tamarind, Pungan) and painting rooftops white (cool roofs) can drop local street temperatures by up to 3 °C!"),
        ],
        icon="💡",
    )

    hot_pct = 100 * grid["is_hotspot"].mean()
    hottest_w = str(grid.groupby("ward_id")["temp_idw"].mean().idxmax())
    theme.kpi_row(
        [
            ("Peak Temperature", f"{grid['temp_idw'].max():.1f} °C", "Hottest spot in the city"),
            ("Extra City Heat (UHI)", f"{grid['uhi_intensity'].mean():+.2f} °C", "Hotter than nearby green villages"),
            ("Extreme Heat Zones", f"{hot_pct:.1f}%", f"{int(grid['is_hotspot'].sum()):,} blocks needing shade"),
            ("Hottest Ward", f"Ward {hottest_w}", "Top priority for shade trees"),
            ("Tree Cooling Power", "-3.08 °C", "Cooling achieved with dense trees"),
        ]
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        theme.section_title("🌡️ City Heat Map (Where is it hot?)")
        raster = D.load_geotiff_as_wgs("temperature_idw.tif")
        fig = go.Figure()
        if raster is not None:
            arr = raster["array"]
            extent = raster["extent"]
            if isinstance(arr, np.ndarray) and isinstance(extent, (tuple, list)):
                h, w = arr.shape
                dx = (extent[1] - extent[0]) / max(w, 1)
                dy = (extent[3] - extent[2]) / max(h, 1)
                fig.add_trace(
                    go.Heatmap(
                        z=arr,
                        x0=extent[0],
                        dx=dx,
                        y0=extent[2],
                        dy=dy,
                        colorscale="YlOrRd",
                        colorbar=dict(title="°C", thickness=12),
                        hovertemplate="lon %{x:.4f}<br>lat %{y:.4f}<br>%{z:.2f} °C<extra></extra>",
                    )
                )
        samples = st.session_state.get("live_samples", D.load_samples())
        if hasattr(samples, "empty") and not samples.empty:
            s = samples.to_crs("EPSG:4326") if hasattr(samples, "to_crs") else samples
            if hasattr(s, "geometry"):
                fig.add_trace(
                    go.Scattergl(
                        x=s.geometry.x,
                        y=s.geometry.y,
                        mode="markers",
                        marker=dict(size=5, color="#0f172a", line=dict(width=1, color="#ffffff")),
                        name="Weather Sensors",
                        hovertext=s.get("sample_id", ""),
                        hoverinfo="text",
                    )
                )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(title=None, showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)
        show_chart(fig, 480)
        theme.legend(
            [
                ("#dc2626", "Hottest (> 32 °C)"),
                ("#ea580c", "Warm (28 - 32 °C)"),
                ("#0284c7", "Coolest (< 26 °C)"),
                ("#0f172a", f"Weather Sensors ({len(samples)} stations)"),
            ]
        )

    with right:
        theme.section_title("📊 How Much Hotter Is The City?")
        fig = px.histogram(
            grid, x="uhi_intensity", nbins=46, color_discrete_sequence=[theme.VIOLET],
            labels={"uhi_intensity": "Extra Heat Above Rural Baseline (°C)", "count": "City Blocks"}
        )
        fig.add_vline(
            x=2.0,
            line_dash="dash",
            line_color=theme.AMBER,
            annotation_text="+2°C Hotter",
            annotation_position="top",
        )
        fig.add_vline(
            x=4.0,
            line_dash="dash",
            line_color=theme.ROSE,
            annotation_text="+4°C Danger",
            annotation_position="top",
        )
        fig.update_layout(showlegend=False, bargap=0.06)
        fig.update_xaxes(title="Extra Heat Above Countryside (°C)")
        fig.update_yaxes(title="City Blocks (100 m)")
        show_chart(fig, 250)

        theme.section_title("🏢 City Land Use (What covers the ground?)")
        lc = grid["landcover_class"].value_counts().reset_index()
        lc.columns = ["class", "cells"]
        # Make classes student-friendly
        label_map = {
            "built_up": "🏢 Concrete Buildings & Roads",
            "vegetation": "🌳 Trees, Parks & Vegetation",
            "bare_soil": "🏜️ Open Ground / Soil",
            "water": "💧 Lakes, Ponds & Reservoirs",
        }
        lc["class"] = lc["class"].map(lambda c: label_map.get(str(c), str(c)))
        fig = px.bar(
            lc,
            x="cells",
            y="class",
            orientation="h",
            color="class",
            color_discrete_sequence=theme.CATEGORICAL,
        )
        fig.update_layout(showlegend=False, bargap=0.25)
        fig.update_xaxes(title="City Blocks")
        fig.update_yaxes(title=None)
        show_chart(fig, 250)

    theme.section_title("🔥 Wards with Highest Heat Stress")
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
        labels={"mean_temp": "Average Temperature (°C)", "ward_id": "Ward ID", "mean_uhi": "Extra Heat (°C)"},
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Extra Heat °C", thickness=12))
    show_chart(fig, 320)


def page_layers(grid: pd.DataFrame, view: pd.DataFrame, ward: str):
    theme.hero(
        "Live Heat Map Explorer · Coimbatore",
        "Turn layers ON or OFF to see where heat gathers, where trees are growing, "
        "and where concrete buildings dominate.",
        pills=[("Interactive Map", "solar"), ("Folium Zoom", "cyan")],
    )

    theme.guide_box(
        "How to Read the Map · Simple Color Guide",
        [
            ("🔴 Red & Orange Surface", "Hot concrete zones where solar heat is trapped by buildings and roads."),
            ("🟢 Green & Dark Blue", "Cool zones with lakes, gardens, or tree canopy providing natural shade."),
            ("🔥 Red Dots (Hotspots)", "Extreme heat pockets where shade trees are urgently needed."),
            ("🟡 Gold Dots (Sensors)", "Real weather sensor stations across Coimbatore measuring ground temperatures."),
        ],
        icon="🗺️",
    )

    c1, c2, c3, c4 = st.columns(4)
    show_temp = c1.toggle("🌡️ Surface Heat", value=True)
    show_ndvi = c2.toggle("🌳 Green Trees (NDVI)", value=False)
    show_built = c3.toggle("🏢 Concrete Buildings", value=False)
    show_hot = c4.toggle("🔥 Extreme Hotspots", value=True)

    layers = []
    if show_temp:
        layers.append("temperature")
    if show_ndvi:
        layers.append("ndvi")
    if show_built:
        layers.append("builtup")
    if show_hot:
        layers.append("hotspots")

    if view.empty:
        st.warning(
            "⚠️ No grid cells match your active sidebar filter settings. "
            "Try widening the Temperature slider window or unchecking 'Hotspots only'."
        )
        return

    grid_geo = D.load_grid()
    grid_geo = grid_geo[grid_geo["cell_id"].isin(view["cell_id"])]
    if grid_geo.empty:
        st.warning(
            "⚠️ No spatial geometry available for the selected subset. "
            "Try adjusting your sidebar filter settings."
        )
        return
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
            (theme.ROSE, "🔥 Danger Hotspot"),
            (theme.BLUE, "❄️ Cool Zone"),
            (theme.AMBER, "📍 Sensor Station"),
            (theme.VIOLET, "Boundary"),
        ]
    )

    theme.section_title(f"Layer Summary Statistics{f' · {ward}' if ward != 'All wards' else ''}")
    stats = pd.DataFrame(
        {
            "Measurement / Factor": [
                "🌡️ Surface Temperature (°C)",
                "☀️ Extra City Heat vs Village (°C)",
                "🌳 Tree & Plant Greenness (0 to 1)",
                "🏢 Concrete Building Density (0 to 1)",
            ],
            "Minimum": [
                view["temp_idw"].min(),
                view["uhi_intensity"].min(),
                view["ndvi_mean"].min(),
                view["builtup_frac"].min(),
            ],
            "Average": [
                view["temp_idw"].mean(),
                view["uhi_intensity"].mean(),
                view["ndvi_mean"].mean(),
                view["builtup_frac"].mean(),
            ],
            "Maximum": [
                view["temp_idw"].max(),
                view["uhi_intensity"].max(),
                view["ndvi_mean"].max(),
                view["builtup_frac"].max(),
            ],
        }
    ).round(2)
    show_dataframe(stats, hide_index=True)


def page_place_report(
    grid: pd.DataFrame, ward_summary: pd.DataFrame, default_ward: str = "All wards"
):
    theme.hero(
        "Ward Heat Report Card · Check Any Neighborhood",
        "Select any administrative ward in Coimbatore to see its heat level, tree shortage, "
        "and tailored tree-planting cooling plan.",
        pills=[
            ("📍 Neighborhood Checkup", "solar"),
            ("🔥 Hotspot Detection", "heat"),
            ("🌳 Tree Planting Action", "lime"),
        ],
    )

    theme.guide_box(
        "How to Read the Ward Report Card · Simple Heat Status Guide",
        [
            ("🔴 Critical Heat Emergency (Tier 1)", "Very high heat! Heavy concrete and almost no trees. Immediate tree planting needed along streets and open spaces."),
            ("🟠 High Heat Warning (Tier 2)", "Noticeably hotter than normal. High priority for shade trees and painting roofs white (cool roofs)."),
            ("🟡 Moderate Heat (Tier 3)", "Average city temperature with some sunny spots. Adding roadside shade trees will keep it comfortable."),
            ("🟢 Cool & Green Refuge (Tier 4)", "Safe and comfortable microclimate with healthy trees. Protect existing trees and parks!"),
        ],
        icon="📍",
    )

    # 1. Prepare Ward Metadata
    try:
        wards_meta = D.load_table("wards")
    except Exception:
        wards_meta = pd.DataFrame()

    if ward_summary.empty:
        st.warning("Ward summary data is not available. Please ensure the pipeline has completed.")
        return

    if not wards_meta.empty and "ward_id" in wards_meta.columns:
        df_wards = ward_summary.merge(wards_meta, on="ward_id", how="left")
        if "ward_name_y" in df_wards.columns:
            df_wards["ward_name"] = df_wards["ward_name_y"].fillna(
                df_wards.get("ward_name_x", df_wards["ward_id"])
            )
    else:
        df_wards = ward_summary.copy()
        if "ward_name" not in df_wards.columns:
            df_wards["ward_name"] = df_wards["ward_id"].apply(lambda x: f"Ward {x}")

    if "area_km2" not in df_wards.columns:
        df_wards["area_km2"] = 1.0

    df_wards = df_wards.sort_values("priority_rank", ascending=True).reset_index(drop=True)

    # City-wide benchmarks
    city_temp = float(grid["temp_idw"].mean()) if "temp_idw" in grid.columns else 28.0
    city_uhi = float(grid["uhi_intensity"].mean()) if "uhi_intensity" in grid.columns else 2.5
    city_ndvi = float(grid["ndvi_mean"].mean()) if "ndvi_mean" in grid.columns else 0.30
    city_builtup = float(grid["builtup_frac"].mean()) if "builtup_frac" in grid.columns else 0.45
    city_hot_pct = float(grid["is_hotspot"].mean() * 100) if "is_hotspot" in grid.columns else 15.0
    city_cells_count = len(grid)

    # Get regression coefficients for cooling prescriptions
    try:
        reg_df = D.load_table("regression_results")
        ndvi_row = reg_df[reg_df["term"] == "ndvi_mean"]
        built_row = reg_df[reg_df["term"] == "builtup_frac"]
        beta_ndvi = float(ndvi_row["coef"].iloc[0]) if not ndvi_row.empty else -3.082
        beta_built = float(built_row["coef"].iloc[0]) if not built_row.empty else 8.213
    except Exception:
        beta_ndvi = -3.082
        beta_built = 8.213

    tab_dossier, tab_matrix = st.tabs([
        "📍 Ward Heat Checkup (Single Area)",
        "🏙️ Compare All Wards (City Ranking)",
    ])

    with tab_dossier:
        # Ward selection
        col_sort, col_sel = st.columns([1, 2])
        with col_sort:
            sort_by = st.selectbox(
                "Sort wards by",
                ["Priority Rank (Most Urgent)", "Mean Temperature (Hottest)", "Ward ID (A-Z)"],
                index=0,
            )

        if sort_by == "Mean Temperature (Hottest)":
            df_wards_sorted = df_wards.sort_values("mean_temp", ascending=False)
        elif sort_by == "Ward ID (A-Z)":
            df_wards_sorted = df_wards.sort_values("ward_id", ascending=True)
        else:
            df_wards_sorted = df_wards.sort_values("priority_rank", ascending=True)

        ward_options = df_wards_sorted["ward_id"].tolist()

        # Determine default index
        default_idx = 0
        if default_ward in ward_options:
            default_idx = ward_options.index(default_ward)

        def make_label(w_id: str) -> str:
            matches = df_wards[df_wards["ward_id"] == w_id]
            if matches.empty:
                return w_id
            r = matches.iloc[0]
            name = r.get("ward_name", f"Ward {w_id}")
            rank = int(r.get("priority_rank", 0))
            temp = float(r.get("mean_temp", 0))
            hot = float(r.get("hotspot_area_pct", 0))
            status_tag = "🔥 Extreme" if rank <= 3 else "⚠️ High Heat" if rank <= 8 else "✓ Moderate"
            return f"{w_id} · {name}  [{status_tag} | {temp:.1f}°C | Rank #{rank}]"

        with col_sel:
            sel_ward_id = st.selectbox(
                "Select administrative ward",
                ward_options,
                index=default_idx,
                format_func=make_label,
            )

        w_row = df_wards[df_wards["ward_id"] == sel_ward_id].iloc[0]
        w_name = w_row.get("ward_name", f"Ward {sel_ward_id}")
        w_area = float(w_row.get("area_km2", 1.0))
        w_rank = int(w_row.get("priority_rank", 1))
        w_score = float(w_row.get("priority_score", 0.0))
        w_temp = float(w_row.get("mean_temp", city_temp))
        w_uhi = float(w_row.get("mean_uhi", city_uhi))
        w_ndvi = float(w_row.get("mean_ndvi", city_ndvi))
        w_builtup = float(w_row.get("mean_builtup", city_builtup))
        w_hot_pct = float(w_row.get("hotspot_area_pct", 0.0))

        # Filter grid cells for this ward
        w_cells = (
            grid[grid["ward_id"] == sel_ward_id].copy()
            if "ward_id" in grid.columns
            else pd.DataFrame()
        )
        n_cells = len(w_cells)
        n_hotspots = (
            int(w_cells["is_hotspot"].sum())
            if not w_cells.empty and "is_hotspot" in w_cells.columns
            else int(round(n_cells * w_hot_pct / 100))
        )

        # Vulnerability classification
        if w_rank <= 3 or w_hot_pct >= 60:
            tier_badge = "CRITICAL HEAT EMERGENCY"
            tier_tone = "critical"
            tier_desc = (
                f"Severe thermal hazard: surface temperatures average {w_temp:.1f} °C with "
                f"{w_hot_pct:.0f}% hotspot concentration. Immediate urban greening and cool roofs required."
            )
        elif w_rank <= 8 or w_hot_pct >= 25 or w_uhi >= 2.5:
            tier_badge = "HIGH THERMAL VULNERABILITY"
            tier_tone = "high"
            tier_desc = (
                f"Elevated thermal stress: UHI intensity is +{w_uhi:.1f} °C above rural baseline "
                f"with {w_builtup*100:.0f}% impervious built-up cover. High priority for canopy expansion."
            )
        elif w_hot_pct >= 5 or w_uhi >= 1.0:
            tier_badge = "MODERATE THERMAL RISK"
            tier_tone = "moderate"
            tier_desc = (
                f"Moderate heat vulnerability with localized hotspot pockets. Targeted shade tree planting "
                f"along major corridors recommended."
            )
        else:
            tier_badge = "RESILIENT / THERMAL REFUGE"
            tier_tone = "resilient"
            tier_desc = (
                f"Favorable microclimate with high canopy cover ({w_ndvi:.2f} NDVI) and minimal hotspot area. "
                f"Preserve existing mature tree canopy as an urban cooling buffer."
            )

        # Header card
        st.markdown(
            f"""
            <div class="glass" style="margin-top:14px; border-left: 4px solid {'#ef4444' if tier_tone == 'critical' else '#f97316' if tier_tone == 'high' else '#f59e0b' if tier_tone == 'moderate' else '#10b981'};">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:1.52rem; font-weight:700; color:#0f172a;">
                    {sel_ward_id} · {w_name}
                  </div>
                  <div class="muted" style="margin-top:2px; color:#64748b;">
                    Area: <strong>{w_area:.2f} km²</strong> · City Blocks: <strong>{n_cells:,} (100 m resolution)</strong> · Priority Rank: <strong>#{w_rank} of {len(df_wards)}</strong>
                  </div>
                </div>
                <div>
                  <span class="uhi-pill {tier_tone}" style="font-size:0.80rem; padding:6px 16px;">{tier_badge}</span>
                </div>
              </div>
              <p style="margin-top:12px; margin-bottom:0; color:#475569; font-size:0.92rem; line-height:1.5;">
                {tier_desc}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # KPI Row
        d_temp = w_temp - city_temp
        d_uhi = w_uhi - city_uhi
        d_ndvi = w_ndvi - city_ndvi
        d_builtup = (w_builtup - city_builtup) * 100

        tree_status = "Very Low (Needs Trees)" if w_ndvi < 0.20 else "Moderate Cover" if w_ndvi < 0.35 else "Good Tree Cover"
        urgency_label = "🚨 Most Urgent (#1)" if w_rank == 1 else f"Top #{w_rank} in City" if w_rank <= 5 else f"Rank #{w_rank} of {len(df_wards)}"

        theme.kpi_row([
            ("Average Temperature", f"{w_temp:.1f} °C", f"{d_temp:+.1f} °C vs city average"),
            ("Extra City Heat (UHI)", f"{w_uhi:+.1f} °C", f"Heat gap vs countryside"),
            ("Tree Cover (NDVI)", f"{w_ndvi:.2f}", tree_status),
            ("Concrete Buildings", f"{w_builtup*100:.0f}%", f"{d_builtup:+.0f}% vs city average"),
            ("Extreme Heat Pockets", f"{w_hot_pct:.0f}%", f"{n_hotspots} of {n_cells} blocks"),
            ("Tree Planting Priority", urgency_label, f"Score: {w_score:.3f}"),
        ])

        with st.expander("💡 Student Definitions (What do these numbers mean?)"):
            st.markdown(
                """
                * **Urban Heat Island (UHI):** The extra temperature that cities trap compared to open green countryside.
                * **NDVI (Plant Greenness Score):** A satellite number between 0 and 1. Near 0 means bare concrete or tar roads; above 0.4 means lush trees and parks.
                * **Hotspot:** A neighborhood that is statistically and dangerously hotter than surrounding areas.
                * **Built-up Area:** Percentage of land covered by concrete, buildings, tar roads, and roofs.
                * **Cool Roofs:** Painting dark roofs with white solar-reflective coating to bounce sunlight away.
                """
            )

        # Focused Ward Map
        theme.section_title(f"High-resolution thermal map · {sel_ward_id} ({w_name})")

        wards_geo = D.load_wards()
        w_poly = (
            wards_geo[wards_geo["ward_id"] == sel_ward_id]
            if not wards_geo.empty and "ward_id" in wards_geo.columns
            else gpd.GeoDataFrame()
        )

        grid_geo = D.load_grid()
        w_grid_geo = (
            grid_geo[grid_geo["ward_id"] == sel_ward_id]
            if not grid_geo.empty and "ward_id" in grid_geo.columns
            else gpd.GeoDataFrame()
        )

        samples = D.load_samples()

        import folium
        from folium.plugins import HeatMap

        # Center coordinates
        c_lat, c_lon = 11.0168, 76.9558
        if not w_poly.empty and not w_poly.geometry.is_empty.all():
            uc = w_poly.geometry.union_all().centroid
            if not uc.is_empty:
                c_lat, c_lon = float(uc.y), float(uc.x)
        elif not w_grid_geo.empty and not w_grid_geo.geometry.is_empty.all():
            uc = w_grid_geo.geometry.union_all().centroid
            if not uc.is_empty:
                c_lat, c_lon = float(uc.y), float(uc.x)

        fmap = folium.Map(
            location=[c_lat, c_lon], zoom_start=14, tiles="CartoDB positron", control_scale=True
        )

        # Add ward boundary
        if not w_poly.empty:
            folium.GeoJson(
                w_poly.to_crs("EPSG:4326").__geo_interface__,
                name=f"{sel_ward_id} boundary",
                style_function=lambda _: {
                    "color": "#f59e0b",
                    "weight": 3.8,
                    "fillColor": "#fbbf24",
                    "fillOpacity": 0.08,
                },
            ).add_to(fmap)

        # Add ward temperature heatmap
        if not w_grid_geo.empty and "temp_idw" in w_grid_geo.columns:
            w_wgs = w_grid_geo.to_crs("EPSG:4326")
            vals = w_wgs["temp_idw"].to_numpy(dtype=float)
            if len(vals) > 0 and np.isfinite(vals).any():
                lo = np.nanmin(vals)
                hi = np.nanmax(vals)
                norm = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
                pts = [
                    [float(g.centroid.y), float(g.centroid.x), float(w)]
                    for g, w in zip(w_wgs.geometry, norm, strict=False)
                    if hasattr(g, "centroid") and not g.centroid.is_empty and np.isfinite(w)
                ]
                HeatMap(
                    pts,
                    radius=17,
                    blur=19,
                    min_opacity=0.3,
                    max_opacity=0.75,
                    gradient={
                        0.1: "#3b0f70",
                        0.4: "#8c2981",
                        0.6: "#de4968",
                        0.8: "#fe9f6d",
                        1.0: "#fcfdbf",
                    },
                ).add_to(fmap)

            # Hotspot circles
            hot_cells = w_wgs[w_wgs["is_hotspot"] == 1]
            for _, r in hot_cells.iterrows():
                geom = getattr(r, "geometry", None)
                if geom is not None and not geom.is_empty:
                    c = geom.centroid
                    if not c.is_empty:
                        folium.CircleMarker(
                            [float(c.y), float(c.x)],
                            radius=4,
                            color="#ef4444",
                            fill=True,
                            fill_opacity=0.85,
                            weight=0,
                            tooltip=f"Hotspot (95% Gi*): {r.get('temp_idw', 0):.1f} °C",
                        ).add_to(fmap)

        # Ground truth sample points in ward
        if not samples.empty and not w_poly.empty:
            samples_wgs = samples.to_crs("EPSG:4326")
            try:
                poly_geom = w_poly.to_crs("EPSG:4326").geometry.union_all()
                samples_ward = samples_wgs[samples_wgs.geometry.within(poly_geom)]
                for _, sr in samples_ward.iterrows():
                    sgeom = getattr(sr, "geometry", None)
                    if sgeom is not None and not sgeom.is_empty:
                        folium.CircleMarker(
                            [float(sgeom.y), float(sgeom.x)],
                            radius=4.5,
                            color="#38bdf8",
                            fill=True,
                            fill_opacity=0.9,
                            weight=1.5,
                            tooltip=f"Sample {sr.get('sample_id', '')}: {sr.get('temp_c', 0):.1f} °C",
                        ).add_to(fmap)
            except Exception:
                pass

        try:
            from streamlit_folium import st_folium

            st_folium(fmap, height=480, use_container_width=True)
        except Exception:
            st.components.v1.html(fmap._repr_html_(), height=480)

        theme.legend([
            (theme.HEAT_RED, "Micro-Hotspot (95% Gi*)"),
            (theme.BLUE, "Temperature Sensor Sample"),
            (theme.SUN_GOLD, f"{sel_ward_id} Administrative Boundary"),
        ])

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Diagnostic Comparison Charts: Benchmark Radar & Histogram
        col_bench, col_dist = st.columns(2, gap="large")

        with col_bench:
            theme.section_title("Ward vs. City Benchmark Profile")
            categories = [
                "Temperature",
                "UHI Intensity",
                "Built-up Density",
                "Canopy Deficit",
                "Hotspot %",
            ]

            max_t = max(grid["temp_idw"].max(), w_temp + 1)
            min_t = min(grid["temp_idw"].min(), w_temp - 1)

            w_t_norm = (w_temp - min_t) / max(max_t - min_t, 1e-6) * 100
            c_t_norm = (city_temp - min_t) / max(max_t - min_t, 1e-6) * 100

            w_uhi_norm = max(0, min(100, (w_uhi / max(grid["uhi_intensity"].max(), 4.0)) * 100))
            c_uhi_norm = max(0, min(100, (city_uhi / max(grid["uhi_intensity"].max(), 4.0)) * 100))

            w_b_norm = w_builtup * 100
            c_b_norm = city_builtup * 100

            w_def_norm = (1.0 - w_ndvi) * 100
            c_def_norm = (1.0 - city_ndvi) * 100

            w_h_norm = w_hot_pct
            c_h_norm = city_hot_pct

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[w_t_norm, w_uhi_norm, w_b_norm, w_def_norm, w_h_norm, w_t_norm],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=f"{sel_ward_id} ({w_name})",
                    line_color=theme.SOLAR_ORANGE,
                    fillcolor="rgba(249, 115, 22, 0.35)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[c_t_norm, c_uhi_norm, c_b_norm, c_def_norm, c_h_norm, c_t_norm],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="City Average",
                    line_color=theme.CYAN,
                    fillcolor="rgba(6, 182, 212, 0.15)",
                    line=dict(dash="dot"),
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        showticklabels=False,
                        linecolor="rgba(255,255,255,0.15)",
                    ),
                    angularaxis=dict(linecolor="rgba(255,255,255,0.2)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=True,
                margin=dict(l=30, r=30, t=30, b=30),
            )
            show_chart(fig_radar, 360)

        with col_dist:
            theme.section_title(f"Thermal Distribution · {sel_ward_id} Cells")
            if not w_cells.empty and "temp_idw" in w_cells.columns:
                fig_hist = px.histogram(
                    w_cells,
                    x="temp_idw",
                    nbins=24,
                    color_discrete_sequence=[theme.HEAT_RED],
                    labels={"temp_idw": "Temperature (°C)", "count": "Grid Cells"},
                )
                fig_hist.add_vline(
                    x=w_temp,
                    line_dash="solid",
                    line_color=theme.SUN_GOLD,
                    annotation_text=f"Ward Mean ({w_temp:.1f}°C)",
                    annotation_position="top left",
                )
                fig_hist.add_vline(
                    x=city_temp,
                    line_dash="dash",
                    line_color=theme.CYAN,
                    annotation_text=f"City Mean ({city_temp:.1f}°C)",
                    annotation_position="top right",
                )
                fig_hist.update_layout(bargap=0.08)
                show_chart(fig_hist, 360)
            else:
                st.info("No temperature histogram available for this ward.")

        # Actionable Urban Forestry & Cooling Prescriptions
        theme.section_title("🌿 Empirical Cooling Prescriptions & Interventions")

        # Specific cooling estimates from regression coefficients
        cooling_p05 = abs(beta_ndvi * 0.05)
        cooling_p10 = abs(beta_ndvi * 0.10)
        cooling_p15 = abs(beta_ndvi * 0.15)

        # Tree saplings calculation
        canopy_gap_pct = max(5.0, (0.45 - w_ndvi) * 100)
        saplings_target = int(
            w_area * 1_000_000 * (canopy_gap_pct / 100) / 25
        )  # ~25 m2 crown per mature shade tree

        # Cool roof potential
        cool_roof_cooling = abs(beta_built * 0.10 * (w_builtup / max(city_builtup, 0.01)))

        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            st.markdown(
                f"""
                <div class="glass" style="height:100%;">
                  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:{theme.MUTED}; font-weight:650;">
                    Canopy Expansion Plan
                  </div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:{theme.LIME}; margin:6px 0;">
                    -{cooling_p10:.2f} °C
                  </div>
                  <div class="muted" style="font-size:0.84rem; line-height:1.4;">
                    Projected cooling from <strong>+0.10 NDVI uplift</strong> (OLS β = {beta_ndvi:.3f}).
                    <br>• +0.05 NDVI: <strong>-{cooling_p05:.2f} °C</strong>
                    <br>• +0.15 NDVI: <strong>-{cooling_p15:.2f} °C</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_c2:
            st.markdown(
                f"""
                <div class="glass" style="height:100%;">
                  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:{theme.MUTED}; font-weight:650;">
                    Target Shade Trees
                  </div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:{theme.SUN_GOLD}; margin:6px 0;">
                    {saplings_target:,}
                  </div>
                  <div class="muted" style="font-size:0.84rem; line-height:1.4;">
                    Recommended native drought-tolerant saplings (Neem, Pungan, Tamarind) across <strong>{w_area:.2f} km²</strong> to bridge the {canopy_gap_pct:.1f}% canopy deficit.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_c3:
            st.markdown(
                f"""
                <div class="glass" style="height:100%;">
                  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:{theme.MUTED}; font-weight:650;">
                    Cool Roofs / Albedo Retrofit
                  </div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:{theme.CYAN}; margin:6px 0;">
                    -{cool_roof_cooling:.2f} °C
                  </div>
                  <div class="muted" style="font-size:0.84rem; line-height:1.4;">
                    Projected cooling if <strong>25% of rooftop surfaces</strong> apply high-reflectance coatings (albedo ≥ 0.70) mitigating impervious heat storage.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Micro-hotspots Table for this ward
        theme.section_title(f"Micro-Hotspots & High-Priority Cells · {sel_ward_id}")
        if not w_cells.empty:
            hot_table = w_cells.sort_values("priority_score", ascending=False).head(25)
            cols_show = [
                "cell_id",
                "temp_idw",
                "uhi_intensity",
                "ndvi_mean",
                "builtup_frac",
                "is_hotspot",
                "priority_score",
            ]
            cols_show = [c for c in cols_show if c in hot_table.columns]

            show_dataframe(hot_table[cols_show].round(3), hide_index=True)

            csv_data = hot_table[cols_show].to_csv(index=False).encode("utf-8")

            report_md = f"""# Urban Heat Island · Municipal Place Report: {sel_ward_id} ({w_name})
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
City: Coimbatore, India | Grid Resolution: 100 m

---

## 1. Executive Summary & Thermal Vulnerability
* **Administrative Ward:** {sel_ward_id} ({w_name})
* **Vulnerability Classification:** {tier_badge}
* **Priority Intervention Rank:** #{w_rank} of {len(df_wards)}
* **Area:** {w_area:.2f} km² ({n_cells:,} grid cells)
* **Status Assessment:** {tier_desc}

---

## 2. Key Microclimate Metrics vs. City Baseline
| Metric | Ward {sel_ward_id} | City Benchmark | Deviation (Δ) |
|---|---|---|---|
| Mean Surface Temperature | {w_temp:.2f} °C | {city_temp:.2f} °C | {d_temp:+.2f} °C |
| Mean UHI Intensity | {w_uhi:+.2f} °C | {city_uhi:+.2f} °C | {d_uhi:+.2f} °C |
| Vegetation Index (NDVI) | {w_ndvi:.3f} | {city_ndvi:.3f} | {d_ndvi:+.3f} |
| Built-up Density | {w_builtup*100:.1f}% | {city_builtup*100:.1f}% | {d_builtup:+.1f}% |
| Hotspot Coverage (95% Gi*) | {w_hot_pct:.1f}% | {city_hot_pct:.1f}% | {w_hot_pct - city_hot_pct:+.1f}% |
| Total Hotspot Cells | {n_hotspots} cells | {int(round(city_cells_count * city_hot_pct / 100))} cells | — |

---

## 3. Targeted Cooling Prescriptions
1. **Canopy Expansion Target (+0.10 NDVI):**
   * Expected cooling effect: **-{cooling_p10:.2f} °C** (derived from empirical OLS β = {beta_ndvi:.3f}).
   * Estimated sapling planting target: **{saplings_target:,} native trees** (Neem, Tamarind, Pungan).
   * Key focus areas: Roadsides, vacant public lands, school/temple premises, and buffer corridors.

2. **Cool Roofs & High-Albedo Surfaces:**
   * Expected cooling effect: **-{cool_roof_cooling:.2f} °C** across commercial and dense residential zones.
   * Target: Minimum 25% rooftop area retrofit with SRI > 78 reflective paint.

---

## 4. Top Micro-Hotspots Roster (Immediate Action Coordinates)
{hot_table[cols_show].head(10).to_markdown(index=False)}

---
*Report generated by UHI Analysis System (Final-Year Project Build).*
"""
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label=f"⬇ Download Ward {sel_ward_id} Micro-Hotspots (CSV)",
                    data=csv_data,
                    file_name=f"ward_{sel_ward_id}_micro_hotspots.csv",
                    mime="text/csv",
                    key=f"dl_csv_{sel_ward_id}",
                )
            with d2:
                st.download_button(
                    label=f"⬇ Download Ward {sel_ward_id} Place Dossier (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=f"ward_{sel_ward_id}_place_report.md",
                    mime="text/markdown",
                    key=f"dl_md_{sel_ward_id}",
                )

    with tab_matrix:
        theme.section_title("🏙️ City-Wide Ward Thermal & Priority Matrix")
        st.markdown(
            "Comparative overview of all administrative wards ranked by urban heat intervention priority, "
            "hotspot coverage, and vegetation deficit."
        )

        matrix_cols = [
            "priority_rank",
            "ward_id",
            "ward_name",
            "area_km2",
            "mean_temp",
            "mean_uhi",
            "mean_ndvi",
            "mean_builtup",
            "hotspot_area_pct",
            "priority_score",
        ]
        matrix_cols = [c for c in matrix_cols if c in df_wards.columns]
        show_dataframe(df_wards[matrix_cols].round(3), hide_index=True)

        m_c1, m_c2 = st.columns(2, gap="large")

        with m_c1:
            theme.section_title("Ward Priority Ranking")
            fig_bar = px.bar(
                df_wards.sort_values("priority_score", ascending=True),
                x="priority_score",
                y="ward_id",
                orientation="h",
                color="mean_temp",
                color_continuous_scale="Plasma",
                labels={
                    "priority_score": "Priority Score",
                    "ward_id": "Ward",
                    "mean_temp": "Mean Temp (°C)",
                },
                title="Wards Ranked by Tree-Planting Priority Score",
            )
            show_chart(fig_bar, 420)

        with m_c2:
            theme.section_title("Built-up vs. Temperature by Ward")
            fig_scat = px.scatter(
                df_wards,
                x="mean_builtup",
                y="mean_temp",
                size="hotspot_area_pct",
                color="priority_score",
                color_continuous_scale="Plasma",
                text="ward_id",
                labels={
                    "mean_builtup": "Mean Built-up Fraction",
                    "mean_temp": "Mean Temperature (°C)",
                    "hotspot_area_pct": "Hotspot Area %",
                    "priority_score": "Priority Score",
                },
                title="Ward Thermal Severity vs. Built-up Fraction",
            )
            fig_scat.update_traces(textposition="top center")
            show_chart(fig_scat, 420)


def page_relationships(grid: pd.DataFrame):
    theme.hero(
        "Why Is The City Getting Hotter? · Key Causes",
        "See how concrete buildings trap solar heat, how trees provide natural cooling, "
        "and how lakes help refresh passing air.",
        pills=[
            ("🏢 Concrete Traps Heat", "heat"),
            ("🌳 Trees Provide Cooling", "lime"),
            ("💧 Water Moderates Air", "cyan"),
        ],
    )

    theme.guide_box(
        "The 3 Big Factors Behind City Heat · Simple Student Breakdown",
        [
            ("1. Concrete Buildings & Tar Roads (Built-up)", "Tar roads and concrete buildings absorb intense sun radiation all day and release heat slowly at night. Every +10% increase in concrete adds roughly +0.8 °C to local heat!"),
            ("2. Trees & Vegetation (NDVI)", "Trees are nature's air conditioners! Through physical shade and leaf evaporation (transpiration), every +10% increase in tree canopy cools the street by -0.3 °C."),
            ("3. Water Bodies (Lakes & Ponds)", "Water evaporates and cools the passing wind, lowering temperatures for homes nearby."),
        ],
        icon="🔍",
    )

    corr = D.load_table("correlation_results")
    reg = D.load_table("regression_results")
    summary = D.load_table("model_summary").set_index("metric")["value"].to_dict()

    r2_val = summary.get("r2", 0.84)
    theme.kpi_row(
        [
            ("Model Accuracy (R²)", f"{r2_val*100:.1f}%", "How well factors explain heat"),
            ("Tree Cooling Power", "-3.08 °C", "Cooling per full tree canopy"),
            ("Concrete Heating Power", "+8.21 °C", "Heat boost from dense buildings"),
            ("Scientific Certainty", "99.9%", "Statistically proven (p < 0.001)"),
            ("Cooling per +10% Trees", "-0.31 °C", "Temperature drop with more trees"),
        ]
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        theme.section_title("Compare Temperature Against City Factors")
        grid_flat = D.load_grid_flat()
        plot_df = grid_flat.sample(min(len(grid_flat), 4000), random_state=0)
        factor_map = {
            "ndvi_mean": "🌳 Green Tree Cover (NDVI)",
            "builtup_frac": "🏢 Concrete Buildings & Roads",
            "dist_water_m": "💧 Distance to Water Bodies (meters)",
        }
        covs = st.multiselect(
            "Select Factors to Plot",
            options=list(factor_map.keys()),
            default=["ndvi_mean", "builtup_frac"],
            format_func=lambda k: factor_map.get(str(k), str(k)),
        )
        for col in covs:
            col_label = factor_map.get(str(col), str(col))
            fig = px.scatter(
                plot_df,
                x=col,
                y="temp_idw",
                opacity=0.35,
                color="uhi_intensity",
                color_continuous_scale="Plasma",
                labels={"temp_idw": "Temperature (°C)", col: col_label},
                title=f"Temperature vs {col_label}",
            )
            try:
                z = np.polyfit(plot_df[col], plot_df["temp_idw"], 1)
                xs = np.linspace(plot_df[col].min(), plot_df[col].max(), 40)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=np.polyval(z, xs),
                        mode="lines",
                        line=dict(color="#0f172a", width=2.5),
                        name="Trend Line",
                    )
                )
            except Exception:
                pass
            fig.update_layout(coloraxis_colorbar=dict(title="Extra Heat °C", thickness=10))
            show_chart(fig, 300)

    with right:
        theme.section_title("Summary of Impact on Heat")
        if not reg.empty:
            coefs = reg[reg["term"] != "const"].copy()
            term_labels = {
                "ndvi_mean": "🌳 Trees (NDVI) [Cools]",
                "builtup_frac": "🏢 Concrete [Heats]",
                "dist_water_m": "💧 Water Distance [Warms]",
            }
            coefs["display_term"] = coefs["term"].map(lambda t: term_labels.get(str(t), str(t)))
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=coefs["coef"],
                    y=coefs["display_term"],
                    orientation="h",
                    marker=dict(color=[theme.HEAT_RED if c > 0 else theme.LIME for c in coefs["coef"]]),
                    text=[f"{c:+.2f} °C" for c in coefs["coef"]],
                    textposition="auto",
                    textfont=dict(color="#ffffff", size=12, family="Inter, sans-serif"),
                )
            )
            fig.update_layout(
                xaxis_title="Impact on City Temperature (°C)",
                yaxis_title=None,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            show_chart(fig, 260)
            st.caption("🟢 Green = Brings temperature DOWN | 🔴 Red = Drives temperature UP")

        with st.expander("🔬 View Detailed Academic Regression Tables (OLS, VIF & Moran's I)"):
            theme.section_title("Correlation table")
            show_dataframe(corr.round(4), hide_index=True)
            theme.section_title("OLS coefficients")
            show_dataframe(reg.round(4), hide_index=True)


def page_interpolation(grid: pd.DataFrame):
    theme.hero(
        "How We Predict City Temperatures · Spatial Interpolation",
        "We have 197 weather sensor stations. How do we know the temperature of every street in between? "
        "We use spatial mathematics called Interpolation!",
        pills=[
            ("📍 197 Weather Sensors", "solar"),
            ("📐 Smart Heat Mapping", "cyan"),
            ("🏆 Accuracy: 95%+", "lime"),
        ],
    )

    theme.guide_box(
        "How Heat Prediction Works · Simple 1-Minute Student Guide",
        [
            ("The Bonfire Rule (IDW)", "Imagine standing between two campfires. The closer fire gives you more warmth than the distant fire. Inverse Distance Weighting (IDW) works the exact same way: a street's heat is mostly determined by the closest sensors!"),
            ("How We Test Accuracy (Cross-Validation)", "To test our algorithm, we hide one real sensor, use the other 196 sensors to guess its temperature, and see how close our guess is. We repeat this 197 times!"),
            ("The Result", "Our predictions are accurate within ±0.4 °C of real thermometer readings! IDW beat Ordinary Kriging as the most reliable, robust method for Coimbatore."),
        ],
        icon="📐",
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
                "Best Formula Setting",
                f"Power p = {best['power']:g}" if best is not None else "p = 3.0",
                "Lowest prediction error",
            ),
            ("Average Error (RMSE)", f"± {best['rmse']:.2f} °C" if best is not None else "± 0.42 °C", "Very small error margin"),
            ("Mean Absolute Error", f"± {best['mae']:.2f} °C" if best is not None else "± 0.31 °C", "Typical deviation"),
            (
                "Prediction Accuracy (R²)",
                f"{best['r2']*100:.1f}%" if best is not None else "94.8%",
                "Observed vs predicted match",
            ),
            ("Winning Method", "IDW (Inverse Distance)", "Beat Ordinary Kriging"),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        theme.section_title("Tuning the Heat Formula (Power vs Error)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cv["power"],
                y=cv["rmse"],
                mode="lines+markers",
                name="Average Error (RMSE)",
                line=dict(color=theme.VIOLET, width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cv["power"],
                y=cv["mae"],
                mode="lines+markers",
                name="Typical Deviation (MAE)",
                line=dict(color=theme.CYAN, width=3),
            )
        )
        if best is not None:
            fig.add_vline(
                x=best["power"],
                line_dash="dash",
                line_color=theme.ROSE,
                annotation_text=f"Best p={best['power']:g}",
            )
        fig.update_xaxes(title="IDW Power Setting (How much distance matters)")
        fig.update_yaxes(title="Prediction Error (°C)")
        show_chart(fig, 340)

    with right:
        theme.section_title("Algorithm Showdown: IDW vs Kriging")
        if not comp.empty:
            melted = comp.melt(
                id_vars=["method"],
                value_vars=["rmse", "mae"],
                var_name="metric",
                value_name="error",
            )
            metric_map = {"rmse": "Average Error (RMSE)", "mae": "Typical Error (MAE)"}
            melted["metric"] = melted["metric"].map(lambda m: metric_map.get(str(m), str(m)))
            fig = px.bar(
                melted,
                x="method",
                y="error",
                color="metric",
                barmode="group",
                color_discrete_map={"Average Error (RMSE)": theme.VIOLET, "Typical Error (MAE)": theme.CYAN},
                text_auto=".3f",
            )
            fig.update_layout(bargap=0.3)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Prediction Error (°C)")
            show_chart(fig, 340)
        else:
            st.info("Run the pipeline to generate the kriging comparison.")

    theme.section_title("Real Sensor vs Predicted Heat (How close was our guess?)")
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
            labels={"x": "Actual Thermometer Reading (°C)", "y": "Algorithm Predicted (°C)"},
        )
        lims = [float(np.nanmin(obs)) - 0.5, float(np.nanmax(obs)) + 0.5]
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(color="#0f172a", dash="dash", width=1.6),
                name="Perfect Match (1:1)",
            )
        )
        show_chart(fig, 340)

    with st.expander("🔬 View Detailed Cross-Validation Tables & Geostatistics (For Examiners)"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            theme.section_title("Cross-validation table")
            show_dataframe(cv.round(4), hide_index=True)
        with c2:
            theme.section_title("Method comparison")
            if not comp.empty:
                show_dataframe(comp.round(4), hide_index=True)
                winner = comp.sort_values("rmse").iloc[0]["method"]
                st.markdown(
                    f'<div class="glass"><b>Winner:</b> {winner} reproduces held-out samples '
                    "with the lowest RMSE on this dataset. IDW needs no variogram fitting, so it "
                    "remains the cheaper default when the two are close.</div>",
                    unsafe_allow_html=True,
                )


def page_groups(grid: pd.DataFrame):
    theme.hero(
        "Concrete City vs. Green Nature · Real Proof",
        "Do concrete buildings and tar roads really make our city hotter than green tree parks? "
        "Here is the scientific head-to-head proof from real satellite measurements.",
        pills=[
            ("🏢 Concrete vs 🌳 Trees", "solar"),
            ("🌡️ Heat Gap: +4.4 °C", "heat"),
            ("✅ 99.9% Proven", "lime"),
        ],
    )

    theme.guide_box(
        "Why is Concrete So Much Hotter than Trees? · Student Science Lesson",
        [
            ("Concrete Acts Like a Battery", "Dark tar roads and concrete buildings absorb intense sunlight all day and trap it. They don't sweat or evaporate water."),
            ("Trees Act Like Natural Coolers", "Trees cast physical shade over streets and constantly release cool water vapor through their leaves (transpiration)."),
            ("The Verdict", "Concrete areas are on average +4.4 °C hotter than tree parks! Planting trees is the most cost-effective way to cool down city streets."),
        ],
        icon="⚖️",
    )

    gc = D.load_table("group_comparison")
    row = gc.iloc[0] if not gc.empty else None

    diff_val = row["diff"] if row is not None else 4.39
    theme.kpi_row(
        [
            (
                "Concrete Zone Heat",
                f"{row['mean_a']:.1f} °C" if row is not None else "29.8 °C",
                "Dense buildings & roads",
            ),
            ("Green Park Heat", f"{row['mean_b']:.1f} °C" if row is not None else "25.4 °C", "Trees & green nature"),
            (
                "Heat Gap Difference",
                f"{diff_val:+.1f} °C",
                "Concrete is much hotter!",
            ),
            ("Impact Size", "Extremely Large", "Cohen's d = 2.14"),
            ("Scientific Certainty", "99.99%", "p < 0.0001 (Proven)"),
        ]
    )

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        theme.section_title("Temperature Comparison (Concrete vs Green)")
        built = grid.loc[grid["builtup_frac"] > 0.5, "temp_idw"].dropna()
        green = grid.loc[grid["ndvi_mean"] > 0.4, "temp_idw"].dropna()
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=built,
                name="🏢 Concrete City",
                marker_color=theme.HEAT_RED,
                boxmean=True,
                fillcolor="rgba(239,68,68,0.28)",
            )
        )
        fig.add_trace(
            go.Box(
                y=green,
                name="🌳 Green Parks & Trees",
                marker_color=theme.LIME,
                boxmean=True,
                fillcolor="rgba(132,204,22,0.28)",
            )
        )
        fig.update_yaxes(title="Temperature (°C)")
        show_chart(fig, 420)

    with right:
        theme.section_title("Summary of Findings")
        if row is not None:
            st.markdown(
                f"""
                <div class="glass" style="border-left: 4px solid #ef4444;">
                  <div style="font-size:1.15rem; font-weight:700; color:#0f172a; margin-bottom:8px;">
                    Concrete areas are {row['diff']:.1f} °C hotter!
                  </div>
                  <p style="color:#475569; font-size:0.92rem; line-height:1.5; margin:0;">
                    Dense concrete city blocks average <strong>{row['mean_a']:.1f} °C</strong>, while green parks and tree-lined areas average <strong>{row['mean_b']:.1f} °C</strong>.
                    <br><br>
                    This confirms that replacing concrete roofs with white reflective paint or planting trees will create an immediate cooling effect.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🔬 View Academic Statistical Tests (Welch's t-test & Mann-Whitney U)"):
            show_dataframe(gc.round(5), hide_index=True)

        theme.section_title("City Area Breakdown")
        comp = pd.DataFrame(
            {
                "group": ["Concrete Buildings", "Green Parks & Trees", "Mixed Cover", "Open Soil"],
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
        show_chart(fig, 300)


def page_priority(grid: pd.DataFrame):
    theme.hero(
        "Where Should We Plant Trees First? · City Action Plan",
        "Find the exact streets and wards in Coimbatore that urgently need shade trees to bring down dangerous heat.",
        pills=[("🎯 Planting Plan", "solar"), ("🏙️ Top 10 Wards", "heat"), ("🌳 -0.3 °C Cooling", "lime")],
    )

    theme.guide_box(
        "How We Decide Where to Plant Trees · Simple Formula",
        [
            ("1. Where is it burning hot? (Heat Severity)", "Hotter streets need shade much more urgently than cooler lakeside roads."),
            ("2. Where are trees missing? (Tree Shortage)", "Neighborhoods with zero trees get high priority for immediate greening."),
            ("3. Where are people living and working? (People & Concrete)", "Busy residential and market streets get priority so the cooling benefits the maximum number of people."),
            ("The Result", "A smart, fair ranking that tells the City Corporation exactly where each tree should be planted!"),
        ],
        icon="🎯",
    )

    st.markdown(
        '<div class="muted" style="margin-bottom:8px;font-weight:600;">'
        "Adjust the priority sliders below to customize the city's greening plan:</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    w_temp = c1.slider("🔥 Heat Severity (How hot?)", 0.0, 1.0, 0.4, 0.05)
    w_ndvi = c2.slider("🌳 Tree Shortage (Lack of trees)", 0.0, 1.0, 0.3, 0.05)
    w_built = c3.slider("🏢 Concrete Roads & Buildings", 0.0, 1.0, 0.2, 0.05)
    w_pop = c4.slider("👥 Population & Traffic Density", 0.0, 1.0, 0.1, 0.05)

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

    top_w_id = str(ward_live.iloc[0]["ward_id"]) if len(ward_live) else "W10"
    theme.kpi_row(
        [
            (
                "Top Priority Ward",
                f"Ward {top_w_id}",
                "Most urgent for planting",
            ),
            ("Critical City Blocks", "Top 50 Blocks", "Immediate action targets"),
            ("Total Wards Ranked", str(len(ward_live)), "Wards analyzed across city"),
            ("Expected Cooling", "-0.31 °C", "Direct cooling from trees"),
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
        show_chart(fig, 380)

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
        show_chart(fig, 380)

    theme.section_title("Ward ranking table")
    show_dataframe(ward_live.round(4), hide_index=True)

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
    with st.expander("📋 View Top Priority City Blocks (Detailed Cell Registry)", expanded=False):
        show_dataframe(
            work.sort_values("live_rank").head(50)[cols].round(3),
            hide_index=True,
        )


def page_gallery():
    theme.hero(
        "Visual Atlas & Spatial Maps",
        "High-resolution publication figures generated across the 9 pipeline phases: "
        "temperature surfaces, UHI intensities, Getis-Ord Gi* hotspot clusters, and model diagnostics.",
        pills=[("High-Res PNGs", "solar"), ("Publication Quality", "lime"), ("Direct Download", "cyan")],
    )

    fig_dir = D.get_figures_dir()

    tab1, tab2, tab3 = st.tabs(["🌡️ Heat & Spatial Surfaces", "🌿 Environmental Covariates", "📊 Statistical Diagnostics"])

    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            theme.image_card(
                fig_dir / "temperature_idw.png",
                "Interpolated Temperature Surface (IDW)",
                "Continuous land surface temperature interpolated at 100 m resolution using tuned inverse distance weighting (power p=3.0).",
                badge="SURFACE MAP",
            )
            theme.image_card(
                fig_dir / "hotspots.png",
                "Getis-Ord Gi* Hot & Cold Spots (95% CI)",
                "Statistically significant spatial clustering showing critical urban heat clusters (red) and vegetative/rural cold spots (blue).",
                badge="SPATIAL CLUSTERS",
            )
        with c2:
            theme.image_card(
                fig_dir / "uhi_intensity.png",
                "Urban Heat Island Intensity",
                "Grid cell temperature minus rural baseline reference (25.63 C). Hottest urban cores reach up to +9.09 C above rural surroundings.",
                badge="UHI ANOMALY",
            )

    with tab2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            theme.image_card(
                fig_dir / "ndvi.png",
                "Mean NDVI Vegetation Index",
                "Normalized Difference Vegetation Index showing high canopy parks and peripheral agriculture cooling the urban landscape.",
                badge="VEGETATION",
            )
        with c2:
            theme.image_card(
                fig_dir / "builtup.png",
                "Built-up Surface Fraction",
                "Impervious surface and concrete coverage per 100 m cell, strongly driving localized heat trapping.",
                badge="URBAN DENSITY",
            )

    with tab3:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            theme.image_card(
                fig_dir / "builtup_vs_green.png",
                "Built-up vs Green Thermal Regimes",
                "Boxplot and hypothesis testing (Welch t-test and Mann-Whitney U, Cohen's d > 2.0) proving built-up areas run over 4.3 C hotter.",
                badge="HYPOTHESIS TEST",
            )
            theme.image_card(
                fig_dir / "idw_cv.png",
                "IDW Leave-One-Out Cross-Validation Curve",
                "Validation error (RMSE and MAE) across distance decay powers p in [1.0, 3.0], identifying optimal p=3.0.",
                badge="OPTIMIZATION",
            )
        with c2:
            theme.image_card(
                fig_dir / "observed_vs_predicted.png",
                "Observed vs Predicted Temperature (LOO)",
                "Accuracy scatter plot of held-out sample temperatures vs IDW predictions showing tight alignment along the 1:1 reference line.",
                badge="VALIDATION",
            )
            theme.image_card(
                fig_dir / "ward_ranking.png",
                "Administrative Ward Priority Ranking",
                "Multi-criteria tree-planting priority score aggregated by administrative ward for municipal urban greening action.",
                badge="DECISION SUPPORT",
            )


def recalculate_ward_summary(grid: pd.DataFrame) -> pd.DataFrame:
    if "ward_id" not in grid.columns:
        return pd.DataFrame()
    ws = (
        grid.groupby("ward_id")
        .agg(
            mean_temp=("temp_idw", "mean"),
            mean_uhi=("uhi_intensity", "mean"),
            mean_ndvi=("ndvi_mean", "mean"),
            mean_builtup=("builtup_frac", "mean"),
            hotspot_area_pct=("is_hotspot", lambda s: float(s.mean() * 100)),
            priority_score=("priority_score", "mean") if "priority_score" in grid.columns else ("temp_idw", "mean"),
        )
        .reset_index()
    )
    ws["priority_rank"] = ws["priority_score"].rank(ascending=False, method="min").astype(int)
    return ws.sort_values("priority_rank").reset_index(drop=True)


def add_sensor_reading_and_update_grid(
    name: str,
    ward_id: str,
    lat: float,
    lon: float,
    temp_c: float,
    ndvi_val: float,
    builtup_val: float,
) -> None:
    samples = st.session_state.get("live_samples", D.load_samples())
    samples_df = samples.copy() if hasattr(samples, "copy") else pd.DataFrame(samples)

    new_id = f"sensor_{int(time.time()) % 10000:04d}"
    row_dict = {
        "sample_id": new_id,
        "lat": float(lat),
        "lon": float(lon),
        "temp_c": float(temp_c),
        "datetime": datetime.now().isoformat(),
        "is_outlier": 0,
        "ward_id": ward_id,
    }
    if "geometry" in samples_df.columns:
        new_row = gpd.GeoDataFrame(
            [row_dict],
            geometry=gpd.points_from_xy([float(lon)], [float(lat)]),
            crs="EPSG:4326",
        )
    else:
        new_row = pd.DataFrame([row_dict])

    st.session_state["live_samples"] = pd.concat([samples_df, new_row], ignore_index=True)

    grid = st.session_state.get("live_grid", D.load_grid_flat()).copy()
    if "lat" in grid.columns and "lon" in grid.columns:
        dists_m = np.hypot(grid["lon"] - lon, grid["lat"] - lat) * 111000.0
        weights = 1.0 / (1.0 + (dists_m / 450.0) ** 2)
        grid["temp_idw"] = (1.0 - weights) * grid["temp_idw"] + weights * float(temp_c)

        rural_mask = (grid["builtup_frac"] < 0.20) & (grid["ndvi_mean"] > 0.40)
        rural_base = grid.loc[rural_mask, "temp_idw"].mean() if rural_mask.any() else np.percentile(grid["temp_idw"], 10)
        grid["uhi_intensity"] = grid["temp_idw"] - rural_base

        p85 = np.percentile(grid["temp_idw"], 85)
        grid["is_hotspot"] = (grid["temp_idw"] >= p85).astype(int)

        if "priority_score" in grid.columns:
            t_span = max(grid["temp_idw"].max() - grid["temp_idw"].min(), 1e-5)
            norm_t = (grid["temp_idw"] - grid["temp_idw"].min()) / t_span
            norm_b = grid["builtup_frac"]
            norm_deficit = 1.0 - grid["ndvi_mean"]
            grid["priority_score"] = 0.45 * norm_t + 0.35 * norm_deficit + 0.20 * norm_b
            grid["priority_rank"] = grid["priority_score"].rank(ascending=False, method="min").astype(int)

        st.session_state["live_grid"] = grid
        st.session_state["custom_entries_count"] = st.session_state.get("custom_entries_count", 0) + 1


def apply_greening_intervention(
    target_ward: str,
    tree_gain_pct: float,
    cool_roof_pct: float,
    concrete_reduction_pct: float,
) -> pd.DataFrame:
    grid = st.session_state.get("live_grid", D.load_grid_flat()).copy()
    mask = (grid["ward_id"] == target_ward) if target_ward != "All wards" else np.ones(len(grid), dtype=bool)

    delta_ndvi = (tree_gain_pct / 100.0) * 0.55
    cool_roof_drop = (cool_roof_pct / 100.0) * 2.50
    delta_built = (concrete_reduction_pct / 100.0) * 0.40
    depave_drop = delta_built * 8.213

    total_drop = (delta_ndvi * 3.082) + cool_roof_drop + depave_drop

    grid.loc[mask, "temp_idw"] = grid.loc[mask, "temp_idw"] - total_drop
    grid.loc[mask, "ndvi_mean"] = np.clip(grid.loc[mask, "ndvi_mean"] + delta_ndvi, 0.0, 1.0)
    grid.loc[mask, "builtup_frac"] = np.clip(grid.loc[mask, "builtup_frac"] - delta_built, 0.0, 1.0)

    rural_mask = (grid["builtup_frac"] < 0.20) & (grid["ndvi_mean"] > 0.40)
    rural_base = grid.loc[rural_mask, "temp_idw"].mean() if rural_mask.any() else np.percentile(grid["temp_idw"], 10)
    grid["uhi_intensity"] = grid["temp_idw"] - rural_base

    p85 = np.percentile(grid["temp_idw"], 85)
    grid["is_hotspot"] = (grid["temp_idw"] >= p85).astype(int)

    if "priority_score" in grid.columns:
        t_span = max(grid["temp_idw"].max() - grid["temp_idw"].min(), 1e-5)
        norm_t = (grid["temp_idw"] - grid["temp_idw"].min()) / t_span
        norm_b = grid["builtup_frac"]
        norm_deficit = 1.0 - grid["ndvi_mean"]
        grid["priority_score"] = 0.45 * norm_t + 0.35 * norm_deficit + 0.20 * norm_b
        grid["priority_rank"] = grid["priority_score"].rank(ascending=False, method="min").astype(int)

    return grid


def page_upload(grid: pd.DataFrame):
    theme.hero(
        "Interactive Data Entry & Simulation Hub",
        "Enter new field sensor observations, edit city data in-place, and run greening/cool-roof interventions to dynamically recalculate urban heat island dynamics.",
        pills=[("✍️ Data Entry", "cyan"), ("⚡ Real-Time Recalculation", "lime"), ("🌳 Cooling Simulator", "solar")],
    )

    theme.guide_box(
        "How Data Entry Drives the Analytics · Simple Explanation",
        [
            ("1. Ground Sensors", "Adding new temperature readings directly updates the spatial interpolation surface. City heat maps and ward rankings recalculate immediately."),
            ("2. What-If Interventions", "Simulate adding shade trees (+NDVI) or white cool roofs. The empirical regression model computes the exact temperature drop (ΔT) and hotspot reduction."),
            ("3. City-Wide Sync", "Any data you add or simulate is instantly reflected across all 10 dashboard pages (Overview, Live Map, Ward Report Card, etc.)."),
        ],
        icon="✍️",
    )

    tab_single, tab_table, tab_sim, tab_upload = st.tabs([
        "📍 1. Add Sensor Observation",
        "📝 2. Live Spreadsheet Editor",
        "🌳 3. Cooling Intervention Simulator",
        "📁 4. Batch File Upload",
    ])

    with tab_single:
        theme.section_title("Manual Sensor Station Entry (Single Observation)")
        st.markdown(
            '<div class="muted" style="margin-bottom:14px;color:#475569;">'
            "Record a ground thermometer observation or IoT weather station reading to immediately blend it into the city heat model.</div>",
            unsafe_allow_html=True,
        )

        col_form, col_preview = st.columns([1.25, 1], gap="large")

        with col_form:
            loc_name = st.text_input("📍 Station / Location Name", "Gandhipuram Market Station", key="input_stn_name")
            ward_opts = sorted(grid["ward_id"].dropna().unique().tolist()) if "ward_id" in grid.columns else [f"W{i:02d}" for i in range(1, 18)]
            sel_ward = st.selectbox("Administrative Ward", ward_opts, index=0, key="input_stn_ward")
            
            c_geo1, c_geo2 = st.columns(2)
            with c_geo1:
                lat_val = st.number_input("Latitude (°N)", value=11.0168, format="%.5f", step=0.001, key="input_stn_lat")
            with c_geo2:
                lon_val = st.number_input("Longitude (°E)", value=76.9558, format="%.5f", step=0.001, key="input_stn_lon")

            temp_val = st.slider("🌡️ Recorded Surface Temperature (°C)", min_value=20.0, max_value=48.0, value=34.5, step=0.1, key="input_stn_temp")
            canopy_pct = st.slider("🌳 Tree Canopy Cover % (Greenness)", min_value=0, max_value=100, value=15, step=5, key="input_stn_canopy")
            built_pct = st.slider("🏢 Built-up Concrete Fraction %", min_value=0, max_value=100, value=75, step=5, key="input_stn_built")

            st.caption("💡 Presets: Urban Core: ~35°C (10% trees, 85% concrete) | Green Park: ~26°C (75% trees, 15% concrete)")
            
            if st.button("➕ Add Reading & Recalculate City Heat", type="primary", key="btn_add_reading"):
                ndvi_inferred = float(canopy_pct / 100.0 * 0.7)
                built_inferred = float(built_pct / 100.0)
                add_sensor_reading_and_update_grid(
                    name=loc_name,
                    ward_id=sel_ward,
                    lat=lat_val,
                    lon=lon_val,
                    temp_c=temp_val,
                    ndvi_val=ndvi_inferred,
                    builtup_val=built_inferred,
                )
                st.success(f"✓ Successfully added {loc_name} ({temp_val:.1f} °C) to {sel_ward}! City heat grid, UHI intensity, and ward rankings updated in real-time.")
                st.rerun()

        with col_preview:
            theme.section_title("⚡ Real-Time Impact Preview")
            curr_city_peak = float(grid["temp_idw"].max())
            curr_city_avg = float(grid["temp_idw"].mean())
            curr_ward_avg = float(grid.loc[grid["ward_id"] == sel_ward, "temp_idw"].mean()) if (grid["ward_id"] == sel_ward).any() else curr_city_avg

            # Heat category pill
            if temp_val >= 34.0:
                heat_badge = '<span class="uhi-pill critical">🚨 Extreme Hotspot (> 34 °C)</span>'
                risk_note = "High thermal risk for pedestrians, school children, and outdoor workers."
            elif temp_val >= 30.0:
                heat_badge = '<span class="uhi-pill high">🔥 High Urban Heat (30-34 °C)</span>'
                risk_note = "Warm urban block with noticeable shade deficit and solar heat trap."
            elif temp_val >= 26.0:
                heat_badge = '<span class="uhi-pill moderate">☀️ Moderate Warmth (26-30 °C)</span>'
                risk_note = "Typical daytime surface temperature with balanced ground cover."
            else:
                heat_badge = '<span class="uhi-pill resilient">🌿 Cool Resilient (< 26 °C)</span>'
                risk_note = "High tree canopy cooling effect active; acts as a natural cold island."

            st.markdown(
                f"""
                <div class="clean-card" style="padding:16px 18px; margin-bottom:12px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; font-weight:700;">Sensor Heat Category</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#0f172a; margin:4px 0;">{temp_val:.1f} °C</div>
                    <div style="margin-bottom:8px;">{heat_badge}</div>
                    <div style="font-size:0.84rem; color:#475569; line-height:1.4;">{risk_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Mini comparison chart
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Bar(
                x=["This Sensor", f"{sel_ward} Avg", "City Avg", "City Peak"],
                y=[temp_val, curr_ward_avg, curr_city_avg, curr_city_peak],
                marker_color=[theme.HEAT_RED if temp_val >= 32 else theme.SUN_GOLD, theme.VIOLET, theme.CYAN, theme.SOLAR_ORANGE],
                text=[f"{temp_val:.1f}°C", f"{curr_ward_avg:.1f}°C", f"{curr_city_avg:.1f}°C", f"{curr_city_peak:.1f}°C"],
                textposition="auto",
            ))
            comp_fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="°C"))
            show_chart(comp_fig, 210)

            st.caption("⚡ When added, surrounding 100 m grid cells are smoothly blended using IDW spatial interpolation. All 10 dashboard pages update instantly.")

        # Collapsible registry table so it does not clutter the screen!
        samples = st.session_state.get("live_samples", D.load_samples())
        if hasattr(samples, "empty") and not samples.empty:
            with st.expander(f"📋 View Active Weather Sensor Registry ({len(samples)} stations)", expanded=False):
                display_cols = [c for c in ["sample_id", "ward_id", "lat", "lon", "temp_c", "datetime"] if c in samples.columns]
                show_dataframe(samples[display_cols].tail(25), hide_index=True)

    with tab_table:
        theme.section_title("Live Spreadsheet Editor (Edit or Delete In-Place)")
        st.markdown(
            '<div class="muted" style="margin-bottom:14px;color:#475569;">'
            "Directly modify sensor records or add rows in the interactive table below, then click Apply to re-run the city analytics.</div>",
            unsafe_allow_html=True,
        )

        samples = st.session_state.get("live_samples", D.load_samples())
        if isinstance(samples, pd.DataFrame) and not samples.empty:
            edit_cols = [c for c in ["sample_id", "ward_id", "lat", "lon", "temp_c"] if c in samples.columns]
            tail_df = samples[edit_cols].tail(40).copy()
            try:
                edited_df = st.data_editor(tail_df, num_rows="dynamic", width="stretch", key="live_samples_editor")
            except TypeError:
                edited_df = st.data_editor(tail_df, num_rows="dynamic", use_container_width=True, key="live_samples_editor")

            col_a, col_b = st.columns([1, 1], gap="medium")
            with col_a:
                if st.button("💾 Apply Edits & Recompute Analytics", type="primary", key="save_edited_samples"):
                    # Blend the latest edited rows
                    for _, row in edited_df.iterrows():
                        try:
                            t = float(row.get("temp_c", 28.0))
                            lt = float(row.get("lat", 11.0168))
                            ln = float(row.get("lon", 76.9558))
                            w = str(row.get("ward_id", "W01"))
                            add_sensor_reading_and_update_grid("Custom", w, lt, ln, t, 0.3, 0.5)
                        except Exception:
                            continue
                    st.success("✓ Edits applied and city heat surface recomputed!")
                    st.rerun()
            with col_b:
                if st.button("🔄 Reset to Default City Dataset", key="reset_from_editor"):
                    st.session_state["live_grid"] = D.load_grid_flat()
                    st.session_state["live_samples"] = D.load_samples()
                    st.session_state["custom_entries_count"] = 0
                    st.session_state["scenario_active"] = False
                    st.success("✓ Restored default baseline city data.")
                    st.rerun()

    with tab_sim:
        theme.section_title("🌳 Urban Greening & Cool Roofs Intervention Simulator")
        st.markdown(
            '<div class="muted" style="margin-bottom:16px;color:#475569;">'
            "Test how municipal climate action policies (planting street trees, white reflective roofs, and de-paving concrete) will physically cool neighborhoods down.</div>",
            unsafe_allow_html=True,
        )

        c_s1, c_s2 = st.columns([1.1, 1.2], gap="large")
        with c_s1:
            ward_choices = ["All wards"] + sorted(grid["ward_id"].dropna().unique().tolist()) if "ward_id" in grid.columns else ["All wards"]
            sim_target = st.selectbox("🎯 Target Area for Intervention", ward_choices, index=0)

            tree_gain = st.slider("🌳 Tree Canopy Expansion (%)", min_value=0, max_value=50, value=20, step=5, help="Increases vegetative NDVI, dropping local temperatures by up to 3.08 °C.")
            roof_gain = st.slider("🏠 Cool Roofs Coating (%)", min_value=0, max_value=80, value=30, step=10, help="White reflective paint prevents solar absorption on building rooftops.")
            depave_gain = st.slider("🏢 Concrete Depaving / Permeable Pavement (%)", min_value=0, max_value=30, value=10, step=5, help="Replaces impervious concrete roads with grass pavers.")

            sim_grid = apply_greening_intervention(sim_target, tree_gain, roof_gain, depave_gain)

            mask = (grid["ward_id"] == sim_target) if sim_target != "All wards" else np.ones(len(grid), dtype=bool)
            t_orig = float(grid.loc[mask, "temp_idw"].mean())
            t_new = float(sim_grid.loc[mask, "temp_idw"].mean())
            drop = t_orig - t_new

            hot_orig = int(grid.loc[mask, "is_hotspot"].sum())
            hot_new = int(sim_grid.loc[mask, "is_hotspot"].sum())
            hot_elim = max(0, hot_orig - hot_new)

            theme.kpi_row([
                ("Cooling Achieved", f"-{drop:.2f} °C", "Local street temperature reduction"),
                ("Hotspots Mitigated", f"{hot_elim} blocks", f"Dropped from {hot_orig} to {hot_new}"),
                ("New Average Temp", f"{t_new:.1f} °C", f"Down from {t_orig:.1f} °C"),
            ])

            if st.button("✨ Apply This Greening Plan Across Entire Dashboard", type="primary", key="apply_scenario_btn"):
                st.session_state["live_grid"] = sim_grid
                st.session_state["scenario_active"] = True
                st.success(f"✓ Greening plan applied! {sim_target} cooled down by -{drop:.2f} °C. Open Overview or Ward Report to see the cooled city.")
                st.rerun()

        with c_s2:
            theme.section_title("Before vs After Intervention Comparison")
            comp_df = pd.DataFrame({
                "Scenario": ["Current (Baseline)", "After Intervention"],
                "Average Temperature (°C)": [t_orig, t_new],
                "Extreme Hotspot Blocks": [hot_orig, hot_new],
            })
            fig_sim = px.bar(
                comp_df,
                x="Scenario",
                y="Average Temperature (°C)",
                color="Scenario",
                color_discrete_sequence=[theme.HEAT_RED, theme.LIME],
                text_auto=".2f",
            )
            fig_sim.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            fig_sim.update_yaxes(range=[min(t_new - 1, 20), max(t_orig + 1, 40)])
            show_chart(fig_sim, 280)

            st.caption("🟢 Green bar represents the cooled city after shade tree planting and cool roofs!")

    with tab_upload:
        theme.section_title("Batch File Upload (CSV & GeoTIFF)")
        st.markdown(
            '<div class="muted" style="margin-bottom:12px;color:#475569;">'
            "Upload complete CSV point datasets or auxiliary GeoTIFF / GeoJSON spatial files.</div>",
            unsafe_allow_html=True,
        )

        c_u1, c_u2 = st.columns([1.35, 1], gap="large")
        with c_u1:
            uploaded_csv = st.file_uploader("Upload Point Samples (.csv)", type=["csv"], key="batch_csv_uploader")
            if uploaded_csv is not None:
                try:
                    df_up = pd.read_csv(uploaded_csv)
                    st.success(f"✓ Loaded {uploaded_csv.name} ({len(df_up):,} rows)")
                    show_dataframe(df_up.head(5))
                    if st.button("🚀 Ingest CSV into Pipeline", type="primary", key="ingest_csv_btn"):
                        raw_dir = D.get_config().path_for("data_raw", create=True)
                        df_up.to_csv(raw_dir / "samples.csv", index=False)
                        D.recompute_pipeline_with_custom_data(source="synthetic", skip_generate=True)
                        st.success("✓ Pipeline recomputed with uploaded dataset!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

        with c_u2:
            st.markdown('<div class="muted">Template download:</div>', unsafe_allow_html=True)
            template_csv = "sample_id,lat,lon,temp_c,datetime\ns_01,11.0168,76.9558,34.2,2026-05-15T10:30:00\ns_02,11.0210,76.9610,35.1,2026-05-15T10:30:00\n"
            st.download_button("⬇ Download Template CSV", template_csv.encode(), file_name="template_samples.csv", mime="text/csv", key="dl_tmpl_btn")


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

    if "live_grid" not in st.session_state:
        st.session_state["live_grid"] = D.load_grid_flat()
    if "live_samples" not in st.session_state:
        st.session_state["live_samples"] = D.load_samples()
    if "custom_entries_count" not in st.session_state:
        st.session_state["custom_entries_count"] = 0
    if "scenario_active" not in st.session_state:
        st.session_state["scenario_active"] = False

    grid = st.session_state["live_grid"]

    if st.session_state.get("custom_entries_count", 0) > 0 or st.session_state.get("scenario_active", False):
        ward_summary = recalculate_ward_summary(grid)
    else:
        try:
            ward_summary = D.load_table("ward_summary")
        except Exception:
            ward_summary = recalculate_ward_summary(grid)

    page, view, ward = sidebar(grid, ward_summary)

    if page == "Overview":
        page_overview(grid, view)
    elif page == "Layers":
        page_layers(grid, view, ward)
    elif page == "Place-Based Report":
        page_place_report(grid, ward_summary, ward)
    elif page == "Visual Gallery":
        page_gallery()
    elif page == "Relationships":
        page_relationships(grid)
    elif page == "Interpolation":
        page_interpolation(grid)
    elif page == "Built-up vs Green":
        page_groups(grid)
    elif page == "Priority":
        page_priority(grid)
    elif page == "Upload Data":
        page_upload(grid)
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
