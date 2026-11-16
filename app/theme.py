"""Shared visual theme for the Streamlit dashboard: Sun & Dry Heatwave Edition.

Design language: Deep warm obsidian base with radiant sun glow, solar corona
gradients, heatwave thermal shimmer, glassmorphic cards with amber/crimson heat
borders, and vivid thermal visualization styling.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

# ---------------------------------------------------------------- palette (Sun & Dry Thermal)
INK = "#0c0a0d"  # deep warm obsidian
SURFACE = "rgba(26, 20, 24, 0.78)"
SURFACE_SOLID = "#1a1418"
BORDER = "rgba(251, 146, 60, 0.28)"  # warm solar amber-orange border
TEXT = "#f8fafc"
MUTED = "#cbd5e1"

# Warm solar & heat colors
SUN_GOLD = "#fbbf24"
SOLAR_ORANGE = "#f97316"
HEAT_RED = "#ef4444"
WARM_AMBER = "#d97706"
DESERT_SAND = "#fde047"
CRIMSON_GLOW = "#dc2626"

# Contrasting vegetation & water cooling accents
CYAN = "#06b6d4"   # water cooling
LIME = "#84cc16"   # green vegetation
VIOLET = "#a855f7" # spatial boundary
ROSE = "#f97316"   # heat accent
BLUE = "#38bdf8"
AMBER = "#fbbf24"

# Solar heatwave gradient: Sun Gold -> Solar Flare Orange -> Thermal Heat Red
ACCENT_GRADIENT = f"linear-gradient(135deg, {SUN_GOLD} 0%, {SOLAR_ORANGE} 50%, {HEAT_RED} 100%)"
SUN_GLOW = "radial-gradient(circle, rgba(251,191,36,0.35) 0%, rgba(249,115,22,0.18) 50%, transparent 75%)"

# Categorical palette for maps and charts
CATEGORICAL = [SOLAR_ORANGE, CYAN, LIME, SUN_GOLD, HEAT_RED, "#f43f5e", "#10b981", "#8b5cf6"]


def inject_theme() -> None:
    """Inject the global solar & dry heat stylesheet once per session."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {{
            --ink: {INK};
            --surface: {SURFACE};
            --border: {BORDER};
            --text: {TEXT};
            --muted: {MUTED};
            --sun-gold: {SUN_GOLD};
            --solar-orange: {SOLAR_ORANGE};
            --heat-red: {HEAT_RED};
            --warm-amber: {WARM_AMBER};
            --cyan: {CYAN};
            --lime: {LIME};
        }}

        .stApp {{
            background:
                /* Radiant Glowing Sun Orb in upper right */
                radial-gradient(circle 420px at 85% 4%, rgba(254,240,138,0.38) 0%, rgba(251,191,36,0.25) 35%, rgba(249,115,22,0.14) 55%, transparent 75%),
                /* Heatwave atmospheric thermal haze across the top */
                radial-gradient(ellipse 1000px 420px at 50% -8%, rgba(239,68,68,0.20), transparent 65%),
                /* Warm solar flare on left */
                radial-gradient(circle 500px at 10% 12%, rgba(245,158,11,0.15), transparent 55%),
                /* Arid dry earth ground reflection at bottom */
                radial-gradient(ellipse 1100px 520px at 50% 112%, rgba(180,83,9,0.16), transparent 70%),
                {INK} !important;
            color: {TEXT} !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* Clean header styling */
        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}
        .block-container {{ padding: 1.4rem 2.2rem 4rem !important; max-width: 1540px !important; }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            letter-spacing: -0.02em;
            color: {TEXT} !important;
        }}
        h1 {{ font-weight: 700; }}
        h2 {{ font-weight: 650; margin-top: 0.4rem; }}
        p, span, label, li, td, th {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}
        a {{ color: {SUN_GOLD}; text-decoration: none; font-weight: 500; }}
        a:hover {{ color: {SOLAR_ORANGE}; text-decoration: underline; }}

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"], [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1c1418 0%, #100a0d 100%) !important;
            border-right: 1px solid rgba(251,146,60,0.26) !important;
            box-shadow: 4px 0 24px rgba(0,0,0,0.4);
        }}
        [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

        /* ---------- sun & dry hero banner ---------- */
        .uhi-hero {{
            position: relative;
            border-radius: 26px;
            padding: 26px 32px;
            background: linear-gradient(135deg, rgba(239,68,68,0.24) 0%, rgba(249,115,22,0.22) 45%, rgba(251,191,36,0.18) 100%);
            border: 1px solid rgba(251,146,60,0.42);
            box-shadow: 0 14px 40px rgba(239,68,68,0.14), inset 0 1px 0 rgba(255,255,255,0.15);
            margin-bottom: 22px;
            backdrop-filter: blur(16px);
        }}
        .uhi-hero h1 {{
            margin: 0 0 8px;
            font-size: 2.15rem;
            background: linear-gradient(120deg, #ffffff 30%, #fde047 70%, #f97316 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .uhi-hero p {{ margin: 0; color: #e2e8f0 !important; font-size: 1.0rem; max-width: 68ch; line-height: 1.5; }}
        .uhi-pill {{
            display: inline-block; margin-top: 14px; margin-right: 8px;
            padding: 5px 14px; border-radius: 999px; font-size: 0.74rem; font-weight: 650;
            letter-spacing: 0.04em; text-transform: uppercase;
            background: rgba(249,115,22,0.18); border: 1px solid rgba(249,115,22,0.52);
            color: #fed7aa !important;
        }}
        .uhi-pill.solar {{ background: rgba(251,191,36,0.22); border-color: rgba(251,191,36,0.65); color: #fef08a !important; }}
        .uhi-pill.cyan {{ background: rgba(6,182,212,0.16); border-color: rgba(6,182,212,0.55); color: #cffafe !important; }}
        .uhi-pill.lime {{ background: rgba(132,204,22,0.16); border-color: rgba(132,204,22,0.55); color: #ecfccb !important; }}
        .uhi-pill.heat {{ background: rgba(239,68,68,0.22); border-color: rgba(239,68,68,0.65); color: #fecaca !important; }}
        .uhi-pill.critical {{ background: rgba(239,68,68,0.32); border-color: #ef4444; color: #fee2e2 !important; font-weight: 700; box-shadow: 0 0 12px rgba(239,68,68,0.5); }}
        .uhi-pill.high {{ background: rgba(249,115,22,0.28); border-color: #f97316; color: #ffedd5 !important; font-weight: 700; }}
        .uhi-pill.moderate {{ background: rgba(245,158,11,0.24); border-color: #f59e0b; color: #fef3c7 !important; }}
        .uhi-pill.resilient {{ background: rgba(16,185,129,0.24); border-color: #10b981; color: #d1fae5 !important; }}

        /* ---------- KPI cards ---------- */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin: 6px 0 22px; }}
        .kpi {{
            border-radius: 20px; padding: 18px 20px;
            background: {SURFACE};
            border: 1px solid {BORDER};
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 26px rgba(0,0,0,0.32);
            transition: transform 180ms cubic-bezier(.2,.7,.3,1), border-color 180ms, box-shadow 180ms;
        }}
        .kpi:hover {{
            transform: translateY(-4px);
            border-color: rgba(251,191,36,0.6);
            box-shadow: 0 14px 34px rgba(249,115,22,0.24);
        }}
        .kpi .k-label {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.09em; color: {MUTED} !important; font-weight: 650; }}
        .kpi .k-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.82rem; font-weight: 700; margin-top: 6px;
                         background: {ACCENT_GRADIENT}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .kpi .k-sub {{ font-size: 0.78rem; color: #94a3b8 !important; margin-top: 4px; }}

        /* ---------- glass cards ---------- */
        .glass {{
            border-radius: 22px; padding: 22px 24px; background: {SURFACE};
            border: 1px solid {BORDER}; backdrop-filter: blur(16px); margin-bottom: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }}
        .glass h3 {{ margin-top: 0; color: #fff !important; }}
        .section-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.22rem;
            font-weight: 650;
            margin: 6px 0 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #fff !important;
        }}

        /* ---------- image card gallery ---------- */
        .image-card {{
            border-radius: 20px;
            padding: 16px 18px;
            background: linear-gradient(180deg, rgba(32,24,30,0.85) 0%, rgba(20,15,20,0.92) 100%);
            border: 1px solid rgba(251,146,60,0.32);
            margin-bottom: 20px;
            transition: transform 200ms ease, box-shadow 200ms ease;
        }}
        .image-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(251,191,36,0.65);
            box-shadow: 0 12px 36px rgba(249,115,22,0.22);
        }}

        /* ---------- metrics / tabs ---------- */
        div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: {SUN_GOLD} !important; }}
        div[data-testid="stMetric"] {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-radius: 18px; padding: 12px 16px; backdrop-filter: blur(12px);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{
            height: 42px; border-radius: 12px 12px 0 0; padding: 0 18px;
            color: {MUTED} !important; font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, rgba(249,115,22,0.24) 0%, transparent 100%);
            color: #fff !important;
            border-bottom: 2px solid {SOLAR_ORANGE};
        }}

        /* ---------- controls ---------- */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 14px; border: 1px solid rgba(251,191,36,0.5);
            background: linear-gradient(120deg, rgba(245,158,11,0.35), rgba(239,68,68,0.28));
            color: #ffffff !important; font-weight: 650; padding: 0.52rem 1.2rem;
            box-shadow: 0 4px 18px rgba(245,158,11,0.22);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px); box-shadow: 0 10px 32px rgba(249,115,22,0.45);
            border-color: rgba(251,191,36,0.9);
            color: #ffffff !important;
        }}
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
            background: rgba(28,22,26,0.85) !important; border-radius: 12px !important;
            border-color: {BORDER} !important; color: {TEXT} !important;
        }}
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {SUN_GOLD}; box-shadow: 0 0 12px rgba(251,191,36,0.8); }}

        /* ---------- dataframes ---------- */
        div[data-testid="stDataFrame"] {{
            border-radius: 16px; overflow: hidden;
            border: 1px solid {BORDER};
            box-shadow: 0 6px 20px rgba(0,0,0,0.28);
        }}

        /* ---------- misc ---------- */
        .legend-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
        .legend-dot {{ display: inline-flex; align-items: center; gap: 7px; font-size: 0.82rem; color: #cbd5e1 !important; font-weight: 500; }}
        .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px currentColor; }}
        .muted, .muted * {{ color: {MUTED} !important; font-size: 0.86rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    """Render the solar radiant hero header."""
    pill_html = ""
    for text, tone in pills or []:
        cls = "uhi-pill" + (f" {tone}" if tone else "")
        pill_html += f'<span class="{cls}">{text}</span>'
    st.markdown(
        f"""
        <div class="uhi-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """Render a responsive row of KPI cards: (label, value, sub)."""
    cards = "".join(
        f'<div class="kpi"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>'
        f'<div class="k-sub">{sub}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def glass_open() -> None:
    st.markdown('<div class="glass">', unsafe_allow_html=True)


def glass_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    """Small colour-key row: (colour, label)."""
    html = "".join(
        f'<span class="legend-dot"><span class="dot" style="background:{c}; color:{c}"></span>{label}</span>'
        for c, label in items
    )
    st.markdown(f'<div class="legend-row">{html}</div>', unsafe_allow_html=True)


def image_card(img_path: str | Path, title: str, description: str, badge: str = "MAP") -> None:
    """Render an image inside a stylized solar glass card with captions and download."""
    p = Path(img_path)
    if not p.exists():
        return
    st.markdown(
        f"""
        <div class="image-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.08rem; font-weight:700; color:#fff;">{title}</span>
                <span class="uhi-pill solar" style="margin:0; font-size:0.68rem;">{badge}</span>
            </div>
            <p style="color:#cbd5e1 !important; font-size:0.86rem; margin-bottom:12px; line-height:1.4;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        st.image(str(p), width="stretch")
    except TypeError:
        st.image(str(p), use_container_width=True)
    st.download_button(
        f"⬇  Download {p.name}",
        p.read_bytes(),
        file_name=p.name,
        mime="image/png",
        key=f"dl_card_{p.stem}",
    )


def guide_box(title: str, items: list[tuple[str, str]], icon: str = "💡") -> None:
    """Render a student-friendly concept explanation box with bullet points."""
    body = "".join(
        f'<div style="margin-bottom:8px;"><strong style="color:#fbbf24;">{h}:</strong> '
        f'<span style="color:#e2e8f0;">{desc}</span></div>'
        for h, desc in items
    )
    st.markdown(
        f"""
        <div class="glass" style="border-left: 4px solid #f59e0b; padding: 16px 20px; margin-bottom: 20px; background: rgba(30, 22, 26, 0.72);">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
            <span style="font-size:1.25rem;">{icon}</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:1.04rem; font-weight:700; color:#fff;">{title}</span>
          </div>
          <div style="font-size:0.90rem; line-height:1.55;">
            {body}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
