"""Shared visual theme for the Streamlit dashboard: Clean, Modern & Crisp Light SaaS Edition.

Strictly Light Theme:
- Canvas: Pure slate-50 (#f8fafc)
- Surfaces / Cards: Pure white (#ffffff) with subtle 1px border (#e2e8f0) and soft shadow
- Typography: High-contrast rich slate-900 (#0f172a), slate-600 secondary (#475569)
- Inputs, Selects, Buttons, Tables: 100% Light Theme styling (zero black patches)
- Semantic accents: Solar heat red (#dc2626), vibrant green (#16a34a), ocean blue (#2563eb)
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

# ---------------------------------------------------------------- palette (Crisp Modern Light)
INK = "#f8fafc"           # soft slate-50 light background
SURFACE = "#ffffff"       # pure white card surface
SURFACE_SOLID = "#ffffff"
BORDER = "#e2e8f0"        # clean slate-200 border
BORDER_FOCUS = "#3b82f6"  # active blue border
TEXT = "#0f172a"          # deep rich slate-900 text (maximum legibility)
MUTED = "#64748b"         # slate-500 secondary text
MUTED_LIGHT = "#94a3b8"

# Vibrant semantic colors
SUN_GOLD = "#d97706"      # amber-600
SOLAR_ORANGE = "#ea580c"  # orange-600
HEAT_RED = "#dc2626"      # red-600
WARM_AMBER = "#b45309"    # amber-700
CRIMSON_GLOW = "#be123c"

# Contrasting vegetation & cooling accents
CYAN = "#0284c7"          # sky-600 water cooling
LIME = "#16a34a"          # green-600 vegetation
VIOLET = "#7c3aed"        # violet-600 spatial boundary
ROSE = "#e11d48"          # rose-600 heat accent
BLUE = "#2563eb"          # blue-600
AMBER = "#d97706"

# Light gradient accents
ACCENT_GRADIENT = "linear-gradient(135deg, #ea580c 0%, #dc2626 100%)"
PRIMARY_GRADIENT = "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)"
HERO_GRADIENT = "linear-gradient(135deg, #eff6ff 0%, #f0fdf4 50%, #fff7ed 100%)"

# Categorical palette for light mode charts
CATEGORICAL = [
    "#ea580c",  # solar orange
    "#0284c7",  # sky blue
    "#16a34a",  # emerald green
    "#d97706",  # amber
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#e11d48",  # rose
]


def inject_theme() -> None:
    """Inject the global clean and clear light stylesheet."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* Force Root Light Mode Variables */
        :root, html, body, .stApp {{
            color-scheme: light !important;
            --background-color: #f8fafc !important;
            --secondary-background-color: #ffffff !important;
            --text-color: #0f172a !important;
            --primary-color: #2563eb !important;
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
            background-color: #f8fafc !important;
            color: #0f172a !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* Clean header styling */
        header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid #e2e8f0 !important;
        }}
        #MainMenu, footer {{ visibility: hidden; }}
        .block-container {{ padding: 1.2rem 2.0rem 3.5rem !important; max-width: 1480px !important; }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            letter-spacing: -0.02em;
            color: {TEXT} !important;
        }}
        h1 {{ font-weight: 700; }}
        h2 {{ font-weight: 650; margin-top: 0.3rem; }}
        p, span, label, li, td, th {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}
        a {{ color: {BLUE}; text-decoration: none; font-weight: 500; }}
        a:hover {{ color: {SOLAR_ORANGE}; text-decoration: underline; }}

        /* ---------- sidebar (Crisp White SaaS Style) ---------- */
        section[data-testid="stSidebar"], [data-testid="stSidebar"] {{
            background-color: #ffffff !important;
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
            box-shadow: 2px 0 8px rgba(15, 23, 42, 0.03);
        }}
        [data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}
        [data-testid="stSidebar"] .muted {{
            color: {MUTED} !important;
        }}

        /* ---------- neat & clear hero banner ---------- */
        .uhi-hero {{
            position: relative;
            border-radius: 16px;
            padding: 20px 24px;
            background: {HERO_GRADIENT};
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
            margin-bottom: 18px;
        }}
        .uhi-hero h1 {{
            margin: 0 0 6px;
            font-size: 1.85rem;
            color: #0f172a !important;
            background: none !important;
            -webkit-text-fill-color: #0f172a !important;
        }}
        .uhi-hero p {{
            margin: 0;
            color: #475569 !important;
            font-size: 0.96rem;
            max-width: 76ch;
            line-height: 1.5;
        }}
        .uhi-pill {{
            display: inline-block; margin-top: 10px; margin-right: 6px;
            padding: 4px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 650;
            letter-spacing: 0.03em; text-transform: uppercase;
            background: #f1f5f9; border: 1px solid #cbd5e1;
            color: #334155 !important;
        }}
        .uhi-pill.solar {{ background: #fff7ed; border-color: #fdba74; color: #c2410c !important; }}
        .uhi-pill.cyan {{ background: #f0f9ff; border-color: #7dd3fc; color: #0369a1 !important; }}
        .uhi-pill.lime {{ background: #f0fdf4; border-color: #86efac; color: #15803d !important; }}
        .uhi-pill.heat {{ background: #fef2f2; border-color: #fca5a5; color: #b91c1c !important; }}
        .uhi-pill.critical {{ background: #fee2e2; border-color: #ef4444; color: #991b1b !important; font-weight: 700; }}
        .uhi-pill.high {{ background: #ffedd5; border-color: #f97316; color: #9a3412 !important; font-weight: 700; }}
        .uhi-pill.moderate {{ background: #fef3c7; border-color: #f59e0b; color: #92400e !important; }}
        .uhi-pill.resilient {{ background: #dcfce7; border-color: #22c55e; color: #166534 !important; }}

        /* ---------- KPI cards (Clean White Elevation) ---------- */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin: 4px 0 18px; }}
        .kpi {{
            border-radius: 14px; padding: 16px 18px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}
        .kpi:hover {{
            transform: translateY(-2px);
            border-color: #cbd5e1;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        }}
        .kpi .k-label {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: #64748b !important; font-weight: 650;
        }}
        .kpi .k-value {{
            font-family: 'Space Grotesk', sans-serif; font-size: 1.70rem; font-weight: 700; margin-top: 4px;
            color: #0f172a !important;
            background: {ACCENT_GRADIENT}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .kpi .k-sub {{ font-size: 0.76rem; color: #64748b !important; margin-top: 2px; }}

        /* ---------- clean cards ---------- */
        .glass, .clean-card {{
            border-radius: 16px; padding: 20px 22px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }}
        .glass h3, .clean-card h3 {{ margin-top: 0; color: #0f172a !important; }}
        .section-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 650;
            margin: 8px 0 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #0f172a !important;
        }}

        /* ---------- controls, inputs, selects (100% Crisp Light SaaS Styling) ---------- */
        /* Selectbox container and wrapper */
        div[data-testid="stSelectbox"],
        div[data-testid="stSelectbox"] > div,
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div {{
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] * {{
            color: #0f172a !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: #0f172a !important;
            color: #0f172a !important;
        }}

        /* Dropdown popover menu */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        ul[data-baseweb="menu"],
        ul[data-baseweb="menu"] *,
        li[data-baseweb="menu-item"],
        div[role="listbox"] {{
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #0f172a !important;
        }}
        ul[data-baseweb="menu"] {{
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10) !important;
            border-radius: 8px !important;
        }}
        li[data-baseweb="menu-item"]:hover,
        li[data-baseweb="menu-item"][aria-selected="true"] {{
            background-color: #f1f5f9 !important;
            background: #f1f5f9 !important;
            color: #2563eb !important;
        }}

        /* Text and Number Inputs */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="input"],
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"],
        input[type="text"],
        input[type="number"] {{
            background-color: #ffffff !important;
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }}

        /* NumberInput stepper buttons (+ / -) */
        div[data-testid="stNumberInput"] button,
        button[data-testid*="stNumberInput"],
        div[data-baseweb="input"] button,
        div[data-baseweb="input"] ~ div button {{
            background-color: #f1f5f9 !important;
            background: #f1f5f9 !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
        }}
        div[data-testid="stNumberInput"] button:hover,
        button[data-testid*="stNumberInput"]:hover {{
            background-color: #e2e8f0 !important;
            background: #e2e8f0 !important;
            color: #0f172a !important;
        }}
        div[data-testid="stNumberInput"] button svg,
        button[data-testid*="stNumberInput"] svg {{
            fill: #0f172a !important;
            color: #0f172a !important;
        }}

        /* Sliders */
        .stSlider [data-baseweb="slider"] div[role="slider"] {{
            background: #2563eb !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4) !important;
        }}
        .stSlider [data-baseweb="slider"] * {{
            color: #0f172a !important;
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px !important;
            border: none !important;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            padding: 0.52rem 1.3rem !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
            transition: all 160ms ease !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.38) !important;
            color: #ffffff !important;
        }}

        /* Dataframes & Tables */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0 !important;
            background-color: #ffffff !important;
            background: #ffffff !important;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }}
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {{
            color: #0f172a !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid #e2e8f0;
            background: transparent !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 42px;
            border-radius: 8px 8px 0 0;
            padding: 0 18px;
            color: #64748b !important;
            font-weight: 600;
            background: transparent !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: #ffffff !important;
            color: #2563eb !important;
            border: 1px solid #e2e8f0;
            border-bottom: 2px solid #2563eb !important;
            box-shadow: 0 -2px 6px rgba(15, 23, 42, 0.02);
        }}

        /* Radio buttons in sidebar */
        [data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: #ffffff !important;
            border: 1px solid #f1f5f9;
            padding: 6px 12px;
            border-radius: 8px;
            margin-bottom: 4px;
            transition: all 120ms ease;
        }}
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: #f8fafc !important;
            border-color: #cbd5e1;
        }}
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
            background: #eff6ff !important;
            border-color: #93c5fd !important;
        }}

        /* Expanders */
        div[data-testid="stExpander"] {{
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
            margin-bottom: 14px;
        }}
        div[data-testid="stExpander"] summary {{
            color: #0f172a !important;
            font-weight: 600;
        }}

        /* Misc */
        .legend-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px; }}
        .legend-dot {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.80rem; color: #475569 !important; font-weight: 500; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .muted, .muted * {{ color: {MUTED} !important; font-size: 0.85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    """Render the clean and clear hero header."""
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
    """Render a responsive row of clean white KPI cards: (label, value, sub)."""
    cards = "".join(
        f'<div class="kpi"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>'
        f'<div class="k-sub">{sub}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def clean_card_open() -> None:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)


def clean_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    """Small colour-key row: (colour, label)."""
    html = "".join(
        f'<span class="legend-dot"><span class="dot" style="background:{c}; border: 1px solid rgba(0,0,0,0.1)"></span>{label}</span>'
        for c, label in items
    )
    st.markdown(f'<div class="legend-row">{html}</div>', unsafe_allow_html=True)


def guide_box(title: str, items: list[tuple[str, str]], icon: str = "💡") -> None:
    """Render a simple, clean concept explanation box in light mode."""
    body = "".join(
        f'<div style="margin-bottom:6px;"><strong style="color:#0f172a;">{h}:</strong> '
        f'<span style="color:#334155;">{desc}</span></div>'
        for h, desc in items
    )
    st.markdown(
        f"""
        <div class="clean-card" style="border-left: 4px solid #2563eb; padding: 14px 18px; margin-bottom: 16px;">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span style="font-size:1.15rem;">{icon}</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:0.98rem; font-weight:700; color:#0f172a;">{title}</span>
          </div>
          <div style="font-size:0.88rem; line-height:1.5;">
            {body}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_card(img_path: str | Path, title: str, description: str, badge: str = "MAP") -> None:
    """Render an image inside a clean white card with captions and download."""
    p = Path(img_path)
    if not p.exists():
        return
    st.markdown(
        f"""
        <div class="clean-card" style="padding:14px 16px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.02rem; font-weight:700; color:#0f172a;">{title}</span>
                <span class="uhi-pill solar" style="margin:0; font-size:0.68rem;">{badge}</span>
            </div>
            <p style="color:#475569 !important; font-size:0.84rem; margin-bottom:10px; line-height:1.4;">{description}</p>
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
