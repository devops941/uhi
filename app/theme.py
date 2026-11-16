"""Shared 2026-trend visual theme for the Streamlit dashboard.

Design language: deep "midnight" dark base with a soft aurora gradient,
glassmorphic cards, a single vivid accent ramp (violet -> cyan), generous
radii, tight display typography and quiet motion on interaction. All styling
lives here so every page looks like one product instead of seven.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------- palette
INK = "#05070f"
SURFACE = "rgba(19, 24, 40, 0.66)"
SURFACE_SOLID = "#131828"
BORDER = "rgba(148, 163, 184, 0.16)"
TEXT = "#e8ecf7"
MUTED = "#9aa6c2"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
LIME = "#a3e635"
AMBER = "#fbbf24"
ROSE = "#fb7185"
BLUE = "#60a5fa"

ACCENT_GRADIENT = f"linear-gradient(120deg, {VIOLET} 0%, {CYAN} 100%)"

# a compact, colour-blind-safe ramp for the choropleth figures
CATEGORICAL = [VIOLET, CYAN, LIME, AMBER, ROSE, BLUE, "#f472b6", "#34d399"]


def inject_theme() -> None:
    """Inject the global stylesheet once per session."""
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
            --violet: {VIOLET};
            --cyan: {CYAN};
            --lime: {LIME};
            --amber: {AMBER};
            --rose: {ROSE};
        }}

        .stApp {{
            background:
                radial-gradient(1100px 600px at 8% -8%, rgba(139,92,246,0.20), transparent 60%),
                radial-gradient(900px 620px at 100% 0%, rgba(34,211,238,0.16), transparent 55%),
                radial-gradient(900px 700px at 50% 120%, rgba(163,230,53,0.10), transparent 60%),
                var(--ink);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* hide the default Streamlit chrome for a clean product feel */
        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}
        .block-container {{ padding: 1.4rem 2rem 4rem; max-width: 1500px; }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            letter-spacing: -0.02em;
            color: var(--text);
        }}
        h1 {{ font-weight: 700; }}
        h2 {{ font-weight: 650; margin-top: 0.4rem; }}
        p, span, label, li, td, th {{ font-family: 'Inter', sans-serif; }}
        a {{ color: var(--cyan); }}

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(19,24,40,0.96), rgba(9,12,22,0.98));
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] * {{ color: var(--text); }}

        /* ---------- hero ---------- */
        .uhi-hero {{
            position: relative;
            border-radius: 26px;
            padding: 26px 30px;
            background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.14) 55%, rgba(163,230,53,0.10));
            border: 1px solid var(--border);
            overflow: hidden;
            margin-bottom: 18px;
        }}
        .uhi-hero::after {{
            content: "";
            position: absolute; inset: -40% -10% auto auto; width: 420px; height: 420px;
            background: radial-gradient(circle, rgba(34,211,238,0.28), transparent 62%);
            filter: blur(6px);
            pointer-events: none;
        }}
        .uhi-hero h1 {{ margin: 0 0 6px; font-size: 2.05rem; }}
        .uhi-hero p {{ margin: 0; color: var(--muted); font-size: 0.98rem; max-width: 62ch; }}
        .uhi-pill {{
            display: inline-block; margin-top: 14px; margin-right: 8px;
            padding: 5px 13px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
            letter-spacing: 0.03em; text-transform: uppercase;
            background: rgba(139,92,246,0.16); border: 1px solid rgba(139,92,246,0.42);
            color: #d9c9ff;
        }}
        .uhi-pill.cyan {{ background: rgba(34,211,238,0.14); border-color: rgba(34,211,238,0.42); color: #bdf3fb; }}
        .uhi-pill.lime {{ background: rgba(163,230,53,0.14); border-color: rgba(163,230,53,0.42); color: #e2f8b4; }}

        /* ---------- KPI cards ---------- */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 14px; margin: 6px 0 20px; }}
        .kpi {{
            border-radius: 20px; padding: 16px 18px;
            background: var(--surface);
            border: 1px solid var(--border);
            backdrop-filter: blur(14px);
            transition: transform 180ms cubic-bezier(.2,.7,.3,1), border-color 180ms;
        }}
        .kpi:hover {{ transform: translateY(-3px); border-color: rgba(139,92,246,0.5); }}
        .kpi .k-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--muted); font-weight: 600; }}
        .kpi .k-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.72rem; font-weight: 700; margin-top: 6px;
                         background: {ACCENT_GRADIENT}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .kpi .k-sub {{ font-size: 0.76rem; color: var(--muted); margin-top: 3px; }}

        /* ---------- glass cards ---------- */
        .glass {{
            border-radius: 22px; padding: 20px 22px; background: var(--surface);
            border: 1px solid var(--border); backdrop-filter: blur(14px); margin-bottom: 16px;
        }}
        .glass h3 {{ margin-top: 0; }}
        .section-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.16rem; font-weight: 650; margin: 4px 0 10px; }}

        /* ---------- metrics / tabs ---------- */
        div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; }}
        div[data-testid="stMetric"] {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 18px; padding: 12px 16px; backdrop-filter: blur(12px);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px; border-radius: 12px 12px 0 0; padding: 0 16px;
            color: var(--muted); font-weight: 550;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, rgba(139,92,246,0.20), transparent);
            color: var(--text) !important;
            border-bottom: 2px solid var(--violet);
        }}

        /* ---------- controls ---------- */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 14px; border: 1px solid rgba(139,92,246,0.45);
            background: linear-gradient(120deg, rgba(139,92,246,0.30), rgba(34,211,238,0.24));
            color: var(--text); font-weight: 600; padding: 0.5rem 1.1rem;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px); box-shadow: 0 10px 30px rgba(139,92,246,0.32);
            border-color: rgba(34,211,238,0.6);
        }}
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
            background: rgba(19,24,40,0.8) !important; border-radius: 12px !important;
            border-color: var(--border) !important; color: var(--text) !important;
        }}
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--cyan); }}

        /* ---------- dataframes ---------- */
        div[data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; border: 1px solid var(--border); }}

        /* ---------- misc ---------- */
        .legend-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }}
        .legend-dot {{ display: inline-flex; align-items: center; gap: 7px; font-size: 0.8rem; color: var(--muted); }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .muted {{ color: var(--muted); font-size: 0.85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    """Render the gradient hero header."""
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
        f'<span class="legend-dot"><span class="dot" style="background:{c}"></span>{label}</span>'
        for c, label in items
    )
    st.markdown(f'<div class="legend-row">{html}</div>', unsafe_allow_html=True)
