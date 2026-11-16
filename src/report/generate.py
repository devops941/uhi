"""Auto-generated results report (Markdown, optional PDF via weasyprint)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parent
TEMPLATE_NAME = "template.md.j2"


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df is None or len(df) == 0:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return out.to_markdown(index=False)


def build_context(cfg, results: dict) -> dict:
    """Turn the pipeline results dict into the template context."""
    summary = results.get("model_summary", {}) or {}
    corr = results.get("correlations")
    reg = results.get("regression")
    std = results.get("standardised")
    groups = results.get("group_comparison")
    cv = results.get("cv_table")
    wards = results.get("ward_summary")
    top_cells = results.get("top_cells")
    uhi_stats = results.get("uhi_stats", {}) or {}

    thresholds = cfg.get("uhi.thresholds_c", [2.0, 4.0])
    uhi_rows = []
    for t in thresholds:
        uhi_rows.append(
            {
                "threshold": f"{t:g}",
                "area_km2": f"{uhi_stats.get(f'area_above_{t:g}c_km2', float('nan')):.2f}",
                "pct": f"{uhi_stats.get(f'pct_above_{t:g}c', float('nan')):.1f}",
            }
        )

    def fmt(value, spec="{:.3f}"):
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return "n/a"
            return spec.format(value)
        except (TypeError, ValueError):
            return str(value)

    # Place-based vulnerability tier classification table
    place_table_md = "_No place-based data._"
    if wards is not None and not wards.empty:
        pw = wards.copy()

        def assign_tier(row):
            rank = row.get("priority_rank", 99)
            hot = row.get("hotspot_area_pct", 0)
            uhi = row.get("mean_uhi", 0)
            if rank <= 3 or hot >= 60:
                return "Tier 1: Critical Emergency"
            elif rank <= 8 or hot >= 25 or uhi >= 2.5:
                return "Tier 2: High Vulnerability"
            elif hot >= 5 or uhi >= 1.0:
                return "Tier 3: Moderate Risk"
            else:
                return "Tier 4: Resilient / Refuge"

        pw["vulnerability_tier"] = pw.apply(assign_tier, axis=1)
        pw["target_cooling_p10"] = "-0.31 °C"
        display_cols = [
            "priority_rank",
            "ward_id",
            "mean_temp",
            "mean_uhi",
            "hotspot_area_pct",
            "vulnerability_tier",
            "target_cooling_p10",
        ]
        display_cols = [c for c in display_cols if c in pw.columns]
        place_table_md = _df_to_md(pw[display_cols])

    ctx = {
        "project": {
            "name": cfg.get("project.name"),
            "city": cfg.get("project.city"),
            "country": cfg.get("project.country"),
        },
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": results.get("source", "synthetic"),
        "crs_metric": results.get("crs_metric", ""),
        "n_samples": results.get("n_samples", 0),
        "n_grid_cells": results.get("n_grid_cells", 0),
        "grid_cell_m": results.get("grid_cell_m", 100),
        "mean_temp": fmt(results.get("mean_temp"), "{:.2f}"),
        "rural_baseline": fmt(uhi_stats.get("rural_baseline_c"), "{:.2f}"),
        "max_uhi": fmt(uhi_stats.get("max_uhi_c"), "{:.2f}"),
        "mean_uhi": fmt(uhi_stats.get("mean_uhi_c"), "{:.2f}"),
        "n_hotspots": results.get("n_hotspots", 0),
        "n_coldspots": results.get("n_coldspots", 0),
        "hotspot_pct": fmt(results.get("hotspot_pct"), "{:.1f}"),
        "idw_power": results.get("selected_power", ""),
        "idw_rmse": fmt(results.get("idw_rmse"), "{:.3f}"),
        "r2": fmt(summary.get("r2"), "{:.3f}"),
        "adj_r2": fmt(summary.get("adj_r2"), "{:.3f}"),
        "max_vif": fmt(summary.get("max_vif"), "{:.2f}"),
        "moran_i": fmt(summary.get("moran_i_residuals"), "{:.3f}"),
        "ndvi_uplift": fmt(summary.get("ndvi_uplift"), "{:g}"),
        "cooling_estimate": fmt(summary.get("cooling_estimate_c"), "{:.3f}"),
        "correlation_table": _df_to_md(corr),
        "regression_table": _df_to_md(reg),
        "standardised_table": _df_to_md(std),
        "group_table": _df_to_md(groups),
        "cv_table": _df_to_md(cv),
        "interpolation_comparison_table": _df_to_md(results.get("interpolation_comparison")),
        "best_interpolation": results.get("best_interpolation", "n/a"),
        "variogram_model": (results.get("variogram") or {}).get("model", "n/a"),
        "variogram_nugget": fmt((results.get("variogram") or {}).get("nugget"), "{:.3f}"),
        "variogram_psill": fmt((results.get("variogram") or {}).get("psill"), "{:.3f}"),
        "variogram_range": fmt((results.get("variogram") or {}).get("range"), "{:.1f}"),
        "variogram_note": (
            "Kriging weights neighbours by the fitted variogram, so it can beat IDW when "
            "the surface is smooth; IDW stays competitive and needs no variogram fitting."
        ),
        "ward_table": _df_to_md(wards),
        "place_based_dossier_table": place_table_md,
        "top_cells_table": _df_to_md(top_cells),
        "uhi_threshold_rows": uhi_rows,
        "priority_weights": results.get("priority_weights", {}),
        "figures": results.get("figures", []),
    }
    return ctx


def render_markdown(cfg, results: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)
    return template.render(**build_context(cfg, results))


def write_report(cfg, results: dict, out_path=None, as_pdf: bool = False) -> Path:
    """Render the report to Markdown (and optionally PDF)."""
    out_dir = cfg.path_for("outputs", create=True)
    path = Path(out_path) if out_path else out_dir / "report.md"
    markdown = render_markdown(cfg, results)
    path.write_text(markdown, encoding="utf-8")

    if as_pdf:
        try:
            import markdown as md

            html = md.markdown(markdown, extensions=["tables", "fenced_code"])
            from weasyprint import HTML

            HTML(string=html).write_pdf(path.with_suffix(".pdf"))
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[report] PDF skipped ({exc.__class__.__name__}: {exc})")
    return path
