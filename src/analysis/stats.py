"""Phase 4 - statistical analysis.

* Pearson and Spearman correlation of temperature against each covariate
* OLS regression ``temp_c ~ ndvi + builtup_frac + dist_water_m`` with R^2,
  adjusted R^2, confidence intervals, VIF and Moran's I of the residuals
* Built-up vs green group comparison (Welch's t-test, Mann-Whitney U, Cohen's d)

The regression uses standardised predictors for the effect-size table and the
raw-scale coefficients for the cooling estimate, so both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

if TYPE_CHECKING:
    import geopandas as gpd

PREDICTORS = ["ndvi_mean", "builtup_frac", "dist_water_m"]


@dataclass
class StatsResult:
    correlations: pd.DataFrame
    regression: pd.DataFrame
    model_summary: dict
    group_comparison: pd.DataFrame
    regression_model: object | None = None
    fitted: pd.DataFrame | None = None
    standardised: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------
def correlate(df: pd.DataFrame, target: str = "temp_c", predictors=None) -> pd.DataFrame:
    """Pearson + Spearman correlation of ``target`` against each predictor."""
    predictors = predictors or PREDICTORS
    rows = []
    for col in predictors:
        if col not in df.columns:
            continue
        sub = df[[target, col]].dropna()
        n = len(sub)
        if n < 3 or sub[col].nunique() < 2:
            rows.append(
                {
                    "variable": col,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "n": n,
                }
            )
            continue
        pr, pp = stats.pearsonr(sub[target], sub[col])
        sr, sp = stats.spearmanr(sub[target], sub[col])
        rows.append(
            {
                "variable": col,
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
                "n": int(n),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# regression
# ----------------------------------------------------------------------
def _moran_i_residuals(residuals: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """Moran's I of the OLS residuals using k-nearest-neighbour weights."""
    from esda.moran import Moran
    from libpysal.weights import KNN

    if len(residuals) < k + 2:
        return float("nan")
    try:
        w = KNN(coords, k=k)
        w.transform = "r"
        mi = Moran(residuals, w, permutations=99)
        return float(mi.I)
    except Exception:  # pragma: no cover - defensive
        return float("nan")


def regress(
    df: pd.DataFrame,
    target: str = "temp_c",
    predictors=None,
    coords: np.ndarray | None = None,
    moran_k: int = 8,
) -> tuple[pd.DataFrame, dict, object, pd.DataFrame]:
    """OLS with VIF, standardised effect sizes and Moran's I of the residuals.

    Returns ``(coefficient_table, model_summary, fitted_model, standardised_table)``.
    """
    predictors = predictors or PREDICTORS
    predictors = [p for p in predictors if p in df.columns]

    work = df[[target] + predictors].dropna().copy()
    if len(work) <= len(predictors) + 1:
        raise ValueError("not enough complete rows for regression")

    y = work[target].to_numpy(dtype=float)
    X_raw = work[predictors].to_numpy(dtype=float)
    X = sm.add_constant(X_raw, has_constant="add")

    model = sm.OLS(y, X).fit()

    # --- VIF (computed without the intercept) ----------------------------
    vif_values = {}
    if len(predictors) > 1:
        for i, name in enumerate(predictors):
            try:
                vif_values[name] = float(variance_inflation_factor(X_raw, i))
            except Exception:  # pragma: no cover
                vif_values[name] = float("nan")
    else:
        vif_values[predictors[0]] = 1.0

    conf = model.conf_int()
    rows = []
    names = ["const"] + predictors
    for i, name in enumerate(names):
        rows.append(
            {
                "term": name,
                "coef": float(model.params[i]),
                "std_err": float(model.bse[i]),
                "p_value": float(model.pvalues[i]),
                "ci_low": float(conf[i][0]),
                "ci_high": float(conf[i][1]),
                "vif": vif_values.get(name, np.nan),
            }
        )
    coef_table = pd.DataFrame(rows)

    # --- standardised predictors (effect sizes) --------------------------
    std = work[predictors].std(ddof=0).replace(0, np.nan)
    X_std = (work[predictors] - work[predictors].mean()) / std
    X_std = sm.add_constant(X_std.to_numpy(dtype=float), has_constant="add")
    model_std = sm.OLS(y, X_std).fit()
    std_rows = []
    for i, name in enumerate(predictors):
        std_rows.append(
            {
                "term": name,
                "std_coef": float(model_std.params[i + 1]),
                "p_value": float(model_std.pvalues[i + 1]),
            }
        )
    std_table = pd.DataFrame(std_rows)

    residuals = np.asarray(model.resid, dtype=float)
    moran_i = (
        _moran_i_residuals(residuals, np.asarray(coords), k=moran_k)
        if coords is not None
        else float("nan")
    )

    summary = {
        "n_obs": int(len(work)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_stat": float(model.fvalue) if model.fvalue is not None else np.nan,
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "aic": float(model.aic),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "moran_i_residuals": float(moran_i),
        "max_vif": float(max(vif_values.values())) if vif_values else np.nan,
    }

    fitted = work.copy()
    fitted["fitted"] = model.fittedvalues
    fitted["residual"] = residuals
    return coef_table, summary, model, fitted, std_table


# ----------------------------------------------------------------------
# group comparison
# ----------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def compare_groups(
    df: pd.DataFrame,
    temp_col: str = "temp_idw",
    builtup_col: str = "builtup_frac",
    ndvi_col: str = "ndvi_mean",
    cfg=None,
) -> pd.DataFrame:
    """Built-up cells vs green cells: Welch t-test, Mann-Whitney U and Cohen's d."""
    builtup_thr = 0.5
    ndvi_thr = 0.4
    if cfg is not None:
        builtup_thr = float(cfg.get("groups.builtup_threshold", 0.5))
        ndvi_thr = float(cfg.get("groups.ndvi_threshold", 0.4))

    built = df.loc[df[builtup_col] > builtup_thr, temp_col].dropna()
    green = df.loc[df[ndvi_col] > ndvi_thr, temp_col].dropna()

    if len(built) < 2 or len(green) < 2:
        return pd.DataFrame(
            [
                {
                    "group_a": "built_up",
                    "group_b": "green",
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "diff": np.nan,
                    "welch_p": np.nan,
                    "mannwhitney_p": np.nan,
                    "cohens_d": np.nan,
                }
            ]
        )

    welch_t, welch_p = stats.ttest_ind(built, green, equal_var=False)
    mw = stats.mannwhitneyu(built, green, alternative="two-sided")
    return pd.DataFrame(
        [
            {
                "group_a": "built_up",
                "group_b": "green",
                "mean_a": float(built.mean()),
                "mean_b": float(green.mean()),
                "diff": float(built.mean() - green.mean()),
                "welch_p": float(welch_p),
                "mannwhitney_p": float(mw.pvalue),
                "cohens_d": cohens_d(built, green),
            }
        ]
    )


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def run_stats(
    cfg, samples: gpd.GeoDataFrame, features: pd.DataFrame, moran_k: int = 8
) -> StatsResult:
    """Join samples with their features and run every Phase-4 test."""
    df = samples.drop(columns=["geometry"]).merge(features, on="sample_id", how="left")
    df = df[df["is_outlier"] == 0].copy()

    corr = correlate(df)

    coords = np.column_stack([samples.geometry.x, samples.geometry.y])
    # align coords with the filtered frame
    keep_idx = samples["sample_id"].isin(df["sample_id"]).to_numpy()
    coords = coords[keep_idx]

    coef_table, summary, model, fitted, std_table = regress(df, coords=coords, moran_k=moran_k)

    # cooling estimate: dT/dNDVI from the raw-scale coefficient
    ndvi_coef = float(coef_table.loc[coef_table["term"] == "ndvi_mean", "coef"].iloc[0])
    uplift = float(cfg.get("priority.ndvi_uplift", 0.1))
    summary["ndvi_coef"] = ndvi_coef
    summary["cooling_estimate_c"] = abs(ndvi_coef) * uplift
    summary["ndvi_uplift"] = uplift

    return StatsResult(
        correlations=corr,
        regression=coef_table,
        model_summary=summary,
        group_comparison=compare_groups(df, temp_col="temp_c", cfg=cfg),
        regression_model=model,
        fitted=fitted,
        standardised=std_table,
        extra={"joined": df, "ndvi_coef": ndvi_coef},
    )
