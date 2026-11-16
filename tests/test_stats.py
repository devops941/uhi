"""Tests for the statistics module, including synthetic-truth recovery."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import stats as st


def _toy_frame(n: int = 60, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ndvi = rng.uniform(0, 1, n)
    built = rng.uniform(0, 1, n)
    water = rng.uniform(0, 2000, n)
    temp = 30 + 7 * built - 5 * ndvi - 0.002 * water + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"temp_c": temp, "ndvi_mean": ndvi, "builtup_frac": built, "dist_water_m": water}
    )


# ----------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------
def test_correlation_recovers_known_signs():
    df = _toy_frame()
    corr = st.correlate(df).set_index("variable")
    assert corr.loc["builtup_frac", "pearson_r"] > 0
    assert corr.loc["ndvi_mean", "pearson_r"] < 0
    assert corr.loc["dist_water_m", "pearson_r"] < 0
    assert (corr["n"] == 60).all()


def test_correlation_reports_nan_for_constant_column():
    df = _toy_frame()
    df["ndvi_mean"] = 0.4
    corr = st.correlate(df).set_index("variable")
    assert np.isnan(corr.loc["ndvi_mean", "pearson_r"])


# ----------------------------------------------------------------------
# regression
# ----------------------------------------------------------------------
def test_regression_recovers_built_in_coefficients():
    df = _toy_frame(400, seed=11)
    coef, summary, model, fitted, std = st.regress(df)
    by_term = coef.set_index("term")
    assert by_term.loc["builtup_frac", "coef"] == pytest.approx(7.0, abs=0.5)
    assert by_term.loc["ndvi_mean", "coef"] == pytest.approx(-5.0, abs=0.5)
    assert summary["r2"] > 0.95
    assert summary["adj_r2"] <= summary["r2"] + 1e-9


def test_regression_vif_is_one_for_single_predictor():
    df = _toy_frame()
    coef, summary, model, fitted, std = st.regress(df, predictors=["builtup_frac"])
    assert coef.set_index("term").loc["builtup_frac", "vif"] == pytest.approx(1.0)
    assert np.isnan(summary["moran_i_residuals"])


def test_regression_confidence_interval_brackets_coef():
    df = _toy_frame()
    coef, *_ = st.regress(df)
    for _, row in coef.iterrows():
        if row["term"] == "const":
            continue
        assert row["ci_low"] <= row["coef"] <= row["ci_high"]


def test_regression_computes_moran_i_with_coords():
    df = _toy_frame(120)
    coords = np.column_stack([np.arange(120) * 10.0, np.zeros(120)])
    coef, summary, *_ = st.regress(df, coords=coords)
    assert np.isfinite(summary["moran_i_residuals"])


def test_regression_rejects_too_few_rows():
    df = _toy_frame(3)
    with pytest.raises(ValueError, match="not enough complete rows"):
        st.regress(df)


def test_standardised_coefficients_are_bounded_and_signed():
    df = _toy_frame(300)
    coef, summary, model, fitted, std = st.regress(df)
    by_term = std.set_index("term")
    assert by_term.loc["builtup_frac", "std_coef"] > 0
    assert by_term.loc["ndvi_mean", "std_coef"] < 0


# ----------------------------------------------------------------------
# group comparison
# ----------------------------------------------------------------------
def test_cohens_d_positive_when_first_group_hotter():
    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert st.cohens_d(a, b) > 0


def test_cohens_d_nan_for_tiny_groups():
    assert np.isnan(st.cohens_d(np.array([1.0]), np.array([2.0, 3.0])))


def test_compare_groups_detects_a_difference():
    grid = pd.DataFrame(
        {
            "temp_idw": [40, 41, 39, 40, 25, 24, 26, 25],
            "builtup_frac": [0.9, 0.8, 0.7, 0.85, 0.1, 0.05, 0.2, 0.1],
            "ndvi_mean": [0.1, 0.15, 0.2, 0.1, 0.7, 0.8, 0.6, 0.75],
        }
    )
    out = st.compare_groups(grid).iloc[0]
    assert out["diff"] > 0
    assert out["welch_p"] < 0.05
    assert out["cohens_d"] > 0


def test_compare_groups_handles_empty_groups():
    grid = pd.DataFrame(
        {"temp_idw": [30.0, 31.0], "builtup_frac": [0.0, 0.0], "ndvi_mean": [0.0, 0.0]}
    )
    out = st.compare_groups(grid).iloc[0]
    assert np.isnan(out["diff"])


# ----------------------------------------------------------------------
# synthetic truth recovery (end-to-end on the generated city)
# ----------------------------------------------------------------------
def test_regression_recovers_synthetic_truth(prepared, synthetic_inputs, fast_config):
    from src.features import extract

    feats = extract.extract_sample_features(
        fast_config,
        prepared.samples,
        prepared.buildings,
        synthetic_inputs.landcover_polygons,
        synthetic_inputs.ndvi_path,
        synthetic_inputs.landcover_path,
        prepared.crs,
    )
    res = st.run_stats(fast_config, prepared.samples, feats)
    by_term = res.regression.set_index("term")

    truth = json.loads((fast_config.path_for("data_raw") / "synthetic_truth.json").read_text())
    # the built-in coefficient signs must be recovered
    assert by_term.loc["builtup_frac", "coef"] > 0
    assert by_term.loc["ndvi_mean", "coef"] < 0
    # and the magnitudes must be in the right ballpark (generous, noisy data)
    assert by_term.loc["builtup_frac", "coef"] == pytest.approx(
        truth["coef_builtup"], abs=0.6 * abs(truth["coef_builtup"])
    )
    assert by_term.loc["ndvi_mean", "coef"] == pytest.approx(
        truth["coef_ndvi"], abs=0.6 * abs(truth["coef_ndvi"])
    )
    assert res.model_summary["r2"] > 0.5
    assert np.isfinite(res.model_summary["moran_i_residuals"])
    assert res.model_summary["cooling_estimate_c"] > 0
    assert len(res.group_comparison) == 1
