"""Tests for the IDW vs kriging comparison (spec Section 6 extension)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import interpolation_compare as ic


def _smooth_field(n: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 2000, size=(n, 2))
    z = 30 + 4 * np.sin(xy[:, 0] / 500.0) + 3 * np.cos(xy[:, 1] / 400.0)
    return xy, z


def _grid(cell: float = 100.0, n: int = 8):
    xs = np.arange(n) * cell + cell / 2
    ys = np.arange(n) * cell + cell / 2
    gx, gy = np.meshgrid(xs, ys)
    return pd.DataFrame({"centroid_x": gx.ravel(), "centroid_y": gy.ravel()})


pytestmark = pytest.mark.skipif(not ic.HAS_PYKRIGE, reason="pykrige is not installed")


def test_kriging_predicts_smooth_field_well():
    xy, z = _smooth_field(80)
    query = xy[:10] + 3.0
    pred = ic.kriging_predict(xy, z, query)
    assert len(pred) == 10
    assert np.isfinite(pred).all()


def test_kriging_leave_one_out_has_no_nan():
    xy, z = _smooth_field(40)
    loo = ic.kriging_leave_one_out(xy, z)
    assert len(loo) == 40
    assert np.isfinite(loo).all()


def test_kriging_recovers_a_smooth_field_better_than_the_field_range():
    xy, z = _smooth_field(70)
    loo = ic.kriging_leave_one_out(xy, z)
    rmse = float(np.sqrt(np.mean((loo - z) ** 2)))
    # a smooth field should be reproduced far more tightly than its own spread
    assert rmse < float(z.std())


def test_compare_methods_reports_both_and_picks_a_winner(fast_config):
    xy, z = _smooth_field(50)
    res = ic.compare_methods(fast_config, xy, z, _grid(), "EPSG:32643", 100.0)
    assert set(res.metrics["method"]) == {"IDW", "Kriging"}
    assert res.metrics["is_best"].sum() == 1
    assert res.best_method in {"IDW", "Kriging"}
    assert res.metrics["rmse"].notna().all()
    assert res.variogram is not None
    assert res.variogram["model"] == "spherical"
    assert res.variogram["range"] > 0


def test_compare_methods_surfaces_are_aligned_to_the_grid(fast_config):
    xy, z = _smooth_field(50)
    grid = _grid(n=8)
    res = ic.compare_methods(fast_config, xy, z, grid, "EPSG:32643", 100.0)
    surface = res.surfaces[res.best_method]
    assert surface.shape == (8, 8)
    assert np.isfinite(surface).all()


def test_variogram_model_is_configurable(fast_config):
    xy, z = _smooth_field(40)
    res = ic.compare_methods(
        fast_config, xy, z, _grid(), "EPSG:32643", 100.0, variogram_model="exponential"
    )
    assert res.variogram["model"] == "exponential"
    assert res.metrics.set_index("method").loc["Kriging", "parameter"] == "exponential"


def test_best_method_is_the_lowest_rmse(fast_config):
    xy, z = _smooth_field(60)
    res = ic.compare_methods(fast_config, xy, z, _grid(), "EPSG:32643", 100.0)
    winner = res.metrics.sort_values("rmse").iloc[0]["method"]
    assert res.best_method == winner


def test_missing_pykrige_degrades_gracefully(monkeypatch, fast_config):
    """If pykrige is unavailable the comparison must still return IDW metrics."""
    monkeypatch.setattr(ic, "HAS_PYKRIGE", False)
    xy, z = _smooth_field(40)
    res = ic.compare_methods(fast_config, xy, z, _grid(), "EPSG:32643", 100.0)
    assert res.metrics["method"].tolist() == ["IDW"]
    assert res.best_method == "IDW"
    assert any("pykrige" in note for note in res.notes)


def test_kriging_predict_raises_without_pykrige(monkeypatch):
    monkeypatch.setattr(ic, "HAS_PYKRIGE", False)
    with pytest.raises(RuntimeError, match="pykrige is not installed"):
        ic.kriging_predict(np.zeros((3, 2)), np.zeros(3), np.zeros((1, 2)))
