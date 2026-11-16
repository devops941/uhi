"""Tests for IDW interpolation (spec Section 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import idw


# ----------------------------------------------------------------------
# exactness and bounds
# ----------------------------------------------------------------------
def test_idw_is_exact_at_sample_points():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    z = np.array([10.0, 20.0, 30.0, 40.0])
    pred = idw.idw_predict(xy, z, xy, power=2.0, k=4)
    np.testing.assert_allclose(pred, z, atol=1e-9)


def test_idw_stays_within_neighbour_range():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, size=(40, 2))
    z = rng.uniform(20, 40, size=40)
    query = rng.uniform(0, 1000, size=(30, 2))
    pred = idw.idw_predict(xy, z, query, power=2.0, k=6)
    assert pred.min() >= z.min() - 1e-9
    assert pred.max() <= z.max() + 1e-9


def test_higher_power_gives_more_local_surface():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    z = np.array([0.0, 100.0])
    query = np.array([[1.0, 0.0]])
    low = idw.idw_predict(xy, z, query, power=1.0, k=2)[0]
    high = idw.idw_predict(xy, z, query, power=3.0, k=2)[0]
    # the near sample (z=0) dominates more strongly as power rises
    assert high < low


def test_idw_single_training_point_returns_that_value():
    xy = np.array([[5.0, 5.0]])
    z = np.array([33.0])
    pred = idw.idw_predict(xy, z, np.array([[100.0, 100.0]]), power=2.0, k=1)
    assert pred[0] == pytest.approx(33.0)


def test_idw_rejects_empty_training_set():
    with pytest.raises(ValueError, match="at least one training point"):
        idw.idw_predict(np.empty((0, 2)), np.empty(0), np.array([[0.0, 0.0]]))


def test_idw_k_larger_than_n_is_clamped():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    z = np.array([1.0, 2.0])
    pred = idw.idw_predict(xy, z, np.array([[5.0, 0.0]]), power=2.0, k=50)
    assert 1.0 <= pred[0] <= 2.0


# ----------------------------------------------------------------------
# leave-one-out
# ----------------------------------------------------------------------
def test_leave_one_out_never_uses_the_point_itself():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    z = np.array([0.0, 100.0])
    loo = idw.idw_leave_one_out(xy, z, power=2.0, k=1)
    # each point is predicted from the other, so neither prediction is exact
    np.testing.assert_allclose(loo, [100.0, 0.0])


def test_tune_power_selects_lowest_rmse():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 500, size=(60, 2))
    z = 30 + 5 * np.sin(xy[:, 0] / 120.0)
    cv = idw.tune_power(xy, z, powers=[1.0, 2.0, 3.0], k=8)
    assert cv["is_selected"].sum() == 1
    selected = cv.loc[cv["is_selected"] == 1, "power"].iloc[0]
    assert selected == cv.loc[cv["rmse"].idxmin(), "power"]


def test_tune_power_reports_all_candidate_powers():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(20, 2))
    z = rng.uniform(20, 30, size=20)
    powers = [1.0, 1.5, 2.0, 2.5, 3.0]
    cv = idw.tune_power(xy, z, powers=powers, k=5)
    assert cv["power"].tolist() == powers
    assert cv[["rmse", "mae", "r2"]].notna().all().all()


def test_metrics_are_perfect_for_identical_arrays():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    m = idw._metrics(z, z)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# surface + raster alignment
# ----------------------------------------------------------------------
def test_grid_to_array_places_values_at_correct_index():
    grid = pd.DataFrame(
        {
            "centroid_x": [50.0, 150.0, 50.0, 150.0],
            "centroid_y": [950.0, 950.0, 850.0, 850.0],
        }
    )
    values = np.array([1.0, 2.0, 3.0, 4.0])
    transform, array = idw.grid_to_array(grid, values, cell_m=100.0)
    # row 0 is the northernmost (y=950), row 1 the southern (y=850)
    assert array[0, 0] == 1.0
    assert array[0, 1] == 2.0
    assert array[1, 0] == 3.0
    assert array[1, 1] == 4.0
    # transform origin is the north-west corner of the north-west cell
    assert transform.c == pytest.approx(0.0)
    assert transform.f == pytest.approx(1000.0)


def test_interpolate_to_grid_aligns_and_writes(prepared, fast_config, tmp_path):
    xy = np.column_stack([prepared.samples.geometry.x, prepared.samples.geometry.y])
    z = prepared.samples["temp_c"].to_numpy(dtype=float)
    res = idw.interpolate_to_grid(
        fast_config, xy, z, prepared.grid, prepared.crs, prepared.grid_res_m
    )
    assert res.surface.shape[0] > 0 and res.surface.shape[1] > 0
    assert np.isfinite(res.surface).all()
    assert res.selected_power in fast_config.get("idw.power_candidates")
    assert res.cv_table["is_selected"].sum() == 1
    assert res.loo_metrics["rmse"] > 0

    out = tmp_path / "surface.tif"
    idw.write_geotiff(out, res.surface, res.transform, res.crs)
    assert out.exists()

    import rasterio

    with rasterio.open(out) as src:
        assert src.crs.to_string() == "EPSG:32643"
        assert src.count == 1
        assert abs(src.res[0] - prepared.grid_res_m) < 1e-6


def test_surface_is_within_observed_range(prepared, fast_config):
    xy = np.column_stack([prepared.samples.geometry.x, prepared.samples.geometry.y])
    z = prepared.samples["temp_c"].to_numpy(dtype=float)
    res = idw.interpolate_to_grid(
        fast_config, xy, z, prepared.grid, prepared.crs, prepared.grid_res_m
    )
    assert res.surface.min() >= z.min() - 1e-6
    assert res.surface.max() <= z.max() + 1e-6
