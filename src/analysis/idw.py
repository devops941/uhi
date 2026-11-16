"""Phase 5 - inverse distance weighting interpolation.

Implements IDW directly on top of ``scipy.spatial.cKDTree``:

    z(x) = sum(w_i * z_i) / sum(w_i),  w_i = 1 / d_i ** p

using the k nearest samples. The power ``p`` is chosen by leave-one-out
cross-validation, and the selected surface is written as a GeoTIFF aligned to
the 100 m analysis grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# core IDW
# ----------------------------------------------------------------------
def idw_predict(
    train_xy: np.ndarray,
    train_z: np.ndarray,
    query_xy: np.ndarray,
    power: float = 2.0,
    k: int = 12,
) -> np.ndarray:
    """IDW prediction at ``query_xy`` using the k nearest training points.

    At a query point that coincides with a training point the exact training
    value is returned (distance 0), matching the spec's "exact at sample
    points" requirement.
    """
    train_xy = np.asarray(train_xy, dtype=float)
    train_z = np.asarray(train_z, dtype=float)
    query_xy = np.atleast_2d(np.asarray(query_xy, dtype=float))

    if len(train_xy) == 0:
        raise ValueError("IDW needs at least one training point")

    k = int(min(k, len(train_xy)))
    tree = cKDTree(train_xy)
    dist, idx = tree.query(query_xy, k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    # exact hits: distance below a tiny epsilon -> copy the sample value
    exact = dist[:, 0] <= 1e-9
    weights = np.zeros_like(dist)
    with np.errstate(divide="ignore"):
        weights = 1.0 / np.power(np.where(dist <= 0, 1e-12, dist), power)
    weights_sum = weights.sum(axis=1)
    pred = (weights * train_z[idx]).sum(axis=1) / weights_sum
    pred[exact] = train_z[idx[exact, 0]]
    return pred


def idw_leave_one_out(xy: np.ndarray, z: np.ndarray, power: float, k: int = 12) -> np.ndarray:
    """Leave-one-out predictions: each point is predicted from all the others."""
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(xy)
    if n < 2:
        return np.full(n, np.nan)

    kk = int(min(k, n - 1))
    preds = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        preds[i] = idw_predict(xy[mask], z[mask], xy[i], power=power, k=kk)[0]
    return preds


# ----------------------------------------------------------------------
# cross-validation
# ----------------------------------------------------------------------
def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    o, p = observed[mask], predicted[mask]
    if len(o) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    resid = p - o
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def tune_power(
    xy: np.ndarray,
    z: np.ndarray,
    powers: list[float] | None = None,
    k: int = 12,
) -> pd.DataFrame:
    """Leave-one-out CV over candidate powers; flags the lowest-RMSE winner."""
    powers = powers or [1.0, 1.5, 2.0, 2.5, 3.0]
    rows = []
    for p in powers:
        preds = idw_leave_one_out(xy, z, power=float(p), k=k)
        m = _metrics(z, preds)
        rows.append({"power": float(p), **m})
    df = pd.DataFrame(rows)
    best = df["rmse"].idxmin()
    df["is_selected"] = 0
    df.loc[best, "is_selected"] = 1
    return df


@dataclass
class IDWResult:
    cv_table: pd.DataFrame
    selected_power: float
    surface: np.ndarray
    transform: Affine
    crs: str
    loo_predictions: np.ndarray | None = None
    loo_metrics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# surface
# ----------------------------------------------------------------------
def interpolate_to_grid(
    cfg,
    samples_xy: np.ndarray,
    samples_z: np.ndarray,
    grid: pd.DataFrame,
    crs: str,
    cell_m: float,
) -> IDWResult:
    """Tune the power, interpolate onto the analysis grid and return the surface.

    ``grid`` must carry ``centroid_x`` / ``centroid_y`` in the metric CRS. The
    output raster is a regular grid aligned to the cell size, so the array index
    maps directly onto ``cell_id`` via the returned transform.
    """
    k = int(cfg.get("idw.k", 12))
    powers = list(cfg.get("idw.power_candidates", [1.0, 1.5, 2.0, 2.5, 3.0]))

    cv = tune_power(samples_xy, samples_z, powers=powers, k=k)
    best_row = cv.loc[cv["is_selected"] == 1].iloc[0]
    selected_power = float(best_row["power"])

    loo = idw_leave_one_out(samples_xy, samples_z, power=selected_power, k=k)
    loo_metrics = _metrics(samples_z, loo)

    query = np.column_stack([grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy()])
    surface_values = idw_predict(samples_xy, samples_z, query, power=selected_power, k=k)

    transform, array = grid_to_array(grid, surface_values, cell_m)
    return IDWResult(
        cv_table=cv,
        selected_power=selected_power,
        surface=array,
        transform=transform,
        crs=crs,
        loo_predictions=loo,
        loo_metrics=loo_metrics,
    )


def grid_to_array(
    grid: pd.DataFrame, values: np.ndarray, cell_m: float
) -> tuple[Affine, np.ndarray]:
    """Rasterise grid-cell values onto a regular array aligned to cell_m.

    Rows run north-to-south (row 0 is the northernmost row) and columns
    west-to-east, matching the GeoTIFF convention used by the returned
    transform.
    """
    xs = np.sort(grid["centroid_x"].unique())
    ys = np.sort(grid["centroid_y"].unique())[::-1]  # descending: row 0 = north
    ncols, nrows = len(xs), len(ys)

    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}

    array = np.full((nrows, ncols), np.nan, dtype="float32")
    for (cx, cy), val in zip(
        zip(grid["centroid_x"], grid["centroid_y"], strict=False), values, strict=False
    ):
        array[y_index[cy], x_index[cx]] = val

    # top-left corner of the north-west cell
    transform = Affine.translation(xs[0] - cell_m / 2.0, ys[0] + cell_m / 2.0) * Affine.scale(
        cell_m, -cell_m
    )
    return transform, array


def write_geotiff(path, array: np.ndarray, transform: Affine, crs: str, nodata=np.nan) -> None:
    """Write a single-band float32 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array.astype("float32"), 1)
