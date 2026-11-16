"""Interpolation comparison: IDW vs Ordinary Kriging (spec Section 6 extension).

Both methods are scored on the *same* leave-one-out split so the comparison is
fair, then the better one is used for the final surface. PyKrige is optional:
if it is not installed the module degrades to IDW only and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis import idw as idw_mod

try:  # pragma: no cover - optional dependency
    from pykrige.ok import OrdinaryKriging

    HAS_PYKRIGE = True
except Exception:  # pragma: no cover
    OrdinaryKriging = None
    HAS_PYKRIGE = False


@dataclass
class InterpolationComparison:
    """Metrics for every method plus the winner and its surface."""

    metrics: pd.DataFrame
    best_method: str
    surfaces: dict[str, np.ndarray] = field(default_factory=dict)
    transform: object | None = None
    variogram: dict | None = None
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# kriging
# ----------------------------------------------------------------------
def kriging_predict(
    train_xy: np.ndarray,
    train_z: np.ndarray,
    query_xy: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 8,
) -> np.ndarray:
    """Ordinary kriging predictions at ``query_xy``."""
    if not HAS_PYKRIGE:
        raise RuntimeError("pykrige is not installed; pip install pykrige")
    ok = OrdinaryKriging(
        train_xy[:, 0],
        train_xy[:, 1],
        train_z,
        variogram_model=variogram_model,
        nlags=nlags,
        verbose=False,
        enable_plotting=False,
    )
    z, _ = ok.execute("points", query_xy[:, 0], query_xy[:, 1])
    return np.asarray(z, dtype=float)


def kriging_leave_one_out(
    xy: np.ndarray, z: np.ndarray, variogram_model: str = "spherical", nlags: int = 8
) -> np.ndarray:
    """Leave-one-out predictions - one kriging fit per held-out point.

    Ordinary kriging has no closed form for LOO, so this refits n times. That is
    fine for the ~200 sample points this project uses.
    """
    if not HAS_PYKRIGE:
        raise RuntimeError("pykrige is not installed; pip install pykrige")
    preds = np.full(len(z), np.nan, dtype=float)
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool)
        mask[i] = False
        try:
            preds[i] = kriging_predict(
                xy[mask],
                z[mask],
                xy[i : i + 1],
                variogram_model=variogram_model,
                nlags=nlags,
            )[0]
        except Exception:  # a single bad fold should not kill the comparison
            preds[i] = np.nan
    return preds


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------
def compare_methods(
    cfg,
    samples_xy: np.ndarray,
    samples_z: np.ndarray,
    grid: pd.DataFrame,
    crs: str,
    cell_m: float,
    variogram_model: str = "spherical",
) -> InterpolationComparison:
    """Score IDW and kriging with the same LOO protocol, then build the surface."""
    idw_res = idw_mod.interpolate_to_grid(cfg, samples_xy, samples_z, grid, crs, cell_m)

    rows = [
        {
            "method": "IDW",
            "parameter": f"p={idw_res.selected_power:g}",
            **idw_res.loo_metrics,
        }
    ]
    notes: list[str] = []
    surfaces = {"IDW": idw_res.surface}

    if HAS_PYKRIGE:
        loo = kriging_leave_one_out(samples_xy, samples_z, variogram_model)
        rows.append(
            {"method": "Kriging", "parameter": variogram_model, **idw_mod._metrics(samples_z, loo)}
        )
    else:
        notes.append("pykrige is not installed, so only IDW was evaluated.")

    metrics = pd.DataFrame(rows)
    metrics["is_best"] = (metrics["rmse"] == metrics["rmse"].min()).astype(int)

    best_method = str(metrics.sort_values("rmse").iloc[0]["method"])

    query = np.column_stack([grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy()])
    if best_method == "Kriging" and HAS_PYKRIGE:
        preds = kriging_predict(samples_xy, samples_z, query, variogram_model)
        transform, array = idw_mod.grid_to_array(grid, preds, cell_m)
        surfaces["Kriging"] = array
        transform_used = transform
    else:
        transform_used = idw_res.transform

    # variogram parameters, for the notebook and the report
    variogram = None
    if HAS_PYKRIGE:
        try:
            ok = OrdinaryKriging(
                samples_xy[:, 0],
                samples_xy[:, 1],
                samples_z,
                variogram_model=variogram_model,
                nlags=8,
                verbose=False,
                enable_plotting=False,
            )
            variogram = {
                "model": variogram_model,
                "psill": float(ok.variogram_model_parameters[0]),
                "range": float(ok.variogram_model_parameters[1]),
                "nugget": float(ok.variogram_model_parameters[2]),
            }
        except Exception:  # pragma: no cover - defensive
            variogram = None

    notes.append(
        "Both methods use the same leave-one-out split, so lower RMSE means a better "
        "reproduction of held-out sample temperatures."
    )
    return InterpolationComparison(
        metrics=metrics,
        best_method=best_method,
        surfaces=surfaces,
        transform=transform_used,
        variogram=variogram,
        notes=notes,
    )
