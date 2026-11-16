"""Shared pytest fixtures.

The synthetic city is generated once per session into a temporary directory so
the full pipeline can run without touching the repository's data folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def base_config():
    return load_config()


@pytest.fixture(scope="session")
def fast_config(tmp_path_factory):
    """A smaller, faster configuration for pipeline-level tests."""
    work = tmp_path_factory.mktemp("uhi")
    cfg = load_config()
    data = cfg.as_dict()
    data["synthetic"]["n_samples"] = 90
    data["synthetic"]["size_m"] = 2400.0
    data["synthetic"]["urban_core_radius_m"] = 700.0
    data["synthetic"]["grid_size_m"] = 120.0
    data["synthetic"]["raster_res_m"] = 40.0
    data["paths"] = {
        "data_raw": str(work / "raw"),
        "data_processed": str(work / "processed"),
        "sqlite_db": str(work / "uhi.db"),
        "geopackage": str(work / "uhi_layers.gpkg"),
        "outputs": str(work / "outputs"),
        "figures": str(work / "outputs" / "figures"),
    }
    data["hotspots"]["permutations"] = 99
    from src.config import Config

    return Config(data, cfg.path)


@pytest.fixture(scope="session")
def synthetic_inputs(fast_config):
    """Generate the small synthetic city once and return the validated inputs."""
    from src.data import ingest
    from src.data.synthetic import generate_synthetic_city

    generate_synthetic_city(fast_config)
    return ingest.load_inputs(fast_config)


@pytest.fixture(scope="session")
def prepared(fast_config, synthetic_inputs):
    from src.data import preprocess

    return preprocess.preprocess(fast_config, synthetic_inputs)


@pytest.fixture(scope="session")
def tiny_raster(tmp_path_factory):
    """A 3x3 raster with hand-checkable values, in EPSG:32643 (metric)."""
    import rasterio
    from affine import Affine

    path = tmp_path_factory.mktemp("raster") / "tiny_ndvi.tif"
    values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype="float32")
    transform = Affine.translation(500000, 1000000) * Affine.scale(100, -100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)
    return path, values, transform
