"""Build the project notebooks programmatically (keeps them reproducible)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


SETUP = """\
import sys
from pathlib import Path

# notebooks live one level below the project root
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import db as db_mod

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
%matplotlib inline
"""


def notebook_01() -> nbf.NotebookNode:
    return nb(
        [
            (
                "md",
                "# 01 - Exploratory data analysis\n\n"
                "What the raw inputs look like before any modelling: the sample "
                "distribution, the land-cover mix and the covariates attached to "
                "each temperature sample.",
            ),
            ("code", SETUP),
            (
                "code",
                "cfg = load_config(ROOT / 'config.yaml')\n"
                "engine = db_mod.get_engine(cfg)\n"
                "db_mod.summarise_db(engine)",
            ),
            ("md", "## Temperature samples"),
            (
                "code",
                "samples = db_mod.read_table(engine, 'samples')\n"
                "samples[['temp_c', 'lat', 'lon']].describe().round(3)",
            ),
            (
                "code",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
                "axes[0].hist(samples['temp_c'], bins=30, color='#8b5cf6')\n"
                "axes[0].set_title('Temperature distribution')\n"
                "axes[0].set_xlabel('Temperature (C)')\n"
                "axes[1].scatter(samples['lon'], samples['lat'], c=samples['temp_c'],\n"
                "                cmap='inferno', s=18)\n"
                "axes[1].set_title('Sample locations')\n"
                "axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')\n"
                "plt.tight_layout(); plt.show()",
            ),
            ("md", "## Covariates attached to each sample"),
            (
                "code",
                "feats = db_mod.read_table(engine, 'sample_features')\n"
                "joined = samples.merge(feats, on='sample_id')\n"
                "joined.groupby('landcover_class')[['temp_c', 'ndvi_mean', 'builtup_frac',\n"
                "                                   'dist_water_m']].mean().round(3)",
            ),
            (
                "code",
                "joined[['temp_c', 'ndvi_mean', 'builtup_frac', 'dist_water_m']].corr().round(3)",
            ),
            ("md", "## The analysis grid"),
            (
                "code",
                "grid = db_mod.read_table(engine, 'grid_cells')\n"
                "print('cells:', len(grid))\n"
                "grid[['ndvi_mean', 'builtup_frac', 'temp_idw', 'uhi_intensity']].describe().round(3)",
            ),
            (
                "md",
                "**Read this.** Temperature rises with built-up fraction and falls "
                "with NDVI, which is the relationship the rest of the project "
                "quantifies.",
            ),
        ]
    )


def notebook_02() -> nbf.NotebookNode:
    return nb(
        [
            (
                "md",
                "# 02 - Statistics\n\n"
                "Correlation, the OLS regression with VIF and Moran's I, and the "
                "built-up versus green comparison.",
            ),
            ("code", SETUP),
            ("code", "cfg = load_config(ROOT / 'config.yaml')\n" "engine = db_mod.get_engine(cfg)"),
            ("md", "## Correlation (Pearson and Spearman)"),
            ("code", "db_mod.read_table(engine, 'correlation_results').round(4)"),
            ("md", "## OLS regression with VIF"),
            ("code", "db_mod.read_table(engine, 'regression_results').round(4)"),
            (
                "code",
                "summary = db_mod.read_table(engine, 'model_summary').set_index('metric')['value']\n"
                "summary.round(4)",
            ),
            ("md", "### Residual diagnostics"),
            (
                "code",
                "samples = db_mod.read_table(engine, 'samples')\n"
                "feats = db_mod.read_table(engine, 'sample_features')\n"
                "df = samples.merge(feats, on='sample_id')\n"
                "df = df[df['is_outlier'] == 0]\n"
                "df = df.dropna(subset=['temp_c', 'ndvi_mean', 'builtup_frac', 'dist_water_m'])\n"
                "\n"
                "from src.analysis import stats as st\n"
                "coef, summ, model, fitted, std = st.regress(df)\n"
                "fitted.head()",
            ),
            (
                "code",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
                "axes[0].scatter(fitted['fitted'], fitted['residual'], s=16, alpha=0.6)\n"
                "axes[0].axhline(0, color='k', ls='--')\n"
                "axes[0].set_xlabel('Fitted (C)'); axes[0].set_ylabel('Residual (C)')\n"
                "axes[0].set_title('Residuals vs fitted')\n"
                "axes[1].hist(fitted['residual'].dropna(), bins=25, color='#22d3ee')\n"
                "axes[1].set_title('Residual distribution')\n"
                "plt.tight_layout(); plt.show()",
            ),
            ("md", "## Built-up versus green"),
            ("code", "db_mod.read_table(engine, 'group_comparison').round(5)"),
            (
                "md",
                "**Read this.** A positive Moran's I on the residuals means nearby "
                "samples are more alike than chance, so the OLS p-values are "
                "optimistic. The effect sizes are still large and consistent.",
            ),
        ]
    )


def notebook_03() -> nbf.NotebookNode:
    return nb(
        [
            (
                "md",
                "# 03 - Interpolation: IDW vs kriging\n\n"
                "How the power is chosen, how IDW compares with Ordinary Kriging on "
                "the same leave-one-out split, and what the final surface looks like.",
            ),
            ("code", SETUP),
            ("code", "cfg = load_config(ROOT / 'config.yaml')\n" "engine = db_mod.get_engine(cfg)"),
            ("md", "## IDW power selection by leave-one-out CV"),
            ("code", "cv = db_mod.read_table(engine, 'idw_cv_results')\n" "cv.round(4)"),
            (
                "code",
                "fig, ax = plt.subplots(figsize=(7, 4))\n"
                "ax.plot(cv['power'], cv['rmse'], marker='o', label='RMSE', color='#8b5cf6')\n"
                "ax.plot(cv['power'], cv['mae'], marker='s', label='MAE', color='#22d3ee')\n"
                "best = cv.loc[cv['is_selected'] == 1, 'power'].iloc[0]\n"
                "ax.axvline(best, ls='--', color='#fb7185', label=f'selected p={best:g}')\n"
                "ax.set_xlabel('power'); ax.set_ylabel('error (C)'); ax.legend()\n"
                "ax.set_title('IDW cross-validation'); plt.show()",
            ),
            ("md", "## IDW vs Ordinary Kriging"),
            ("code", "db_mod.read_table(engine, 'interpolation_comparison').round(4)"),
            ("md", "## Rebuild the comparison in-memory to inspect the variogram"),
            (
                "code",
                "from src.analysis import interpolation_compare as ic\n"
                "from src.data import ingest, preprocess\n"
                "from src.features import extract\n"
                "\n"
                "raw = ingest.load_inputs(cfg)\n"
                "prep = preprocess.preprocess(cfg, raw)\n"
                "grid = extract.extract_grid_features(\n"
                "    cfg, prep.grid, prep.buildings, raw.landcover_polygons,\n"
                "    raw.ndvi_path, raw.landcover_path, prep.crs)\n"
                "xy = np.column_stack([prep.samples.geometry.x, prep.samples.geometry.y])\n"
                "z = prep.samples['temp_c'].to_numpy(dtype=float)\n"
                "comp = ic.compare_methods(cfg, xy, z, grid, prep.crs, prep.grid_res_m)\n"
                "print('best:', comp.best_method)\n"
                "print('variogram:', comp.variogram)",
            ),
            ("md", "## The final surface"),
            (
                "code",
                "import rasterio\n"
                "with rasterio.open(ROOT / 'outputs' / 'temperature_idw.tif') as src:\n"
                "    surface = src.read(1)\n"
                "fig, ax = plt.subplots(figsize=(7, 6))\n"
                "im = ax.imshow(surface, cmap='inferno')\n"
                "ax.set_title('Interpolated temperature (IDW)'); ax.set_xticks([]); ax.set_yticks([])\n"
                "plt.colorbar(im, label='C'); plt.show()",
            ),
            (
                "md",
                "**Read this.** Kriging usually wins on a smooth field because it "
                "fits a variogram, while IDW is cheaper and needs no fitting. When "
                "the two are close, IDW is the sensible default.",
            ),
        ]
    )


def notebook_04() -> nbf.NotebookNode:
    return nb(
        [
            (
                "md",
                "# 04 - UHI hotspots and tree-planting priority\n\n"
                "Where the heat islands are, and where planting trees first would "
                "help most.",
            ),
            ("code", SETUP),
            (
                "code",
                "cfg = load_config(ROOT / 'config.yaml')\n"
                "engine = db_mod.get_engine(cfg)\n"
                "grid = db_mod.read_table(engine, 'grid_cells')",
            ),
            ("md", "## UHI intensity and hotspots"),
            (
                "code",
                "print('hotspot cells:', int(grid['is_hotspot'].sum()))\n"
                "print('coldspot cells:', int((grid['is_hotspot'] == 0).sum()))\n"
                "print('mean UHI:', round(grid['uhi_intensity'].mean(), 3))\n"
                "print('max UHI:', round(grid['uhi_intensity'].max(), 3))",
            ),
            (
                "code",
                "fig, ax = plt.subplots(figsize=(7.5, 6.5))\n"
                "sc = ax.scatter(grid['centroid_x'], grid['centroid_y'], c=grid['uhi_intensity'],\n"
                "                cmap='coolwarm', s=4)\n"
                "hot = grid[grid['is_hotspot'] == 1]\n"
                "ax.scatter(hot['centroid_x'], hot['centroid_y'], s=6, facecolors='none',\n"
                "           edgecolors='#e11d48', linewidths=0.6, label='hotspot (95%)')\n"
                "ax.set_title('UHI intensity with Gi* hotspots'); ax.legend()\n"
                "ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(sc, label='UHI (C)'); plt.show()",
            ),
            ("md", "## Ward ranking for tree planting"),
            ("code", "db_mod.read_table(engine, 'ward_summary').round(4)"),
            ("md", "## Top priority cells"),
            (
                "code",
                "cols = ['cell_id', 'ward_id', 'temp_idw', 'uhi_intensity', 'ndvi_mean',\n"
                "        'builtup_frac', 'landcover_class', 'priority_score', 'priority_rank']\n"
                "grid.nsmallest(20, 'priority_rank')[cols].round(3)",
            ),
            ("md", "## Where the hotspots sit by land cover"),
            (
                "code",
                "grid.groupby('landcover_class').agg(\n"
                "    cells=('cell_id', 'size'),\n"
                "    hotspot_pct=('is_hotspot', lambda s: round(100 * s.mean(), 1)),\n"
                "    mean_uhi=('uhi_intensity', 'mean'),\n"
                ").round(2)",
            ),
            (
                "md",
                "**Read this.** Hotspots should sit on built-up land in the core, "
                "and the priority ranking should point at hot, bare, built-up cells. "
                "If a hotspot landed on the lake or a park, something would be wrong.",
            ),
        ]
    )


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_eda.ipynb": notebook_01(),
        "02_statistics.ipynb": notebook_02(),
        "03_interpolation_idw_vs_kriging.ipynb": notebook_03(),
        "04_uhi_and_priority.ipynb": notebook_04(),
    }
    for name, notebook in notebooks.items():
        path = NB_DIR / name
        nbf.write(notebook, path)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
