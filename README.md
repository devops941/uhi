# Urban Heat Island Mapping with Spatial Interpolation

Cities run hotter than their rural surroundings because concrete, asphalt and
buildings store heat while vegetation and water cool the air. This project maps
that Urban Heat Island (UHI) for a configurable Indian city, explains it with
land cover and vegetation, and ranks the places where planting trees would help
most.

It is a complete, runnable pipeline: point temperature samples in, an
interpolated heat map, a statistical explanation, a hotspot map and a
tree-planting priority list out, with an interactive dashboard and an
auto-generated report on top.

**Domain:** Environment & Climate · **Stack:** Python, GeoPandas, SciPy,
statsmodels, rasterio, esda/libpysal, Folium, SQLite + GeoPackage, Streamlit

---

## What it answers

1. **Where are the hottest zones?** An IDW-interpolated temperature surface plus
   Getis-Ord Gi* hotspots at 95% confidence.
2. **How much do land cover, NDVI and built-up density explain the heat?**
   Pearson/Spearman correlation and an OLS regression with VIF, confidence
   intervals, Moran's I of the residuals and standardised effect sizes.
3. **Where should trees be planted first?** A configurable priority score per
   grid cell and per ward, with a cooling estimate derived from the regression.

## Results on the synthetic city

| Metric | Value |
|---|---|
| Temperature samples used | 197 |
| Analysis grid | 2,500 cells at 100 m |
| Regression R² | 0.91 |
| Built-up coefficient | +8.2 C per unit fraction |
| NDVI coefficient | -3.1 C per unit |
| Max UHI intensity | +9.1 C |
| Hotspot cells (95%) | 598 (23.9%) |
| Best interpolation (LOO RMSE) | Kriging 0.89 C vs IDW 1.11 C |

The synthetic generator plants known coefficients, and the regression recovers
their signs and approximate magnitudes — that is a tested guarantee, not a
coincidence (see `tests/test_stats.py`).

---

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the whole project on the synthetic city (no downloads needed)
python -m src.pipeline --source synthetic

# 3. explore the results
streamlit run app/dashboard.py
```

The pipeline writes:

```
outputs/
|-- temperature_idw.tif      interpolated temperature surface
|-- uhi_intensity.tif        UHI intensity (urban minus rural baseline)
|-- hotspots.tif             Gi* hotspot mask
|-- priority_score.tif       tree-planting priority
|-- report.md                auto-generated results report
|-- figures/                 static maps and charts
`-- dashboard/               precomputed dashboard payload
data/
|-- uhi.db                   SQLite: every table from spec Section 8
`-- uhi_layers.gpkg          GeoPackage: boundary, wards, samples, grid, buildings, hotspots
```

## Command line

```bash
python -m src.pipeline --source synthetic              # generate + run
python -m src.pipeline --source synthetic --skip-generate   # reuse raw files
python -m src.pipeline --source real                   # use your own inputs
python -m src.pipeline --source synthetic --report-pdf # also emit PDF (needs weasyprint)
```

`--config` points at an alternative `config.yaml`.

## Project layout

```
src/
|-- config.py               config loader with dotted access
|-- pipeline.py             end-to-end CLI (phases 1-9)
|-- data/
|   |-- synthetic.py        synthetic city generator (the mandatory fallback)
|   |-- ingest.py           load + validate inputs
|   |-- preprocess.py       reproject, clip, clean, build the 100 m grid
|   `-- db.py               SQLAlchemy + GeoPackage helpers
|-- features/extract.py     spatial joins, NDVI zonal stats, built-up fraction
|-- analysis/
|   |-- stats.py            correlation, OLS + VIF, Moran's I, group tests
|   |-- idw.py              IDW + leave-one-out cross-validation
|   |-- interpolation_compare.py   IDW vs Ordinary Kriging
|   |-- uhi.py              UHI intensity, Getis-Ord Gi* hotspots
|   `-- priority.py         tree-planting priority score
|-- viz/maps.py             matplotlib + Folium maps
|-- report/generate.py      Jinja2 -> Markdown/PDF report
`-- report/template.md.j2
app/
|-- dashboard.py            Streamlit app (7 pages)
|-- data.py                 cached data access
`-- theme.py                2026-trend UI theme
tests/                      111 tests, 93% coverage
notebooks/                  4 executed analysis notebooks
```

## Data

The project runs end-to-end with **no downloads** thanks to the synthetic city
generator (5 km x 5 km, urban core, parks, lake, rural fringe, ~200 sample
points, temperature generated from known coefficients). Real data plugs into the
same schemas.

| Input | Format | File |
|---|---|---|
| Temperature samples | CSV: `sample_id, lat, lon, temp_c, datetime` | `data/raw/samples.csv` |
| Vegetation | single-band GeoTIFF, NDVI in -1..1 | `data/raw/ndvi.tif` |
| Land cover | GeoTIFF of class codes, or GeoJSON polygons | `data/raw/landcover.tif` |
| Buildings | GeoJSON polygons | `data/raw/buildings.geojson` |
| Boundary / wards | GeoJSON with `ward_id`, `ward_name` | `data/raw/boundary.geojson` |

### Preparing real data

1. **LST samples** — export Landsat 8/9 Band 10 surface temperature from Earth
   Explorer or Earth Engine, sample random points, and save as
   `samples.csv` with `temp_c` in degrees Celsius.
2. **NDVI** — compute `(NIR - Red) / (NIR + Red)` from Sentinel-2 B8/B4 or
   Landsat B5/B4 and save as a single-band GeoTIFF.
3. **Land cover** — download ESA WorldCover or Dynamic World, clip to the city,
   and set the class codes in `config.yaml`.
4. **Buildings and boundary** — fetch with `osmnx` and save as GeoJSON.

Then run `python -m src.pipeline --source real`. The ingestion layer validates
the schemas and rejects bad CRS, missing columns and out-of-range temperatures.

**CRS rule:** raw data is stored in EPSG:4326, but everything is reprojected to
a metric CRS (`estimate_utm_crs`) before any distance, buffer, area or
interpolation operation. The code raises rather than interpolating in degrees.

## Dashboard

```bash
streamlit run app/dashboard.py
```

Seven pages: Overview, Layers, Relationships, Interpolation, Built-up vs Green,
Priority and Download. The Priority page recomputes the ranking live as you move
the weight sliders.

## Tests

```bash
python -m pytest tests/ -q
python -m pytest tests/ --cov=src --cov-report=term-missing
```

111 tests, 93% coverage. They include hand-checked 3x3 raster cases, IDW
exactness at sample points, CRS guards that must raise on degrees, and a
synthetic-truth test that checks the regression recovers the built-in
coefficient signs.

## Method notes

* Interpolation is IDW with the power chosen by leave-one-out CV, benchmarked
  against Ordinary Kriging on the same split.
* Satellite LST is **surface** temperature, not air temperature. This matters
  and is stated in the report.
* A positive Moran's I on the residuals means the OLS p-values are optimistic; a
  spatial lag/error model is listed as future work.
* The cooling estimate is an indicative effect size from the regression
  coefficient, not a guarantee.

## Limitations and future work

Out of scope by design: physical urban-climate simulation (WRF-UCM, ENVI-met),
real-time sensor networks and 3D building models. Future work: regression
kriging, Random-Forest LST downscaling, multi-season and time-of-day analysis,
tree-planting scenario simulation, and integration with population data.

## References

Oke (1982); Shepard (1968); Getis & Ord (1992); Weng (2009); Voogt & Oke (2003).
Library documentation: GeoPandas, rasterio, PySAL (esda, libpysal), statsmodels,
osmnx, ESA WorldCover, USGS Landsat Collection 2.
