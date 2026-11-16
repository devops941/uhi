# Urban Heat Island Mapping with Spatial Interpolation

### Final-Year Project Report

**Domain:** Environment & Climate · **Study area:** Coimbatore, India
(configurable) · **Level:** Intermediate

---

## Abstract

Cities are warmer than their rural surroundings because built surfaces store
heat while vegetation and water cool the air. This project quantifies that Urban
Heat Island (UHI) for a configurable Indian city. Point temperature samples are
interpolated into a continuous heat surface using inverse distance weighting
(IDW) with the power tuned by leave-one-out cross-validation, and the result is
benchmarked against Ordinary Kriging. Land cover, vegetation (NDVI) and built-up
density are related to temperature through correlation, ordinary least-squares
regression with variance-inflation and spatial-autocorrelation diagnostics, and
a Welch / Mann-Whitney comparison of built-up and green cells. Getis-Ord Gi*
hotspots are located at 95% confidence, and a configurable priority score ranks
grid cells and wards for tree planting. On the synthetic 5 km study city the
regression explains 91% of the temperature variance, built-up fraction warms by
+8.2 C per unit and NDVI cools by -3.1 C per unit, and 24% of the city is a
significant hotspot. The whole pipeline runs end-to-end with no downloads, is
covered by 111 tests at 93% coverage, and ships with an interactive Streamlit
dashboard and an auto-generated results report.

---

## 1. Introduction

### 1.1 Problem statement

Concrete, asphalt and buildings trap heat, while vegetation and water cool the
air. The resulting Urban Heat Island (UHI) effect raises surface and air
temperatures in cities relative to their rural surroundings, with consequences
for energy demand, public health and thermal comfort. Planners need answers to
three practical questions:

1. Where are the hottest zones in the city?
2. How strongly do land cover, vegetation (NDVI) and built-up density explain
   temperature differences?
3. Where should trees and green cover be planted first to reduce heat most
   effectively?

### 1.2 Objectives

1. Load and validate temperature sample points.
2. Prepare land-cover, NDVI and built-up layers for the study city.
3. Use spatial joins to attach land cover, NDVI and built-up fraction to each
   sample.
4. Quantify relationships with Pearson and Spearman correlation and multiple
   linear regression.
5. Build a continuous temperature surface with IDW, the power tuned by
   cross-validation.
6. Compute UHI intensity and identify hotspots with Getis-Ord Gi*.
7. Compare built-up and green zones statistically.
8. Produce a tree-planting priority score per cell and ward and visualise it in
   an interactive dashboard.

### 1.3 Scope

In scope: point-to-surface interpolation, spatial joins, correlation and
regression, hotspot detection, priority ranking, maps, dashboard and report.
Out of scope (listed as future work): physical urban-climate simulation
(WRF-UCM, ENVI-met), real-time sensor networks and 3D building models.

---

## 2. Literature review

Oke (1982) established the energetic basis of the urban heat island: the urban
surface stores and re-releases heat, and reduced sky-view factor and ventilation
suppress nocturnal cooling. Weng (2009) and Voogt & Oke (2003) show that thermal
infrared remote sensing, and Landsat's thermal band in particular, provides a
practical way to map land surface temperature (LST) across a whole city, while
noting that LST is a surface quantity, not air temperature.

Shepard (1968) formalised inverse distance weighting, the workhorse of
point-to-surface interpolation: a value at an unknown location is a
distance-weighted mean of nearby observations, with the weight
`w = 1 / d^p`. Getis & Ord (1992) introduced the local Gi* statistic that flags
statistically significant clusters of high or low values, which is exactly the
hotspot question a heat map raises.

This project sits in that tradition: it uses Landsat-style LST samples,
Shepard's IDW, Oke's UHI definition and Getis-Ord's local statistic, and adds a
priority score to turn the analysis into planning advice. It also follows the
modern practice of benchmarking IDW against geostatistical kriging, since the
two make different assumptions about spatial structure.

---

## 3. Data and methodology

### 3.1 Data sources

| Input | Source | Format |
|---|---|---|
| Temperature samples | Landsat 8/9 Band 10 LST, weather stations or field traverse | CSV: `sample_id, lat, lon, temp_c, datetime` |
| Vegetation (NDVI) | Sentinel-2 (B8, B4) or Landsat (B5, B4) | single-band GeoTIFF, -1..1 |
| Land cover | ESA WorldCover, Dynamic World, OSM landuse | GeoTIFF or GeoJSON |
| Built-up density | OSM building footprints (osmnx) | GeoJSON polygons |
| Boundary / wards | OSM or municipal GIS | GeoJSON with `ward_id`, `ward_name` |

**Mandatory fallback.** A synthetic city generator builds a 5 km x 5 km area
with an urban core, parks, a lake and a rural fringe, plus ~200 stratified
temperature sample points. Temperature follows
`T = base + a*builtup - b*NDVI - c*water_proximity + noise` with known
coefficients, so the whole project runs end-to-end with no downloads and the
regression can be checked against a known truth. Real data plugs into the same
schemas.

### 3.2 CRS rule

Raw data is stored in EPSG:4326, but everything is reprojected to a metric
projected CRS (`estimate_utm_crs`) before any distance, buffer, area or
interpolation operation. The code raises rather than interpolating in degrees.

### 3.3 Methods

1. **Cleaning.** Drop duplicate coordinates and out-of-range temperatures
   (10-60 C), flag outliers beyond three standard deviations from the local
   median of the eight nearest neighbours, and keep points inside the boundary.
2. **Analysis grid.** A 100 m x 100 m grid covering the boundary; each cell
   stores mean NDVI, built-up fraction, dominant land cover and `ward_id`.
3. **Spatial join / zonal statistics.** `geopandas.sjoin` for polygon attributes,
   `rasterstats.zonal_stats` for mean NDVI inside a 100 m buffer, and
   `area(buildings ∩ cell) / area(cell)` for built-up fraction.
4. **Correlation.** Pearson and Spearman between `temp_c` and each of
   `ndvi_mean`, `builtup_frac`, `dist_water_m`, with p-values and scatter plots.
5. **Regression.** OLS `temp_c ~ ndvi + builtup_frac + dist_water_m` with R²,
   adjusted R², coefficients with confidence intervals, VIF, a residual plot and
   Moran's I of the residuals (esda). Predictors are standardised to compare
   effect sizes.
6. **IDW interpolation.** `z(x) = Σ w_i z_i / Σ w_i`, `w_i = 1 / d_i^p`, using
   the `k = 12` nearest samples (SciPy `cKDTree`). Power `p` is tuned over
   {1, 1.5, 2, 2.5, 3} by leave-one-out CV, reporting RMSE, MAE and R² for each
   and choosing the lowest RMSE. Output is a raster aligned to the grid.
   *Extension:* the same LOO split scores Ordinary Kriging (PyKrige), and the
   better method is reported.
7. **UHI intensity.** `UHI = T_cell - mean(T of rural reference cells)`, where
   rural reference cells have built-up fraction < 10% and lie near the outskirts.
   Max, mean and area above +2 C and +4 C are reported.
8. **Hotspots.** Getis-Ord Gi* on grid temperatures (esda `G_Local` with queen
   weights); hotspots at 95% confidence.
9. **Built-up vs green.** Built-up if `builtup_frac > 0.5`, green if `NDVI >
   0.4`; compare mean temperature with Welch's t-test and Mann-Whitney U, and
   report the mean difference and Cohen's d.
10. **Priority score.**
    `priority = 0.4*norm(temp) + 0.3*(1 - norm(NDVI)) + 0.2*norm(builtup) +
    0.1*norm(pop_proxy)`, with configurable weights. Output the top 10 wards and
    top 50 cells, plus a cooling estimate for a +0.1 NDVI uplift from the
    regression coefficient.

### 3.4 System flow

```
Raw data (CSV / rasters / GeoJSON) or synthetic generator
        |
[1] Ingestion & validation (schema, CRS, geometry, temp range)
        |
[2] Preprocessing (reproject to UTM, clip, clean, build 100 m grid)
        |
[3] Feature extraction (buffer 100 m -> NDVI, built-up, land cover, dist to water)
        |
[4] Store in SQLite + GeoPackage
        |
        +--> [5] Statistics (correlation, OLS, VIF, Moran's I, group tests)
        +--> [6] IDW + LOO-CV -> temperature raster (+ kriging comparison)
        +--> [7] UHI intensity + Gi* hotspots
        |
[8] Priority scoring (temperature, low NDVI, built-up, pop proxy)
        |
[9] Outputs (GeoTIFF, GeoPackage, tables, figures)
        |
[10] Streamlit dashboard + auto-generated report
```

---

## 4. Implementation

### 4.1 Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Vector data | GeoPandas, Shapely, pyproj |
| Raster data | rasterio, rasterstats, numpy |
| Interpolation | SciPy (cKDTree), PyKrige (comparison) |
| Statistics | statsmodels, scipy.stats, esda, libpysal |
| OSM data | osmnx |
| Visualisation | matplotlib, folium, plotly |
| Database | SQLite (SQLAlchemy) + GeoPackage |
| Dashboard | Streamlit (+ streamlit-folium) |
| Reporting | Jinja2, markdown, optional weasyprint |
| Config | PyYAML |
| Testing / quality | pytest, pytest-cov, ruff, black |

### 4.2 Module map

```
src/
|-- data/     synthetic.py, ingest.py, preprocess.py, db.py
|-- features/ extract.py
|-- analysis/ stats.py, idw.py, interpolation_compare.py, uhi.py, priority.py
|-- viz/      maps.py
|-- report/   generate.py, template.md.j2
`-- pipeline.py   end-to-end CLI (phases 1-9)
app/            dashboard.py, data.py, theme.py
```

### 4.3 Database design

SQLite (`data/uhi.db`) holds the tabular tables: `wards`, `samples`,
`sample_features`, `grid_cells`, `correlation_results`, `regression_results`,
`model_summary`, `idw_cv_results`, `ward_summary`, `group_comparison`, and the
`interpolation_comparison` extension table. The GeoPackage
(`data/uhi_layers.gpkg`) holds the spatial layers: `boundary`, `wards`,
`samples`, `grid`, `buildings` and `hotspots`. Rasters are written to
`outputs/` as GeoTIFF: `temperature_idw.tif`, `uhi_intensity.tif`, `ndvi.tif`,
`hotspots.tif`, `priority_score.tif`.

### 4.4 Reproducibility

A single CLI runs the whole project:

```bash
python -m src.pipeline --source synthetic
streamlit run app/dashboard.py
```

Every tunable value (study city, sample count, coefficients, thresholds,
weights) lives in `config.yaml`. The test suite uses a smaller `fast_config`
fixture that shrinks the grid so it runs in seconds.

---

## 5. Results and discussion

All numbers below come from the synthetic study city (seed 42, ~200 samples,
2,500 grid cells at 100 m). They are regenerated by
`python -m src.pipeline --source synthetic`.

### 5.1 Sample and grid summary

| Metric | Value |
|---|---|
| Temperature samples used | 197 |
| Analysis grid cells (100 m) | 2,500 |
| Mean IDW temperature | 26.33 C |
| Rural baseline temperature | 25.63 C |
| Max UHI intensity | 9.09 C |
| Mean UHI intensity | 0.70 C |
| Hotspot cells (95%) | 598 (23.9%) |
| Coldspot cells | 809 |

### 5.2 Correlation

| Variable | Pearson r | Spearman rho | n |
|---|---|---|---|
| `ndvi_mean` | -0.775 | -0.808 | 192 |
| `builtup_frac` | +0.894 | +0.884 | 192 |
| `dist_water_m` | -0.244 | -0.276 | 192 |

Vegetation cools and built-up surface warms, both strongly. The water-distance
term is weaker and, after controlling for the other two, changes sign — built-up
areas happen to sit close to the lake, so the simple correlation is confounded.
This is a good example of why the regression matters.

### 5.3 OLS regression

| Term | Coef | p-value | VIF |
|---|---|---|---|
| const | 25.17 | < 1e-140 | |
| `ndvi_mean` | -3.08 | 2.1e-13 | 2.60 |
| `builtup_frac` | +8.21 | 3.2e-60 | 2.75 |
| `dist_water_m` | +0.00159 | 9.0e-32 | 1.45 |

R² = 0.911, adjusted R² = 0.910, maximum VIF = 2.75, Moran's I of residuals =
0.310. The signs match the generator's built-in coefficients (built-up positive,
NDVI negative), which the test suite checks explicitly. VIF is well below 10, so
multicollinearity is not a problem. The positive Moran's I means nearby samples
are more alike than chance, so the OLS p-values are optimistic — a spatial
lag/error model is listed as future work.

**Estimated cooling from a +0.1 NDVI uplift: 0.31 C** (from the NDVI
coefficient; an indicative effect size, not a guarantee).

### 5.4 Built-up vs green

| Group A | Group B | Mean A | Mean B | Diff | Welch p | Mann-Whitney p | Cohen's d |
|---|---|---|---|---|---|---|---|
| built_up | green | 30.98 | 26.59 | +4.39 | 1.1e-23 | 4.3e-21 | 2.12 |

Built-up cells are 4.4 C hotter than green cells, a very large effect (d > 2)
that is significant under both tests.

### 5.5 Interpolation

| Power | RMSE | MAE | R² |
|---|---|---|---|
| 1.0 | 1.222 | 0.907 | 0.820 |
| 1.5 | 1.160 | 0.832 | 0.838 |
| 2.0 | 1.128 | 0.787 | 0.847 |
| 2.5 | 1.114 | 0.763 | 0.851 |
| 3.0 | 1.111 | 0.757 | 0.852 |

The selected power is 3.0 (lowest LOO RMSE). On the same leave-one-out split:

| Method | Parameter | RMSE | MAE | R² |
|---|---|---|---|---|
| IDW | p = 3 | 1.111 | 0.757 | 0.852 |
| Kriging | spherical | 0.889 | 0.594 | 0.905 |

Kriging wins here because the synthetic temperature field is smooth and the
fitted variogram (nugget 0.0, sill 7.4, range 1306 m) captures that structure.
IDW stays competitive and needs no variogram fitting, so it remains the cheaper
default when the two are close.

### 5.6 UHI intensity and hotspots

| Threshold | Area (km²) | Share of city |
|---|---|---|
| +2 C | 6.21 | 24.8% |
| +4 C | 2.07 | 8.3% |

Gi* finds 598 hotspot cells (23.9%) and 809 coldspots. Hotspots concentrate on
built-up land in the core and avoid the parks and lake, which is exactly the
acceptance criterion from the project plan and is verified by a test.

### 5.7 Tree-planting priority

Top wards by priority score: **W10, W17, W11**. W11 stands out with a mean UHI
of +5.9 C and 100% of its area flagged as hotspot; W10 combines the highest mean
built-up fraction (0.79) with a low mean NDVI (0.26). The top individual cells
are all built-up cells in W10 and W11 with NDVI near zero and UHI above +8 C —
hot, bare, impervious ground, exactly where planting buys the most cooling.

### 5.8 Discussion

The three questions from the introduction have clear answers on this dataset:
the heat is concentrated in the dense built-up core (Section 5.6), built-up
fraction and NDVI together explain 91% of the temperature variance
(Section 5.3), and the priority ranking points at the hot, bare, dense wards
(Section 5.7). The comparison of IDW and kriging shows that the choice of
interpolator matters at the margin (RMSE 0.89 vs 1.11) but does not change the
planning conclusion, because both surfaces put the same cells at the top of the
priority list. The confounded water-distance sign (Section 5.2) is a useful
reminder that marginal correlations can mislead and that the regression is the
right tool for attribution.

---

## 6. Conclusion and future work

The project delivers a complete, reproducible Urban Heat Island analysis: it
validates and cleans temperature samples, attaches land-cover and vegetation
features through spatial joins and zonal statistics, explains temperature with
correlation and regression, interpolates a heat surface with tuned IDW and a
kriging benchmark, locates hotspots with Getis-Ord Gi*, and ranks tree-planting
priority per cell and ward. Everything runs end-to-end without downloads, is
covered by 111 tests at 93% coverage, and ships with an interactive dashboard
and an auto-generated report.

Future work:

* Regression kriging (regression plus IDW on the residuals) and a fuller
  geostatistical treatment of the residuals.
* Random-Forest LST downscaling and machine-learning emulation of the surface.
* Multi-season and time-of-day analysis to separate day and night UHI.
* Tree-planting scenario simulation that re-runs the model with raised NDVI.
* Integration with population and vulnerability data for equity-aware planning.

---

## 7. References

1. Oke, T. R. (1982). The energetic basis of the urban heat island. *Quarterly
   Journal of the Royal Meteorological Society*, 108(455), 1-24.
2. Shepard, D. (1968). A two-dimensional interpolation function for
   irregularly-spaced data. *Proceedings of the 1968 ACM National Conference*,
   517-524.
3. Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of
   distance statistics. *Geographical Analysis*, 24(3), 189-206.
4. Weng, Q. (2009). Thermal infrared remote sensing for urban climate and
   environmental studies. *ISPRS Journal of Photogrammetry and Remote Sensing*,
   64(4), 335-344.
5. Voogt, J. A., & Oke, T. R. (2003). Thermal remote sensing of urban climates.
   *Remote Sensing of Environment*, 86(3), 370-384.
6. Documentation: GeoPandas, rasterio, PySAL (esda, libpysal), statsmodels,
   osmnx, ESA WorldCover, USGS Landsat Collection 2, PyKrige.

---

## Appendix A - Final deliverables checklist

- [x] Working code with README and setup steps
- [x] SQLite database and GeoPackage with all layers
- [x] Heat map GeoTIFF and hotspot layers
- [x] Streamlit dashboard
- [x] Auto-generated results report (`outputs/report.md`)
- [x] Final project report (this document)
- [x] Presentation slides and demo script (`docs/`)
- [x] Future work documented

## Appendix B - How to reproduce

```bash
pip install -r requirements.txt
python -m src.pipeline --source synthetic     # ~16 s
python -m pytest tests/ -q                    # 111 tests
python -m pytest tests/ --cov=src             # ~93% coverage
streamlit run app/dashboard.py
```
