# Presentation slides

A 12-slide deck for a 10-minute talk. Each slide gives the on-screen title, the
bullets to show, and a speaker note. Numbers match the synthetic study city.

---

## Slide 1 — Title

**Urban Heat Island Mapping with Spatial Interpolation**
Final-year project · Environment & Climate
Your name · Supervisor · Date

> *Note:* One sentence: "Cities are hotter than the countryside; this project
> maps where, explains why, and says where to plant trees first."

---

## Slide 2 — The problem

- Cities run hotter than their rural surroundings (the UHI effect).
- Concrete and asphalt store heat; vegetation and water cool the air.
- Planners ask three things:
  1. Where is it hottest?
  2. What explains the heat?
  3. Where should we plant trees first?

> *Note:* Land on the three questions — the whole deck answers them in order.

---

## Slide 3 — What the project does

- Validates temperature samples, builds a 100 m analysis grid.
- Attaches land cover, NDVI and built-up fraction to every sample.
- Interpolates a continuous heat map with tuned IDW.
- Explains temperature with correlation and regression.
- Finds hotspots (Getis-Ord Gi*) and ranks tree-planting priority.

> *Note:* This is the pipeline in five lines; the flow diagram is on the next
> slide.

---

## Slide 4 — System flow

```
Samples + NDVI + land cover + buildings
   -> validate -> preprocess -> features
   -> statistics | IDW + kriging | UHI + hotspots
   -> priority -> GeoTIFF, GeoPackage, dashboard, report
```

- One CLI runs everything: `python -m src.pipeline --source synthetic`.
- No downloads needed: a synthetic city generator is built in.

> *Note:* Stress reproducibility — one command, one config file.

---

## Slide 5 — Data

| Input | Source |
|---|---|
| Temperature | Landsat 8/9 Band 10 LST samples |
| Vegetation | Sentinel-2 / Landsat NDVI |
| Land cover | ESA WorldCover |
| Built-up | OSM building footprints |
| Boundary | OSM / municipal GIS |

- Synthetic fallback: 5 km city, ~200 samples, known coefficients.
- CRS rule: raw in EPSG:4326, everything metric before distance work.

> *Note:* The synthetic fallback is what makes the project runnable anywhere.

---

## Slide 6 — Method: interpolation

- IDW: `z(x) = Σ w_i z_i / Σ w_i`, `w_i = 1 / d_i^p`, `k = 12` neighbours.
- Power tuned over {1, 1.5, 2, 2.5, 3} by leave-one-out CV.
- Extension: Ordinary Kriging on the **same** LOO split.

| Method | RMSE | R² |
|---|---|---|
| IDW (p=3) | 1.111 | 0.852 |
| Kriging (spherical) | 0.889 | 0.905 |

> *Note:* Kriging wins on a smooth field; IDW is the cheaper default.

---

## Slide 7 — Method: statistics

- Pearson and Spearman correlation per covariate.
- OLS `temp ~ ndvi + builtup + dist_water`, with VIF and Moran's I.
- Built-up vs green: Welch's t-test, Mann-Whitney U, Cohen's d.

| Term | Coef | VIF |
|---|---|---|
| NDVI | -3.08 | 2.60 |
| built-up | +8.21 | 2.75 |

R² = 0.911 · max VIF 2.75 · Moran's I (residuals) 0.310

> *Note:* VIF is fine; Moran's I says p-values are optimistic — say so.

---

## Slide 8 — Where is it hot?

- Max UHI intensity: **+9.1 C**, mean +0.7 C.
- 598 hotspot cells (23.9%), 809 coldspots, at 95% confidence.
- Area above +2 C: 24.8% of the city; above +4 C: 8.3%.
- Hotspots sit on the built-up core and avoid the park and lake.

> *Note:* Show the hotspot figure here.

---

## Slide 9 — Why is it hot?

- Built-up cells are **+4.4 C** hotter than green cells (Cohen's d = 2.1).
- Regression: built-up warms (+8.2 C/unit), NDVI cools (-3.1 C/unit).
- The generator's known signs are recovered — a tested guarantee.

> *Note:* Tie back to the objectives: this is the "why".

---

## Slide 10 — Where to plant first?

- Priority = 0.4·temp + 0.3·low NDVI + 0.2·built-up + 0.1·pop proxy.
- Top wards: **W10, W17, W11** — hot, bare, dense.
- Top cells: built-up, NDVI near zero, UHI above +8 C.
- Estimated cooling for +0.1 NDVI: ~0.31 C (an estimate).

> *Note:* This is the planning payoff; the dashboard sliders make it interactive.

---

## Slide 11 — Testing and quality

- 111 tests, **93% coverage**.
- Hand-checked 3x3 raster cases and IDW exactness at sample points.
- Synthetic-truth test: regression recovers the built-in signs.
- CRS guards raise on degree coordinates.
- Smoke test drives every dashboard page.

> *Note:* Quality is a deliverable, not an afterthought.

---

## Slide 12 — Conclusion and future work

**Done:** full pipeline, database + GeoPackage, heat map and hotspots,
dashboard, auto report, 93% test coverage.

**Next:** regression kriging, Random-Forest downscaling, multi-season analysis,
planting-scenario simulation, population-aware planning.

> *Note:* Close on the honest caveat: LST is surface temperature, and residual
> autocorrelation is flagged and deferred to a spatial model.
