# Demo script

A short, reliable walkthrough for the viva or a recorded demo. Timings assume a
laptop that has already run `pip install -r requirements.txt`. Total: about
6 minutes.

---

## 0. Setup (before the demo, one time)

```bash
python -m src.pipeline --source synthetic   # ~16 s, pre-builds every artefact
python -m pytest tests/ -q                  # 111 passed
```

Do this once so the live demo never waits on a cold run.

---

## 1. The pitch (45 s)

> "Cities run hotter than the countryside because concrete stores heat and
> vegetation cools it. This project maps that heat island for a city, explains
> it with land cover and NDVI, and ranks the places where planting trees would
> help most. It runs end-to-end with no downloads, and it is tested."

Say the three questions the project answers: where it is hot, why it is hot, and
where to plant first.

---

## 2. Run the pipeline live (90 s)

```bash
python -m src.pipeline --source synthetic
```

Narrate the phases as they scroll past:

* **Phase 1-2** — synthetic city generated, inputs validated, 100 m grid built.
* **Phase 3** — each sample gets NDVI, built-up fraction, land cover, distance to
  water.
* **Phase 4** — correlation, OLS regression, VIF, Moran's I, group tests.
* **Phase 5** — IDW with the power chosen by leave-one-out CV, then the same
  split scores kriging.
* **Phase 6-7** — UHI intensity, Gi* hotspots, tree-planting priority.
* **Phase 8-9** — GeoTIFFs, figures, dashboard payload, auto report.

Point at two lines: `R2=0.911` and `better method on LOO: Kriging`. That is the
headline result and the honest comparison in one screen.

---

## 3. Show the outputs (60 s)

```bash
ls outputs
```

Open `outputs/report.md`. Walk through:

1. **Headline table** — 197 samples, 2,500 cells, R² 0.91, max UHI +9.1 C.
2. **Section 5.1** — IDW vs kriging table, kriging wins on RMSE (0.89 vs 1.11).
3. **Section 7** — the top-ward priority table: W10, W17, W11.

Open one figure, e.g. `outputs/figures/hotspots.png`, and note the hotspots sit
on the built-up core and avoid the park and lake.

---

## 4. The dashboard (120 s)

```bash
streamlit run app/dashboard.py
```

Click through the seven pages:

1. **Overview** — the interpolated heat surface, sample points, KPI cards.
2. **Layers** — toggle temperature, NDVI, built-up, land cover, hotspots.
3. **Relationships** — temperature vs NDVI and vs built-up, correlation table,
   regression summary, residual map. Note the residual Moran's I caveat.
4. **Interpolation** — power vs CV error chart and the IDW-vs-kriging bar chart.
5. **Built-up vs Green** — box plot and the Welch / Mann-Whitney results.
6. **Priority** — drag the weight sliders and watch the ranking recompute live.
   This is the "so what" of the project.
7. **Download** — GeoTIFF, GeoPackage and CSV tables.

---

## 5. Prove it is tested (45 s)

```bash
python -m pytest tests/ -q --cov=src --cov-report=term-missing
```

Point out `111 passed` and `TOTAL ... 93%`. Mention two tests by name:

* `test_synthetic_truth_recovers_coefficient_signs` — the regression recovers the
  generator's built-in signs.
* `test_idw_is_exact_at_sample_points` — IDW returns the observed value where a
  sample sits.

---

## 6. Real data and close (60 s)

Explain that real Landsat LST samples, Sentinel-2 NDVI, ESA WorldCover and OSM
buildings drop into the same schemas and the same CLI:

```bash
python -m src.pipeline --source real
```

Close on the honest caveats and future work: LST is surface not air temperature;
residual autocorrelation means OLS p-values are optimistic; kriging and
regression kriging, Random-Forest downscaling and scenario simulation are the
next steps.

---

## Backup answers

* **"Why IDW and not just kriging?"** IDW is cheap, needs no variogram fitting,
  and on this data gives the same planning conclusion. We benchmark both on the
  same LOO split and report which wins.
* **"Why is Moran's I positive?"** Neighbouring cells are more alike than chance;
  the OLS standard errors are optimistic, so we flag it and propose a spatial
  model.
* **"How do you know the regression is right?"** The synthetic generator plants
  known coefficients and a test checks the recovered signs and magnitudes.
* **"What if a file is missing?"** Ingestion validates schemas and rejects bad
  CRS, missing columns and out-of-range temperatures before anything runs.
