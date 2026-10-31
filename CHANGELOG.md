# Changelog

All notable changes to `itchi-core` will be documented in this file.

The format follows the general structure of [Keep a Changelog](https://keepachangelog.com/),
and this project uses semantic versioning principles while the API is under active development.

---

## [0.2.1] - 2026-10-30

Consistency fixes, a single v0.2 entry point and the Block 1 experiments
(P95 as maximum, R24/A72 thresholds, wind-tail cut). The v0.2 formulation
(matrix, W06 tail, h34) is unchanged and remains the default; the
attribution radius changes from R_W06 (verification studies) to ROCLOUD with
an R_W06 floor. Next development line: v0.3 (`docs/plan_v0.3.md`).

### Added

- `pipeline.compute_itchi_snapshot_v02`: ITCHI v0.2 from P6, R24, A72 and
  their 6/24/72-h percentile climatologies. Computes the precipitation
  matrix, the direct radius (observed R34; RMW or parametric R34 as
  fallback), R_W06, the attribution radius and the W06 wind with a tail that
  ends at R_W06. Reports `H_P_missing`, radii and status by quadrant,
  `itchi_formulation = "v0.2"`. Keyword defaults equal `configs/default.yaml`.
- `radii.resolve_attribution_radius_with_w06` and `AttributionRadii`: role
  of R_W06 in the attribution radius by quadrant, modes `"never"`,
  `"fallback"`, `"floor"` (default) and `"always"`; ROCLOUD <= 0 counts as
  no detection; never below R_direct_q.
- `parametric_radii.compute_w06_reference_radii` and `R_W06_DEFAULTS`:
  R_W06 (V = 2 m/s, PA21 configuration), parametric R34 and Rmax of one
  record; same as `pa21_r2_km` of the v0.2 verification.
- `parametric_radii.compute_w06_wind_hazard_with_tail(tail_end_wind_kt=...)`:
  optional cut of the tail at a wind speed (decision D1); the tail never
  extends beyond the tail-end radius. Default unchanged.
- `config.itchi_v02_settings_from_config`: keyword arguments of the v0.2
  entry point from a configuration.
- `climatology.load_percentile_climatology` and
  `climatology.quantile_dataarray_to_dict`: read the
  `mswep_v3_{6,24,72}h_percentiles_may_nov.nc` files as {"Q75": ..., ...}.
- `tests/test_v02_snapshot.py`, including a test that the defaults of the
  v0.2 entry point equal `configs/default.yaml` and a test that the entry
  point reproduces the setting of the v0.2 studies.

- Block 1 (`docs/experimento_b1.md`): `precipitation.compute_precipitation_class_table_hazard`,
  a class table P6 x R24 x A72 against each variable's own percentiles,
  interpolated multilinearly (non-decreasing in every input); spec by base
  ramp and rules or explicit table, optional P75 gate.
  `PRECIP_TABLE_V02` reproduces the v0.2 matrix to 1e-12;
  `PRECIP_TABLES_B1` holds the candidates with P95 as maximum (M0 6 h only,
  M1 v0.2 escalation one class earlier, M2 gated maximum of durations).
  Helpers `compute_percentile_coordinate`, `build_precipitation_class_table`,
  `compute_precipitation_hazard_from_spec`.
- `compute_itchi_snapshot_v02` accepts a class-table spec as `precip_matrix`
  (`precipitation_hazard_source = "class_table"`).
- Configuration: `precipitation.hazard_model: "class_table"` with
  `precipitation.class_table`; `wind.tail_end_wind_ms`;
  `config.load_experiment_config` and `config.merge_config` (overlays);
  `configs/experiments/b1/p{0,1}_w{0,1}.yaml` (2 x 2 design: v0.2 / M1 x
  tail to R_W06 / cut at 12 m/s).
- `scripts/experimento_b1.py`: candidates and 2 x 2 on the
  municipality-cyclone pairs (cell table P(D | P6 x R24 x A72), M1 threshold
  grid and ablations, M3 calibrated table with leave-one-year-out
  validation, wind variants, AUC by year, PSS, SEDI, bootstrap by cyclone,
  POD at equal POFD).
- `scripts/negativos_w06.py --guardar-pasos`: per-step table for the
  experiment; `lat_ct` added to the steps.
- `scripts/figuras_b1.py` and `notebooks/third test/09_bloque1_experimentos.ipynb`:
  figures of each experiment (part A, no data: hazard of each scheme, class
  maps P6 x R24, wind tails; part B: cell evidence, candidates, ROC,
  threshold grid, 2 x 2, class frequencies) and the decision table.
- `tests/test_class_table.py`, `tests/test_experimento_b1.py`.
- `scripts/declaratorias_6h.local_path`: `D:/...` paths also work from WSL;
  `negativos_w06.py` stops with a clear message if no 6-h file is read.

### Changed

- `configs/default.yaml`: header states which function uses it;
  precipitation as a 6-h accumulation ending at T (`precipitation_window`,
  `temporal_alignment: "window_end"`, tolerance 0 h) instead of the v0.1
  snapshot fields; `climatology_variable: "p_wet_raw"` (the variable of all
  v0.2 results); `geometry.attribution_floor: "R_W06"`;
  `parametric_radii.attribution_radius_mode: "floor"` with the documented
  modes; new `parametric_radii.r_w06`, `wind.tail_end_radius`,
  `wind.tail_end_wind_kt`, `precipitation.matrix_v02.nan_hazard_value` and
  `geometry.radius_fill_strategy`.
- README: v0.2 entry point and attribution rule; `compute_itchi_snapshot_from_grid`
  documented as keeping the v0.1 defaults.
- CHANGELOG: "Planned" section updated to the current state.

### Fixed

- The configuration declared `matrix_v02` and `w06_tail` as defaults while
  the pipeline defaulted to v0.1 and did not read it; v0.2 now has its own
  entry point whose defaults are tested against the file.
- The v0.2 precipitation matrix had to be computed outside the pipeline and
  passed as `precipitation_hazard`.


### Notes (Block 1 results, `docs/experimento_b1.md` section 9)

- 381 929 municipality-cyclone pairs, 2 % with declaration, index evaluated
  at every step of the event.
- P95 of 6 h as maximum (M0) does not improve discrimination: PSS class 5
  0.553 -> 0.546 (90 % CI of the difference [-0.023, +0.008]), POD at equal
  POFD -0.004 (same ROC curve), AUC 0.869 -> 0.842, +61 % class-5 alerts.
  Not adopted.
- R24/A72 below P95 add little: best +0.006 PSS over M0 (R24 >= P95 alone);
  A72 >= P75 alone -0.008. Above P95 they matter: P(D) with P6 >= P95 and
  R24 >= P95 goes from 5.7 % (A72 < P75) to 21.8 % (A72 >= P95).
- Wind tail cut: 12 m/s restores class 1 (PSS 0.000 -> 0.472) with AUC
  0.887; 17 kt gives 0.480 and 0.895. Rain and wind effects are additive.
- Declarations are a noisy, coarse impact record; from v0.3 the hazard is
  defined climatologically (return periods, absolute wind thresholds) and
  declarations are used only as an ordinal check.
---

## [0.2.0] - 2026-10-26

ITCHI v0.2 keeps the v0.1 structure (direct, indirect and exterior regions;
bounded unions) and changes the precipitation and wind hazards. All v0.2
options are opt-in in the pipeline; the defaults reproduce v0.1.

### Added

- `precipitation.compute_precipitation_hazard_ramp`: general node ramp
  (R3 = Q90/Q95/Q99 reproduces v0.1; `RAMP_R4_NODES` = Q75/Q90/Q95/Q99).
- `precipitation.compute_precipitation_matrix_hazard`: P6 ramp R4 escalated
  towards Q95 -> 0.8 by persistence (R24 from Q90 to Q95 of its climatology)
  and antecedent rain (A72 from 0 to Q75), weights combined as a product
  (`PRECIP_MATRIX_V02`). Continuous form of "P95-P99 with R24 >= P95 and
  A72 >= P75 -> maximum", calibrated with municipality-cyclone pairs
  (maximum-level PSS 0.413 -> 0.514; 0.550 -> 0.670 for disasters).
- `precipitation.compute_antecedent_precipitation`: exclusive antecedent
  A_d(T) over (T-d-6h, T-6h] or running sum R_d(T), NaN across gaps.
- `wind.normalize_wind_hazard_with_tail` and
  `parametric_radii.compute_w06_wind_hazard_with_tail`: W06 wind calibrated to
  the observed mean R34, quadrant-normalized radius (V = 34 kt at each
  R_direct_q), hazard h34 + (1 - h34)(V - 34)/(Vmax - 34) inside R_direct and a
  tail h34 (V - V_out)/(34 - V_out) that vanishes at the attribution radius.
- `components`: optional indirect wind; H_ind = 1 - (1 - H_Wind)(1 - H_Pind).
  Outputs `H_Wdir` and `H_Wind` when enabled.
- `pipeline.compute_itchi_snapshot_from_grid`: `precipitation_hazard`,
  `wind_model` ("v01" | "w06_tail"), `wind_tail_h34`, `r34_mean_km`,
  `w06_profile_kwargs`; result reports `itchi_version`,
  `precipitation_hazard_source` and W06 metadata.
- `tests/test_v02.py`.
- Scripts: MSWEP V3.16 6-h accumulations and 6/24/72-h percentiles with
  effective-sample sufficiency (`scripts/percentiles_6h.py --durations`),
  TC-associated disaster declarations vs 6/24/72-h rain
  (`declaratorias_6h.py`), and municipality-cyclone verification pairs inside
  the W06 (PA21) radius with positives and negatives (`negativos_w06.py`).
- Notebooks (`notebooks/second test/`): `05_itchi_v02_mejoras` (Cristóbal 2020
  and Odile 2014, v0.1 vs v0.2), `06_verificacion_metricas_figuras`
  (verification metrics explained and all figures of the national
  verification), `07_itchi_v02_peligro_por_snapshot` (hazard by snapshot,
  attribution of changes by step, duration and timing of alerts) and
  `08_itchi_v01_v02_mapas_por_posicion` (v0.1 vs v0.2 maps at every position).
- `docs/verification_v02.md`: data, design, metrics, results, case studies,
  limitations, reproduction and open decisions for v0.3.

### Changed

- Version 0.2.0 (`pyproject.toml`, `constants.ITCHI_VERSION`, `configs/default.yaml`).
- `configs/default.yaml` documents the v0.2 matrix and wind tail.
- README, `docs/methodology.md`, `docs/declaratorias.md`, `notebooks/README.md`
  and `.gitignore` updated for v0.2.
- With `wind_model="w06_tail"`, tropical depressions get wind hazard below
  h34 (v0.1 normalized V/Vmax inside the fallback direct radius).

### Notes

- The escalation rule was selected among five candidates on the same pairs;
  cross-validation by year/storm and bootstrap intervals are pending.
  h34 = 0.2 is provisional.
- Declaration-cyclone association is by dates (nearest active cyclone);
  about 30 % of declared pairs were mis-associated and excluded.
- 72-h P99/P99.9 are not robust (overlapping windows) and are not used.
- At the maximum level the escalation raises PSS and SEDI but lowers CSI, ETS
  and HSS (frequency bias 3.1 -> 4.7): it trades false alarms for detection on
  purpose, because declarations under-report impacts.
- The default wind tail (to the attribution radius) gives wind hazard to 96 %
  of the verification pairs and removes the skill of SIAT class 1
  (PSS 0.45 -> 0.03); cutting it at 17 kt keeps the wind information
  (AUC 0.85) and restores class 1 (PSS 0.49). Open decision D1 for v0.3.
- Case studies (Cristóbal 2020, Odile 2014): POD = 1 in v0.1 and v0.2, so the
  event-level PSS can only reflect the extent of the alert; v0.2 does not
  improve it. Odile has 6 municipalities with a declaration.
- Open decisions for v0.3 are listed in `docs/verification_v02.md`.

---

## [0.1.0] - Initial development version

### Added

- Initial repository structure for `itchi-core`.
- Python package structure under `src/itchi/`.
- Project configuration with:
  - `pyproject.toml`
  - `environment.yml`
  - `.pre-commit-config.yaml`
  - `.gitignore`

### Core modules

- Added `constants.py` for shared constants and canonical labels.
- Added `config.py` for configuration loading utilities.
- Added `units.py` for unit conversion and quadrant-key normalization.
- Added `geometry.py` for cyclone-centered distance and quadrant calculations.
- Added `precipitation.py` for precipitation-hazard normalization.
- Added `masks.py` for direct, indirect and exterior region masks.
- Added `components.py` for direct and indirect hazard components.
- Added `index.py` for final ITCHI calculation.
- Added `aggregation.py` for event-level products:
  - `ITCHI_max`
  - `ITCHI_acc`
- Added `radii.py` for quadrant-radius resolution:
  - partial `R34` filling;
  - partial `ROCLOUD` filling;
  - `R_direct_q` resolution;
  - `RMW` fallback for tropical depressions without `R34`.
- Added `quality_control.py` for consistency checks.
- Added `io.py` for NetCDF/Zarr input-output utilities.
- Added `tracks.py` for tropical cyclone track table handling.
- Added `rocloud.py` for ROCLOUD table handling.
- Added `wind.py` for normalized wind-hazard calculation.
- Added `pipeline.py` for snapshot-level ITCHI computation.
- Added `compiler.py` for event-level orchestration.

### Methodology

- Defined precipitation as snapshot-based rather than accumulated.
- Defined ITCHI as a bounded index in `[0, 1]`.
- Defined direct, indirect and exterior regions.
- Introduced `R_direct_q` as the effective direct-region radius.
- Distinguished observed `R34_q` from effective `R_direct_q`.
- Added support for missing quadrant radii using available-quadrant mean.
- Added support for tropical depression cases using `RMW` as direct-radius fallback.
- Added optional internal computation of wind hazard from `Vmax` and `RMW`.

### Tests

- Added unit tests for:
  - precipitation hazard;
  - masks;
  - components;
  - index calculation;
  - geometry;
  - units;
  - aggregation;
  - radii;
  - quality control;
  - input-output;
  - tracks;
  - ROCLOUD;
  - wind;
  - pipeline;
  - compiler.

### Documentation

- Added initial `README.md`.
- Added `docs/architecture.md`.
- Added `docs/api.md`.
- Added synthetic validation notebook:
  - `notebooks/00_smoke_test_synthetic.ipynb`
- Added executable synthetic example:
  - `examples/smoke_test_synthetic.py`
- Added `examples/README.md`.

### Development workflow

- Added formatting and linting workflow through `pre-commit`.
- Added support for local testing with `pytest`.
- Added development installation using:

```bash
pip install -e .
````

---

## Planned

### Done since v0.1

- GitHub Actions workflow, `docs/methodology.md`, real cases (Willa 2018,
  Cristóbal 2020, Odile 2014), MSWEP V3.16 adapters and 6/24/72-h
  climatology, validation against disaster declarations.

### v0.2.1 (experiments on top of `compute_itchi_snapshot_v02`)

* Timing convention of P6/R24/A72 against IBTrACS (late alerts in Odile).
* Declaration–cyclone association by storm name (D5).
* Information added by the escalation at equal POFD; cross-validation by
  year and bootstrap by cyclone (D6).
* Wind-tail cut and h34 in the full pipeline (D1, D2).
* Hours-in-alert event product (D7).
* New cyclone cases, risk cuts, landslide records, lead time of the alert
  relative to the declaration, ITCHI aggregated by municipality.

### Later (v0.3)

* Brief vs persistent rain at >= P99 (D4), soil moisture instead of A72 (D8),
  wind normalization (D9).
* Performance optimization for large `xarray`/`dask` grids.
