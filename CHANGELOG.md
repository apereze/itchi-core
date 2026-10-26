# Changelog

All notable changes to `itchi-core` will be documented in this file.

The format follows the general structure of [Keep a Changelog](https://keepachangelog.com/),
and this project uses semantic versioning principles while the API is under active development.

---

## [0.2.0] - 2026-10-26

ITCHI v0.2 keeps the v0.1 structure (direct, indirect and exterior regions;
bounded unions) and changes the precipitation and wind hazards. All v0.2
options are opt-in in the pipeline; the defaults reproduce v0.1.

### Added

- `precipitation.compute_precipitation_hazard_ramp`: general node ramp
  (R3 = Q90/Q95/Q99 reproduces v0.1; `RAMP_R4_NODES` = Q75/Q90/Q95/Q99).
- `precipitation.compute_precipitation_matrix_hazard`: continuous P6 x A72
  matrix (`PRECIP_MATRIX_V02`). Columns anchored at Q75/Q90/Q95 of the 72-h
  climatology, interpolated linearly in A72; without antecedent rain it is R4.
  Node values are provisional pending calibration.
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

### Changed

- Version 0.2.0 (`pyproject.toml`, `constants.ITCHI_VERSION`, `configs/default.yaml`).
- `configs/default.yaml` documents the v0.2 matrix and wind tail.
- With `wind_model="w06_tail"`, tropical depressions get wind hazard below
  h34 (v0.1 normalized V/Vmax inside the fallback direct radius).

### Notes

- The matrix nodes and h34 = 0.2 are provisional; calibrate them with the
  municipality-cyclone pairs (POD, FAR, Peirce) before operational use.
- 72-h P99/P99.9 are not robust (overlapping windows); the matrix only uses
  Q75, Q90 and Q95 of A72.

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

### Near-term

* Add GitHub Actions continuous integration.
* Add a formal `docs/methodology.md`.
* Add a first real-case workflow with one tropical cyclone.
* Add notebook for single-snapshot diagnostics.
* Add notebook for single-event synthetic diagnostics.
* Add examples for NetCDF output writing.

### Future

* Add climatology utilities for Q90/Q95/Q99.
* Add precipitation snapshot alignment utilities.
* Add real-data adapters for MSWEP, IBTrACS and ROCLOUD.
* Add validation routines using observed disaster/emergency records.
* Add performance optimization for large `xarray`/`dask` grids.
