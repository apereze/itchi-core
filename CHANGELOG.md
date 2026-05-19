# Changelog

All notable changes to `itchi-core` will be documented in this file.

The format follows the general structure of [Keep a Changelog](https://keepachangelog.com/),
and this project uses semantic versioning principles while the API is under active development.

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
