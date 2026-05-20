# ITCHI Core

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-itchi--core-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](tests/)
[![License: TBD](https://img.shields.io/badge/License-TBD-yellow.svg)](#license)

**Integrated Tropical Cyclone Hazard Index**

Repository for the development of the computational core of **ITCHI v0.1**, a physically based tropical-cyclone hazard index.

---

## Table of contents

- [Quick start](#quick-start)
- [Overview](#overview)
- [Repository scope](#repository-scope)
- [Methodological principle](#methodological-principle)
- [Precipitation treatment](#precipitation-treatment)
- [Temporal resolution](#temporal-resolution)
- [Precipitation normalization](#precipitation-normalization)
- [Index components](#index-components)
- [Conceptual formula](#conceptual-formula)
- [Computational workflow](#computational-workflow)
- [Repository structure](#repository-structure)
- [Expected inputs](#expected-inputs)
- [Expected outputs](#expected-outputs)
- [Current project status](#current-project-status)
- [Installation](#installation)
- [Reproducible examples](#reproducible-examples)
- [Tests and quality control](#tests-and-quality-control)
- [Data requirements](#data-requirements)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Authorship](#authorship)
- [License](#license)
- [References](#references)

---

## Quick start

```bash
# Clone the repository
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

# Create and activate environment
conda env create -f environment.yml
conda activate itchi

# Install package in development mode
pip install -e .

# Verify installation
python -c "import itchi; print('ITCHI installed successfully')"

# Run tests
python -m pytest tests/

# Run synthetic smoke test
python examples/smoke_test_synthetic.py
```

Run the table-driven workflow:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

Read and validate an exported compiled event:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

Optional diagnostic figure:

```bash
python examples/smoke_test_from_tables.py \
  --plot \
  --figure-path outputs/figures/smoke_test_from_tables.png \
  --output-path outputs/events/smoke_test_from_tables.nc
```

The `outputs/` directory is ignored by Git.

---

## Overview

**ITCHI** stands for **Integrated Tropical Cyclone Hazard Index**.

The initial version, **ITCHI v0.1**, is designed as a continuous physically based index bounded between `0` and `1`. It represents tropical-cyclone hazard over a spatial grid and at cyclone-centered synoptic times.

The index integrates three main hazard sources:

1. **Direct precipitation**, associated with rainfall inside the cyclone direct region.
2. **Wind**, estimated from a normalized wind-hazard field.
3. **Indirect precipitation**, associated with external rainbands inside the cyclone-attributable precipitation region.

The objective of this repository is to build the ITCHI calculation in a modular, traceable, reproducible and scientifically explicit way.

---

## Repository scope

This repository contains the computational elements required to build the **ITCHI core index**.

### Included

- Reading and standardization of tropical-cyclone track tables.
- Reading and cleaning of wind radii, especially `R34`.
- Resolution of the effective direct radius `R_direct`.
- Treatment of tropical depressions without `R34` using `RMW` or an explicit fallback.
- Reading and cleaning of external structural radii such as `ROCLOUD`.
- Snapshot-based precipitation selection.
- Reading and writing scientific products with `xarray`.
- Normalized precipitation-hazard calculation.
- Normalized wind-hazard calculation or ingestion.
- Construction of spatial masks:
  - direct region;
  - indirect region;
  - exterior region.
- Computation of physical hazard components.
- Final `ITCHI` calculation.
- Physical and computational quality control.
- Compilation of multiple snapshots into an event.
- Event-level derived products: `ITCHI_max` and `ITCHI_acc`.
- Synthetic scripts and notebooks for validation.
- Table-driven synthetic workflows that approximate the future real-data workflow.

### Not included yet

- Validation against emergency or disaster declarations.
- Machine-learning models.
- Advanced statistical calibration.
- Final publication-quality visualization products.
- Dashboard or web application.
- Full scientific manuscript.

These elements may be developed later in separate modules, notebooks or repositories.

---

## Methodological principle

ITCHI v0.1 distinguishes three regions around a tropical cyclone:

| Region | Conceptual condition | Interpretation |
|---|---|---|
| Direct region | `r <= R_direct_q` | Direct hazard region driven by wind and precipitation |
| Indirect region | `R_direct_q < r <= ROCLOUD_q` | External precipitation region attributable to the cyclone |
| Exterior region | `r > ROCLOUD_q` | Region with no contribution to the index |

The conceptual separation is:

```text
Direct hazard = wind + precipitation inside R_direct
```

```text
Indirect hazard = precipitation between R_direct and ROCLOUD
```

The exterior region does not contribute to the index.

### `R34` versus `R_direct`

The repository explicitly distinguishes between:

| Variable | Meaning |
|---|---|
| `R34` | Observed 34-kt wind radius |
| `R_direct` | Effective radius used to define the direct region |
| `RMW` | Radius of maximum wind |
| `ROCLOUD` | External cloud/precipitation attribution radius |

Main rules:

```text
If R34 exists:
    R_direct_q = R34_q

If R34 is partial:
    missing R34 quadrants are filled from available quadrants

If R34 does not exist and Vmax < 34 kt:
    R_direct_q = RMW

If neither R34 nor RMW is available:
    fallback_direct_radius_km is used, if explicitly configured
```

This avoids assigning artificial `R34` values to tropical depressions that physically do not reach 34-kt winds.

---

## Precipitation treatment

In ITCHI v0.1, precipitation is treated as a **snapshot**, not as a temporal accumulation.

This means the precipitation field represents the precipitation state or intensity at a given valid time. ITCHI therefore does not assume that precipitation should be accumulated from 3-hourly to 6-hourly fields.

This decision keeps the framework compatible with:

- historical MSWEP products;
- numerical forecasts;
- hindcasts;
- downscaled products;
- bias-corrected products.

Critical rule:

```text
Do not sum precipitation across valid_time unless a future product explicitly defines accumulated precipitation as the intended input.
```

---

## Temporal resolution

The index is calculated at cyclone-associated synoptic times:

```text
00, 06, 12 and 18 UTC
```

This maintains temporal consistency among:

- cyclone track;
- intensity;
- wind radii;
- `R34`;
- `RMW`;
- `ROCLOUD`;
- radial wind profile;
- precipitation snapshot.

---

## Precipitation normalization

Precipitation hazard is computed using local climatological percentiles:

| Percentile | Interpretation |
|---|---|
| `P90` / `Q90` | Onset of low or occasional hazard |
| `P95` / `Q95` | High precipitation hazard |
| `P99` / `Q99` | Maximum precipitation hazard threshold |

The normalization should be computed by:

- spatial cell;
- month;
- synoptic hour or comparable valid-time window.

The piecewise function is:

```text
P < Q90        -> H_P = 0
Q90 <= P < Q95 -> H_P increases from 0 to 0.5
Q95 <= P < Q99 -> H_P increases from 0.5 to 1
P >= Q99       -> H_P = 1
```

Important methodological interpretation:

```text
Q95 is not saturation.
Q99 is the saturation threshold.
```

---

## Index components

| Component | Description |
|---|---|
| `H_P` | Normalized precipitation hazard |
| `H_W` | Normalized wind hazard |
| `H_Pdir` | Direct precipitation inside `R_direct` |
| `H_Pind` | Indirect precipitation between `R_direct` and `ROCLOUD` |
| `H_dir` | Direct component |
| `H_ind` | Indirect component |
| `ITCHI` | Final integrated index |

---

## Conceptual formula

Direct component:

```text
H_dir = 1 - (1 - H_W)^alpha * (1 - H_Pdir)^beta
```

Indirect component:

```text
H_ind = H_Pind
```

Final index:

```text
ITCHI = 1 - (1 - H_dir)^lambda_direct * (1 - H_ind)^mu_indirect
```

Base version:

```text
alpha = beta = lambda_direct = mu_indirect = 1.0
```

---

## Computational workflow

```text
Cyclone track
    center_lon, center_lat, vmax_kt, rmw_km, R34
        ↓
ROCLOUD by quadrant
        ↓
snapshot_inputs.py
    table records + precipitation snapshot + climatology
        ↓
radii.py
    resolution of R_direct_q and ROCLOUD_q
        ↓
geometry.py
    radius_km, quadrant
        ↓
precipitation.py
    H_P from precipitation snapshot and Q90/Q95/Q99
        ↓
wind.py
    V* provided or computed from Vmax/RMW
        ↓
masks.py
    M_direct, M_indirect, M_exterior
        ↓
components.py
    H_Pdir, H_Pind, H_W, H_dir, H_ind
        ↓
index.py
    ITCHI_g,h,t
        ↓
quality_control.py
    physical and structural validation
        ↓
compiler.py + aggregation.py
    snapshots, metadata, ITCHI_max, ITCHI_acc
        ↓
io.py
    NetCDF/Zarr export and product reading
```

---

## Repository structure

```text
itchi-core/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── docs/
│   ├── architecture.md
│   ├── data_contracts.md
│   └── workflows.md
│
├── examples/
│   ├── README.md
│   ├── read_compiled_event.py
│   ├── smoke_test_from_tables.py
│   └── smoke_test_synthetic.py
│
├── notebooks/
│   ├── 00_smoke_test_synthetic.ipynb
│   └── 01_read_compiled_event.ipynb
│
├── src/
│   └── itchi/
│       ├── __init__.py
│       ├── aggregation.py
│       ├── climatology.py
│       ├── compiler.py
│       ├── components.py
│       ├── config.py
│       ├── constants.py
│       ├── geometry.py
│       ├── index.py
│       ├── io.py
│       ├── masks.py
│       ├── pipeline.py
│       ├── precipitation.py
│       ├── precipitation_snapshots.py
│       ├── quality_control.py
│       ├── radii.py
│       ├── rocloud.py
│       ├── snapshot_inputs.py
│       ├── tracks.py
│       ├── units.py
│       └── wind.py
│
└── tests/
    ├── test_aggregation.py
    ├── test_climatology.py
    ├── test_compiler.py
    ├── test_components.py
    ├── test_geometry.py
    ├── test_index.py
    ├── test_io.py
    ├── test_masks.py
    ├── test_pipeline.py
    ├── test_precipitation.py
    ├── test_precipitation_snapshots.py
    ├── test_quality_control.py
    ├── test_radii.py
    ├── test_rocloud.py
    ├── test_snapshot_inputs.py
    ├── test_tracks.py
    ├── test_units.py
    └── test_wind.py
```

---

## Expected inputs

Minimum inputs required to calculate ITCHI:

1. Tropical-cyclone track data.
2. Cyclone center by time: `center_lon`, `center_lat`.
3. Intensity: `vmax_kt`.
4. Radius of maximum wind: `rmw_km`, when available.
5. Wind radii, especially `R34`.
6. External `ROCLOUD` radii.
7. Snapshot precipitation field.
8. Local climatological percentiles: `Q90`, `Q95`, `Q99`.
9. Normalized wind field `V*` or parameters to construct it.

Detailed input contracts are available in [`docs/data_contracts.md`](docs/data_contracts.md).

---

## Expected outputs

| Output | Description |
|---|---|
| `ITCHI_g,h,t` | Index by grid cell and time |
| `H_P` | Precipitation hazard |
| `H_W` | Wind hazard |
| `H_Pdir` | Direct precipitation component |
| `H_Pind` | Indirect precipitation component |
| `H_dir` | Direct component |
| `H_ind` | Indirect component |
| `M_direct` | Direct-region mask |
| `M_indirect` | Indirect-region mask |
| `M_exterior` | Exterior-region mask |
| `ITCHI_max` | Event-level maximum |
| `ITCHI_acc` | Bounded accumulation or persistence across event snapshots |

Compiled events may be exported as NetCDF or Zarr with spatial fields, event products and serialized metadata.

---

## Current project status

| Component | Status | Description |
|---|---|---|
| Base repository | Done | Repository, environment, configuration and installable package |
| Core modules | Done | Precipitation, geometry, masks, components and index |
| Radii and units | Done | Unit conversion, quadrants, `R_direct`, `ROCLOUD` |
| Wind | Done | Simple radial profile and `V*` normalization |
| Snapshot precipitation | Done | Valid-time snapshot selection without accumulation |
| Climatology utilities | Done | Q90/Q95/Q99 computation and extraction utilities |
| Integrated pipeline | Done | Snapshot-level calculation with optional quality control |
| Aggregation | Done | `ITCHI_max` and `ITCHI_acc` |
| Compiler | Done | Multi-snapshot event orchestration |
| I/O | Done | `xarray.Dataset` conversion, reading and writing |
| Tracks and ROCLOUD | Done | Tabular standardization and radius extraction |
| Snapshot input builders | Done | Track + ROCLOUD + precipitation assembly |
| Unit tests | Done | Module-level coverage with `pytest` |
| Synthetic scripts | Done | Direct and table-driven smoke tests |
| Product reader | Done | Independent compiled-event validation |
| Notebooks | Done | Synthetic calculation and compiled-event inspection |
| Data contracts | Done | Formal input/output documentation |
| Workflow documentation | Done | Reproducible execution guide |
| Real cyclone case | Pending | Integration with real cyclone data |
| Formal gridded climatology workflow | Pending | Robust Q90/Q95/Q99 loading and alignment |
| Impact validation | Pending | Validation against observed impacts or disaster declarations |

---

## Installation

### Requirements

- Python 3.11 or higher
- Conda or pip
- Git

### Option 1: Conda

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

conda env create -f environment.yml
conda activate itchi

pip install -e .
```

### Option 2: pip/venv

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Verify installation

```bash
python -c "import itchi; print('ITCHI installed successfully')"
python -m pytest tests/
pre-commit run --all-files
```

---

## Reproducible examples

### Direct synthetic smoke test

```bash
python examples/smoke_test_synthetic.py
```

Export compiled event:

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc
```

Diagnostic figure:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

### Table-driven synthetic smoke test

```bash
python examples/smoke_test_from_tables.py
```

Export compiled event:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

Validate exported event:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

### Synthetic notebook

```bash
jupyter lab notebooks/00_smoke_test_synthetic.ipynb
```

### Compiled-event inspection notebook

```bash
jupyter lab notebooks/01_read_compiled_event.ipynb
```

For the complete workflow sequence, see [`docs/workflows.md`](docs/workflows.md).

---

## Tests and quality control

Run all tests:

```bash
python -m pytest tests/
```

Run pre-commit:

```bash
pre-commit run --all-files
```

Run specific module tests:

```bash
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_compiler.py
python -m pytest tests/test_snapshot_inputs.py
```

Consistency rules verified by the test suite and quality-control utilities:

- `ITCHI` remains in `[0, 1]`.
- `H_P`, `H_W`, `H_dir`, `H_ind`, `H_Pdir` and `H_Pind` remain in `[0, 1]`.
- `R_direct_q <= ROCLOUD_q`.
- The exterior region does not contribute to the index.
- Direct, indirect and exterior masks do not overlap.
- Fields preserve dimensions and coordinates when `xarray` is used.
- Geometric units are normalized to kilometers.

---

## Data requirements

The repository expects data in the following general formats.

### 1. Tropical-cyclone tracks

**Format:** IBTrACS, local best-track, CSV or Parquet.

Minimum variables:

- `storm_id`
- `time`
- `lat`
- `lon`
- `vmax_kt`
- optional `pmin_hpa`
- optional or conditional `rmw_km`

### 2. Wind radii

Expected variables:

- `R34_NE`, `R34_SE`, `R34_SW`, `R34_NW`, or equivalent names.
- Typical unit: nautical miles (`nm`).

Internally, quadrant labels are normalized to:

```text
RNE, RSE, RSW, RNW
```

### 3. ROCLOUD radii

Expected variables:

- `ROCLOUD_RNE`, `ROCLOUD_RSE`, `ROCLOUD_RSW`, `ROCLOUD_RNW`, or equivalent names.
- Expected unit: kilometers (`km`).

### 4. Precipitation fields

**Format:** NetCDF or Zarr.

Specifications:

- Target resolution: 0.1° × 0.1°, when using MSWEP-like products.
- Type: snapshot, not temporal accumulation.
- Units: must match the climatology used to derive Q90/Q95/Q99.
- Coverage: sufficient spatial domain around the cyclone center.

Compatible sources may include:

- historical MSWEP;
- numerical forecasts;
- hindcasts;
- downscaled products;
- bias-corrected products.

### 5. Climatological percentiles

Required variables:

- `Q90`
- `Q95`
- `Q99`

They should be computed by spatial cell and by a comparable temporal window.

For detailed requirements, see [`docs/data_contracts.md`](docs/data_contracts.md).

---

## Documentation

Main documentation files:

- [`docs/architecture.md`](docs/architecture.md): project architecture and computational design.
- [`docs/data_contracts.md`](docs/data_contracts.md): expected input/output contracts.
- [`docs/workflows.md`](docs/workflows.md): reproducible execution workflows.
- [`examples/README.md`](examples/README.md): example scripts and usage.
- [`notebooks/README.md`](notebooks/README.md): notebook descriptions, if available.

---

## Contributing

Contributions are welcome.

### Development setup

```bash
git clone https://github.com/YOUR_USERNAME/itchi-core.git
cd itchi-core

conda env create -f environment.yml
conda activate itchi
pip install -e .
pre-commit install
```

### Code conventions

- Formatting: `black`, maximum line length of 88 characters.
- Linting: `ruff`.
- Type hints are recommended for public functions.
- NumPy-style docstrings are preferred.
- New functionality should include tests.
- Do not commit heavy data or derived products.

### Commit style

Suggested format:

```text
type: brief description
```

Recommended types:

- `feat`
- `fix`
- `docs`
- `test`
- `refactor`
- `style`
- `chore`

Example:

```text
feat: add event-level ITCHI compiler
```

---

## Authorship

**Adolfo Perez-Estrada**

Universidad Nacional Autónoma de México (UNAM)
Instituto de Ciencias de la Atmósfera y Cambio Climático (ICACC)

---

## License

License to be defined.

For guidance on scientific software licenses, see [choosealicense.com](https://choosealicense.com/).

---

## References

### Key resources

- **IBTrACS**: International Best Track Archive for Climate Stewardship.
- **MSWEP**: Multi-Source Weighted-Ensemble Precipitation.
- **ROCLOUD**: Database for the outer sizes of tropical cyclones over the Middle Americas.

### Related documentation

- [Project architecture](docs/architecture.md)
- [Data contracts](docs/data_contracts.md)
- [Reproducible workflows](docs/workflows.md)
- [Examples](examples/)
- [Synthetic smoke test notebook](notebooks/00_smoke_test_synthetic.ipynb)
- [Compiled-event inspection notebook](notebooks/01_read_compiled_event.ipynb)

### Scientific references

- Pérez-Estrada & Dominguez (2025): A database for the outer sizes of tropical cyclones over the Middle Americas.
- Pérez-Alarcón et al. (2021): Comparative climatology of outer tropical cyclone size using radial wind profiles.
- Knapp et al. (2010): The International Best Track Archive for Climate Stewardship (IBTrACS).
- Beck et al. (2019): MSWEP V2 global 3-hourly 0.1° precipitation.
