# ITCHI Core

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-itchi--core-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](tests/)
[![License: TBD](https://img.shields.io/badge/License-TBD-yellow.svg)](#license)

**Integrated Tropical Cyclone Hazard Index**

Repository for the development of the computational core of **ITCHI**, a physically based tropical-cyclone hazard index. Current version: **0.2.0**.

**What is new in v0.2** (opt-in; the defaults reproduce v0.1). Details in [`docs/methodology.md`](docs/methodology.md), results in [`docs/verification_v02.md`](docs/verification_v02.md), changes in [`CHANGELOG.md`](CHANGELOG.md):

- **Precipitation matrix.** The 6-h rain (P6) is compared with the pixel's own percentiles (alert from P75: P75 → 0, P90 → 0.25, P95 → 0.5, P99 → 1). The P95–P99 band is escalated towards the maximum (P95 → 0.8) when the rain persists (24-h accumulation R24 ≥ P95) over wet ground (72-h antecedent A72 ≥ P75). Continuous form in `precipitation.compute_precipitation_matrix_hazard`.
- **Calibration with disaster declarations.** 381 929 municipality–cyclone pairs (2000–2025, 2 % with a TC-associated declaration): the escalation raises the maximum-level Peirce skill score from 0.41 to 0.51 (0.55 → 0.67 for disasters) and leaves the lower levels unchanged.
- **W06 wind with an outer tail.** W06 profile calibrated to the observed R34 by quadrant; hazard decays from `h34` at R34 to zero at the attribution radius and enters the indirect component (`wind_model="w06_tail"`).
- **Multi-duration climatology and verification scripts.** MSWEP V3.16 percentiles of 6, 24 and 72 h; declarations vs rain; municipality–cyclone pairs.
- **Case studies.** Cristóbal 2020 and Odile 2014, v0.1 vs v0.2, by snapshot and by position (notebooks 05–08).
- **Open for v0.3.** The default wind tail gives wind hazard to 96 % of the pairs and removes the skill of the lowest class; cutting it at 17 kt is proposed. See the decision list in [`docs/verification_v02.md`](docs/verification_v02.md#6-open-decisions-for-v03).

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

**ITCHI v0.2** keeps this structure and changes how precipitation and wind hazards are computed: a precipitation matrix that escalates intense rain by persistence and antecedent rain, calibrated against disaster declarations, and a W06 wind profile whose tail extends the wind hazard into the indirect region.

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
- v0.2: precipitation matrix (P6 escalated by R24 and A72), W06 wind with outer tail, indirect wind component.
- v0.2: MSWEP V3.16 6-h accumulations and 6/24/72-h local percentiles.
- v0.2: verification against TC-associated disaster declarations (municipality–cyclone pairs) and real cases (Cristóbal 2020, Odile 2014).

### Not included yet

- Cross-validated calibration and confidence intervals (bootstrap by cyclone).
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

In the v0.2 studies the attribution radius is the W06 (PA21) radius where the wind falls to 2 m/s (`R_W06`, median 743 km over the verification pairs); it is passed through the same `rocloud_by_quadrant` argument as ROCLOUD.

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

Precipitation enters ITCHI as the **6-h accumulation ending at the synoptic time** T, `P6 = (T − 6 h, T]` (MSWEP V3.16 6-h accumulations in the v0.2 studies). It is not accumulated over the event.

v0.2 adds two rolling windows, used only to escalate the precipitation hazard:

| Symbol | Window | Role |
|---|---|---|
| `P6` | (T − 6 h, T] | Intensity |
| `R24` | (T − 24 h, T], includes P6 | Persistence |
| `A72` | (T − 78 h, T − 6 h], excludes P6 | Antecedent wetness |

The framework stays compatible with historical MSWEP, numerical forecasts, hindcasts, downscaled and bias-corrected products, provided the climatology is computed with the same accumulation windows.

Critical rule:

```text
Each window is compared only with percentiles of the same duration.
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

Precipitation hazard is computed with **local percentiles of each pixel** (MSWEP V3.16, May–November 2000–2025, computed on wet windows ≥ 1 mm, variable `p_wet_raw`), one climatology per duration (6, 24 and 72 h).

v0.1 ramp (6 h):

```text
P < P90        -> H_P = 0
P90 <= P < P95 -> H_P increases from 0 to 0.5
P95 <= P < P99 -> H_P increases from 0.5 to 1
P >= P99       -> H_P = 1
```

v0.2 matrix (base ramp with alert from P75, escalated at P95):

| P6 | Base `H_P` | Escalated `H_P` | Level |
|---|---|---|---|
| < P75 | 0 | 0 | 0 |
| P75–P90 | 0 → 0.25 | same | 1 |
| P90–P95 | 0.25 → 0.5 | same | 2 |
| P95–P99 | 0.5 → 1 | 0.8 → 1 | 3 → 4 with escalation |
| ≥ P99 | 1 | 1 | 4 |

```text
H_P = (1 − f) · H_base(P6) + f · H_esc(P6),   f = f_R24 · f_A72
f_R24: 0 at R24 = P90(24 h) → 1 at R24 = P95(24 h)
f_A72: 0 at A72 = 0 mm      → 1 at A72 = P75(72 h)
```

The discrete rule behind it, calibrated with disaster declarations: **P6 in P95–P99 with R24 ≥ P95 and A72 ≥ P75 → maximum level.** P99 is the saturation threshold in both versions.

---

## Wind hazard

| Version | Inside `R_direct` | Outside `R_direct` |
|---|---|---|
| v0.1 | Modified Rankine profile (inner exponent 1, outer decay 0.5), `H_W = (V − 34)/(Vmax − 34)` | 0 |
| v0.2 (`wind_model="w06_tail"`) | W06 calibrated to the observed R34, `H_W = h34 + (1 − h34)(V − 34)/(Vmax − 34)` | Tail `H_W = h34 (V − V_out)/(34 − V_out)`, zero at the attribution radius |

`h34 = 0.2` by default, so the tail alone stays in the lowest SIAT-style class. In both versions the hazard inside `R_direct` is relative to the storm's own Vmax.

---

## Index components

| Component | Description |
|---|---|
| `H_P` | Normalized precipitation hazard |
| `H_W` | Normalized wind hazard |
| `H_Wdir`, `H_Wind` | v0.2: wind hazard in the direct and indirect regions |
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
v0.1: H_ind = H_Pind
v0.2: H_ind = 1 - (1 - H_Wind) * (1 - H_Pind)
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
    H_P: v0.1 ramp or v0.2 matrix (P6, R24, A72 vs their percentiles)
        ↓
wind.py + parametric_radii.py
    v0.1 Rankine V* inside R_direct, or v0.2 W06 with outer tail
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
├── README.md, CHANGELOG.md, environment.yml, pyproject.toml, .gitignore
├── configs/default.yaml            # v0.1 defaults and v0.2 options
├── docs/
│   ├── methodology.md              # formulas (v0.1 and v0.2 extensions)
│   ├── verification_v02.md         # v0.2 calibration, results and open decisions
│   ├── declaratorias.md            # disaster-declaration processing
│   ├── architecture.md, api.md, data_contracts.md, workflows.md
│   ├── mswep_adapter.md, rocloud_format.md, real_case_validation.md
│   └── development.md, dev_log.md
├── examples/                       # synthetic and table-driven smoke tests
├── notebooks/
│   ├── 00–03 *.ipynb               # synthetic test, Willa, Cristóbal/Odile v0.1
│   ├── declaratorias_6h_*.ipynb    # declaration exploration
│   ├── first test/, second test/   # real-case iterations; second test/05–08 = v0.2
│   └── README.md
├── scripts/
│   ├── mswep_v3_6h_accum.py        # MSWEP V3.16 6-h accumulations
│   ├── percentiles_6h.py           # 6/24/72-h local percentiles
│   ├── declaratorias_6h.py         # TC declarations vs 6/24/72-h rain
│   ├── negativos_w06.py            # municipality–cyclone verification pairs
│   └── percentiles.py, prueba_rapida_6h.py
├── src/itchi/
│   ├── precipitation.py            # ramps, v0.2 matrix, antecedent windows
│   ├── wind.py, wind_profiles.py, parametric_radii.py, chavas_outer.py
│   ├── components.py, index.py, masks.py, pipeline.py, compiler.py, aggregation.py
│   ├── radii.py, rocloud.py, tracks.py, ibtracs.py, mswep.py, climatology.py
│   ├── declaratorias_6h.py, itchi_size_helpers.py, plotting.py
│   └── geometry.py, units.py, time_utils.py, io.py, quality_control.py, ...
└── tests/                          # pytest; tests/test_v02.py covers v0.2
```

`data/` and `outputs/` keep only `.gitkeep`; data and derived products are never committed.

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
8. Local climatological percentiles: `Q90`, `Q95`, `Q99` (v0.1) or P75–P99 of 6, 24 and 72 h (v0.2).
9. Normalized wind field `V*` or parameters to construct it (v0.2: Vmax, RMW, R34 and latitude for W06).
10. For verification: TC-associated disaster declarations by municipality and municipal polygons.

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
| `H_Wdir`, `H_Wind` | v0.2: wind hazard in the direct and indirect regions |

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
| Real cyclone cases | Done | Willa 2018, Cristóbal 2020, Odile 2014 (notebooks) |
| Gridded climatology | Done | MSWEP V3.16 6/24/72-h local percentiles (`scripts/percentiles_6h.py`) |
| v0.2 precipitation matrix | Done | Escalation by R24 and A72, calibrated with declarations |
| v0.2 wind tail | Done, under review | W06 with outer tail; cut at 17 kt proposed (D1) |
| Impact validation | In progress | 381 929 municipality–cyclone pairs; cross-validation and association by storm name pending |

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

### ITCHI v0.2 snapshot

```python
from itchi.precipitation import compute_antecedent_precipitation, compute_precipitation_matrix_hazard
from itchi.pipeline import compute_itchi_snapshot_from_grid

# p6: DataArray (time, lat, lon) of 6-h accumulations
# q6, q24, q72: {"Q75": DataArray, "Q90": ..., "Q95": ..., "Q99": ...} on the same grid
r24 = compute_antecedent_precipitation(p6, hours=24, exclude_current=False)
a72 = compute_antecedent_precipitation(p6, hours=72, exclude_current=True)
hp = compute_precipitation_matrix_hazard(p6, q6, {"R24": r24, "A72": a72}, {"R24": q24, "A72": q72})

res = compute_itchi_snapshot_from_grid(
    p6.values[i], None, None, None,
    precipitation_hazard=hp.fillna(0.0).values[i],
    lon=lon2d, lat=lat2d, center_lon=lon_c, center_lat=lat_c,
    r34_by_quadrant=r34_nm, r34_unit="nm",
    rocloud_by_quadrant=attribution_km, rocloud_unit="km",
    vmax_kt=vmax, rmw_km=rmw, wind_model="w06_tail",
)
```

Without `precipitation_hazard` and `wind_model`, the result is identical to v0.1. The full v0.2 workflow (scripts and notebooks 05–08) is in [`docs/verification_v02.md`](docs/verification_v02.md#reproduction).

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
- Type: 6-h accumulation ending at the synoptic time (v0.2: MSWEP V3.16 6-h).
- Units: must match the climatology used to derive the percentiles.
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

They should be computed by spatial cell and by a comparable temporal window. v0.2 uses P75, P90, P95, P99 and P99.9 of 6, 24 and 72 h (`data/processed/mswep_v3_{6,24,72}h_percentiles_may_nov.nc`, variable `p_wet_raw`).

### 6. Disaster declarations (verification only)

CENAPRED declarations of emergency, disaster and climatological contingency filtered to tropical cyclones, and INEGI municipal polygons. See [`docs/declaratorias.md`](docs/declaratorias.md).

For detailed requirements, see [`docs/data_contracts.md`](docs/data_contracts.md).

---

## Documentation

Main documentation files:

- [`docs/methodology.md`](docs/methodology.md): formulas of v0.1 and the v0.2 extensions.
- [`docs/verification_v02.md`](docs/verification_v02.md): v0.2 calibration, verification results, case studies and open decisions for v0.3.
- [`docs/declaratorias.md`](docs/declaratorias.md): processing of disaster declarations.
- [`docs/architecture.md`](docs/architecture.md): project architecture and computational design.
- [`docs/data_contracts.md`](docs/data_contracts.md): expected input/output contracts.
- [`docs/workflows.md`](docs/workflows.md): reproducible execution workflows.
- [`examples/README.md`](examples/README.md): example scripts and usage.
- [`notebooks/README.md`](notebooks/README.md): notebook descriptions.

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
- Willoughby, Darling & Rahn (2006): Parametric representation of the primary hurricane vortex. Part II. *Mon. Wea. Rev.*, 134, 1102–1120, https://doi.org/10.1175/MWR3106.1.
- Jolliffe & Stephenson (2012): *Forecast Verification: A Practitioner's Guide in Atmospheric Science*, 2nd ed., Wiley.
- Ferro & Stephenson (2011): Extremal Dependence Indices. *Weather and Forecasting*, 26, 699–713, https://doi.org/10.1175/WAF-D-10-05030.1.
