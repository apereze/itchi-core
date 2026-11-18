# ITCHI Core

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-itchi--core-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](tests/)
[![License: TBD](https://img.shields.io/badge/License-TBD-yellow.svg)](#license)

**Integrated Tropical Cyclone Hazard Index**

Computational core of **ITCHI**, a bounded index in [0, 1] of the hazard of
tropical cyclones that affect Mexico, on a 0.1° grid and by municipality, at
synoptic times. Version **0.3.0.dev0**: development line v0.3
([`docs/plan_v0.3.md`](docs/plan_v0.3.md)) on the scheme agreed in v0.2.2;
changes in [`CHANGELOG.md`](CHANGELOG.md).

## Current scheme

Defined in [`configs/default.yaml`](configs/default.yaml), which equals the
keyword defaults of `pipeline.compute_itchi_snapshot`; details in
[`docs/methodology.md`](docs/methodology.md).

| Component | Definition |
|---|---|
| Precipitation | P6 × (R24, A72) matrix against local MSWEP V3 percentiles; maximum at the 6-h P95; with R24 ≥ P95₂₄ and A72 ≥ P75₇₂ the P90 node rises from 0.25 to 0.8 |
| Wind | Absolute hazard homologated with Saffir-Simpson on the local W06 wind (calibrated to R34 by quadrant): tail (12 m/s to 34 kt) → 1, TS → 2, H1 → 3, H2 → 4, ≥ H3 → 5 |
| Regions | Direct: R34; attribution: ROCLOUD with R_W06 (W06, 2 m/s, PA21) as a floor by quadrant |
| Index | Bounded union of the direct and indirect components; classes in steps of 0.2 |
| Municipality | Maximum over the cells of the municipality |

The precipitation thresholds in percentiles are provisional: v0.3 replaces
them by return periods (F1, [`docs/periodos_retorno.md`](docs/periodos_retorno.md)).

## Quick start

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core
conda env create -f environment.yml
conda activate itchi
pip install -e .
python -m pytest tests/
python examples/smoke_test_synthetic.py
```

One synoptic time:

```python
from itchi.climatology import load_percentile_climatology
from itchi.config import itchi_settings_from_config, load_experiment_config
from itchi.pipeline import compute_itchi_snapshot
from itchi.precipitation import compute_antecedent_precipitation

# p6: DataArray (time, lat, lon) of 6-h accumulations ending at each time stamp
r24 = compute_antecedent_precipitation(p6, hours=24, exclude_current=False)
a72 = compute_antecedent_precipitation(p6, hours=72, exclude_current=True)
q6, q24, q72 = (load_percentile_climatology(f"data/processed/mswep_v3_{d}h_percentiles_may_nov.nc",
                                            variable="p_wet_raw", template=p6.isel(time=0))
                for d in (6, 24, 72))
t = "2020-06-07T18:00"
res = compute_itchi_snapshot(
    p6.sel(time=t), r24.sel(time=t), a72.sel(time=t),
    quantiles_6h=q6, quantiles_24h=q24, quantiles_72h=q72,
    lon=lon2d, lat=lat2d, center_lon=-90.4, center_lat=29.1, vmax_kt=45.0,
    r34_by_quadrant={"RNE": 150, "RSE": 140, "RSW": 0, "RNW": 90}, r34_unit="nm",
    rocloud_by_quadrant={"RNE": 900.0, "RSE": 800.0, "RSW": None, "RNW": 600.0},
)
res["ITCHI"], res["H_P"], res["H_W"], res["R_attr_by_quadrant"]

# An experiment is an overlay on configs/default.yaml with only the keys that change
settings = itchi_settings_from_config(load_experiment_config("configs/experiments/mi_experimento.yaml"))
res_exp = compute_itchi_snapshot(..., **settings)
```

Case studies on the grid and by municipality (Emily 2005, Alex 2010,
Manuel 2013, Odile 2014, Patricia 2015, Earl 2016, Cristóbal 2020):

```bash
cd scripts
python casos.py                                       # cache in outputs/diagnostics/casos/
python casos.py --casos ODILE2014 --forzar
```

## Workflow and data

Run the scripts from `scripts/` in the `itchi` environment (Anaconda Prompt
on Windows), as their docstrings show.

| Step | Script | Product |
|---|---|---|
| 6-h accumulations of MSWEP V3.16 | `scripts/mswep_v3_6h_accum.py` | `D:/MSWEP_V316_6H/mswep_v3_6h_AAAA.nc` |
| Local percentiles of 6, 24 and 72 h | `scripts/percentiles_6h.py` | `data/processed/mswep_v3_{6,24,72}h_percentiles_may_nov.nc` |
| Return periods (F1) | `scripts/periodos_retorno.py`, `scripts/figuras_retorno.py` | `data/processed/mswep_v3_return_periods_may_nov.nc` |
| Declaratorias associated with cyclones | `scripts/declaratorias_6h.py` | `outputs/diagnostics/declaratorias_6h/` ([`docs/declaratorias.md`](docs/declaratorias.md)) |
| Case studies | `scripts/casos.py` | `outputs/diagnostics/casos/<caso>.nc` |

Inputs: IBTrACS v04r01 (`data/raw/tracks/`), ROCLOUD
(`data/external/Rocloud/`, [`docs/rocloud_format.md`](docs/rocloud_format.md)),
MSWEP V3 ([`docs/mswep_adapter.md`](docs/mswep_adapter.md)), INEGI
municipalities `mun23gw` (`data/external/mun23gw/`). Paths in
`configs/default.yaml`.

## Repository structure

```text
configs/default.yaml        current scheme and paths (experiments: overlays in configs/experiments/)
src/itchi/
  pipeline.py               compute_itchi_snapshot: one synoptic time on a grid
  precipitation.py          P6 x (R24, A72) matrix, antecedent windows
  wind.py                   absolute Saffir-Simpson wind hazard
  parametric_radii.py       W06/PA21 radii and wind with tail
  wind_profiles.py          Willoughby et al. (2006) profile
  radii.py, geometry.py, masks.py, components.py, index.py
  climatology.py            percentile files
  return_periods.py         GEV and POT by L-moments (F1)
  cases.py, municipal.py    case studies and municipal aggregation
  ibtracs.py, tracks.py, rocloud.py, mswep.py, precipitation_snapshots.py, units.py, time_utils.py
  declaratorias_6h.py, plotting.py, io.py, quality_control.py, config.py, constants.py
scripts/                    command-line workflows (table above)
notebooks/v03/              v0.3 experiments, with figures and reading guide
docs/                       methodology, plan v0.3, data formats; docs/historial/ for v0.1–v0.2.2
examples/                   synthetic smoke test
tests/                      pytest
```

## History

The code, notebooks and configurations of v0.1, v0.2.0 and the v0.2.1
experiments (Block 1, progress report) were retired at the start of v0.3 and
are kept in the git tags `v0.2.0`, `v0.2.1` and `v0.2.2`; their documents
are in [`docs/historial/`](docs/historial/README.md).

## Development

Guide in [`docs/development.md`](docs/development.md). Scientific functions are
NumPy-based and have tests in `tests/`; each v0.3 experiment gets a notebook
in `notebooks/v03/` and its criteria are written before seeing the results.

## Authorship

**Adolfo Perez-Estrada**

Universidad Nacional Autónoma de México (UNAM)
Instituto de Ciencias de la Atmósfera y Cambio Climático (ICACC)

## License

License to be defined.

## References

- Pérez-Estrada & Dominguez (2025): A database for the outer sizes of tropical cyclones over the Middle Americas.
- Pérez-Alarcón et al. (2021): Comparative climatology of outer tropical cyclone size using radial wind profiles.
- Knapp et al. (2010): The International Best Track Archive for Climate Stewardship (IBTrACS).
- Beck et al. (2019): MSWEP V2 global 3-hourly 0.1° precipitation.
- Willoughby, Darling & Rahn (2006): Parametric representation of the primary hurricane vortex. Part II. *Mon. Wea. Rev.*, 134, 1102–1120, https://doi.org/10.1175/MWR3106.1.
- NHC: Saffir-Simpson Hurricane Wind Scale, https://www.nhc.noaa.gov/aboutsshws.php.
