# ITCHI examples usage

This file documents the executable examples available in `itchi-core`.

Run all commands from the repository root.

```bash
cd itchi-core
```

Generated outputs should be written under `outputs/`, which is ignored by Git.

Recommended structure:

```text
outputs/events/
outputs/figures/
outputs/notebooks/
```

---

## Available scripts

| Script | Purpose |
|---|---|
| `smoke_test_synthetic.py` | Direct synthetic end-to-end ITCHI calculation. |
| `smoke_test_from_tables.py` | Table-driven workflow using synthetic track and ROCLOUD tables. |
| `real_data_template.py` | Parameterized real-data workflow using IBTrACS, ROCLOUD and MSWEP. |
| `read_compiled_event.py` | Read and validate an exported compiled event. |

---

## Direct synthetic smoke test

`smoke_test_synthetic.py` validates the end-to-end ITCHI workflow using synthetic data and manually constructed snapshot inputs.

Run:

```bash
python examples/smoke_test_synthetic.py
```

Export a compiled event:

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc
```

Generate a diagnostic figure:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

Export and plot:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png \
  --output-path outputs/events/smoke_test_synthetic.nc
```

---

## Table-driven synthetic smoke test

`smoke_test_from_tables.py` validates the workflow that builds ITCHI snapshot inputs from synthetic track and ROCLOUD tables.

This is closer to the expected real-data workflow because it uses:

- a precipitation object with `valid_time`;
- a standardized track table;
- a standardized ROCLOUD table;
- `build_event_snapshot_inputs_from_tables`;
- `compile_itchi_event`;
- `write_compiled_event`.

Run:

```bash
python examples/smoke_test_from_tables.py
```

Export a compiled event:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

Export and plot:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc \
  --plot \
  --figure-path outputs/figures/smoke_test_from_tables.png
```

---

## Real-data workflow template

`real_data_template.py` is a parameterized template for connecting real sources:

- IBTrACS NetCDF for track metadata and R34 radii;
- ROCLOUD `.dat`, `.txt` or `.dot` files for the external precipitation-attribution radius;
- MSWEP NetCDF for precipitation snapshots;
- scalar or gridded `Q90`, `Q95` and `Q99` precipitation thresholds.

Example using scalar thresholds for an initial smoke run:

```bash
python examples/real_data_template.py \
  --ibtracs-path data/ibtracs/ibtracs.nc \
  --rocloud-path data/rocloud/EP_TCSize_2000_2024.dat \
  --mswep-path data/mswep/event_precip.nc \
  --precipitation-variable precipitation \
  --storm-id EP182023 \
  --start-time 2023-10-24T00:00:00 \
  --end-time 2023-10-25T06:00:00 \
  --q90 10 \
  --q95 20 \
  --q99 30 \
  --output-path outputs/events/EP182023_itchi.nc
```

Example with gridded climatological thresholds:

```bash
python examples/real_data_template.py \
  --ibtracs-path data/ibtracs/ibtracs.nc \
  --rocloud-path data/rocloud/EP_TCSize_2000_2024.dat \
  --mswep-path data/mswep/event_precip.nc \
  --precipitation-variable precipitation \
  --storm-id EP182023 \
  --start-time 2023-10-24T00:00:00 \
  --end-time 2023-10-25T06:00:00 \
  --q90 data/climatology/q90.nc \
  --q90-variable q90 \
  --q95 data/climatology/q95.nc \
  --q95-variable q95 \
  --q99 data/climatology/q99.nc \
  --q99-variable q99 \
  --lon-min -110 \
  --lon-max -90 \
  --lat-min 5 \
  --lat-max 30 \
  --pad-degrees 2 \
  --output-path outputs/events/EP182023_itchi.nc
```

Important: the real-data template treats MSWEP as valid-time snapshots. It does not accumulate or sum 3-hourly precipitation into 6-hourly totals.

---

## Read compiled event

`read_compiled_event.py` validates that an exported compiled event can be consumed without recomputing ITCHI.

Read the direct synthetic workflow output:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc
```

Read the table-driven workflow output:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

Read a real-data template output:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/EP182023_itchi.nc
```

Read and plot:

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc \
  --plot \
  --figure-path outputs/figures/read_compiled_event.png
```

The reader checks that required variables exist, that `ITCHI`, `ITCHI_max` and `ITCHI_acc` remain bounded in `[0, 1]`, and that event-level products do not retain the time dimension.

---

## Recommended validation sequence

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc

python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc

python -m pytest tests/
pre-commit run --all-files
```

---

## Expected compiled-event variables

A compiled event exported by the examples should include variables such as:

```text
ITCHI(time, y, x)
H_P(time, y, x)
H_W(time, y, x)
H_dir(time, y, x)
H_ind(time, y, x)
ITCHI_max(y, x)
ITCHI_acc(y, x)
```

The exact set of variables depends on the snapshot variables included during event compilation.

---

## Notes

The smoke tests are synthetic. They validate computational structure, data flow, export mechanics and product consumption.

The real-data template is intentionally parameterized and should be treated as the first executable bridge to real sources, not as a fully tuned production workflow.

For the full reproducible workflow guide, see `docs/workflows.md`.

For input and output requirements, see `docs/data_contracts.md`.
