# ITCHI reproducible workflows

This document describes the minimum reproducible workflows currently supported by `itchi-core`.

It complements [`docs/data_contracts.md`](data_contracts.md), which defines the expected input and output contracts. This document focuses on execution order, validation commands and expected products.

---

## Workflow levels

`itchi-core` currently supports three workflow levels:

1. **Direct synthetic workflow**
   - Builds synthetic precipitation and metadata directly in Python.
   - Useful for validating the mathematical core and export utilities.

2. **Table-driven synthetic workflow**
   - Builds snapshot inputs from synthetic track and ROCLOUD tables.
   - Closer to the expected real-data workflow.

3. **Compiled event consumption workflow**
   - Reads a previously exported NetCDF or Zarr event.
   - Validates that ITCHI products can be consumed without recomputing the index.

---

## Recommended environment setup

From the repository root:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e .[dev]
```

Run the test suite:

```bash
python -m pytest tests/
```

Run formatting and linting hooks:

```bash
pre-commit run --all-files
```

---

## Direct synthetic workflow

### Purpose

Validate the end-to-end ITCHI calculation using synthetic fields and manually constructed snapshot inputs.

This workflow checks:

- synthetic lon/lat grid generation;
- synthetic precipitation snapshots;
- snapshot-level ITCHI calculation;
- event compilation;
- event-level products such as `ITCHI_max` and `ITCHI_acc`;
- optional export to NetCDF or Zarr;
- optional diagnostic plotting.

### Run without export

```bash
python examples/smoke_test_synthetic.py
```

### Export compiled event

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc
```

### Export and plot

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png \
  --output-path outputs/events/smoke_test_synthetic.nc
```

### Expected output

The exported file should contain time-stacked snapshot fields and event products, including:

```text
ITCHI(time, y, x)
H_P(time, y, x)
H_W(time, y, x)
H_dir(time, y, x)
H_ind(time, y, x)
ITCHI_max(y, x)
ITCHI_acc(y, x)
```

Metadata are serialized in Dataset attributes.

---

## Table-driven synthetic workflow

### Purpose

Validate the workflow that builds ITCHI snapshot inputs from tabular cyclone metadata and ROCLOUD radii.

This workflow is closer to the intended operational structure because it uses:

- a precipitation object with `valid_time`;
- a standardized track table;
- a standardized ROCLOUD table;
- `build_event_snapshot_inputs_from_tables`;
- `compile_itchi_event`;
- `write_compiled_event`.

### Run without export

```bash
python examples/smoke_test_from_tables.py
```

### Export compiled event

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

### Export and plot

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc \
  --plot \
  --figure-path outputs/figures/smoke_test_from_tables.png
```

### Validate exported file

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

### Expected behavior

The workflow should print:

```text
Table-driven synthetic ITCHI smoke test passed.
```

The exported event should pass the read/validation workflow without recomputing ITCHI.

---

## Compiled event consumption workflow

### Purpose

Validate that an already exported event can be read, inspected and checked independently.

This workflow does not recompute ITCHI. It is intended to test downstream consumption.

### Read synthetic direct workflow output

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc
```

### Read table-driven workflow output

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

### Read and plot

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc \
  --plot \
  --figure-path outputs/figures/read_compiled_event.png
```

### Validation checks

The read workflow validates:

1. required variables are present;
2. `ITCHI`, `ITCHI_max` and `ITCHI_acc` remain bounded in `[0, 1]`;
3. `ITCHI` has a temporal dimension;
4. `ITCHI_max` and `ITCHI_acc` are event-level fields without time;
5. serialized metadata are present when exported.

---

## Notebook workflows

Notebook workflows are intended for exploratory inspection, not as the primary test mechanism.

Scripts in `examples/` should remain the primary reproducible validation layer.

### Synthetic calculation notebook

```text
notebooks/00_smoke_test_synthetic.ipynb
```

This notebook computes ITCHI from synthetic inputs and inspects intermediate results.

### Compiled event inspection notebook

```text
notebooks/01_read_compiled_event.ipynb
```

This notebook reads a previously exported NetCDF/Zarr event and inspects variables, metadata and diagnostic maps.

Before running it, create an exported event:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

---

## Full validation sequence

From the repository root, a complete local validation can be run as:

```bash
python -m pytest tests/

python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc

python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc

pre-commit run --all-files
```

If all commands pass, the repository has validated:

- mathematical components;
- spatial masks;
- precipitation hazard scaling;
- wind hazard calculation;
- geometry and quadrant assignment;
- quality control;
- event compilation;
- NetCDF/Zarr export;
- independent product consumption;
- table-driven snapshot input construction.

---

## Output directory convention

Generated files should be written under:

```text
outputs/
```

Recommended structure:

```text
outputs/events/
outputs/figures/
outputs/notebooks/
```

The `outputs/` directory should remain ignored by Git.

---

## Current boundary of the repository

The current workflows are synthetic and structural. They validate the core mechanics of ITCHI but do not yet claim a fully real-data operational product.

The next logical development blocks are:

1. formal real-data ingestion examples for track tables;
2. formal real-data ingestion examples for ROCLOUD tables;
3. MSWEP snapshot selection examples;
4. gridded climatology loading and alignment;
5. event-level validation against observed impacts or disaster declarations.

---

## Failure interpretation

### Coordinate-alignment errors

If quality control raises a coordinate error, check that `precipitation`, `lon`, `lat`, climatology fields and wind fields share dimensions and compatible coordinates.

`itchi.snapshot_inputs` can copy missing dimension coordinates only when dimensions and shape already match. It does not regrid.

### Missing wind metadata

If `wind_hazard_normalized` is not provided, the workflow needs:

```text
vmax_kt
rmw_km
```

If either is missing, provide a precomputed wind hazard field or add valid wind metadata.

### Missing ROCLOUD radii

If all ROCLOUD quadrants are missing, the pipeline should fail because no external precipitation-attribution boundary exists.

If only some quadrants are missing, the configured fill strategy may resolve them and record metadata.

### R34 and weak systems

For systems weaker than tropical-storm strength, R34 should not be interpreted as a physical direct wind boundary. The pipeline should resolve the direct radius using RMW or an explicit fallback.
