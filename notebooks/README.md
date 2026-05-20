# ITCHI notebooks

This directory contains exploratory notebooks for validating and inspecting
`itchi-core` outputs.

Scripts in `examples/` are the preferred reproducible validation layer.
Notebooks are intended for interactive inspection and scientific exploration.

## 00 — Synthetic smoke test

```text
00_smoke_test_synthetic.ipynb
````

This notebook computes ITCHI from synthetic precipitation, cyclone geometry, wind metadata and fixed precipitation thresholds.

It is useful for inspecting:

- precipitation hazard;
- wind hazard;
- direct and indirect masks;
- ITCHI snapshots;
- event-level products.

## 01 — Read compiled event

``````
01_read_compiled_event.ipynb
``````

This notebook reads a previously exported compiled event and inspects:

- Dataset dimensions;
- variables;
- metadata;
- ITCHI;
- ITCHI_max;
- ITCHI_acc.

Before running it, generate a compiled event:

``````
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
``````

Then open:

``````
jupyter lab notebooks/01_read_compiled_event.ipynb

``````

Recommended order
``````
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc

jupyter lab notebooks/01_read_compiled_event.ipynb
``````

### Notes

Notebook outputs should not be treated as the primary validation mechanism.
For reproducible validation, use:
``````
python -m pytest tests/
python examples/smoke_test_from_tables.py
python examples/read_compiled_event.py
``````
Luego:

``````
git add notebooks/README.md
git commit -m "Add notebooks documentation"
git push
``````