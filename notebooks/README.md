# ITCHI notebooks

This directory contains exploratory notebooks for validating and inspecting `itchi-core` outputs.

Scripts in `examples/` are the preferred reproducible validation layer. Notebooks are intended for interactive inspection and scientific exploration.

---

## Available notebooks

| Notebook | Purpose |
|---|---|
| `00_smoke_test_synthetic.ipynb` | Compute ITCHI from synthetic precipitation, cyclone geometry, wind metadata and fixed thresholds. |
| `01_read_compiled_event.ipynb` | Read and inspect a previously exported compiled ITCHI event. |

---

## 00 — Synthetic smoke test

```text
00_smoke_test_synthetic.ipynb
```

This notebook computes ITCHI from synthetic precipitation, cyclone geometry, wind metadata and fixed precipitation thresholds.

It is useful for inspecting:

- precipitation hazard;
- wind hazard;
- direct and indirect masks;
- ITCHI snapshots;
- event-level products.

Open it with:

```bash
jupyter lab notebooks/00_smoke_test_synthetic.ipynb
```

---

## 01 — Read compiled event

```text
01_read_compiled_event.ipynb
```

This notebook reads a previously exported compiled event and inspects:

- Dataset dimensions;
- variables;
- metadata;
- `ITCHI`;
- `ITCHI_max`;
- `ITCHI_acc`.

Before running it, generate a compiled event:

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

Then open:

```bash
jupyter lab notebooks/01_read_compiled_event.ipynb
```

---

## Recommended notebook workflow

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc

jupyter lab notebooks/01_read_compiled_event.ipynb
```

---

## Notes

Notebook outputs should not be treated as the primary validation mechanism.

For reproducible validation, use:

```bash
python -m pytest tests/
python examples/smoke_test_from_tables.py
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

For the complete reproducible workflow guide, see `docs/workflows.md`.
