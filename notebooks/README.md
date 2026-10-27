# ITCHI notebooks

This directory contains exploratory notebooks for validating and inspecting `itchi-core` outputs.

Scripts in `examples/` are the preferred reproducible validation layer. Notebooks are intended for interactive inspection and scientific exploration.

---

## Available notebooks

| Notebook | Purpose |
|---|---|
| `00_smoke_test_synthetic.ipynb` | Compute ITCHI from synthetic precipitation, cyclone geometry, wind metadata and fixed thresholds. |
| `01_read_compiled_event.ipynb` | Read and inspect a previously exported compiled ITCHI event. |
| `01_willa_parametric_radii_itchi.ipynb` | Willa 2018 with parametric radii. |
| `02_*`, `03_*` and `first test/` | Cristóbal 2020 and Odile 2014 with v0.1: attribution radii (ROCLOUD, W06, Chavas) and ramps. |
| `declaratorias_6h_*.ipynb` | Exploration of TC disaster declarations against 6-h rain. |
| `second test/02.1`–`04` | MSWEP V3.16 6-h cases and declaration synthesis. |
| `second test/05_itchi_v02_mejoras.ipynb` | v0.2: Cristóbal and Odile, v0.1 vs v0.2, maps and municipal PSS. |
| `second test/06_verificacion_metricas_figuras.ipynb` | v0.2: verification metrics explained and national verification figures. |
| `second test/07_itchi_v02_peligro_por_snapshot.ipynb` | v0.2: hazard by snapshot, attribution of changes by step, alert duration and timing. Saves the fields used by 08. |
| `second test/08_itchi_v01_v02_mapas_por_posicion.ipynb` | v0.2: v0.1 vs v0.2 maps at every position and event maps. Run 07 first. |

The v0.2 notebooks read the outputs of `scripts/` (see `docs/verification_v02.md`, "Reproduction") and write figures to `outputs/`, which is not versioned.

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
