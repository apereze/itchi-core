# Examples

| Script | What it does |
|---|---|
| `smoke_test_synthetic.py` | Synthetic grid, rain and a category-3 hurricane; computes ITCHI with `pipeline.compute_itchi_snapshot` (scheme of `configs/default.yaml`) and checks bounds, exterior and classes. `--plot` saves H_P, H_W and ITCHI |

Run from the repository root:

```bash
python examples/smoke_test_synthetic.py
python examples/smoke_test_synthetic.py --plot --figure-path outputs/figures/smoke_test_synthetic.png
```

Real cases: `cd scripts && python casos.py` (see `src/itchi/cases.py`).
