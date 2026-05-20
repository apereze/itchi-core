# ITCHI examples

This directory contains executable examples for validating and demonstrating
the `itchi-core` workflow.

## Synthetic smoke test

Run:

```bash
python examples/smoke_test_synthetic.py
````

This example validates the end-to-end synthetic workflow:

1. create a synthetic lon/lat grid;
2. create two synthetic precipitation snapshots;
3. compute ITCHI for each snapshot;
4. compile the event;
5. compute `ITCHI_max` and `ITCHI_acc`;
6. print metadata and numerical checks.

To generate a diagnostic figure:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

The `outputs/` directory is ignored by Git.

````

---

# Paso 33.3 — Probar todo

```bash
python -m pytest tests/
````

Luego:

```bash
python examples/smoke_test_synthetic.py
```

Después:

```bash
pre-commit run --all-files
```

Si modifica archivos:

```bash
git add .
pre-commit run --all-files
```
## Read compiled event

After exporting the synthetic compiled event:

```python
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc
````
validate and inspect the file with:

```python
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc
````

To generate a diagnostic figure:

````bash
  python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc \
  --plot \
  --figure-path outputs/figures/read_compiled_event.png
````

This example verifies that the exported event can be consumed without recomputing ITCHI.


# Ejecutar pruebas

```bash
python -m pytest tests/test_io.py
python -m pytest tests/test_compiler.py
python -m pytest tests/
````

Luego:

````bash
pre-commit run --all-files
````

Y prueba manual:
````
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc
````
