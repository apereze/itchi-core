# ITCHI Core Development Guide

This document describes the recommended development workflow for `itchi-core`.

It is intended for contributors and maintainers working on the computational
core of the Integrated Tropical Cyclone Hazard Index.

---

## 1. Development philosophy

`itchi-core` should remain:

1. **modular** — each module has a clear responsibility;
2. **testable** — every scientific function should have unit tests;
3. **traceable** — methodological assumptions should be documented;
4. **reproducible** — examples and notebooks should run from a clean checkout;
5. **physically interpretable** — code should preserve the scientific logic of ITCHI.

The package should avoid mixing:

| Concern | Should live in |
|---|---|
| Scientific formulas | `src/itchi/*.py` |
| Input/output | `io.py` |
| Event orchestration | `compiler.py` |
| Validation | `quality_control.py` |
| Examples | `examples/` |
| Exploratory validation | `notebooks/` |
| Methodological explanation | `docs/` |

---

## 2. Repository setup

Clone the repository:

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate itchi
```

Install the package in editable mode:

```bash
pip install -e .
```

Install development hooks:

```bash
pre-commit install
```

Verify the installation:

```bash
python -c "import itchi; print('ITCHI imported successfully')"
```

---

## 3. Common development commands

Run all tests:

```bash
python -m pytest tests/
```

Run tests with verbose output:

```bash
python -m pytest tests/ -v
```

Run one test file:

```bash
python -m pytest tests/test_pipeline.py -v
```

Run pre-commit on all files:

```bash
pre-commit run --all-files
```

Run the synthetic terminal example:

```bash
python examples/smoke_test_synthetic.py
```

Run the example and save a diagnostic figure:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

---

## 4. Recommended workflow for new changes

Before editing:

```bash
git status
git pull
```

Create a new branch:

```bash
git checkout -b feature/my-feature-name
```

Make code changes.

Run tests:

```bash
python -m pytest tests/
```

Run linting and formatting checks:

```bash
pre-commit run --all-files
```

If `pre-commit` modifies files, add them and rerun:

```bash
git add .
pre-commit run --all-files
```

Commit:

```bash
git add .
git commit -m "type: concise description"
```

Push:

```bash
git push origin feature/my-feature-name
```

---

## 5. Commit message conventions

Use short, descriptive commit messages.

Recommended types:

| Type | Use case |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `docs` | Documentation change |
| `test` | Tests only |
| `refactor` | Code restructuring without behavior change |
| `style` | Formatting or linting only |
| `chore` | Maintenance tasks |

Examples:

```text
feat: add climatology percentile utilities
```

```text
fix: handle missing ROCLOUD quadrants
```

```text
docs: update methodology after R_direct implementation
```

```text
test: add compiler xarray tests
```

---

## 6. Pull request checklist

Before opening a pull request, verify:

```bash
python -m pytest tests/
pre-commit run --all-files
python examples/smoke_test_synthetic.py
```

The PR should include:

- clear title;
- concise description;
- list of changed modules;
- explanation of scientific or methodological changes;
- tests for new functionality;
- updated documentation when behavior changes.

---

## 7. Coding standards

### 7.1. Type hints

Use type hints for public functions.

Good:

```python
def compute_event_products(
    itchi: np.ndarray | xr.DataArray,
    dim: str = "time",
) -> dict[str, np.ndarray | xr.DataArray]:
    ...
```

Avoid untyped public functions unless there is a strong reason.

### 7.2. Docstrings

Use NumPy-style docstrings for public functions.

A public function should usually include:

- short description;
- parameters;
- returns;
- raises, when relevant;
- notes, when the scientific assumption is important.

### 7.3. Scientific assumptions

Any methodological decision should be documented in either:

```text
docs/methodology.md
```

or in a concise function-level docstring.

Examples of assumptions that must be documented:

- precipitation is treated as snapshot;
- `R_direct_q` may differ from observed `R34_q`;
- missing quadrants are filled from available-quadrant mean;
- tropical depressions may use `RMW` as direct-radius fallback;
- wind hazard below 34 kt can be normalized relative to `Vmax`.

---

## 8. Testing standards

Every new module should have a matching test file.

| Module | Test file |
|---|---|
| `src/itchi/precipitation.py` | `tests/test_precipitation.py` |
| `src/itchi/masks.py` | `tests/test_masks.py` |
| `src/itchi/pipeline.py` | `tests/test_pipeline.py` |
| `src/itchi/compiler.py` | `tests/test_compiler.py` |

Tests should cover:

1. NumPy inputs;
2. xarray inputs when relevant;
3. expected numerical values;
4. bounds in `[0, 1]`;
5. missing-data behavior;
6. physically invalid cases;
7. preservation of dimensions and coordinates.

---

## 9. Notebook standards

Notebooks are for validation and demonstration, not for core logic.

Core logic must live in:

```text
src/itchi/
```

A notebook should:

1. import functions from `itchi`;
2. avoid redefining core algorithms;
3. run from top to bottom;
4. use synthetic data unless explicitly labeled as a real case;
5. document the purpose of each section;
6. avoid committing large outputs.

Recommended notebooks:

| Notebook | Purpose |
|---|---|
| `00_smoke_test_synthetic.ipynb` | End-to-end synthetic validation |
| `01_single_snapshot_synthetic.ipynb` | Single-snapshot diagnostics |
| `02_single_event_synthetic.ipynb` | Multi-snapshot event diagnostics |
| `03_single_event_real_case.ipynb` | First real-case workflow |

Before committing notebooks:

```bash
pre-commit run --all-files
```

---

## 10. Example script standards

Examples should be executable from the repository root.

Good:

```bash
python examples/smoke_test_synthetic.py
```

Examples should:

1. avoid requiring private data;
2. use synthetic data unless clearly documented;
3. print basic numerical diagnostics;
4. optionally write outputs under `outputs/`;
5. avoid writing files tracked by Git.

The `outputs/` directory should remain ignored by Git.

---

## 11. Debugging workflow

When a test fails:

1. Run the specific failing test:

```bash
python -m pytest tests/test_pipeline.py -v
```

2. Run with traceback details:

```bash
python -m pytest tests/test_pipeline.py -vv
```

3. Check whether the failure is numerical, structural or metadata-related.

Common failure types:

| Failure | Likely cause |
|---|---|
| `AssertionError: arrays are not equal` | Expected values need update or algorithm changed |
| `KeyError` | Output dictionary keys changed |
| `ValueError` | Quality-control rule failed |
| `F811` | Duplicate test function name |
| `E501` | Line too long |
| `ModuleNotFoundError` | Package not installed with `pip install -e .` |
| xarray dimension mismatch | Coordinates or dims not preserved |

---

## 12. Pre-commit troubleshooting

Run:

```bash
pre-commit run --all-files
```

If a hook modifies files:

```bash
git add .
pre-commit run --all-files
```

If `ruff` reports line length errors in notebooks, split long lines manually.

Example:

```python
def make_snapshot_input(
    precipitation,
    center_lon,
    center_lat,
    vmax_kt=80.0,
    rmw_km=25.0,
):
    ...
```

If `ruff` reports duplicate definitions:

```text
F811 Redefinition of unused function
```

search for the duplicated function:

```bash
grep -n "function_name" tests/test_pipeline.py
```

Then delete or rename the duplicate.

---

## 13. Quality-control expectations

The following should always be true for valid ITCHI outputs:

1. `ITCHI` is in `[0, 1]`;
2. `H_P`, `H_W`, `H_dir`, `H_ind` are in `[0, 1]`;
3. exterior cells have zero ITCHI contribution;
4. direct, indirect and exterior masks do not overlap;
5. `R_direct_q <= ROCLOUD_q`;
6. fields preserve dimensions and coordinates;
7. radii used in masks are in kilometers.

When these assumptions are intentionally relaxed, the reason should be
documented in the relevant function, test or issue.

---

## 14. Branch strategy

Recommended branch names:

```text
feature/add-climatology
fix/rocloud-missing-values
docs/update-methodology
test/add-wind-xarray-tests
refactor/compiler-output-structure
```

Avoid working directly on `main` for major changes.

---

## 15. Release checklist

Before tagging a release:

```bash
python -m pytest tests/
pre-commit run --all-files
python examples/smoke_test_synthetic.py
```

Then verify:

- `README.md` is current;
- `CHANGELOG.md` includes the release;
- `docs/api.md` matches public functions;
- `docs/methodology.md` matches scientific assumptions;
- notebooks run from top to bottom;
- GitHub Actions pass.

---

## 16. Recommended next development modules

The following modules are planned but not yet required for the core synthetic
workflow:

| Module | Purpose |
|---|---|
| `climatology.py` | Compute or load Q90/Q95/Q99 fields |
| `precipitation_snapshots.py` | Align precipitation snapshots to cyclone times |
| `adapters/ibtracs.py` | Real-data adapter for IBTrACS |
| `adapters/mswep.py` | Real-data adapter for MSWEP |
| `adapters/rocloud.py` | Real-data adapter for ROCLOUD files |

These should be added only after the current core API remains stable.
