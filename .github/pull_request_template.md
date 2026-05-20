# Pull Request

## Summary

Describe the purpose of this pull request.

## Type of change

Select all that apply:

- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Test update
- [ ] Refactor
- [ ] Performance improvement
- [ ] Methodological change
- [ ] Repository maintenance

## Affected modules

Select all affected areas:

- [ ] `constants.py`
- [ ] `config.py`
- [ ] `units.py`
- [ ] `radii.py`
- [ ] `geometry.py`
- [ ] `precipitation.py`
- [ ] `masks.py`
- [ ] `wind.py`
- [ ] `components.py`
- [ ] `index.py`
- [ ] `pipeline.py`
- [ ] `aggregation.py`
- [ ] `compiler.py`
- [ ] `quality_control.py`
- [ ] `io.py`
- [ ] `tracks.py`
- [ ] `rocloud.py`
- [ ] `examples/`
- [ ] `notebooks/`
- [ ] `docs/`
- [ ] GitHub Actions / CI

## Scientific or methodological changes

Does this PR change any scientific assumption or methodological rule?

- [ ] Yes
- [ ] No

If yes, describe the change.

Examples of methodological changes include:

- precipitation treatment;
- percentile thresholds;
- `R_direct_q` definition;
- missing-radius imputation;
- wind normalization;
- aggregation formula;
- quality-control rules.

## Tests

Commands run locally:

```bash
python -m pytest tests/
pre-commit run --all-files
python examples/smoke_test_synthetic.py
````

Test status:

* [ ] All tests pass locally
* [ ] `pre-commit` passes locally
* [ ] Synthetic example runs successfully
* [ ] GitHub Actions pass

## Documentation

Documentation updated:

* [ ] `README.md`
* [ ] `CHANGELOG.md`
* [ ] `docs/architecture.md`
* [ ] `docs/methodology.md`
* [ ] `docs/api.md`
* [ ] `docs/development.md`
* [ ] Not required

## Checklist

* [ ] Code is modular and does not duplicate existing logic
* [ ] Public functions include type hints
* [ ] Public functions include NumPy-style docstrings
* [ ] Tests were added or updated
* [ ] Outputs remain bounded in `[0, 1]` where applicable
* [ ] Metadata is preserved when relevant
* [ ] No large generated files were committed
* [ ] No private data were committed

## Additional notes

Add any additional context for reviewers.
