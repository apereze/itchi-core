---
name: Feature request
about: Suggest a new capability or methodological extension for ITCHI Core
title: "[FEATURE] "
labels: enhancement
assignees: ""
---

## Summary

Describe the proposed feature clearly.

## Motivation

Why is this feature needed?

Explain whether it is related to:

- [ ] Scientific methodology
- [ ] Data input/output
- [ ] Real-data workflow
- [ ] Testing
- [ ] Documentation
- [ ] Performance
- [ ] Usability
- [ ] Other

## Proposed functionality

Describe what the feature should do.

## Proposed API

If applicable, suggest the expected function, class or module name.

```python
# Example
from itchi.module import new_function

result = new_function(...)
````

## Expected inputs

List the expected inputs.

| Input | Type | Description |
| ----- | ---- | ----------- |
|       |      |             |

## Expected outputs

List the expected outputs.

| Output | Type | Description |
| ------ | ---- | ----------- |
|        |      |             |

## Scientific assumptions

Document any relevant methodological assumptions.

Examples:

* precipitation should be treated as snapshot;
* missing quadrants should be filled from available values;
* `R_direct_q` may differ from observed `R34_q`;
* wind hazard may be computed from `Vmax` and `RMW`.

## Alternatives considered

Describe any alternative approaches considered.

## Tests required

Which tests should be added?

* [ ] NumPy input test
* [ ] xarray input test
* [ ] missing-data test
* [ ] bounds test in `[0, 1]`
* [ ] quality-control test
* [ ] real-data or synthetic example

## Documentation required

Which documentation should be updated?

* [ ] `README.md`
* [ ] `docs/architecture.md`
* [ ] `docs/methodology.md`
* [ ] `docs/api.md`
* [ ] `docs/development.md`
* [ ] notebook
* [ ] example script

## Additional context

Add any other relevant information.
