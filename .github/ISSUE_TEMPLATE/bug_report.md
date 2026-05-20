---
name: Bug report
about: Report an error or unexpected behavior in ITCHI Core
title: "[BUG] "
labels: bug
assignees: ""
---

## Description

Describe the bug clearly and concisely.

## Where does it occur?

Select the affected area:

- [ ] `precipitation.py`
- [ ] `geometry.py`
- [ ] `radii.py`
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
- [ ] Documentation
- [ ] Tests
- [ ] Other

## Steps to reproduce

Provide the minimal steps needed to reproduce the issue.

```bash
# Example
python -m pytest tests/test_pipeline.py -v
````

## Expected behavior

Describe what you expected to happen.

## Actual behavior

Describe what actually happened.

## Error message or traceback

```text
Paste the full error message here
````

## Environment

* OS:
* Python version:
* Installation method:

  * [ ] conda
  * [ ] pip
  * [ ] editable install with `pip install -e .`
* Branch or commit:

## Scientific or methodological relevance

Does this bug affect the scientific interpretation of ITCHI?

* [ ] Yes
* [ ] No
* [ ] Unsure

If yes, explain briefly.

## Additional context

Add any other relevant information.
