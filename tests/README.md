# Tests

The pytest suite of DendriNet. How to run it (commands, markers, coverage,
parallel runs) is described in the contributing guide,
[docs/contributing.rst](../docs/contributing.rst). Run it from the repository
root with `PYTHONPATH` pointing at the checkout under test, for example:

```bash
PYTHONPATH=$PWD/src python -m pytest -m "not slow" -n auto
```

## Layout

- `test_<name>.py`: the tests, one file per subsystem. Add new tests the same
  way.
- `conftest.py`: marks every test parametrized with a CUDA `device` as
  `requires_gpu` and skips those tests on hosts without CUDA, skips
  `requires_git_checkout` tests outside a Git checkout; keeps scheduler
  policies and data out of the tests (temporary data root, no site policy).
- `fixtures/`: reference data for tests, including the default-path goldens.
- `golden_cases.py`: the small models of the default-path goldens.

## Default-path goldens

`test_default_path_goldens.py` compares 28 small default-path models with
`fixtures/default_path_goldens.json` (initial state digests, outputs and input
gradients). A failure is a change of default behaviour: explain it before
regenerating the fixture with

```bash
PYTHONPATH=$PWD/src python tests/golden_cases.py --write --reason "<why>"
```

which records the reason, commit and numerical stack in the fixture history.

## Warnings

Warnings are shown, not hidden (`pytest.ini`). A test that expects a warning
asserts it with `pytest.warns`.
