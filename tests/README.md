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
- `distributed_test_utils.py`: bounded spawning for distributed contract tests.
  Workers dump their stacks after two minutes, and the parent stops and reaps
  them on a five-minute deadline or a pytest interruption. Worker exceptions
  still fail the test. This also prevents timed-out children from keeping the
  pytest process alive during interpreter shutdown.

## Default-path goldens

`test_default_path_goldens.py` compares 28 small default-path models with
`fixtures/default_path_goldens.json` (initial state digests, outputs and input
gradients). A failure is a change of default behaviour: explain it before
regenerating the fixture with

```bash
PYTHONPATH=$PWD/src python tests/golden_cases.py --write --reason "<why>"
```

which records the reason, commit and numerical stack in the fixture history.
The cases run in a child process with fixed MKL, oneDNN and PyTorch CPU code
paths (`PINNED_NUMERICS` in `golden_cases.py`), so the recorded bytes hold on
every x86 CPU, with or without AVX-512.

## Warnings

Warnings are shown, not hidden (`pytest.ini`). A test that expects a warning
asserts it with `pytest.warns`.

The format-4 biological default is tested in `test_biological_default.py`.
Historical fixtures that exercise signed gates, shared E/I inputs or other
unconstrained paths now explicitly set `biological_neuron: false`. This keeps
their equations and numerical goldens unchanged. The stored golden outputs
are not regenerated when only this validation declaration changes.
