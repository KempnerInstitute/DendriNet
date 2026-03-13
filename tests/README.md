# Tests

The test suite uses `pytest` from the repository root. The custom
`tests/test_runner.py` wrapper is optional.

## Common Commands

Run the full suite:

```bash
python -m pytest
```

Run a single file:

```bash
python -m pytest tests/test_ei_network_regression.py -q
```

Run selected high-value suites:

```bash
python -m pytest \
    tests/test_unified_ei.py \
    tests/test_recurrent_trainer_amp.py \
    tests/test_local_learning_additive_fairness.py -q
```

Run by marker:

```bash
python -m pytest -m "not slow"
python -m pytest -m unit
python -m pytest -m integration
```

Coverage:

```bash
python -m pytest --cov=src/dendritic_modeling --cov-report=term-missing
python -m pytest --cov=src/dendritic_modeling --cov-report=html
```

## Markers

The current marker definitions are in `pytest.ini`:

- `slow`
- `integration`
- `unit`
- `requires_gpu`
- `requires_data`

## Custom Runner

`tests/test_runner.py` can still be used for ad hoc runs:

```bash
python tests/test_runner.py --check-deps
python tests/test_runner.py --use-pytest
python tests/test_runner.py --modules analysis config
```

## Notes

- This README does not mirror every test file. Use `ls tests/` or
  `pytest --collect-only` for the current inventory.
- Add new tests as `test_<name>.py` and run them with `pytest`.
