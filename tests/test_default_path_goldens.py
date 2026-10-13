"""Default-path states and outputs match the recorded cross-commit goldens.

The cases live in ``tests/golden_cases.py`` and their records in
``tests/fixtures/default_path_goldens.json``. A mismatch is a change of
default behaviour: explain it, and only then regenerate the fixture with
``python tests/golden_cases.py --write --reason "<why>"``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from golden_cases import CASES, FIXTURE, run_case

GOLDENS = json.loads(FIXTURE.read_text())


def test_every_case_has_a_golden_and_a_recorded_reason():
    assert set(GOLDENS["cases"]) == set(CASES)
    assert all(entry["reason"].strip() for entry in GOLDENS["history"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_its_golden(name):
    record, golden = run_case(name), GOLDENS["cases"][name]
    assert record["parameter_count"] == golden["parameter_count"]
    np.testing.assert_allclose(record["output"], golden["output"], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        record["input_grad"], golden["input_grad"], atol=1e-6, rtol=1e-5
    )
    assert (
        record["state_sha256"] == golden["state_sha256"]
    ), f"{name}: the initial state differs from the golden"
