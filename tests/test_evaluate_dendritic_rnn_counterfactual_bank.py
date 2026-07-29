from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.scripts.analysis import (
    evaluate_dendritic_rnn_counterfactual_bank as counterfactual,
)


def test_expected_identity_logits_joins_in_identity_order() -> None:
    payload = {
        "predictions": [
            {"trial_id": "b", "logits": [3.0, 4.0]},
            {"trial_id": "a", "logits": [1.0, 2.0]},
        ]
    }

    result = counterfactual._expected_identity_logits(payload, ["a", "b"])

    assert result.dtype == np.float32
    assert np.array_equal(
        result, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )


def test_expected_identity_logits_rejects_duplicate_ordinary_ids() -> None:
    payload = {
        "predictions": [
            {"trial_id": "a", "logits": [1.0, 2.0]},
            {"trial_id": "a", "logits": [3.0, 4.0]},
        ]
    }

    with pytest.raises(counterfactual.CounterfactualEvaluationError):
        counterfactual._expected_identity_logits(payload, ["a"])


def test_repository_study_config_has_frozen_four_role_design() -> None:
    path = (
        counterfactual.REPO_ROOT
        / "configs/dendritic_rnn/counterfactual_psychometric_v1.yaml"
    )

    payload, resolved = counterfactual.load_study_config(path)

    assert resolved == path.resolve()
    assert payload["study"]["replicates"] == 10
    assert set(payload["study"]["roles"]) == set(counterfactual.ROLE_CELLS)
