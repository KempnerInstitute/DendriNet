from __future__ import annotations

import hashlib
import json

import torch

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_extreme_cube,
    evaluate_dendritic_rnn_frozen_bank as evaluation,
)


def test_terminal_logits_selects_last_valid_timestep():
    logits = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        ]
    )
    lengths = torch.tensor([2, 3])

    terminal = evaluation._terminal_logits(
        logits,
        seq_lengths=lengths,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(terminal, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))


def test_semantic_state_hash_is_key_order_independent():
    first = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([2, 3]),
    }
    second = {"b": first["b"], "a": first["a"]}

    assert evaluation._semantic_state_sha256(
        first
    ) == evaluation._semantic_state_sha256(second)


def test_semantic_state_hash_supports_scalar_tensors():
    state = {"scalar": torch.tensor(1.25), "vector": torch.tensor([1.25])}

    digest = evaluation._semantic_state_sha256(state)

    assert len(digest) == 64


def test_evaluation_runtime_hash_detects_live_source_drift(tmp_path):
    source = tmp_path / "src/dendritic_modeling/models/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")

    before = evaluation.evaluation_runtime_source_sha256(tmp_path)
    source.write_text("VALUE = 2\n")
    after = evaluation.evaluation_runtime_source_sha256(tmp_path)

    assert before != after


def test_frozen_bank_json_hash_uses_canonical_terminal_newline():
    payload = ["trial-0", "trial-1"]
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    expected = hashlib.sha256(canonical).hexdigest()

    assert evaluation._sha256_json(payload) == expected
    assert analyze_dendritic_rnn_extreme_cube._sha256_json(payload) == expected
