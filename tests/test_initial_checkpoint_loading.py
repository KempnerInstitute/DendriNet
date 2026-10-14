"""General hash-bound strict model checkpoint loading."""

import pytest
import torch

from dendritic_modeling.networks.checkpoints import (
    load_initial_model_checkpoint,
    sha256_file,
)


def test_initial_checkpoint_loader_is_hash_bound_and_strict(tmp_path):
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "linear.pt"
    torch.save(source.state_dict(), checkpoint)
    destination = torch.nn.Linear(3, 2)

    record = load_initial_model_checkpoint(
        destination,
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        strict=True,
    )

    assert record["strict"] is True
    assert record["missing_keys"] == []
    assert record["unexpected_keys"] == []
    assert all(
        torch.equal(expected, observed)
        for expected, observed in zip(
            source.state_dict().values(), destination.state_dict().values(), strict=True
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_initial_model_checkpoint(
            destination,
            checkpoint,
            expected_sha256="0" * 64,
        )
