from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.utils.dataset_fingerprint import (
    DATASET_FINGERPRINT_DEFINITION,
    fingerprint_dataset,
    fingerprint_dataset_splits,
)


def test_dataset_fingerprint_is_repeatable_and_content_sensitive():
    first = TensorDataset(torch.arange(12).reshape(4, 3), torch.tensor([0, 1, 0, 1]))
    same = TensorDataset(torch.arange(12).reshape(4, 3), torch.tensor([0, 1, 0, 1]))
    changed = TensorDataset(torch.arange(12).reshape(4, 3), torch.tensor([0, 1, 1, 1]))

    first_record = fingerprint_dataset(first)
    same_record = fingerprint_dataset(same)
    changed_record = fingerprint_dataset(changed)

    assert first_record == same_record
    assert first_record["content_sha256"] != changed_record["content_sha256"]
    assert first_record["definition"] == DATASET_FINGERPRINT_DEFINITION
    assert first_record["fields"]["tensors"]["kind"] == "sequence"


def test_split_fingerprint_binds_names_and_combined_content():
    train = TensorDataset(torch.ones(2, 3), torch.tensor([0, 1]))
    valid = TensorDataset(torch.zeros(1, 3), torch.tensor([1]))

    record = fingerprint_dataset_splits(train=train, valid=valid)
    reversed_call = fingerprint_dataset_splits(valid=valid, train=train)

    assert record == reversed_call
    assert set(record["splits"]) == {"train", "valid"}
    assert len(record["combined_sha256"]) == 64
