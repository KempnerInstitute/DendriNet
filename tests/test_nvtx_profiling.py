"""Tests for optional NVTX instrumentation."""

from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    topk,
)
from dendritic_modeling.utils import profiling


def test_nvtx_range_is_noop_without_cuda(monkeypatch):
    range_push = Mock()
    range_pop = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)

    with profiling.nvtx_range("test/no_cuda"):
        pass

    range_push.assert_not_called()
    range_pop.assert_not_called()


def test_nvtx_range_balances_push_and_pop_on_exception(monkeypatch):
    range_push = Mock()
    range_pop = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)

    with pytest.raises(ValueError, match="expected"):
        with profiling.nvtx_range("test/balanced"):
            raise ValueError("expected")

    range_push.assert_called_once_with("test/balanced")
    range_pop.assert_called_once_with()


def test_topk_forward_labels_dense_masked_computation(monkeypatch):
    entered_ranges = []

    @contextmanager
    def record_range(message):
        entered_ranges.append(message)
        yield

    monkeypatch.setattr(topk, "nvtx_range", record_range)
    layer = topk.TopKLinear(in_features=4, out_features=2, K=2)
    layer.initialize()

    output = layer(torch.ones(3, 4))

    assert output.shape == (3, 2)
    assert entered_ranges == [
        "dendrite/topk/select_mask",
        "dendrite/topk/apply_dense_mask",
        "dendrite/topk/dense_matmul",
    ]
