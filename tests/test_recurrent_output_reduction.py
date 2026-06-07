"""Regression tests for variable-length recurrent output reduction."""

import torch

from dendritic_modeling.networks.architectures.recurrent.baseline_rnn import BaselineRNN
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.networks.architectures.recurrent.population_graph import (
    PopulationGraphNetwork,
)


def _check_reducer(reducer, mode_attr: str, mode_value: str) -> None:
    setattr(reducer, mode_attr, mode_value)
    outputs = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    seq_lengths = torch.tensor([2, 4])

    reduced = reducer._reduce_outputs(outputs, seq_lengths=seq_lengths)

    if mode_value == "mean":
        expected = torch.stack([outputs[0, :2].mean(dim=0), outputs[1].mean(dim=0)])
        assert torch.equal(reduced, expected)
    else:
        assert torch.equal(reduced[0, 2:], torch.zeros_like(reduced[0, 2:]))
        assert torch.equal(reduced[0, :2], outputs[0, :2])
        assert torch.equal(reduced[1], outputs[1])


def _check_zero_length_policy(reducer, mode_attr: str, mode_value: str) -> None:
    setattr(reducer, mode_attr, mode_value)
    outputs = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    seq_lengths = torch.tensor([0, 2])

    reduced = reducer._reduce_outputs(outputs, seq_lengths=seq_lengths)

    if mode_value == "mean":
        expected = torch.stack([torch.zeros(2), outputs[1, :2].mean(dim=0)])
        assert torch.equal(reduced, expected)
    else:
        assert torch.equal(reduced[0], torch.zeros_like(reduced[0]))
        assert torch.equal(reduced[1, :2], outputs[1, :2])
        assert torch.equal(reduced[1, 2:], torch.zeros_like(reduced[1, 2:]))


def test_baseline_rnn_masks_mean_and_all_outputs_with_seq_lengths():
    reducer = BaselineRNN.__new__(BaselineRNN)

    _check_reducer(reducer, "_output_mode", "mean")
    _check_reducer(reducer, "_output_mode", "all")
    _check_zero_length_policy(reducer, "_output_mode", "mean")
    _check_zero_length_policy(reducer, "_output_mode", "all")


def test_ei_network_masks_mean_and_all_outputs_with_seq_lengths():
    reducer = EINetwork.__new__(EINetwork)

    _check_reducer(reducer, "output_mode", "mean")
    _check_reducer(reducer, "output_mode", "all")
    _check_zero_length_policy(reducer, "output_mode", "mean")
    _check_zero_length_policy(reducer, "output_mode", "all")


def test_population_graph_masks_mean_and_all_outputs_with_seq_lengths():
    reducer = PopulationGraphNetwork.__new__(PopulationGraphNetwork)

    _check_reducer(reducer, "output_mode", "mean")
    _check_reducer(reducer, "output_mode", "all")
    _check_zero_length_policy(reducer, "output_mode", "mean")
    _check_zero_length_policy(reducer, "output_mode", "all")
