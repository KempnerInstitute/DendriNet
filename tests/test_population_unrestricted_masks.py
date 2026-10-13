"""Unrestricted sparse streams must not allocate dense temporary masks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.population_layer import (
    _DELAYED,
    _FF_EXC,
    _REC_EXC,
    _SAME_STEP,
    PopulationLayer,
)


def _layer(probability=1.0, enabled=True, self_connection=False):
    layer = PopulationLayer.__new__(PopulationLayer)
    torch.nn.Module.__init__(layer)
    source = "e" if self_connection else "input"
    timing = _DELAYED if self_connection else _SAME_STEP
    pathway = _REC_EXC if self_connection else _FF_EXC
    key = ("e", timing, pathway)
    layer._sources_by_target_pathway = {key: [source]}
    layer._connection_by_target_pathway_source = {
        (*key, source): SimpleNamespace(
            probability=probability, enabled=enabled, seed=404
        )
    }
    layer._population_dims = {"e": 3}
    layer.config = SimpleNamespace(name="test", connection_seed=404)
    layer._source_dim = lambda _: 3
    return layer, {
        "target": "e",
        "timing": timing,
        "pathway": pathway,
        "level_dims": [6, 3],
        "allow_self_recurrence": False,
    }


def test_huge_unrestricted_stream_never_allocates_mask():
    layer, kwargs = _layer()
    kwargs["level_dims"] = [1_450_000, 725_000, 362_500]
    layer._build_mask_level = Mock(side_effect=AssertionError("dense allocation"))
    rng = torch.random.get_rng_state().clone()
    assert layer._build_masks_for_stream(**kwargs) is None
    layer._build_mask_level.assert_not_called()
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "probability,enabled", [(0.5, True), (0.0, True), (1.0, False)]
)
def test_restricted_stream_keeps_exact_masks(probability, enabled):
    layer, kwargs = _layer(probability, enabled)
    masks = layer._build_masks_for_stream(**kwargs)
    expected = [
        layer._build_mask_level(
            **{k: v for k, v in kwargs.items() if k != "level_dims"},
            sources=["input"],
            out_dim=dim,
            level_idx=i,
        )
        for i, dim in enumerate(kwargs["level_dims"])
    ]
    assert masks is not None
    assert all(torch.equal(a, b) for a, b in zip(masks, expected))


def test_delayed_self_exclusion_is_preserved():
    layer, kwargs = _layer(self_connection=True)
    masks = layer._build_masks_for_stream(**kwargs)
    for mask in masks:
        expected = torch.ones_like(mask)
        owner = torch.arange(len(mask)) // (len(mask) // 3)
        expected[torch.arange(len(mask)), owner] = False
        assert torch.equal(mask, expected)
    kwargs["allow_self_recurrence"] = True
    layer._build_mask_level = Mock(side_effect=AssertionError("dense allocation"))
    assert layer._build_masks_for_stream(**kwargs) is None


def _legacy_masks(self, *, target, timing, pathway, level_dims, allow_self_recurrence):
    sources = self._stream_sources(target, timing, pathway)
    if not sources:
        return None
    masks = [
        self._build_mask_level(
            target=target,
            timing=timing,
            pathway=pathway,
            sources=sources,
            out_dim=dim,
            level_idx=i,
            allow_self_recurrence=allow_self_recurrence,
        )
        for i, dim in enumerate(level_dims)
    ]
    return None if all(bool(m.all()) for m in masks) else masks


def test_network_state_rng_forward_and_gradients_unchanged(monkeypatch):
    cfg = {
        "population_network": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 1,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": True,
                        "dbl_init_method": "default",
                        "weight_transform": "exp",
                    },
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ]
        }
    }
    torch.manual_seed(404)
    actual = get_architecture("population_network", cfg, input_dim=4)
    actual_rng = torch.random.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(PopulationLayer, "_build_masks_for_stream", _legacy_masks)
        torch.manual_seed(404)
        reference = get_architecture("population_network", cfg, input_dim=4)
    assert torch.equal(actual_rng, torch.random.get_rng_state())
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for name, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
    x = torch.randn(2, 4)
    outputs, gradients = [], []
    for model in (actual, reference):
        xx = x.clone().requires_grad_()
        yy = model(xx)
        yy.square().sum().backward()
        outputs.append(yy.detach())
        gradients.append([xx.grad] + [p.grad for p in model.parameters()])
    assert torch.equal(*outputs)
    for a, b in zip(*gradients):
        assert (a is None and b is None) or (
            a is not None and b is not None and torch.equal(a, b)
        )
