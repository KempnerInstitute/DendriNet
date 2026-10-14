"""Variance Top-K input statistic: absolute variance or scale-free relative variance."""

from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

N_PIXELS, N_CUES = 6, 8


def _pixels_and_cue(n=4096, seed=0):
    """Large-scale pixel-like inputs under gain variation and a one-hot cue."""
    generator = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.empty(n, 1).uniform_(-3.0, 3.0, generator=generator))
    pixels = gain * torch.rand(n, N_PIXELS, generator=generator)
    cue = torch.nn.functional.one_hot(
        torch.randint(0, N_CUES, (n,), generator=generator), N_CUES
    ).float()
    return torch.cat([pixels, cue], dim=1)


def _layer(score):
    torch.manual_seed(0)
    layer = VarianceTopKLinear(
        N_PIXELS + N_CUES, 3, 2, weight_transform="exp", variance_score=score
    )
    with torch.no_grad():
        layer.pre_w.zero_()  # equal weights: selection follows the input score
    return layer


def test_absolute_score_never_selects_a_small_scale_cue_and_relative_does():
    x = _pixels_and_cue()
    absolute, relative = _layer("absolute"), _layer("relative")
    absolute.train()
    relative.train()
    for layer in (absolute, relative):
        layer(x)
    cue = slice(N_PIXELS, N_PIXELS + N_CUES)
    assert absolute.weight_mask()[:, cue].sum() == 0
    assert relative.weight_mask()[:, cue].sum() == 2 * relative.out_features


def test_relative_score_is_invariant_to_input_scale():
    x = _pixels_and_cue()
    scaled = x.clone()
    scaled[:, N_PIXELS:] *= 50.0  # recode the cue on a larger scale
    a, b = _layer("relative"), _layer("relative")
    a.train()
    b.train()
    a(x)
    b(scaled)
    torch.testing.assert_close(a._input_score(), b._input_score())
    constant_zero = _layer("relative")
    constant_zero.train()
    silent = x.clone()
    silent[:, -1] = 0.0
    constant_zero(silent)
    assert constant_zero._input_score()[-1] == 0


def test_default_absolute_score_matches_the_variance_buffer_and_relative_starts_by_weight():
    layer = _layer("absolute")
    assert layer._input_score() is layer.activation_var
    fresh = _layer("relative")
    torch.testing.assert_close(fresh._input_score(), torch.ones(N_PIXELS + N_CUES))
    with pytest.raises(ValueError, match="variance_score"):
        VarianceTopKLinear(8, 3, 2, variance_score="normalized")


def _variance_model(family, score, topk_type="variance"):
    config = _config(family, "shunting", "endpoint_tanh")
    if family == "population_network":
        options = config["population_network"]["layers"][0]["population_defaults"]
        options.update(topk_type=topk_type, topk_variance_score=score)
    else:
        config["sparsity"] = {"type": topk_type, "variance_score": score}
    torch.manual_seed(0)
    return get_architecture(family, deepcopy(config), input_dim=6)


@pytest.mark.parametrize("family", ["population_network", "einet"])
def test_option_reaches_every_variance_bank_through_the_builders(family):
    model = _variance_model(family, "relative")
    banks = [m for m in model.modules() if isinstance(m, VarianceTopKLinear)]
    assert banks and all(bank.variance_score == "relative" for bank in banks)
    default = _variance_model(family, "absolute")
    assert all(
        bank.variance_score == "absolute"
        for bank in default.modules()
        if isinstance(bank, VarianceTopKLinear)
    )


@pytest.mark.parametrize("family", ["population_network", "einet"])
def test_relative_score_is_rejected_for_other_topk_types(family):
    with pytest.raises(ValueError, match="variance_score"):
        _variance_model(family, "relative", topk_type="standard")
