"""Structured-connectivity rules mean the same through every builder.

A rule is written under ``model.core.connectivity.structured`` (structured
E/I cores, a typed section) or as a population's ``structured_connectivity``
(a plain mapping). Both are loaded from YAML here and built by the factory, and
their realized connection masks are compared.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.config.structured_connectivity import (
    canonical_rule,
    resolve_pathway_rule,
)
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    distance_kernel_probabilities,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

_INPUTS = 20
_COMPARTMENTS = 10
_PRE = [[float(index)] for index in range(_INPUTS)]
_POST = [[2.0 * index] for index in range(_COMPARTMENTS)]

_STRUCTURED = {
    "model": {
        "task": "classification",
        "encoder": {"type": "identity", "params": {"input_dim": _INPUTS}},
        "core": {
            "biological_neuron": False,
            "type": "dendritic_shunting",
            "architecture": {
                "excitatory_layer_sizes": [_COMPARTMENTS],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [1],
            },
            # One synaptic level of _COMPARTMENTS branches, each allowed every
            # input, so the excitatory bank keeps the whole structured mask.
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [_INPUTS],
                "ie_synapses_per_branch_per_layer": [0],
            },
        },
        "decoder": {"type": "identity", "params": {"hidden_dims": []}},
    }
}

_POPULATION = {
    "model": {
        "task": "classification",
        "encoder": {"type": "identity", "params": {"input_dim": _INPUTS}},
        "core": {
            "biological_neuron": False,
            "type": "population_network",
            "population_network": {
                "layers": [
                    {
                        "name": "layer0",
                        "readout_population": "e",
                        # A branch level and the soma, _COMPARTMENTS each.
                        "population_defaults": {
                            "ff_excitatory_synapses": _INPUTS,
                            "ff_inhibitory_synapses": 0,
                        },
                        "populations": [
                            {
                                "name": "e",
                                "polarity": "excitatory",
                                "n_neurons": _COMPARTMENTS,
                                "branch_factors": [1],
                            }
                        ],
                        "connections": [{"source": "input", "target": "e"}],
                    }
                ]
            },
        },
        "decoder": {"type": "identity", "params": {"hidden_dims": []}},
    }
}

_BUILDERS = ("structured", "population")


def _config(builder: str, rule: dict) -> dict:
    data = deepcopy(_STRUCTURED if builder == "structured" else _POPULATION)
    core = data["model"]["core"]
    if builder == "structured":
        core["connectivity"]["structured"] = deepcopy(rule)
    else:
        layer = core["population_network"]["layers"][0]
        layer["population_defaults"]["structured_connectivity"] = deepcopy(rule)
    return data


def _load(tmp_path, data: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(str(path))


def _masks(
    tmp_path, builder: str, rule: dict, *, torch_seed: int = 0
) -> list[torch.Tensor]:
    """Excitatory connection masks of every synaptic level, distal first."""
    config = _load(tmp_path, _config(builder, rule))
    torch.manual_seed(torch_seed)
    network = get_architecture(
        config.model.core.type, config.model.core, input_dim=_INPUTS
    )
    masks = [
        layer.branch_excitation.connection_mask.clone()
        for layer in network.modules()
        if isinstance(layer, DendriticBranchLayer)
        and layer.branch_excitation is not None
        and getattr(layer.branch_excitation, "connection_mask", None) is not None
    ]
    assert masks and all(
        tuple(mask.shape) == (_COMPARTMENTS, _INPUTS) for mask in masks
    )
    return masks


def _rule(pathway_rule: dict, **root) -> dict:
    return {"enabled": True, **root, "pathways": {"ee": pathway_rule}}


def _distance(**options) -> dict:
    return {
        "method": "distance",
        "pre_positions": _PRE,
        "post_positions": _POST,
        **options,
    }


def _same(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right)
    )


# ---------------------------------------------------------------------------
# Rule-key aliases
# ---------------------------------------------------------------------------


def test_aliases_rename_to_their_keys_and_override_typed_root_defaults():
    assert canonical_rule(
        {"p_connect": 0.2, "in_degree": 3, "kernel": "exponential", "sigma": 2.0},
        section="rule",
    ) == {
        "probability": 0.2,
        "indegree": 3,
        "distance_kernel": "exponential",
        "distance_sigma": 2.0,
    }
    # A typed block always carries these canonical root defaults.
    typed_root = {
        "enabled": True,
        "method": "bernoulli",
        "probability": None,
        "indegree": None,
        "distance_kernel": "none",
        "distance_sigma": 1.0,
        "spatial": {},
        "pathways": {
            "e_to_e": _distance(kernel="gaussian", sigma=3.0, base_probability=1.0)
        },
    }
    rule = resolve_pathway_rule(typed_root, "ee")
    assert rule["distance_kernel"] == "gaussian"
    assert rule["distance_sigma"] == 3.0
    assert "kernel" not in rule and "sigma" not in rule
    assert resolve_pathway_rule(typed_root, "ie") == {}


_ALIAS_PAIRS = {
    "distance": (
        _distance(distance_kernel="gaussian", distance_sigma=3.0, base_probability=0.9),
        _distance(kernel="gaussian", sigma=3.0, base_probability=0.9),
    ),
    "bernoulli": ({"probability": 0.3}, {"p_connect": 0.3}),
    "fixed_indegree": (
        {"method": "fixed_indegree", "indegree": 4},
        {"method": "fixed_indegree", "in_degree": 4},
    ),
}


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("method", sorted(_ALIAS_PAIRS))
def test_alias_and_canonical_spellings_build_identical_masks(tmp_path, builder, method):
    canonical, alias = _ALIAS_PAIRS[method]
    expected = _masks(tmp_path, builder, _rule(canonical, seed=7))
    assert not all(bool(mask.all()) for mask in expected)
    assert _same(_masks(tmp_path, builder, _rule(alias, seed=7)), expected)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_a_root_alias_is_a_default_for_every_pathway(tmp_path, builder):
    expected = _masks(
        tmp_path, builder, {"enabled": True, "seed": 7, "probability": 0.3}
    )
    assert _same(
        _masks(tmp_path, builder, {"enabled": True, "seed": 7, "p_connect": 0.3}),
        expected,
    )
    if builder == "structured":
        config = _load(
            tmp_path,
            _config(builder, {"enabled": True, "seed": 7, "in_degree": 4}),
        )
        assert config.model.core.connectivity.structured.indegree == 4


def test_a_population_pathway_alias_overrides_a_canonical_root_key(tmp_path):
    root = {"seed": 7, "distance_kernel": "exponential", "distance_sigma": 0.5}
    expected = _masks(
        tmp_path,
        "population",
        _rule(
            _distance(
                distance_kernel="gaussian", distance_sigma=3.0, base_probability=1.0
            ),
            **root,
        ),
    )
    aliased = _masks(
        tmp_path,
        "population",
        _rule(_distance(kernel="gaussian", sigma=3.0, base_probability=1.0), **root),
    )
    assert _same(aliased, expected)
    root_rule = _masks(
        tmp_path, "population", _rule(_distance(base_probability=1.0), **root)
    )
    assert not _same(root_rule, expected)


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize(
    "rule, message",
    [
        (_rule({"probability": 0.3, "p_connect": 0.3}), "both spellings"),
        (
            _rule(_distance(distance_sigma=2.0, sigma=2.0, base_probability=1.0)),
            "'distance_sigma' and its alias 'sigma'",
        ),
        (
            {
                "enabled": True,
                "pathways": {
                    "ee": {"probability": 0.3},
                    "e_to_e": {"probability": 0.5},
                },
            },
            "names one pathway more than once",
        ),
    ],
)
def test_a_key_or_pathway_written_twice_raises(tmp_path, builder, rule, message):
    with pytest.raises(ValueError, match=message):
        _masks(tmp_path, builder, rule)


# ---------------------------------------------------------------------------
# What the rules realize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_the_distance_rule_follows_the_positions(tmp_path, builder):
    rule = _distance(
        distance_kernel="gaussian", distance_sigma=0.5, base_probability=1.0
    )
    masks = _masks(tmp_path, builder, _rule(rule, seed=7))
    inputs = torch.arange(_INPUTS)
    for mask in masks:
        for row in range(_COMPARTMENTS):
            distance = (inputs - 2 * row).abs()
            # Probability 1 at distance 0 and below 2e-8 from distance 3 on.
            assert bool(mask[row, 2 * row])
            assert not bool(mask[row][distance >= 3].any())
    reversed_rule = dict(rule, post_positions=list(reversed(_POST)))
    flipped = _masks(tmp_path, builder, _rule(reversed_rule, seed=7))
    for mask in flipped:
        for row in range(_COMPARTMENTS):
            assert bool(mask[row, 2 * (_COMPARTMENTS - 1 - row)])


@pytest.mark.parametrize("builder", _BUILDERS)
def test_realized_contact_counts(tmp_path, builder):
    for mask in _masks(
        tmp_path, builder, _rule({"method": "fixed_indegree", "in_degree": 4}, seed=7)
    ):
        assert torch.equal(mask.sum(dim=1), torch.full((_COMPARTMENTS,), 4))
    for mask in _masks(tmp_path, builder, _rule({"p_connect": 0.25}, seed=7)):
        assert 0.15 <= mask.float().mean().item() <= 0.35
    probabilities = distance_kernel_probabilities(
        torch.tensor(_PRE),
        torch.tensor(_POST),
        kernel="gaussian",
        sigma=3.0,
        base_probability=0.9,
    )
    expected = probabilities.sum().item()
    spread = 4.0 * (probabilities * (1 - probabilities)).sum().sqrt().item()
    rule = _distance(kernel="gaussian", sigma=3.0, base_probability=0.9)
    for mask in _masks(tmp_path, builder, _rule(rule, seed=7)):
        assert abs(mask.sum().item() - expected) <= spread


@pytest.mark.parametrize("builder", _BUILDERS)
def test_seeded_masks_do_not_depend_on_the_global_stream(tmp_path, builder):
    rule = _rule({"p_connect": 0.5}, seed=11)
    first = _masks(tmp_path, builder, rule, torch_seed=0)
    assert _same(_masks(tmp_path, builder, rule, torch_seed=1), first)
    other_seed = _masks(tmp_path, builder, _rule({"p_connect": 0.5}, seed=12))
    assert not _same(other_seed, first)


# ---------------------------------------------------------------------------
# The connection probability at distance 0
# ---------------------------------------------------------------------------


def test_a_typed_distance_rule_without_base_probability_fails_at_load(tmp_path):
    rule = _rule(_distance(distance_kernel="gaussian"), seed=7)
    with pytest.raises(
        ValueError,
        match=r"structured\.pathways\.ee uses method 'distance' without "
        "base_probability",
    ):
        _load(tmp_path, _config("structured", rule))


@pytest.mark.parametrize("builder", _BUILDERS)
def test_the_rule_probability_stands_in_for_base_probability(tmp_path, builder):
    explicit = _rule(
        _distance(distance_kernel="gaussian", base_probability=0.4), seed=7
    )
    expected = _masks(tmp_path, builder, explicit)
    fallback = _rule(_distance(distance_kernel="gaussian", probability=0.4), seed=7)
    assert _same(_masks(tmp_path, builder, fallback), expected)


def test_a_population_distance_rule_without_any_probability_uses_one(tmp_path):
    explicit = _rule(
        _distance(distance_kernel="gaussian", base_probability=1.0), seed=7
    )
    omitted = _rule(_distance(distance_kernel="gaussian"), seed=7)
    assert _same(
        _masks(tmp_path, "population", omitted),
        _masks(tmp_path, "population", explicit),
    )


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize(
    "value, message", [(None, "base_probability is null"), ([0.2, 0.3], "one number")]
)
def test_a_base_probability_that_is_not_a_number_raises(
    tmp_path, builder, value, message
):
    rule = _rule(_distance(distance_kernel="gaussian", base_probability=value), seed=7)
    with pytest.raises(ValueError, match=message):
        _masks(tmp_path, builder, rule)


# ---------------------------------------------------------------------------
# seed: global_rng
# ---------------------------------------------------------------------------


def test_a_global_rng_seed_draws_population_masks_from_the_global_stream(tmp_path):
    rule = _rule({"probability": 0.5}, seed="global_rng")
    first = _masks(tmp_path, "population", rule, torch_seed=0)
    unseeded = _rule({"probability": 0.5})
    assert _same(_masks(tmp_path, "population", unseeded, torch_seed=0), first)
    assert not _same(_masks(tmp_path, "population", rule, torch_seed=1), first)


def test_training_keeps_a_global_rng_seed(tmp_path):
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_topology_seed_defaults,
    )

    rule = _rule({"probability": 0.5}, seed="global_rng")
    config = _load(tmp_path, _config("population", rule))
    _apply_topology_seed_defaults(config.model, 17)
    layer = config.model.core.population_network["layers"][0]
    assert layer["population_defaults"]["structured_connectivity"]["seed"] == (
        "global_rng"
    )


@pytest.mark.parametrize(
    "rule, message",
    [
        (_rule({"probability": 0.5}, seed="global_rng"), "applies to population"),
        (
            _rule({"probability": 0.5, "seed": "global_rng"}),
            r"pathways\.ee\.seed='global_rng' applies to population",
        ),
        (_rule({"probability": 0.5}, seed="seven"), "must be an integer"),
    ],
)
def test_structured_e_i_blocks_take_integer_seeds(tmp_path, rule, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _config("structured", rule))


def test_spatial_contacts_need_an_integer_seed(tmp_path):
    data = _config(
        "population",
        {
            "enabled": True,
            "method": "spatial_morphology",
            "seed": "global_rng",
            "spatial": {"input_shape": [1, 4, 5], "split_axes": ["height"]},
        },
    )
    layer = data["model"]["core"]["population_network"]["layers"][0]
    layer["population_defaults"]["topk_type"] = "indexed"
    config = _load(tmp_path, data)
    with pytest.raises(ValueError, match="spatial_morphology places its contacts"):
        get_architecture(config.model.core.type, config.model.core, input_dim=_INPUTS)


# ---------------------------------------------------------------------------
# What omitted keys mean (config_version 3; tests/test_upgrade_config_v3.py
# checks that upgraded version-2 files keep their earlier meaning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_a_distance_rule_without_a_kernel_is_gaussian(tmp_path, builder):
    rule = _rule(_distance(base_probability=1.0), seed=7)
    gaussian = _rule(
        _distance(distance_kernel="gaussian", base_probability=1.0), seed=7
    )
    masks = _masks(tmp_path, builder, rule)
    assert not all(bool(mask.all()) for mask in masks)
    assert _same(masks, _masks(tmp_path, builder, gaussian))
    if builder == "structured":
        config = _load(tmp_path, _config(builder, rule))
        assert config.model.core.connectivity.structured.distance_kernel == "gaussian"


@pytest.mark.parametrize("builder", _BUILDERS)
def test_masks_without_a_seed_follow_the_global_stream_outside_training(
    tmp_path, builder
):
    rule = _rule({"probability": 0.5})
    first = _masks(tmp_path, builder, rule, torch_seed=0)
    assert _same(_masks(tmp_path, builder, rule, torch_seed=0), first)
    assert not _same(_masks(tmp_path, builder, rule, torch_seed=1), first)


def test_training_seeds_every_unseeded_block_from_the_topology_seed(tmp_path):
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_topology_seed_defaults,
    )

    rule = _rule({"probability": 0.5})
    structured = _load(tmp_path, _config("structured", rule))
    _apply_topology_seed_defaults(structured.model, 17)
    assert structured.model.core.connectivity.structured.seed == 17
    population = _load(tmp_path, _config("population", rule))
    _apply_topology_seed_defaults(population.model, 17)
    layer = population.model.core.population_network["layers"][0]
    assert layer["population_defaults"]["structured_connectivity"]["seed"] == 17
