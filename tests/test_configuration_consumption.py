"""Configuration values reach supported builders or fail explicitly."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture


def _build(config, *, family=None, input_dim=6, seed=1234):
    torch.manual_seed(seed)
    core_type = config.get("type", family or "population_network")
    return get_architecture(core_type, deepcopy(config), input_dim=input_dim)


def _population(**population_defaults):
    config = _config("population_network", "shunting", "param_tanh")
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"].update(population_defaults)
    return config


def _make_recurrent(config):
    layer = config["population_network"]["layers"][0]
    layer["recurrent"] = True
    layer["population_defaults"]["rec_excitatory_synapses"] = 1
    layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    return config


def _synapse_grads(model, x):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(7)  # identical release draws for stochastic synapses
    out = model(x)
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(11))
    (out * weights).sum().backward()
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if name.endswith("pre_w") and param.grad is not None
    }


def _assert_same_parameters(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key]), key


def _assert_every_nonzero_gradient_changed(reference, changed):
    assert reference and reference.keys() == changed.keys()
    compared = 0
    for name, gradient in reference.items():
        if torch.count_nonzero(gradient) == 0:
            continue
        compared += 1
        assert not torch.allclose(gradient, changed[name], rtol=1e-3, atol=0), name
    assert compared


_SYNAPSE_TYPES = {
    "standard": {},
    "indexed": {"indexed_seed": 5},
    "indexed_release": {"indexed_seed": 5},
    "deepst": {"excitatory_target_density": 0.5},
}


# Item 1: topk_strategy=conductance_dynamic in feedforward populations.
@pytest.mark.parametrize("topk_type", sorted(_SYNAPSE_TYPES))
def test_feedforward_population_conductance_dynamic_scales_synapse_gradients(
    topk_type,
):
    extra = dict(_SYNAPSE_TYPES[topk_type], topk_type=topk_type)
    plain = _build(_population(**extra))
    scaled = _build(_population(topk_strategy="conductance_dynamic", **extra))
    _assert_same_parameters(plain, scaled)
    x = torch.rand(5, 6, generator=torch.Generator().manual_seed(3))

    unscaled_grads = _synapse_grads(plain, x)
    forward_hook_grads = _synapse_grads(scaled, x)
    _assert_every_nonzero_gradient_changed(unscaled_grads, forward_hook_grads)

    # EINet scales the same layers through backward parameter hooks. Switching
    # this model to that mechanism must reproduce the forward-hook gradients.
    for module in scaled.modules():
        if isinstance(module, DendriticBranchLayer):
            module.set_forward_dynamic_grad_scaling(False)
    backward_hook_grads = _synapse_grads(scaled, x)
    for name, gradient in forward_hook_grads.items():
        torch.testing.assert_close(gradient, backward_hook_grads[name])


@pytest.mark.parametrize("cross_level_mode", ["all_active_star", "parallel_readout"])
def test_feedforward_cross_level_controls_apply_conductance_dynamic(cross_level_mode):
    plain = _build(_population(cross_level_mode=cross_level_mode))
    scaled = _build(
        _population(
            cross_level_mode=cross_level_mode, topk_strategy="conductance_dynamic"
        )
    )
    x = torch.rand(5, 6, generator=torch.Generator().manual_seed(3))
    _assert_every_nonzero_gradient_changed(
        _synapse_grads(plain, x), _synapse_grads(scaled, x)
    )


@pytest.mark.parametrize("cross_level_mode", ["all_active_star", "parallel_readout"])
def test_pooled_soma_gradient_scale_includes_its_pooled_conductance(
    cross_level_mode, monkeypatch
):
    """The soma's scale is its pooled denominator, local terms plus a constant G."""
    from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
        StatefulDendriNet,
    )

    model = _build(
        _population(
            cross_level_mode=cross_level_mode,
            topk_strategy="conductance_dynamic",
            somatic_synapses=True,  # synapses at the soma for the scale to reach
        )
    ).train()
    population = next(m for m in model.modules() if isinstance(m, StatefulDendriNet))
    soma = population.branch_layers[-1]
    synapses = soma.branch_excitation
    scales = []
    register = synapses.register_forward_param_gradient_scale

    def capture(scale):
        scales.append(scale.detach().reshape(-1).clone())
        register(scale)

    monkeypatch.setattr(synapses, "register_forward_param_gradient_scale", capture)
    pooled_minus_local = []
    for seed in (3, 4):
        x = torch.rand(5, 6, generator=torch.Generator().manual_seed(seed))
        model(x)
        with torch.no_grad():
            local = soma.effective_leak + soma.compute_raw_currents(x=x)[0]
        pooled_minus_local.append(scales[-1] - local.mean(dim=0))
    # The pooled conductance G is positive and does not depend on the input.
    assert torch.all(pooled_minus_local[0] > 0)
    torch.testing.assert_close(pooled_minus_local[0], pooled_minus_local[1])


def test_recurrent_population_deepst_receives_conductance_dynamic_scaling():
    extra = {"topk_type": "deepst", "excitatory_target_density": 0.5}
    plain = _build(_make_recurrent(_population(**extra)))
    scaled = _build(
        _make_recurrent(_population(topk_strategy="conductance_dynamic", **extra))
    )
    _assert_same_parameters(plain, scaled)
    x = torch.rand(3, 4, 6, generator=torch.Generator().manual_seed(3))
    _assert_every_nonzero_gradient_changed(
        _synapse_grads(plain, x), _synapse_grads(scaled, x)
    )


def _einet(rule="shunting", **sections):
    """Feedforward EINet config from the shared helper with section overrides."""
    config = _config("einet", rule, "param_tanh")
    for section, values in sections.items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    return config


def _ei_einet():
    """Feedforward EINet with an inhibitory population (needed by ss_mlp)."""
    config = _einet()
    config["architecture"].update(
        {"inhibitory_layer_sizes": [3], "inhibitory_branch_factors": [2]}
    )
    config["connectivity"].update(
        {
            "ee_synapses_per_branch_per_layer": [4],
            "ei_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [1],
        }
    )
    config["transfer"] = {}
    return config


def _typed(core_dict, *, input_dim=6, seed=1234):
    from dendritic_modeling.config.config import _build_core_config

    core = _build_core_config(deepcopy(core_dict))
    torch.manual_seed(seed)
    return get_architecture(core.type, core, input_dim=input_dim)


def _model_payload(core, **model_fields):
    return {
        "task": "classification",
        "encoder": {"type": "identity", "params": {"input_dim": 6}},
        "core": core,
        "decoder": {"type": "identity", "params": {"hidden_dims": []}},
        **model_fields,
    }


# Item 2: n_neurons / branch_factors belong to the population definition.
@pytest.mark.parametrize("location", ["population_defaults", "population"])
@pytest.mark.parametrize("key, value", [("branch_factors", [2]), ("n_neurons", 7)])
def test_population_definition_fields_reject_override_copies(location, key, value):
    _build(_population())
    config = _population()
    layer = config["population_network"]["layers"][0]
    if location == "population_defaults":
        layer["population_defaults"][key] = value
    else:
        layer["populations"][0]["population"] = {key: value}
    with pytest.raises(ValueError, match="population definition"):
        _build(config)


# Item 3: the recurrent bridge has no signed, merged-MLP or flat variant.
@pytest.mark.parametrize("core_type", ["dendritic_signed", "flat_signed", "flat_mlp"])
def test_recurrent_bridge_rejects_signed_and_mlp_aliases(core_type):
    config = _einet("raw")
    config["type"] = core_type
    _build(config)
    config["connectivity"]["rec_ee_synapses_per_branch"] = [1]
    with pytest.raises(ValueError, match="Recurrent mode is not supported"):
        _build(config)


# Item 4: feedforward-only EINet options are not translated by the bridge.
@pytest.mark.parametrize(
    "section, values",
    [
        ("reactivation", {"soma_type": "param_relu"}),
        ("reactivation", {"soma_init_m": 3.0}),
        ("architecture", {"inhibitory_network_type": "mlp"}),
        ("implementation", {"compile_forward": True}),
    ],
)
def test_recurrent_einet_rejects_feedforward_only_options(section, values):
    config = _einet()
    config["connectivity"]["rec_ee_synapses_per_branch"] = [1]
    _build(config)
    config.setdefault(section, {}).update(values)
    with pytest.raises(ValueError, match="feedforward-EINet only"):
        _build(config)


# Item 5: legacy recurrent_ei spellings are shadowed by typed defaults.
@pytest.mark.parametrize(
    "legacy",
    [
        {"recurrent_ei": {"rec_excitatory_synapses": 3}},
        {"recurrent_ei": {"input_projection_dims": [8]}},
        {"recurrent_ei": {"store_routing": True}},
        {"recurrent_ei": {"soma_feedback": {"enabled": True}}},
        {"dynamics": {"soma_feedback": {"enabled": True}}},
    ],
)
def test_typed_loader_rejects_shadowed_legacy_recurrent_keys(legacy):
    from dendritic_modeling.config.config import _build_core_config

    _build_core_config(_einet())
    with pytest.raises(ValueError, match="canonical location"):
        _build_core_config({**_einet(), **legacy})


def test_typed_loader_rejects_retired_recurrent_transfer_inhibitory_mode():
    from dendritic_modeling.config.config import _build_core_config

    legacy = {"recurrent_ei": {"transfer_inhibitory_mode": "all"}}
    with pytest.raises(ValueError, match="transfer_inhibitory_mode was removed"):
        _build_core_config({**_einet(), **legacy})


# Item 6: core.initialization_seed reaches feedforward EINet construction.
def _einet_parameters(config, *, global_seed, input_dim=3):
    model = _build(config, input_dim=input_dim, seed=global_seed)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_einet_initialization_seed_selects_keyed_streams():
    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    config = _einet(architecture={"excitatory_layer_sizes": [3, 3]})
    config["connectivity"]["ee_synapses_per_branch_per_layer"] = [2, 2]
    config["connectivity"]["ie_synapses_per_branch_per_layer"] = [0, 0]
    assert "initialization_seed" not in prepare_ei_network_params(config, 3)
    unseeded = [_einet_parameters(config, global_seed=s) for s in (1, 2)]
    assert any(
        not torch.equal(unseeded[0][name], unseeded[1][name]) for name in unseeded[0]
    )
    keyed = {
        seed: [
            _einet_parameters({**config, "initialization_seed": seed}, global_seed=s)
            for s in (1, 2)
        ]
        for seed in (5, 6)
    }
    pre_w = [name for name in keyed[5][0] if name.endswith("pre_w")]
    for name in keyed[5][0]:
        # Independent of the global stream, and a function of the seed.
        assert torch.equal(keyed[5][0][name], keyed[5][1][name]), name
    assert any(not torch.equal(keyed[5][0][n], keyed[6][0][n]) for n in pre_w)
    # Layers with identical shapes draw from distinct namespaced streams.
    layer0 = keyed[5][0][
        "layers.0.excitatory_cells.branch_layers.0.branch_excitation.pre_w"
    ]
    layer1 = keyed[5][0][
        "layers.1.excitatory_cells.branch_layers.0.branch_excitation.pre_w"
    ]
    assert layer0.shape == layer1.shape and not torch.equal(layer0, layer1)


def test_training_seed_defaults_keep_feedforward_einet_on_global_stream():
    from dendritic_modeling.config.config import _build_model_config
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_initialization_seed_defaults,
    )

    feedforward = _build_model_config(_model_payload(_einet()))
    _apply_initialization_seed_defaults(feedforward, 41)
    assert feedforward.core.initialization_seed is None

    recurrent_core = _einet()
    recurrent_core["connectivity"]["rec_ee_synapses_per_branch"] = [1]
    recurrent = _build_model_config(_model_payload(recurrent_core))
    _apply_initialization_seed_defaults(recurrent, 41)
    assert recurrent.core.initialization_seed == 41

    explicit = _build_model_config(
        _model_payload({**_einet(), "initialization_seed": 7})
    )
    _apply_initialization_seed_defaults(explicit, 41)
    assert explicit.core.initialization_seed == 7


def _tiny_alexnet(**vision):
    config = _einet()
    config["type"] = "dendritic_alexnet"
    config["architecture"].update(
        {"excitatory_layer_sizes": [4], "inhibitory_layer_sizes": [1]}
    )
    config["connectivity"].update(
        {
            "ee_synapses_per_branch_per_layer": [4],
            "ei_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [0],
        }
    )
    config["transfer"] = {"output_activation": "relu"}
    config["vision"] = {
        "input_channels": 3,
        "input_transform": "signed_split",
        "channels": [4, 4, 4, 4, 4],
        "kernels": [3, 3, 3, 3, 3],
        "strides": [1, 1, 1, 1, 1],
        "paddings": [1, 1, 1, 1, 1],
        "pool_after": [0, 1, 4],
        "pool_kernel": 2,
        "pool_stride": 2,
        "adaptive_pool_shape": [1, 1],
        "classifier_widths": [6, 6],
        "inhibitory_ratio": 0.25,
        "num_classes": 3,
        **vision,
    }
    return config


def test_dendritic_alexnet_derives_keyed_block_initialization_from_seed():
    def params(config, global_seed):
        model = _build(config, input_dim=3, seed=global_seed)
        return [p.detach().clone() for p in model.parameters()]

    unseeded = [params(_tiny_alexnet(), s) for s in (1, 2)]
    assert any(not torch.equal(a, b) for a, b in zip(*unseeded))
    seeded = [params(_tiny_alexnet(seed=9), s) for s in (1, 2)]
    # Dropout-free parameters are fully keyed by the configured seed; an
    # explicit null vision.seed falls back to core.initialization_seed.
    fallback = params({**_tiny_alexnet(seed=None), "initialization_seed": 9}, 3)
    for left, right, other in zip(seeded[0], seeded[1], fallback):
        assert torch.equal(left, right) and torch.equal(left, other)


# Item 7: feedforward-only cores reject recurrent and stateful requests.
_FEEDFORWARD_CORES = {
    "tensor_map": ({}, 12),
    "hierarchical_dendritic_conv": (
        {"spatial": {"kernel_size": 3, "stride": 1, "padding": 1}},
        3,
    ),
    "ss_mlp": ({}, 12),
}
_RECURRENT_REQUESTS = [
    {"connectivity": {"rec_ee_synapses_per_branch": [2]}},
    {"synapse_types": {"enabled": True, "excitatory": {"ampa": {"tau": 5.0}}}},
    {"dynamics": {"mode": "spike"}},
    {"soma_feedback": {"enabled": True}},
]


@pytest.mark.parametrize("core_type", sorted(_FEEDFORWARD_CORES))
@pytest.mark.parametrize("request_index", range(len(_RECURRENT_REQUESTS)))
def test_feedforward_cores_reject_recurrent_and_stateful_options(
    core_type, request_index
):
    extra, input_dim = _FEEDFORWARD_CORES[core_type]
    config = {**_ei_einet(), "type": core_type, **deepcopy(extra)}
    _build(config, input_dim=input_dim)
    for section, values in _RECURRENT_REQUESTS[request_index].items():
        config.setdefault(section, {}).update(deepcopy(values))
    with pytest.raises(ValueError, match="builds a feedforward dendritic network"):
        _build(config, input_dim=input_dim)


def test_ss_mlp_rejects_initialization_seed():
    config = {**_ei_einet(), "type": "ss_mlp"}
    _build(config, input_dim=12)
    with pytest.raises(ValueError, match="does not support initialization_seed"):
        _build({**config, "initialization_seed": 3}, input_dim=12)


# Item 8: DeepST regrowth is identical for every registered init method.
def test_deepst_init_method_rejects_equivalent_registered_methods():
    base = {"topk_type": "deepst", "excitatory_target_density": 0.5}
    _build(_population(**base, init_method="xavier_normal"))
    _build(_population(**base, init_method=None))
    with pytest.raises(ValueError, match=r"deepst\.init_method"):
        _build(_population(**base, init_method="kaiming_normal"))


def test_topk_init_method_is_overwritten_by_the_branch_initializer():
    default = _build(_einet())
    other = _build(_einet(sparsity={"init_method": "kaiming_normal"}))
    _assert_same_parameters(default, other)


# Item 9: options not passed to the selected sparse synapse class.
@pytest.mark.parametrize(
    "sparse_type, option",
    [
        ("indexed", {"topk_noise_level": 0.5}),
        ("indexed_rewire", {"topk_noise_level": 0.5}),
        ("deepst", {"topk_noise_level": 0.5}),
        ("stochastic", {"topk_weight_norm_order": 2}),
        ("deepst", {"topk_gamma": 2.0}),
    ],
)
def test_sparse_types_reject_options_they_ignore(sparse_type, option):
    base = {"topk_type": sparse_type, "indexed_seed": 5}
    if sparse_type == "deepst":
        base = {"topk_type": "deepst", "excitatory_target_density": 0.5}
    _build(_population(**base))
    with pytest.raises(ValueError, match="does not apply"):
        _build(_population(**base, **option))


@pytest.mark.parametrize(
    "sparse_type, option",
    [
        ("standard", {"indexed_rewire_frequency": 50}),
        ("indexed", {"indexed_rewire_quantile": 0.1}),
        ("standard", {"release_p_min": 0.3}),
        ("indexed_rewire", {"release_impact_decay": 0.5}),
        ("variance", {"dense_to_sparse_end_step": 500}),
        ("standard", {"indexed_candidate_size": 8}),
    ],
)
def test_options_that_would_change_another_sparse_type_raise(sparse_type, option):
    base = {"topk_type": sparse_type, "indexed_seed": 5}
    _build(_population(**base))
    with pytest.raises(ValueError, match="does not apply"):
        _build(_population(**base, **option))


def test_unread_indexed_execution_options_are_reported_not_rejected(caplog):
    with caplog.at_level("INFO"):
        _build(
            _population(
                topk_type="standard",
                indexed_output_chunk_size=256,
                indexed_index_dtype="int32",
            )
        )
    assert "does not read the indexed execution option" in caplog.text


# Item 10: transfer options read by no TransferLayer path.
def test_feedforward_einet_rejects_non_first_transfer_inhibitory_mode():
    _build(_einet(transfer={"inhibitory_mode": "first"}))
    with pytest.raises(ValueError, match="inhibitory_mode"):
        _build(_einet(transfer={"inhibitory_mode": "all"}))


@pytest.mark.parametrize("family", ["einet", "population_network"])
def test_transfer_input_transform_is_rejected_where_unused(family):
    if family == "einet":
        config = _einet(transfer={"input_transform": "identity"})
        _build(config)
        config["transfer"]["input_transform"] = "signed_split"
    else:
        config = _population()
        config["population_network"].update(
            {"use_transfer": True, "transfer_params": {}}
        )
        _build(config)
        config["population_network"]["transfer_params"][
            "input_transform"
        ] = "signed_split"
    with pytest.raises(ValueError, match="input_transform"):
        _build(config)


# Item 11: output-scale settings for wrappers that apply no output scale.
@pytest.mark.parametrize(
    "fields",
    [
        {"fixed_output_scale": 3.0},
        {"output_scale_mode": "global"},
        {"learned_output_scale": True},
    ],
)
def test_recurrent_wrapper_rejects_output_scale_it_ignores(fields):
    from dendritic_modeling.config.config import _build_model_config
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    core = {**_make_recurrent(_population()), "type": "population_network"}
    matching = {
        "learned_output_scale": False,
        "fixed_output_scale": 1.0,
        "output_scale_mode": "fixed",
    }
    initialize_model(_build_model_config(_model_payload(core, **matching)))
    with pytest.raises(ValueError, match=r"output scale|learned_output_scale"):
        initialize_model(_build_model_config(_model_payload(core, **fields)))


def test_regression_wrapper_rejects_fixed_output_scale():
    from dendritic_modeling.config.config import _build_model_config
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    core = {**_population(), "type": "population_network"}
    payload = _model_payload(core, task="regression")
    initialize_model(_build_model_config(payload))
    with pytest.raises(ValueError, match="regression model wrapper"):
        initialize_model(_build_model_config({**payload, "fixed_output_scale": 2.0}))


# Item 12: initial_child_conductance is read only by two initializers.
def test_initial_child_conductance_requires_an_initializer_that_reads_it():
    changed = {"initial_child_conductance": 2.0}
    neutral = _build(_population(dbl_init_method="mechanism_neutral"))
    scaled = _build(_population(dbl_init_method="mechanism_neutral", **changed))
    assert any(
        not torch.equal(a, b) for a, b in zip(neutral.parameters(), scaled.parameters())
    )
    with pytest.raises(ValueError, match="initial_child_conductance currently"):
        _build(_population(dbl_init_method="analytical_expectation", **changed))


# Item 13: the removed legacy recurrent cell stack.
def test_legacy_recurrent_enabled_is_rejected(caplog):
    from dendritic_modeling.config.config import _build_core_config
    from dendritic_modeling.config.recurrence import is_recurrent_core_config

    assert not is_recurrent_core_config(_einet())
    legacy = {**_einet(), "recurrent": {"enabled": True, "cell_type": "gru"}}
    with pytest.raises(ValueError, match="legacy recurrent cell stack"):
        _build_core_config(legacy)
    with pytest.raises(ValueError, match="legacy recurrent cell stack"):
        is_recurrent_core_config(legacy)
    inert = {**_einet(), "recurrent": {"enabled": False, "cell_type": "dendritic"}}
    with caplog.at_level("WARNING"):
        core = _build_core_config(inert)
    assert "Ignoring model.core.recurrent" in caplog.text
    assert not hasattr(core, "recurrent")


# Item 14: unknown or unused top-level unified_ei keys.
def _unified(**top_level):
    population = {
        "n_neurons": 3,
        "branch_factors": [2],
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 1,
        "rec_inhibitory_synapses": 0,
        "reactivation_init_policy": "fixed",
        "adaptive_initialization": False,
    }
    return {
        "biological_neuron": False,
        "type": "unified_ei",
        "unified_ei": {
            "output_mode": "last",
            "layers": [
                {"recurrent": True, "excitatory": population, "inhibitory": None}
            ],
            **top_level,
        },
    }


@pytest.mark.parametrize(
    "top_level, message",
    [({"ouput_mode": "mean"}, "did you mean 'output_mode'"), ({"dt": 0.5}, "legacy")],
)
def test_unified_ei_rejects_unknown_or_unused_top_level_keys(top_level, message):
    _build(_unified())
    with pytest.raises(ValueError, match=message):
        _build(_unified(**top_level))


# Item 15: strict experiment/training/outputs/distributed sections.
_TRAINING_YAML = """
experiment: {seed: 3}
data: {dataset_name: mnist}
model:
  task: classification
  encoder: {type: identity, params: {input_dim: 6}}
  core:
    type: einet
    architecture: {excitatory_layer_sizes: [3], inhibitory_layer_sizes: []}
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [0]
  decoder: {type: identity, params: {hidden_dims: []}}
training:
  main:
    strategy: standard
    common: {epochs: 1, synapse_decay_rate: 0.0}
"""


@pytest.mark.parametrize(
    "edit, message",
    [
        (("training", "main", "common", "weight_decay_rat"), "synapse_decay_rate"),
        (("experiment", "model_seeed"), "model_seed"),
        (("outputs", "run_nme"), "run_name"),
        (("distributed", "mdoe"), "mode"),
        (("training", "main", "strategies", "standrd"), "standard"),
        (("training", "main", "strategies", "standard", "epochz"), "epoch"),
    ],
)
def test_loader_rejects_unknown_section_keys(tmp_path, edit, message):
    import yaml

    from dendritic_modeling.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(_TRAINING_YAML)
    load_config(str(path))
    data = yaml.safe_load(_TRAINING_YAML)
    node = data
    for key in edit[:-1]:
        node = node.setdefault(key, {})
    node[edit[-1]] = 0 if edit[-1] != "standrd" else {}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_config(str(path))


def test_loader_drops_legacy_experiment_enable_amp_with_warning(tmp_path, caplog):
    import yaml

    from dendritic_modeling.config import load_config

    data = yaml.safe_load(_TRAINING_YAML)
    data["experiment"]["enable_amp"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(str(path))
    assert not hasattr(config.experiment, "enable_amp")
    assert "Ignoring experiment.enable_amp: it has had no effect" in caplog.text


# Item 16: unknown plain-dict options must not vanish into the activation.
@pytest.mark.parametrize(
    "section, key",
    [
        ("morphology", "leak_conductnce"),
        ("implementation", "adaptive_initialisation"),
        ("reactivation", "init_mm"),
    ],
)
def test_plain_dict_einet_rejects_unknown_section_keys(section, key):
    _build(_einet())
    with pytest.raises(ValueError, match="Unknown DendriticBranchLayer option"):
        _build(_einet(**{section: {key: 2.0}}))


def test_branch_layer_rejects_unknown_keyword_options():
    kwargs = {"output_dim": 2, "excitatory_input_dim": 4}
    kwargs["excitatory_synapses_per_branch"] = 2
    DendriticBranchLayer(**kwargs, reactivate=True, memory_efficient=True)
    with pytest.raises(ValueError, match="Unknown DendriticBranchLayer option"):
        DendriticBranchLayer(**kwargs, reactivate=True, leak_conductnce=2.0)
    with pytest.raises(ValueError, match="Unknown DendriticBranchLayer option"):
        DendriticBranchLayer(**kwargs, reactivate=False, memory_efficient=True)


# Item 17: options that the selected builder never applies.
def test_baseline_rnn_rejects_single_layer_dropout():
    config = {"type": "gru", "baseline_rnn": {"hidden_dim": 6, "num_layers": 2}}
    _build({**config, "baseline_rnn": {**config["baseline_rnn"], "dropout": 0.5}})
    with pytest.raises(ValueError, match="num_layers"):
        _build({"type": "gru", "baseline_rnn": {"hidden_dim": 6, "dropout": 0.5}})


def test_population_network_rejects_transfer_params_without_transfer():
    config = _population()
    config["population_network"]["transfer_params"] = {"independent_pathways": True}
    with pytest.raises(ValueError, match="use_transfer"):
        _build(config)
    # The empty block written by the base configs, and the seed training adds
    # to it, configure nothing.
    for params in ({}, {"split_seed": 3}):
        config["population_network"]["transfer_params"] = params
        _build(config)


@pytest.mark.parametrize(
    "section, values",
    [
        ("architecture", {"input_projection_dims": [8]}),
        ("implementation", {"store_routing": True}),
        ("morphology", {"allow_self_recurrence": False}),
    ],
)
def test_feedforward_einet_rejects_recurrent_only_options(section, values):
    with pytest.raises(ValueError, match="Feedforward EINet does not apply"):
        _build(_einet(**{section: values}))


# Item 18: alias conflicts compare explicitly configured values only.
def test_alias_conflict_warning_ignores_typed_defaults(recwarn):
    core = {**_einet("raw"), "type": "dendritic_additive"}
    core["morphology"] = {
        key: value
        for key, value in core["morphology"].items()
        if key not in {"use_shunting", "use_additive_normalization"}
    }
    _typed(core)
    assert not [w for w in recwarn if "overrides" in str(w.message)]
    core["morphology"]["use_shunting"] = True
    with pytest.raises(ValueError, match=r"fixes morphology\.use_shunting=False"):
        _typed(core)


# Item 19: recurrent_ei settings shape a recurrent EINet or fail.
def _two_layer_einet(rec_ee=None, **recurrent_ei):
    config = _einet()
    config["architecture"]["excitatory_layer_sizes"] = [4, 3]
    config["connectivity"] = {
        "ee_synapses_per_branch_per_layer": [2, 2],
        "ie_synapses_per_branch_per_layer": [0, 0],
    }
    if rec_ee is not None:
        config["connectivity"]["rec_ee_synapses_per_branch"] = rec_ee
    if recurrent_ei:
        config["recurrent_ei"] = recurrent_ei
    return config


@pytest.mark.parametrize(
    "settings",
    [
        {"tau_base": 30.0},
        {"dt": 0.5},
        {"output_mode": "mean"},
        {"learnable_tau": True},
        {"recurrent_layers": [True, True]},
    ],
)
def test_feedforward_einet_rejects_recurrent_ei_settings(settings):
    _build(_two_layer_einet(enabled=False))
    key = next(iter(settings))
    with pytest.raises(
        ValueError, match=rf"recurrent_ei\.{key}.*recurrent_ei\.enabled"
    ):
        _build(_two_layer_einet(**settings))


@pytest.mark.parametrize("core_type", sorted(_FEEDFORWARD_CORES))
def test_feedforward_only_cores_reject_recurrent_ei_settings(core_type):
    extra, input_dim = _FEEDFORWARD_CORES[core_type]
    config = {**_ei_einet(), "type": core_type, **deepcopy(extra)}
    config["recurrent_ei"] = {"tau_base": 30.0}
    with pytest.raises(ValueError, match="builds a feedforward network here"):
        _build(config, input_dim=input_dim)


def test_recurrent_layers_select_layers_and_keep_their_synapses():
    model = _build(_two_layer_einet([0, 2], recurrent_layers=[False, True]))
    assert [layer.recurrent for layer in model.layers] == [False, True]
    with pytest.raises(ValueError, match="marks layer 1 as feedforward"):
        _build(_two_layer_einet([2, 2], recurrent_layers=[True, False]))


def test_enabled_recurrence_needs_a_recurrent_layer():
    with pytest.raises(ValueError, match="would run feedforward"):
        _build(_two_layer_einet(enabled=True))
    model = _build(_two_layer_einet(enabled=True, recurrent_layers=[True, True]))
    assert [layer.recurrent for layer in model.layers] == [True, True]


@pytest.mark.parametrize(
    "key, value, target",
    [
        (
            "recurrent_per_layer",
            [True, True],
            "model.core.recurrent_ei.recurrent_layers",
        ),
        ("transfer", {"input_mode": 0}, "model.core.transfer"),
        ("transfer_params", {"input_mode": 0}, "model.core.transfer"),
    ],
)
def test_replaced_recurrent_ei_spellings_fail_at_load_and_build(key, value, target):
    from dendritic_modeling.config.config import _build_core_config

    config = _two_layer_einet([2, 2], **{key: value})
    message = re.escape(f"recurrent_ei.{key} -> {target}")
    with pytest.raises(ValueError, match=message):
        _build(config)
    with pytest.raises(ValueError, match=message):
        _build_core_config(deepcopy(config))


# Item 20: written core input dimensions must match the encoder output.
def test_population_network_input_dim_must_match_the_encoder():
    config = _population()
    config["population_network"]["input_dim"] = 6
    _build(config, input_dim=6)
    config["population_network"]["input_dim"] = 5
    with pytest.raises(ValueError, match="differs from the encoder output"):
        _build(config, input_dim=6)


# Item 21: retired input_mode=1 ran without the inhibitory cells it declares,
# so it raises unless the retired opt-in built them.
@pytest.mark.parametrize("build_flag, raises", [(False, True), (True, False)])
def test_input_mode_one_rejects_inhibitory_cells_it_would_drop(build_flag, raises):
    config = _ei_einet()
    config["transfer"] = {
        "input_mode": 1,
        "input_mode1_build_inhibitory_population": build_flag,
    }
    if raises:
        with pytest.raises(ValueError, match="without the inhibitory cells"):
            _build(config, input_dim=12)
    else:
        _build(config, input_dim=12)


def test_gate_start_values_warn_unless_the_policy_reads_them(
    tmp_path, caplog, monkeypatch
):
    import logging

    import yaml

    from dendritic_modeling.config import load_config, reactivation

    monkeypatch.setattr(reactivation, "_WARNED_UNUSED_GATE_INIT", set())
    caplog.set_level(logging.WARNING, logger="dendritic_modeling.config.reactivation")
    data = yaml.safe_load(_TRAINING_YAML)
    data["model"]["core"]["reactivation"] = {"init_m": 2.0, "init_b": 0.4}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    load_config(str(path))
    assert "init_m, init_b in model.core.reactivation have no effect" in caplog.text

    caplog.clear()
    data["model"]["core"]["reactivation"]["init_policy"] = "fixed"
    path.write_text(yaml.safe_dump(data))
    config = load_config(str(path))
    assert "no effect" not in caplog.text
    assert config.model.core.reactivation.init_m == 2.0

    caplog.clear()
    _build(_population(reactivation_init_m=2.0, reactivation_init_policy="analytical"))
    assert "reactivation_init_m in the population settings" in caplog.text
    caplog.clear()
    model = _build(
        _population(reactivation_init_m=2.0, reactivation_init_policy="fixed")
    )
    assert "no effect" not in caplog.text
    slopes = [
        module.log_m.exp()
        for module in model.modules()
        if hasattr(module, "log_m") and torch.is_tensor(module.log_m)
    ]
    assert slopes and all(torch.allclose(m, torch.tensor(2.0)) for m in slopes)


# Item 6: structured connectivity composes with connection masks and follows
# the network layer of a population.
def _feature_blocks(width: int) -> dict:
    return {
        "enabled": True,
        "pathways": {
            "ee": {
                "method": "level_feature_blocks",
                "ranges_by_level": [[0, width]] * 3,
            }
        },
    }


def _input_masks(module):
    return [
        layer.branch_excitation.connection_mask
        for layer in module.modules()
        if isinstance(layer, DendriticBranchLayer)
        and layer.branch_excitation is not None
        and getattr(layer.branch_excitation, "connection_mask", None) is not None
    ]


@pytest.mark.parametrize("probability", [1.0, 0.8])
def test_connection_probabilities_keep_the_structured_restriction(probability):
    config = _population(structured_connectivity=_feature_blocks(4))
    connection = config["population_network"]["layers"][0]["connections"][0]
    connection["probability"] = probability
    masks = _input_masks(_build(config))
    assert masks
    assert all(int(mask[:, 4:].sum()) == 0 for mask in masks)
    assert all(bool((mask.sum(dim=1) > 0).all()) for mask in masks)


def test_masks_that_leave_a_compartment_without_input_raise():
    config = _population(structured_connectivity=_feature_blocks(2))
    connection = config["population_network"]["layers"][0]["connections"][0]
    connection["probability"] = 0.05
    with pytest.raises(ValueError, match="without an allowed input"):
        _build(config)


def _two_layers(structured):
    config = _population(structured_connectivity=structured)
    layers = config["population_network"]["layers"]
    second = deepcopy(layers[0])
    second["name"] = "layer1"
    layers.append(second)
    return config


def test_structured_layer_lists_follow_the_network_layer():
    structured = {
        "enabled": True,
        "pathways": {"ee": {"method": "bernoulli", "probability": [1.0, 0.5]}},
    }
    model = _build(_two_layers(structured))
    indices = [
        (pop.structured_layer_idx, pop.structured_network_layer_idx)
        for layer in model.layers
        for pop in layer.population_configs.values()
    ]
    assert indices == [(0, 0), (0, 1)]
    first, second = (_input_masks(layer) for layer in model.layers)
    assert all(bool(mask.bool().all()) for mask in first)
    assert not all(bool(mask.bool().all()) for mask in second)


def test_later_network_layers_draw_their_own_structured_masks():
    # A seeded mask is a function of the pathway, layer, level and shape.
    structured = {
        "enabled": True,
        "pathways": {"ee": {"method": "bernoulli", "probability": 0.5, "seed": 3}},
    }
    # Both layers read three features, so their masks have one shape.
    model = _build(_two_layers(structured), input_dim=3)
    first, second = (_input_masks(layer) for layer in model.layers)
    assert [a.shape for a in first] == [b.shape for b in second]
    assert any(not torch.equal(a, b) for a, b in zip(first, second))
    # The first layer draws what a single-layer network draws.
    single = _build(_population(structured_connectivity=structured), input_dim=3)
    assert all(torch.equal(a, b) for a, b in zip(first, _input_masks(single)))


def test_manifest_keys_of_later_network_layers_name_the_layer():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.index_manifest import (
        manifest_entry_key,
    )

    assert manifest_entry_key("ee", 1, 2) == "ee/layer1/level2"
    assert manifest_entry_key("ee", 1, 2, 0) == "ee/layer1/level2"
    assert manifest_entry_key("ee", 1, 2, 3) == "ee/network_layer3/layer1/level2"


# DeepST synapses_per_branch: dendritic branch layers pass each pathway's
# synapse count, which constant_branch DeepST keeps per branch; the written
# value is never read.
def _deepst(**deepst):
    return {"type": "deepst", "deepst": {"use_noise": False, **deepst}}


@pytest.mark.parametrize(
    "deepst, reason",
    [
        ({"rewiring_mode": "constant_branch", "synapses_per_branch": 3}, "2 for this"),
        ({"rewiring_mode": "global", "synapses_per_branch": 2}, "target density"),
    ],
)
def test_deepst_synapses_per_branch_the_layer_does_not_use_raises(deepst, reason):
    with pytest.raises(
        ValueError, match=rf"sparsity\.deepst\.synapses_per_branch=\d.*{reason}"
    ):
        _typed(_einet(sparsity=_deepst(**deepst)))
    with pytest.raises(ValueError, match="only DeepST synapses read it"):
        _typed(_einet(sparsity={"deepst": {"synapses_per_branch": 2}}))


def test_deepst_synapses_per_branch_equal_to_the_pathway_count_is_the_same_model():
    omitted = _typed(_einet(sparsity=_deepst(rewiring_mode="constant_branch")))
    written = _typed(
        _einet(sparsity=_deepst(rewiring_mode="constant_branch", synapses_per_branch=2))
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
        DeepstLinear,
    )

    _assert_same_parameters(omitted, written)
    masks = [m.mask for m in written.modules() if isinstance(m, DeepstLinear)]
    assert masks and all(bool((mask.sum(dim=1) == 2).all()) for mask in masks)


def test_population_deepst_synapses_per_branch_is_checked_per_bank():
    config = _population(
        topk_type="deepst",
        rewiring_mode="constant_branch",
        synapses_per_branch=3,
        use_noise=False,
    )
    with pytest.raises(ValueError, match="2 for this bank"):
        _build(config)


def test_a_deepst_bank_without_a_count_reads_synapses_per_branch():
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
        DeepSTConfig,
        DendriticSynapseConfig,
        TopKConfig,
    )

    config = DendriticSynapseConfig(
        topk=TopKConfig(type="deepst"),
        deepst=DeepSTConfig(
            use_noise=False, rewiring_mode="constant_branch", synapses_per_branch=2
        ),
    )
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=None,
        synapse_config=config,
    )
    assert torch.equal(layer.branch_excitation.mask.sum(dim=1), torch.full((3,), 2))


# Payload errors of the recurrent baselines name their model.core section.
@pytest.mark.parametrize(
    "core_type, section, payload, message",
    [
        ("gru", "baseline_rnn", {"hiden_dim": 8}, "'hidden_dim'"),
        (
            "heterogeneous_leak_ctrnn",
            "heterogeneous_leak_ctrnn",
            {"tau_valuez": [1.0]},
            "'tau_values'",
        ),
        ("legendre_memory", "legendre_memory", {"memory_ordr": 3}, "'memory_order'"),
    ],
)
def test_recurrent_baseline_payload_typos_name_their_section(
    core_type, section, payload, message
):
    from dendritic_modeling.config import config_from_mapping

    core = config_from_mapping(
        {"model": {"core": {"type": core_type, section: payload}}}
    )
    with pytest.raises(
        ValueError, match=rf"model\.core\.{section} option.*did you mean {message}"
    ):
        get_architecture(core_type, core.model.core, input_dim=4)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"cell_type": "gru"}, r"cell_type='gru' conflicts with model\.core\.type"),
        ({"input_dim": 5}, r"baseline_rnn\.input_dim=5 differs from the encoder"),
    ],
)
def test_baseline_rnn_rejects_values_the_core_type_and_encoder_replace(
    payload, message
):
    from dendritic_modeling.config import config_from_mapping

    core = config_from_mapping(
        {"model": {"core": {"type": "lstm", "baseline_rnn": payload}}}
    ).model.core
    with pytest.raises(ValueError, match=message):
        get_architecture("lstm", core, input_dim=4)
    core.baseline_rnn = {"cell_type": "LSTM", "input_dim": 4, "hidden_dim": 8}
    assert get_architecture("lstm", core, input_dim=4).config.cell_type == "lstm"
