"""Contracts for the parameter-matched parallel multiscale control."""

from __future__ import annotations

import hashlib
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationConfig,
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
    StatefulDendriNet,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    sum_level_values_by_output_owner,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.utils.weight_transforms import inverse_weight_transform


def _small_population_config(
    cross_level_mode: str | None = "hierarchical", **overrides
) -> PopulationConfig:
    payload = {
        "n_neurons": 4,
        "branch_factors": [2, 3],
        "ff_excitatory_synapses": 3,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 2,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "use_shunting": True,
        "additive_mode": "raw",
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.25,
        "reactivation_init_b": 0.1,
        "dbl_init_method": "mechanism_neutral",
        "adaptive_initialization": False,
        "weight_transform": "exp",
        "topk_type": "standard",
        "use_noise": False,
        "level_taus": [9.0, 5.0, 2.0],
        "initialization_seed": 1701,
        "initialization_namespace": "test.parallel_multiscale",
    }
    if cross_level_mode is not None:
        payload["cross_level_mode"] = cross_level_mode
    payload.update(overrides)
    return PopulationConfig(**payload)


def _small_population(
    cross_level_mode: str = "hierarchical", **overrides
) -> StatefulDendriNet:
    return StatefulDendriNet(
        pop_config=_small_population_config(cross_level_mode, **overrides),
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
        dt=1.0,
    )


def _assert_state_exact(left: DendriNetState, right: DendriNetState) -> None:
    for field in fields(DendriNetState):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, list):
            assert isinstance(right_value, list)
            assert len(left_value) == len(right_value)
            for left_tensor, right_tensor in zip(left_value, right_value, strict=True):
                assert torch.equal(left_tensor, right_tensor)
        elif isinstance(left_value, dict):
            assert isinstance(right_value, dict)
            assert left_value.keys() == right_value.keys()
            for key in left_value:
                for left_tensor, right_tensor in zip(
                    left_value[key], right_value[key], strict=True
                ):
                    assert torch.equal(left_tensor, right_tensor)
        elif torch.is_tensor(left_value):
            assert torch.equal(left_value, right_value)
        else:
            assert left_value is right_value


def _update_tensor_hash(digest, tensor: torch.Tensor) -> None:
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())


def _update_quantized_tensor_hash(
    digest, tensor: torch.Tensor, *, decimal_places: int = 5
) -> None:
    """Hash a hardware-stable fixed-point representation of a float tensor."""
    scale = 10**decimal_places
    canonical = (
        torch.round(tensor.detach().cpu().to(torch.float64) * scale)
        .to(torch.int64)
        .contiguous()
    )
    digest.update(canonical.numpy().tobytes())


def test_hierarchical_path_matches_frozen_main_rc4_golden_hashes() -> None:
    """Guard legacy construction and rollout independently of parallel support.

    Parameters and inputs are byte-exact. Floating-point rollout values are
    quantized to 1e-5 before hashing because CPU kernels can differ by a few
    ulps across supported hardware while remaining numerically equivalent.
    """
    torch.manual_seed(918)
    population = StatefulDendriNet(
        # RC4's fixture used the additive hierarchical path. Keep that choice
        # explicit now that parallel_readout itself requires shunting.
        pop_config=_small_population_config(None, use_shunting=False),
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
    )

    state_digest = hashlib.sha256()
    for key, tensor in population.state_dict().items():
        state_digest.update(key.encode("utf-8"))
        _update_tensor_hash(state_digest, tensor)

    input_digest = hashlib.sha256()
    rollout_digest = hashlib.sha256()
    generator = torch.Generator().manual_seed(918)
    state = None
    trace_fields = (
        "trace_E",
        "trace_E_rec",
        "trace_I",
        "trace_I_rec",
        "trace_branch",
    )
    for _ in range(3):
        x = torch.rand(2, 6, generator=generator)
        recurrent = torch.rand(2, 4, generator=generator)
        _update_tensor_hash(input_digest, x)
        _update_tensor_hash(input_digest, recurrent)
        output, state = population.step(
            x,
            recurrent_input=recurrent,
            state=state,
        )
        _update_quantized_tensor_hash(rollout_digest, output)
        for field_name in trace_fields:
            for tensor in getattr(state, field_name):
                _update_quantized_tensor_hash(rollout_digest, tensor)

    assert state_digest.hexdigest() == (
        "442aff8300cee05ef19e0f994f96e8dc55f3a90f998db3ebf3fb26c8df88e61c"
    )
    assert input_digest.hexdigest() == (
        "9bde52f2bc6218e989ff484c03c1ea2cd23969fd1ac9a96259a5b74068236acd"
    )
    assert rollout_digest.hexdigest() == (
        "e2c9b0a986360fc2020067a1764664ef8302a940054078845518aeb01b39e59c"
    )


def test_hierarchical_default_is_bitwise_noop() -> None:
    default = _small_population_config(None)
    explicit = _small_population_config("hierarchical")
    assert default.cross_level_mode == "hierarchical"

    reference = StatefulDendriNet(
        pop_config=default,
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
    )
    candidate = StatefulDendriNet(
        pop_config=explicit,
        excitatory_input_dim=6,
        recurrent_excitatory_input_dim=4,
    )

    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for key, reference_value in reference.state_dict().items():
        assert torch.equal(reference_value, candidate.state_dict()[key]), key
    assert not hasattr(reference, "parallel_readout_projections")

    generator = torch.Generator().manual_seed(918)
    state_reference = None
    state_candidate = None
    for _ in range(3):
        x = torch.rand(2, 6, generator=generator)
        recurrent = torch.rand(2, 4, generator=generator)
        output_reference, state_reference = reference.step(
            x, recurrent_input=recurrent, state=state_reference
        )
        output_candidate, state_candidate = candidate.step(
            x, recurrent_input=recurrent, state=state_candidate
        )
        assert torch.equal(output_reference, output_candidate)
        _assert_state_exact(state_reference, state_candidate)


def test_parallel_support_shapes_count_and_g1_initialization() -> None:
    population = _small_population("parallel_readout")
    projections = population.parallel_readout_projections

    assert population.level_dims == [24, 8, 4]
    assert [(p.in_features, p.out_features) for p in projections] == [
        (24, 4),
        (8, 4),
    ]
    assert [p.block_size for p in projections] == [6, 2]
    assert sum(p.log_weight.numel() for p in projections) == 32

    expected_raw = inverse_weight_transform(
        torch.tensor(1.0), population.pop_config.weight_transform
    )
    for projection in projections:
        torch.testing.assert_close(
            projection.log_weight,
            torch.full_like(projection.log_weight, expected_raw),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            projection.weight(),
            torch.ones_like(projection.log_weight),
            rtol=0.0,
            atol=0.0,
        )
        support = projection.block().ne(0)
        assert torch.equal(
            support.sum(dim=1),
            torch.full((projection.out_features,), projection.block_size),
        )
        assert torch.equal(
            support.sum(dim=0), torch.ones(projection.in_features, dtype=torch.long)
        )

    assert all(not layer.input_branches for layer in population.branch_layers)
    assert all(
        not hasattr(layer, "branches_to_output")
        for layer in population.branch_layers[1:]
    )


def test_parallel_denominator_retains_conductance_when_levels_are_silent() -> None:
    population = _small_population("parallel_readout")
    local_numerator = torch.full((2, 4), 0.5)
    local_denominator = torch.full((2, 4), 2.0)
    silent_levels = [
        torch.zeros(2, level_dim) for level_dim in population.level_dims[:-1]
    ]

    voltage = population._parallel_readout_soma_voltage(
        layer=population.branch_layers[-1],
        local_numerator=local_numerator,
        local_denominator=local_denominator,
        level_outputs=silent_levels,
    )

    # g=1 gives G=6+2=8 per soma even though activity current C is zero.
    expected = local_numerator / (
        local_denominator + 8.0 + population.branch_layers[-1].epsilon
    )
    torch.testing.assert_close(voltage, expected, rtol=0.0, atol=0.0)


def test_parallel_equation_reactivates_every_level_exactly_once() -> None:
    population = _small_population("parallel_readout")
    counts = [0] * population.n_levels
    hooks = []
    for level_idx, layer in enumerate(population.branch_layers):
        hooks.append(
            layer.reactivation.register_forward_hook(
                lambda _module, _args, _output, idx=level_idx: counts.__setitem__(
                    idx, counts[idx] + 1
                )
            )
        )

    x = torch.linspace(0.05, 0.65, 12).reshape(2, 6)
    recurrent = torch.linspace(0.1, 0.8, 8).reshape(2, 4)
    output, state = population.step(x, recurrent_input=recurrent)
    for hook in hooks:
        hook.remove()

    assert counts == [1, 1, 1]
    local_voltages = []
    local_denominators = []
    for level_idx, layer in enumerate(population.branch_layers):
        zeros = torch.zeros_like(state.trace_branch[level_idx])
        local_voltage, denominator = layer.voltage_from_currents(
            trace_E=state.trace_E[level_idx],
            trace_E_rec=state.trace_E_rec[level_idx],
            trace_I=state.trace_I[level_idx],
            trace_I_rec=state.trace_I_rec[level_idx],
            trace_branch=zeros,
            branch_conductance=zeros,
        )
        local_voltages.append(local_voltage)
        assert denominator is not None
        local_denominators.append(denominator)

    direct_current = torch.zeros_like(local_voltages[-1])
    direct_conductance = torch.zeros_like(local_voltages[-1][:1])
    for projection, layer, local_voltage in zip(
        population.parallel_readout_projections,
        population.branch_layers[:-1],
        local_voltages[:-1],
        strict=True,
    ):
        direct_current = direct_current + projection(layer.reactivation(local_voltage))
        direct_conductance = direct_conductance + (
            projection.sum_conductances().unsqueeze(0)
        )
    soma_numerator = state.trace_E[-1] + state.trace_E_rec[-1]
    pooled_voltage = (soma_numerator + direct_current) / (
        local_denominators[-1]
        + direct_conductance
        + population.branch_layers[-1].epsilon
    )
    expected = population.branch_layers[-1].reactivation(pooled_voltage)
    assert torch.equal(output, expected)


def test_distal_state_affects_readout_without_changing_middle_or_soma_state() -> None:
    population = _small_population("parallel_readout")
    base_state = population.init_state(2, torch.device("cpu"))
    perturbed_state = base_state.clone()
    perturbed_state.trace_E[0].fill_(0.4)
    x = torch.zeros(2, 6)
    recurrent = torch.zeros(2, 4)

    base_output, base_next = population.step(
        x, recurrent_input=recurrent, state=base_state
    )
    perturbed_output, perturbed_next = population.step(
        x, recurrent_input=recurrent, state=perturbed_state
    )

    assert not torch.equal(base_output, perturbed_output)
    assert not torch.equal(base_next.trace_E[0], perturbed_next.trace_E[0])
    for trace_name in (
        "trace_E",
        "trace_E_rec",
        "trace_I",
        "trace_I_rec",
        "trace_branch",
    ):
        base_traces = getattr(base_next, trace_name)
        perturbed_traces = getattr(perturbed_next, trace_name)
        for level_idx in (1, 2):
            assert torch.equal(base_traces[level_idx], perturbed_traces[level_idx]), (
                trace_name,
                level_idx,
            )


def test_hierarchical_distal_state_reaches_soma_only_through_serial_traces() -> None:
    population = _small_population("hierarchical")
    base_state = population.init_state(2, torch.device("cpu"))
    perturbed_state = base_state.clone()
    perturbed_state.trace_E[0].fill_(0.4)
    x = torch.zeros(2, 6)
    recurrent = torch.zeros(2, 4)

    base_output, base_next = population.step(
        x, recurrent_input=recurrent, state=base_state
    )
    perturbed_output, perturbed_next = population.step(
        x, recurrent_input=recurrent, state=perturbed_state
    )

    assert not torch.equal(base_output, perturbed_output)
    assert not torch.equal(base_next.trace_branch[1], perturbed_next.trace_branch[1])
    assert not torch.equal(base_next.trace_branch[2], perturbed_next.trace_branch[2])


def test_parallel_soma_silencing_zeroes_the_final_pooled_output() -> None:
    population = _small_population("parallel_readout")
    population.set_level_silencing(torch.tensor([0.0, 0.0, 1.0]))

    output, _ = population.step(
        torch.full((2, 6), 0.25),
        recurrent_input=torch.full((2, 4), 0.15),
    )

    assert torch.equal(output, torch.zeros_like(output))


def test_parallel_direct_forward_rejects_implicit_one_step_attenuation() -> None:
    population = _small_population("parallel_readout")

    with pytest.raises(RuntimeError, match="recurrent-only; call step"):
        population(torch.full((2, 6), 0.25))


def test_both_parallel_projections_receive_finite_nonzero_gradients() -> None:
    population = _small_population("parallel_readout")
    state = None
    for step in range(3):
        x = torch.linspace(0.1 + step * 0.01, 0.7 + step * 0.01, 12).reshape(2, 6)
        recurrent = torch.linspace(0.05, 0.45, 8).reshape(2, 4)
        output, state = population.step(x, recurrent_input=recurrent, state=state)
    decoder = torch.tensor(
        [[-0.8, -0.3, 0.2, 0.7], [0.6, 0.1, -0.4, -0.9]],
        dtype=output.dtype,
    )
    logits = output @ decoder.T
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()

    for projection in population.parallel_readout_projections:
        gradient = projection.log_weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def _paper_population_network(cross_level_mode: str) -> PopulationNetwork:
    defaults = {
        "ff_excitatory_synapses": 0,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "use_shunting": True,
        "use_additive_normalization": False,
        "additive_mode": "raw",
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.5,
        "reactivation_init_b": 0.5,
        "dbl_init_method": "mechanism_neutral",
        "adaptive_initialization": False,
        "weight_transform": "exp",
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_noise_level": 0.0,
        "topk_strategy": "none",
        "blocklinear_strategy": "none",
        "indexed_seed": 215058547,
        "initialization_seed": 1319312133,
        "learnable_tau": False,
        "allow_self_recurrence": False,
        "efficient_blocklinear": True,
        "use_noise": False,
        "dynamics_mode": "rate",
        "dendritic_spikes_enabled": False,
        "soma_feedback_enabled": False,
        "synapse_types": {},
    }
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=11,
            use_transfer=False,
            output_mode="last",
            input_projection_dims=[64],
            readout_layer="layer0",
            readout_population="pyr",
            layers=[
                PopulationLayerConfig(
                    name="layer0",
                    recurrent=True,
                    readout_population="pyr",
                    dt=1.0,
                    connection_seed=215058547,
                    population_defaults=defaults,
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 64,
                            "branch_factors": [3, 3],
                            "population": {
                                "cross_level_mode": cross_level_mode,
                                "initialization_namespace": (
                                    "population_network.layer0.pyr"
                                ),
                                "ff_excitatory_synapses_by_level": [3, 9, 50],
                                "ff_inhibitory_synapses_by_level": [0, 0, 0],
                                "rec_excitatory_synapses_by_level": [7, 4, 3],
                                "rec_inhibitory_synapses_by_level": [3, 3, 3],
                                "level_taus": [125.0, 25.0, 5.0],
                            },
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 16,
                            "branch_factors": [],
                            "population": {
                                "initialization_namespace": (
                                    "population_network.layer0.pv"
                                ),
                                "ff_excitatory_synapses_by_level": [8],
                                "ff_inhibitory_synapses_by_level": [0],
                                "rec_excitatory_synapses_by_level": [4],
                                "rec_inhibitory_synapses_by_level": [2],
                                "level_taus": [5.0],
                            },
                        },
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "pv"},
                        {"source": "pyr", "target": "pyr", "timing": "delayed"},
                        {"source": "pyr", "target": "pv", "timing": "delayed"},
                        {"source": "pv", "target": "pyr", "timing": "delayed"},
                        {"source": "pv", "target": "pv", "timing": "delayed"},
                    ],
                )
            ],
        )
    )


def test_paper_control_matches_all_declared_resources_and_recurrent_source() -> None:
    hierarchical = _paper_population_network("hierarchical")
    parallel = _paper_population_network("parallel_readout")
    hierarchical_resources = model_resource_summary(hierarchical)
    parallel_resources = model_resource_summary(parallel)

    exact_resources = {
        "total_parameters": 125344,
        "trainable_parameters": 125344,
        "synapse_candidate_parameters": 122112,
        "candidate_synapse_slots": 121264,
        "active_synapses": 14368,
        "persistent_state_scalars_per_sample": 4320,
    }
    for name, expected in exact_resources.items():
        assert hierarchical_resources[name] == expected
        assert parallel_resources[name] == expected
    assert hierarchical.get_effective_params() == parallel.get_effective_params()

    hierarchical_pyr = hierarchical.layers[0].populations["pyr"]
    parallel_pyr = parallel.layers[0].populations["pyr"]
    parallel_pv = parallel.layers[0].populations["pv"]
    assert hierarchical_pyr.cross_level_mode == "hierarchical"
    assert parallel_pyr.cross_level_mode == "parallel_readout"
    assert parallel_pv.cross_level_mode == "hierarchical"
    assert parallel_pyr.level_dims == [576, 192, 64]
    assert [
        (projection.in_features, projection.out_features, projection.block_size)
        for projection in parallel_pyr.parallel_readout_projections
    ] == [(576, 64, 9), (192, 64, 3)]
    assert (
        sum(
            projection.log_weight.numel()
            for projection in parallel_pyr.parallel_readout_projections
        )
        == 768
    )

    for hierarchical_layer, parallel_layer in zip(
        hierarchical_pyr.branch_layers,
        parallel_pyr.branch_layers,
        strict=True,
    ):
        for pathway in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        ):
            hierarchical_synapse = getattr(hierarchical_layer, pathway, None)
            parallel_synapse = getattr(parallel_layer, pathway, None)
            if hierarchical_synapse is None:
                assert parallel_synapse is None
                continue
            assert hierarchical_synapse.pre_w.shape == parallel_synapse.pre_w.shape
            assert torch.equal(hierarchical_synapse.pre_w, parallel_synapse.pre_w)
            assert torch.equal(
                hierarchical_synapse.weight_mask(),
                parallel_synapse.weight_mask(),
            )

    x = torch.linspace(0.0, 1.0, 22).reshape(2, 11)
    projected = parallel.input_projection(x)
    output, state = parallel.layers[0].step(projected)
    assert output.shape == (2, 64)
    assert torch.equal(output, state.outputs["pyr"])
    parallel_pyr_state = state.population_states["pyr"]
    assert (
        sum(
            trace.numel() // trace.shape[0] for trace in parallel_pyr_state.trace_branch
        )
        == 832
    )
    assert all(
        torch.count_nonzero(trace) == 0 for trace in parallel_pyr_state.trace_branch
    )

    _, hierarchical_state = hierarchical.layers[0].step(
        hierarchical.input_projection(x)
    )
    hierarchical_pyr_state = hierarchical_state.population_states["pyr"]
    assert torch.count_nonzero(hierarchical_pyr_state.trace_branch[0]) == 0
    assert torch.count_nonzero(hierarchical_pyr_state.trace_branch[1]) > 0
    assert torch.count_nonzero(hierarchical_pyr_state.trace_branch[2]) > 0

    # Both controls allocate 4,320 state scalars, but the direct control's 832
    # E trace_branch slots are deliberately dormant. Hierarchy uses its 192+64
    # parent-level branch traces; equal allocation is not equal functional state.
    assert (
        sum(trace.shape[1] for trace in hierarchical_pyr_state.trace_branch[1:]) == 256
    )


def test_paper_control_operating_point_and_ce_gradients_are_nonsaturated() -> None:
    torch.manual_seed(741)
    network = _paper_population_network("parallel_readout")
    pyr = network.layers[0].populations["pyr"]
    soma_reactivation = pyr.branch_layers[-1].reactivation
    preactivations: list[torch.Tensor] = []
    postactivations: list[torch.Tensor] = []

    def _capture_soma_gate(_module, args, output) -> None:
        preactivations.append(args[0].detach())
        postactivations.append(output.detach())

    hook = soma_reactivation.register_forward_hook(_capture_soma_gate)
    values = torch.arange(4 * 12 * 11, dtype=torch.float32).reshape(4, 12, 11)
    sequence = 0.05 + 0.9 * ((values % 41.0) / 40.0)
    output, hidden = network(sequence, return_hidden=True)
    hook.remove()

    assert len(preactivations) == sequence.shape[1]
    preactivation = torch.stack(preactivations)
    postactivation = torch.stack(postactivations)
    assert torch.isfinite(preactivation).all()
    assert torch.isfinite(postactivation).all()
    assert 0.0 <= float(preactivation.min())
    assert float(preactivation.max()) < 1.0
    assert float(preactivation.std()) > 1e-5
    assert float(postactivation.std()) > 1e-5
    saturation_fraction = (
        ((postactivation <= 0.01) | (postactivation >= 0.99)).float().mean()
    )
    assert float(saturation_fraction) == 0.0

    decoder = torch.stack(
        (
            torch.linspace(-0.7, 0.5, 64),
            torch.linspace(0.6, -0.8, 64),
        )
    )
    logits = output @ decoder.T
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    assert torch.isfinite(loss)
    loss.backward()

    for projection in pyr.parallel_readout_projections:
        gradient = projection.log_weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    for parameter in (
        soma_reactivation.log_m,
        soma_reactivation.b,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    for branch_layer in pyr.branch_layers:
        gradient = branch_layer.branch_excitation.pre_w.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    assert hidden is not None
    final_pyr_state = hidden[0].population_states["pyr"]
    assert all(
        torch.count_nonzero(trace) == 0 for trace in final_pyr_state.trace_branch
    )


def test_all_active_star_is_parameter_state_and_initialization_matched() -> None:
    torch.manual_seed(817)
    hierarchical = _paper_population_network("hierarchical")
    torch.manual_seed(817)
    star = _paper_population_network("all_active_star")

    assert hierarchical.state_dict().keys() == star.state_dict().keys()
    for key, reference_value in hierarchical.state_dict().items():
        assert torch.equal(reference_value, star.state_dict()[key]), key

    exact_resources = {
        "total_parameters": 125344,
        "trainable_parameters": 125344,
        "synapse_candidate_parameters": 122112,
        "candidate_synapse_slots": 121264,
        "active_synapses": 14368,
        "persistent_state_scalars_per_sample": 4320,
    }
    hierarchical_resources = model_resource_summary(hierarchical)
    star_resources = model_resource_summary(star)
    for name, expected in exact_resources.items():
        assert hierarchical_resources[name] == expected
        assert star_resources[name] == expected

    star_pyr = star.layers[0].populations["pyr"]
    assert star_pyr.is_all_active_star
    assert not hasattr(star_pyr, "parallel_readout_projections")
    assert [
        (
            layer.branches_to_output.in_features,
            layer.branches_to_output.out_features,
            layer.branches_to_output.block_size,
        )
        for layer in star_pyr.branch_layers[1:]
    ] == [(576, 192, 3), (192, 64, 3)]


def test_all_active_star_reaches_every_serial_shaped_branch_state() -> None:
    network = _paper_population_network("all_active_star")
    pyr = network.layers[0].populations["pyr"]
    x = torch.linspace(0.05, 0.95, 22).reshape(2, 11)

    _, state = network.layers[0].step(network.input_projection(x))
    pyr_state = state.population_states["pyr"]

    assert torch.count_nonzero(pyr_state.trace_branch[0]) == 0
    assert torch.count_nonzero(pyr_state.trace_branch[1]) > 0
    assert torch.count_nonzero(pyr_state.trace_branch[2]) > 0
    assert sum(trace.shape[-1] for trace in pyr_state.trace_branch[1:]) == 256
    assert [trace.shape[-1] for trace in pyr_state.trace_branch] == pyr.level_dims


def test_all_active_star_arm_state_is_isolated_from_receiver_local_path() -> None:
    population = _small_population("all_active_star")
    base_state = population.init_state(2, torch.device("cpu"))
    perturbed_state = base_state.clone()
    perturbed_state.trace_branch[1].fill_(0.4)
    x = torch.zeros(2, 6)
    recurrent = torch.zeros(2, 4)
    middle_outputs: list[torch.Tensor] = []

    hook = population.branch_layers[1].reactivation.register_forward_hook(
        lambda _module, _args, output: middle_outputs.append(output.detach().clone())
    )
    base_output, base_next = population.step(
        x, recurrent_input=recurrent, state=base_state
    )
    perturbed_output, perturbed_next = population.step(
        x, recurrent_input=recurrent, state=perturbed_state
    )
    hook.remove()

    assert not torch.equal(base_output, perturbed_output)
    assert not torch.equal(base_next.trace_branch[1], perturbed_next.trace_branch[1])
    assert torch.equal(base_next.trace_branch[2], perturbed_next.trace_branch[2])
    assert len(middle_outputs) == 2
    assert torch.equal(middle_outputs[0], middle_outputs[1])
    for trace_name in ("trace_E", "trace_E_rec", "trace_I", "trace_I_rec"):
        base_traces = getattr(base_next, trace_name)
        perturbed_traces = getattr(perturbed_next, trace_name)
        for level_idx in range(population.n_levels):
            assert torch.equal(base_traces[level_idx], perturbed_traces[level_idx]), (
                trace_name,
                level_idx,
            )


def test_all_active_star_soma_equation_and_single_reactivation() -> None:
    population = _small_population("all_active_star")
    counts = [0] * population.n_levels
    hooks = []
    for level_idx, layer in enumerate(population.branch_layers):
        hooks.append(
            layer.reactivation.register_forward_hook(
                lambda _module, _args, _output, idx=level_idx: counts.__setitem__(
                    idx, counts[idx] + 1
                )
            )
        )

    x = torch.linspace(0.05, 0.65, 12).reshape(2, 6)
    recurrent = torch.linspace(0.1, 0.8, 8).reshape(2, 4)
    output, state = population.step(x, recurrent_input=recurrent)
    for hook in hooks:
        hook.remove()

    assert counts == [1, 1, 1]
    soma = population.branch_layers[-1]
    zeros = torch.zeros_like(state.trace_branch[-1])
    _, local_denominator = soma.voltage_from_currents(
        trace_E=state.trace_E[-1],
        trace_E_rec=state.trace_E_rec[-1],
        trace_I=state.trace_I[-1],
        trace_I_rec=state.trace_I_rec[-1],
        trace_branch=zeros,
        branch_conductance=zeros,
    )
    assert local_denominator is not None

    direct_current = torch.zeros_like(output)
    direct_conductance = torch.zeros_like(output[:1])
    for level_idx in range(1, population.n_levels):
        aggregation = population.branch_layers[level_idx].branches_to_output
        direct_current = direct_current + sum_level_values_by_output_owner(
            state.trace_branch[level_idx], population.n_soma
        )
        direct_conductance = direct_conductance + sum_level_values_by_output_owner(
            aggregation.sum_conductances().unsqueeze(0), population.n_soma
        )
    soma_numerator = state.trace_E[-1] + state.trace_E_rec[-1]
    pooled_voltage = (soma_numerator + direct_current) / (
        local_denominator + direct_conductance + soma.epsilon
    )
    expected = soma.reactivation(pooled_voltage)
    assert torch.equal(output, expected)


def test_all_active_star_level_silencing_removes_arm_conductance() -> None:
    population = _small_population("all_active_star")
    silencing = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    population.set_level_silencing(silencing)

    x = torch.linspace(0.05, 0.65, 6).repeat(2, 1)
    recurrent = torch.linspace(0.1, 0.8, 4).repeat(2, 1)
    output, state = population.step(x, recurrent_input=recurrent)

    # The dynamic current of the silenced sample/arm is removed during trace
    # integration. Its matching static conductance must also leave the soma
    # denominator; otherwise the nominally silent arm still shunts the output.
    assert torch.count_nonzero(state.trace_branch[1][0]) == 0
    assert torch.count_nonzero(state.trace_branch[1][1]) > 0

    soma = population.branch_layers[-1]
    zeros = torch.zeros_like(state.trace_branch[-1])
    _, local_denominator = soma.voltage_from_currents(
        trace_E=state.trace_E[-1],
        trace_E_rec=state.trace_E_rec[-1],
        trace_I=state.trace_I[-1],
        trace_I_rec=state.trace_I_rec[-1],
        trace_branch=zeros,
        branch_conductance=zeros,
    )
    assert local_denominator is not None

    direct_current = torch.zeros_like(output)
    masked_conductance = torch.zeros_like(output)
    unmasked_conductance = torch.zeros_like(output[:1])
    for level_idx in range(1, population.n_levels):
        aggregation = population.branch_layers[level_idx].branches_to_output
        direct_current = direct_current + sum_level_values_by_output_owner(
            state.trace_branch[level_idx], population.n_soma
        )
        owner_conductance = sum_level_values_by_output_owner(
            aggregation.sum_conductances().unsqueeze(0), population.n_soma
        )
        keep = 1.0 - silencing[:, level_idx].unsqueeze(-1)
        masked_conductance = masked_conductance + owner_conductance * keep
        unmasked_conductance = unmasked_conductance + owner_conductance

    soma_numerator = state.trace_E[-1] + state.trace_E_rec[-1]
    expected = soma.reactivation(
        (soma_numerator + direct_current)
        / (local_denominator + masked_conductance + soma.epsilon)
    )
    stale_static_shunt = soma.reactivation(
        (soma_numerator + direct_current)
        / (local_denominator + unmasked_conductance + soma.epsilon)
    )

    assert torch.equal(output, expected)
    assert not torch.allclose(output[0], stale_static_shunt[0])
    assert torch.equal(output[1], stale_static_shunt[1])


def test_all_active_star_aggregators_receive_finite_nonzero_gradients() -> None:
    population = _small_population("all_active_star")
    state = None
    for step in range(3):
        x = torch.linspace(0.1 + step * 0.01, 0.7 + step * 0.01, 12).reshape(2, 6)
        recurrent = torch.linspace(0.05, 0.45, 8).reshape(2, 4)
        output, state = population.step(x, recurrent_input=recurrent, state=state)
    decoder = torch.tensor(
        [[-0.8, -0.3, 0.2, 0.7], [0.6, 0.1, -0.4, -0.9]],
        dtype=output.dtype,
    )
    loss = F.cross_entropy(output @ decoder.T, torch.tensor([0, 1]))
    loss.backward()

    for layer in population.branch_layers[1:]:
        gradient = layer.branches_to_output.log_weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_output_owner_sum_preserves_owner_major_coordinates() -> None:
    values = torch.arange(48, dtype=torch.float32).reshape(2, 24)
    expected = values.reshape(2, 4, 6).sum(dim=-1)
    assert torch.equal(sum_level_values_by_output_owner(values, 4), expected)

    with pytest.raises(ValueError, match="feature dimension"):
        sum_level_values_by_output_owner(torch.tensor(1.0), 4)
    with pytest.raises(ValueError, match="positive integer"):
        sum_level_values_by_output_owner(values, 0)
    with pytest.raises(ValueError, match="must be divisible"):
        sum_level_values_by_output_owner(values, 5)


def test_all_active_star_validation_is_fail_closed() -> None:
    cases = [
        ({"branch_factors": []}, "at least two levels"),
        ({"dynamics_mode": "spike"}, "spiking soma dynamics"),
        ({"dendritic_spikes_enabled": True}, "dendritic spikes"),
        ({"soma_feedback_enabled": True}, "soma feedback"),
        ({"use_shunting": False}, "additive/non-shunting dynamics"),
        ({"rec_excitatory_synapses": 0}, "requires a recurrent population"),
        (
            {"blocklinear_strategy": "conductance_dynamic"},
            "blocklinear gradient scaling",
        ),
        ({"weight_transform": "identity"}, "positive weight_transform"),
        (
            {
                "synapse_types": {
                    "enabled": True,
                    "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                    "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
                }
            },
            "typed synapses",
        ),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            _small_population("all_active_star", **overrides)

    population = _small_population("all_active_star")
    with pytest.raises(RuntimeError, match="recurrent-only; call step"):
        population(torch.full((2, 6), 0.25))


def test_parallel_mode_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="cross_level_mode"):
        _small_population_config("serialish")

    cases = [
        ({"branch_factors": []}, "at least two levels"),
        ({"dynamics_mode": "spike"}, "spiking soma dynamics"),
        ({"dendritic_spikes_enabled": True}, "dendritic spikes"),
        ({"soma_feedback_enabled": True}, "soma feedback"),
        ({"use_shunting": False}, "additive/non-shunting dynamics"),
        (
            {"rec_excitatory_synapses": 0},
            "requires a recurrent population",
        ),
        (
            {"blocklinear_strategy": "conductance_dynamic"},
            "blocklinear gradient scaling",
        ),
        ({"weight_transform": "identity"}, "positive weight_transform"),
        (
            {
                "synapse_types": {
                    "enabled": True,
                    "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                    "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
                }
            },
            "typed synapses",
        ),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            _small_population("parallel_readout", **overrides)
