"""Equation-level tests for fair additive comparison controls."""

import pytest
import torch

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)


def _configured_layer(
    *,
    use_shunting: bool = False,
    additive_mode: str = "raw",
    additive_tangent_n0=None,
    additive_tangent_t0=None,
    excitatory_connection_mask=None,
) -> DendriticBranchLayer:
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=2,
        inhibitory_synapses_per_branch=1,
        input_branch_factor=2,
        use_shunting=use_shunting,
        additive_mode=additive_mode,
        additive_tangent_n0=additive_tangent_n0,
        additive_tangent_t0=additive_tangent_t0,
        excitatory_connection_mask=excitatory_connection_mask,
        reactivate=False,
        weight_transform="identity",
        dbl_init_method="default",
        adaptive_initialization=False,
    )
    with torch.no_grad():
        layer.branch_excitation.pre_w.copy_(
            torch.tensor([[2.0, 1.0, 0.1], [0.2, 1.5, 0.5]])
        )
        layer.branch_inhibition.pre_w.copy_(torch.tensor([[0.75, 0.1], [0.1, 1.25]]))
        layer.branches_to_output.log_weight.copy_(
            torch.tensor([[0.4, 0.6], [0.3, 0.7]])
        )
    return layer


def _inputs():
    return (
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 0.2]]),
        torch.tensor([[2.0, 1.0], [1.0, 0.5]]),
        torch.tensor([[0.5, 1.5, 2.0, 1.0], [1.0, 0.25, 0.5, 1.5]]),
    )


def _components(layer, x, inhibitory_input, branch_input):
    E = layer.branch_excitation(x)
    inhibitory = layer.branch_inhibition(inhibitory_input)
    C = layer.branches_to_output(branch_input)
    G = layer.branches_to_output.sum_conductances()
    return E, inhibitory, C, G


def test_raw_additive_default_preserves_historical_equation():
    layer = _configured_layer()
    x, inhibitory_input, branch_input = _inputs()

    voltage = layer(x, inhibitory_input, branch_input)
    E, inhibitory, C, _G = _components(layer, x, inhibitory_input, branch_input)
    expected = 0
    expected = expected + E
    expected = expected + C
    expected = expected - inhibitory

    assert layer.additive_mode == "raw"
    assert torch.equal(voltage, expected)


def test_shunting_default_preserves_historical_operation_order():
    layer = _configured_layer(use_shunting=True)
    x, inhibitory_input, branch_input = _inputs()

    voltage = layer(x, inhibitory_input, branch_input)
    E, inhibitory, C, G = _components(layer, x, inhibitory_input, branch_input)
    numerator = 0
    denominator = 1
    numerator = numerator + E
    denominator = denominator + E
    numerator = numerator + C
    denominator = denominator + G
    denominator = denominator + inhibitory
    expected = numerator / (denominator + layer.epsilon)

    assert torch.equal(voltage, expected)


def test_conductance_normalized_additive_uses_total_conductance_not_zscore():
    layer = _configured_layer(additive_mode="conductance_normalized")
    x, inhibitory_input, branch_input = _inputs()

    voltage = layer(x, inhibitory_input, branch_input)
    E, inhibitory, C, G = _components(layer, x, inhibitory_input, branch_input)
    expected = (E - inhibitory + C) / (1 + E + inhibitory + G + layer.epsilon)

    torch.testing.assert_close(voltage, expected)
    # Feature z-scoring would force every row mean to zero; this control does not.
    assert not torch.allclose(voltage.mean(dim=-1), torch.zeros(voltage.shape[0]))


def _voltage_and_gradients(layer):
    E = torch.full((1, 2), 0.4, requires_grad=True)
    inhibitory = torch.full((1, 2), 0.6, requires_grad=True)
    C = torch.full((1, 2), 0.3, requires_grad=True)
    G = torch.full((1, 2), 0.5, requires_grad=True)
    voltage, _ = layer.voltage_from_currents(
        trace_E=E,
        trace_E_rec=0,
        trace_I=inhibitory,
        trace_I_rec=0,
        trace_branch=C,
        branch_conductance=G,
    )
    gradients = torch.autograd.grad(voltage.sum(), (E, inhibitory, C, G))
    return voltage, gradients


def test_tangent_matched_additive_matches_shunt_value_and_jacobian_at_anchor():
    # N0 = E0 + C0 = 0.7; T0 = E0 + I0 + G0 = 1.5.
    shunting = _configured_layer(use_shunting=True)
    tangent = _configured_layer(
        additive_mode="tangent_matched",
        additive_tangent_n0=0.7,
        additive_tangent_t0=1.5,
    )

    shunt_voltage, shunt_gradients = _voltage_and_gradients(shunting)
    tangent_voltage, tangent_gradients = _voltage_and_gradients(tangent)

    torch.testing.assert_close(tangent_voltage, shunt_voltage)
    for tangent_gradient, shunt_gradient in zip(
        tangent_gradients, shunt_gradients, strict=True
    ):
        torch.testing.assert_close(tangent_gradient, shunt_gradient)


def test_tangent_mode_never_estimates_an_anchor_implicitly():
    layer = _configured_layer(additive_mode="tangent_matched")
    x, inhibitory_input, branch_input = _inputs()

    assert layer.additive_operating_point is None
    with pytest.raises(RuntimeError, match="requires a frozen operating point"):
        layer(x, inhibitory_input, branch_input)
    assert layer.additive_operating_point is None


def test_diagnostics_expose_circuit_decomposition_and_realized_k():
    connection_mask = torch.tensor(
        [[True, False, False], [True, True, False]], dtype=torch.bool
    )
    layer = _configured_layer(
        additive_mode="conductance_normalized",
        excitatory_connection_mask=connection_mask,
    )
    layer.set_branch_diagnostics(True)
    x, inhibitory_input, branch_input = _inputs()

    layer(x, inhibitory_input, branch_input)
    diagnostics = layer.get_branch_diagnostics()

    assert {"E", "I", "C", "G", "N", "T", "V", "realized_K"} <= set(diagnostics)
    torch.testing.assert_close(diagnostics["N"], diagnostics["E"] + diagnostics["C"])
    torch.testing.assert_close(
        diagnostics["T"],
        diagnostics["E"] + diagnostics["I"] + diagnostics["G"],
    )
    expected = (diagnostics["E"] - diagnostics["I"] + diagnostics["C"]) / (
        1 + diagnostics["T"] + layer.epsilon
    )
    torch.testing.assert_close(diagnostics["V"], expected)
    torch.testing.assert_close(diagnostics["realized_K"]["E"], torch.tensor([1, 2]))
    torch.testing.assert_close(diagnostics["realized_K"]["I"], torch.tensor([1, 1]))
    assert diagnostics["realized_K"]["E_rec"] is None
    assert diagnostics["realized_K"]["I_rec"] is None
    assert all(not diagnostics[key].requires_grad for key in "EICGNTV")
    assert layer._diag_g_tot is None
    assert layer._diag_numerator is None


def test_realized_k_uses_the_exact_indexed_dynamic_forward_sample():
    torch.manual_seed(7)
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=6,
        excitatory_synapses_per_branch=2,
        use_shunting=False,
        additive_mode="raw",
        reactivate=False,
        weight_transform="identity",
        dbl_init_method="default",
        adaptive_initialization=False,
        topk_type="indexed_dynamic",
        indexed_candidate_size=6,
        indexed_selection="stochastic",
        topk_temperature=1.0,
        topk_ultrafast=False,
    )
    with torch.no_grad():
        layer.branch_excitation.pre_w.fill_(1.0)
    layer.set_branch_diagnostics(True)

    voltage = layer(torch.ones(1, 6))
    realized_k = layer.get_branch_diagnostics()["realized_K"]["E"]

    # Unit inputs and unit active weights make the voltage equal the number of
    # connections selected in that same stochastic forward sample.
    torch.testing.assert_close(realized_k.to(voltage.dtype), voltage.squeeze(0))


def test_explicit_calibration_uses_shunting_diagnostic_means():
    shunting = _configured_layer(use_shunting=True)
    shunting.set_branch_diagnostics(True)
    x, inhibitory_input, branch_input = _inputs()
    shunting(x, inhibitory_input, branch_input)
    diagnostics = shunting.get_branch_diagnostics()

    tangent = _configured_layer(additive_mode="tangent_matched")
    operating_point = tangent.calibrate_additive_operating_point(diagnostics)

    assert operating_point == pytest.approx(
        (diagnostics["N"].mean().item(), diagnostics["T"].mean().item())
    )


def test_new_modes_cannot_be_confused_with_legacy_feature_zscore():
    with pytest.raises(ValueError, match="legacy per-sample, across-output z-score"):
        DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=2,
            excitatory_synapses_per_branch=1,
            use_shunting=False,
            additive_mode="conductance_normalized",
            use_additive_normalization=True,
            reactivate=False,
        )


def test_population_config_bridge_round_trips_additive_control_semantics():
    population = PopulationConfig(
        use_shunting=False,
        use_additive_normalization=False,
        additive_mode="tangent_matched",
        additive_tangent_n0=0.45,
        additive_tangent_t0=1.75,
    )

    config = DendriticSynapseConfig.from_population_config(population)
    kwargs = config.to_kwargs()

    assert kwargs["additive_mode"] == "tangent_matched"
    assert kwargs["additive_tangent_n0"] == pytest.approx(0.45)
    assert kwargs["additive_tangent_t0"] == pytest.approx(1.75)


def test_tangent_anchor_persists_and_old_state_dicts_still_load_strictly():
    source = _configured_layer(
        additive_mode="tangent_matched",
        additive_tangent_n0=0.7,
        additive_tangent_t0=1.5,
    )
    target = _configured_layer(additive_mode="tangent_matched")
    target.load_state_dict(source.state_dict())
    assert target.additive_operating_point == pytest.approx((0.7, 1.5))

    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if "_additive_tangent_" not in key
    }
    raw_target = _configured_layer()
    raw_target.load_state_dict(old_state, strict=True)
    assert raw_target.additive_operating_point is None


@pytest.mark.parametrize("initial_child_conductance", [1.0, 4.0, 64.0])
def test_mechanism_neutral_initialization_is_bit_identical_across_rules(
    initial_child_conductance,
):
    def build(use_shunting: bool) -> DendriticBranchLayer:
        torch.manual_seed(9281)
        return DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=9,
            excitatory_synapses_per_branch=3,
            inhibitory_input_dim=7,
            inhibitory_synapses_per_branch=2,
            input_branch_factor=3,
            use_shunting=use_shunting,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_policy="fixed",
            reactivation_init_m=1.25,
            reactivation_init_b=0.25,
            weight_transform="softplus",
            topk_type="indexed",
            indexed_seed=17,
            dbl_init_method="mechanism_neutral",
            adaptive_initialization=False,
            initial_child_conductance=initial_child_conductance,
        )

    shunting = build(True)
    additive = build(False)

    assert shunting.state_dict().keys() == additive.state_dict().keys()
    for key, shunting_value in shunting.state_dict().items():
        torch.testing.assert_close(
            shunting_value,
            additive.state_dict()[key],
            rtol=0.0,
            atol=0.0,
        )
    assert sum(parameter.numel() for parameter in shunting.parameters()) == sum(
        parameter.numel() for parameter in additive.parameters()
    )
    torch.testing.assert_close(
        shunting.branches_to_output.weight(),
        torch.full_like(
            shunting.branches_to_output.weight(), initial_child_conductance
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_mechanism_neutral_child_conductance_default_is_backward_compatible():
    torch.manual_seed(9281)
    layer = DendriticBranchLayer(
        output_dim=4,
        excitatory_input_dim=9,
        excitatory_synapses_per_branch=3,
        input_branch_factor=3,
        weight_transform="softplus",
        dbl_init_method="mechanism_neutral",
        adaptive_initialization=False,
    )

    torch.testing.assert_close(
        layer.branches_to_output.weight(),
        torch.ones_like(layer.branches_to_output.weight()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_initial_child_conductance_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="initial_child_conductance"):
        DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=9,
            excitatory_synapses_per_branch=3,
            input_branch_factor=3,
            weight_transform="softplus",
            dbl_init_method="mechanism_neutral",
            adaptive_initialization=False,
            initial_child_conductance=value,
        )
