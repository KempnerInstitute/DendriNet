"""Whole-model accounting and architectural controls for the scaling harness."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.benchmarks.parameter_budget import (
    FAMILIES,
    build_model,
    match_parameter_budget,
    model_report,
)


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _spec(family: str, **overrides) -> dict:
    spec = {
        "family": family,
        "input_dim": 8,
        "output_dim": 3,
        "width": 8,
        "branch_factors": [2],
        "contacts_e": 4,
        "contacts_i": (
            2
            if family in {"dendritic_additive", "dendritic_shunting", "grouped_gated"}
            else 0
        ),
        "seed": 11,
    }
    return {**spec, **overrides}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("network_depth", [1, 2])
def test_whole_model_counts_finite_gradients_and_static_support(family, network_depth):
    model = build_model(_spec(family, network_depth=network_depth))
    report = model_report(model)
    actual_parameters = list(model.parameters())
    assert report["total_parameters"] == sum(p.numel() for p in actual_parameters)
    assert report["total_parameters"] == report["trainable_parameters"]
    assert (
        report["total_parameters"]
        == report["hidden_parameters"] + report["readout_parameters"]
    )
    assert report["readout_parameters"] == 8 * 3 + 3
    assert report["parameter_bytes"] == 4 * report["total_parameters"]
    assert report["buffer_bytes"] == sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    assert report["encoded_input_dim"] == 16
    before = report["topology_sha256"]
    x = torch.randn(9, 8, generator=torch.Generator().manual_seed(19))
    output = model(x)
    assert output.shape == (9, 3)
    assert torch.isfinite(output).all()
    nn.functional.cross_entropy(output, torch.arange(9) % 3).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in actual_parameters
    )
    assert any(p.grad.abs().sum() > 0 for p in model.hidden.parameters())
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert model_report(model)["topology_sha256"] == before
    for layer in model.modules():
        if isinstance(layer, IndexedSparseLinear):
            assert layer.pre_w.numel() == layer.out_features * layer.K
    if family.startswith("dendritic_"):
        assert (
            sum(isinstance(m, ConfigurableEINetwork) for m in model.modules())
            == network_depth
        )
        for layer in report["layer_receipts"]:
            assert layer["production_config"]["reactivation"]["enabled"] is True


@pytest.mark.parametrize("family", ["dense", "sparse"])
def test_point_control_is_nonlinear_on_nonnegative_domain(family):
    model = build_model(
        _spec(
            family,
            input_dim=2,
            output_dim=1,
            width=2,
            contacts_e=2,
            input_encoding="nonnegative",
        )
    )
    affine = model.hidden[0][0]
    with torch.no_grad():
        if isinstance(affine, nn.Linear):
            affine.weight.fill_(1.0)
        else:
            affine.projection.pre_w.fill_(1.0)
        affine.bias.fill_(-1.0)
        model.readout.weight.fill_(1.0)
        model.readout.bias.zero_()
    x = torch.zeros(1, 2)
    y = torch.ones(1, 2)
    assert not torch.allclose(model((x + y) / 2), (model(x) + model(y)) / 2)


@pytest.mark.parametrize("family", FAMILIES)
def test_initialization_reproducibility_and_rng_isolation(family):
    spec = _spec(family)
    state = torch.random.get_rng_state().clone()
    first = build_model(spec)
    assert torch.equal(state, torch.random.get_rng_state())
    second = build_model(spec)
    assert all(
        torch.equal(first.state_dict()[k], v) for k, v in second.state_dict().items()
    )
    assert (
        model_report(first)["topology_sha256"]
        == model_report(second)["topology_sha256"]
    )
    changed = build_model({**spec, "seed": 12, "topology_seed": 11})
    assert (
        model_report(first)["topology_sha256"]
        == model_report(changed)["topology_sha256"]
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(first.parameters(), changed.parameters())
    )


@pytest.mark.parametrize("family", [f for f in FAMILIES if f != "dense"])
def test_topology_seed_changes_support(family):
    first = build_model(_spec(family, topology_seed=31))
    second = build_model(_spec(family, topology_seed=32))
    assert (
        model_report(first)["topology_sha256"]
        != model_report(second)["topology_sha256"]
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_budget_matching_uses_instantiated_total_and_preserves_other_axes(family):
    spec = _spec(family)
    original = dict(spec)
    resolved, report = match_parameter_budget(spec, 2048, tolerance=0.02)
    assert spec == original
    assert abs(report["total_parameters"] - 2048) <= 0.02 * 2048
    assert (
        model_report(build_model(resolved))["total_parameters"]
        == report["total_parameters"]
    )
    assert report["budget_matched"] is True
    assert resolved["contacts_e"] == spec["contacts_e"]
    assert resolved["branch_factors"] == spec["branch_factors"]
    assert resolved["network_depth"] == 1
    exact = model_report(build_model(spec))["total_parameters"]
    _, exact_report = match_parameter_budget(spec, exact, tolerance=0)
    assert exact_report["total_parameters"] == exact


def test_budget_unattainable_and_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="No width matches"):
        match_parameter_budget(_spec("dense"), 1, tolerance=0)
    with pytest.raises(ValueError, match="No width matches"):
        match_parameter_budget(_spec("dense"), 100, tolerance=0)
    for tolerance in (-0.1, float("nan"), float("inf"), True, 1):
        with pytest.raises(ValueError, match="tolerance"):
            match_parameter_budget(_spec("dense"), 100, tolerance=tolerance)
    with pytest.raises(ValueError, match="contacts_i=0"):
        build_model(_spec("dendritic_signed", contacts_i=1))
    with pytest.raises(ValueError, match="Unknown model spec"):
        build_model(_spec("dense", widht=4))


@pytest.mark.parametrize("family", FAMILIES)
def test_tiny_inputs_require_explicit_contact_clipping(family):
    spec = _spec(family, input_dim=1, width=1, network_depth=2)
    if family != "dense":
        with pytest.raises(ValueError, match="sources are available"):
            build_model(spec)
    model = build_model({**spec, "clip_contacts": True})
    assert torch.isfinite(model(torch.tensor([[-1.0], [1.0]]))).all()
    for receipt in model_report(model)["layer_receipts"]:
        assert receipt["contacts_e"] <= receipt["input_dim"]
        assert receipt["contacts_i"] <= receipt["input_dim"]


@pytest.mark.parametrize(
    "family", ["dendritic_signed", "dendritic_additive", "dendritic_shunting"]
)
def test_flat_dendritic_units_keep_input_contacts_and_nonlinear_somas(family):
    model = build_model(_spec(family, branch_factors=[]))
    output = model(torch.randn(4, 8))
    output.square().mean().backward()
    report = model_report(model)
    assert report["compartments_per_soma"] == 1
    assert (
        report["layer_receipts"][0]["production_config"]["morphology"][
            "somatic_synapses"
        ]
        is True
    )
    assert all(p.grad is not None for p in model.parameters())


def test_input_encoding_preserves_sign_and_nonnegative_contract_is_checked():
    model = build_model(_spec("dense", input_dim=2))
    x = torch.tensor([[-2.0, 3.0]])
    encoded = model.encoding(x)
    assert torch.equal(encoded[:, :2] - encoded[:, 2:], x)
    with pytest.raises(ValueError, match="negative input"):
        build_model(_spec("dense", input_encoding="nonnegative"))(
            torch.full((2, 8), -1.0)
        )
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, 2, 2))


@pytest.mark.parametrize(
    "family", ["dendritic_signed", "dendritic_additive", "dendritic_shunting"]
)
def test_equal_site_morphologies_have_equal_actual_parameter_inventory(family):
    reports = [
        model_report(build_model(_spec(family, branch_factors=factors)))
        for factors in ([14], [2, 6], [2, 2, 2])
    ]
    assert {report["compartments_per_soma"] for report in reports} == {15}
    assert len({report["total_parameters"] for report in reports}) == 1


def test_additive_shunting_contrast_matches_support_and_parameter_shapes():
    additive = model_report(
        build_model(_spec("dendritic_additive", branch_factors=[2, 2]))
    )
    shunting = model_report(
        build_model(_spec("dendritic_shunting", branch_factors=[2, 2]))
    )
    assert additive["total_parameters"] == shunting["total_parameters"]
    assert additive["parameter_shapes"] == shunting["parameter_shapes"]
    assert additive["topology_sha256"] == shunting["topology_sha256"]


@pytest.mark.parametrize(
    "family", ["dense", "sparse", "dendritic_signed", "grouped_gated"]
)
def test_raw_input_preserves_data_and_records_encoded_dimension(family):
    model = build_model(_spec(family, input_encoding="raw"))
    x = torch.randn(4, 8)
    assert model.encoding(x) is x
    assert torch.isfinite(model(x)).all()
    report = model_report(model)
    assert report["input_encoding"] == "raw"
    assert report["encoded_input_dim"] == 8
    assert report["layer_receipts"][0]["input_dim"] == 8
    if family == "dense":
        split_model = build_model(_spec(family))
        assert (
            model_report(split_model)["total_parameters"] - report["total_parameters"]
            == 8 * 8
        )


@pytest.mark.parametrize("family", ["dendritic_additive", "dendritic_shunting"])
def test_raw_input_rejected_for_positive_conductance_families(family):
    with pytest.raises(ValueError, match="conductance-input contract"):
        build_model(_spec(family, input_encoding="raw"))


@pytest.mark.parametrize(
    "family", ["dense", "sparse", "dendritic_signed", "grouped_gated"]
)
def test_fan_in_initialization_preserves_weight_and_signal_scale_across_width(family):
    x = torch.randn(256, 128, generator=torch.Generator().manual_seed(410))
    signal_variances = []
    for width in (128, 1024):
        model = build_model(
            _spec(
                family,
                input_dim=128,
                width=width,
                contacts_e=16,
                contacts_i=8 if family == "grouped_gated" else 0,
                input_encoding="raw",
                initialization="fan_in",
            )
        )
        if family == "dense":
            projection = model.hidden[0][0]
            weight, fan_in = projection.weight, 128
            assert torch.count_nonzero(projection.bias) == 0
        elif family == "sparse":
            projection = model.hidden[0][0].projection
            weight, fan_in = projection.pre_w, 16
        elif family == "grouped_gated":
            projection = model.hidden[0].value.projection
            weight, fan_in = projection.pre_w, 16
            gate_weight = model.hidden[0].gate.projection.pre_w
            assert float(gate_weight.detach().var()) == pytest.approx(1.0 / 8, rel=0.2)
        else:
            projection = next(
                m for m in model.modules() if isinstance(m, IndexedSparseLinear)
            )
            weight, fan_in = projection.pre_w, 16
        assert float(weight.detach().var()) == pytest.approx(2.0 / fan_in, rel=0.2)
        with torch.no_grad():
            variance = float(projection(x).var(dim=0).mean())
        assert variance == pytest.approx(2.0, rel=0.25)
        signal_variances.append(variance)
    assert signal_variances[1] / signal_variances[0] == pytest.approx(1.0, rel=0.25)


@pytest.mark.parametrize("family", FAMILIES)
def test_initialization_recipe_is_explicit_and_preserves_budget_and_support(family):
    fan_in = model_report(build_model(_spec(family)))
    legacy = model_report(build_model(_spec(family, initialization="legacy_xavier")))
    assert fan_in["initialization"] == "fan_in"
    assert legacy["initialization"] == "legacy_xavier"
    assert fan_in["total_parameters"] == legacy["total_parameters"]
    assert fan_in["topology_sha256"] == legacy["topology_sha256"]
    if family.startswith("dendritic_"):
        assert (
            fan_in["layer_receipts"][0]["production_config"]["sparsity"]["init_method"]
            == "kaiming_normal"
        )
        assert (
            legacy["layer_receipts"][0]["production_config"]["sparsity"]["init_method"]
            == "xavier_normal"
        )


def test_signed_couplings_and_hidden_signal_do_not_shrink_with_population_width():
    x = torch.randn(256, 128, generator=torch.Generator().manual_seed(512))
    variances = []
    for width in (128, 1024):
        model = build_model(
            _spec(
                "dendritic_signed",
                input_dim=128,
                width=width,
                contacts_e=16,
                input_encoding="raw",
                initialization="fan_in",
                branch_factors=[2, 2],
            )
        )
        couplings = [m for m in model.modules() if isinstance(m, BlockLinear)]
        assert couplings
        for coupling in couplings:
            assert float(coupling.log_weight.detach().var()) == pytest.approx(
                2.0 / coupling.block_size, rel=0.25
            )
        with torch.no_grad():
            variances.append(float(model.hidden(model.encoding(x)).var(dim=0).mean()))
        assert model_report(model)["layer_receipts"][0][
            "final_signed_parameter_fan_ins"
        ]
    assert variances[1] / variances[0] == pytest.approx(1.0, rel=0.35)


@pytest.mark.parametrize("family", FAMILIES)
def test_projection_backend_defaults_preserve_explicit_eager_state(family):
    default = build_model(_spec(family))
    explicit = build_model(_spec(family, projection_backend=" EAGER "))
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    report = model_report(default)
    assert report["projection_backend"] == "eager"
    assert report["resolved_spec"]["projection_backend"] == "eager"
    if family == "dense":
        assert report["indexed_projection_backends"] == {}


@pytest.mark.parametrize("backend", ["triton_transposed", "triton_fused", "triton_ell"])
@pytest.mark.parametrize("family", [value for value in FAMILIES if value != "dense"])
def test_accelerator_backend_routes_all_banks_without_changing_initial_state(
    family, backend
):
    spec = _spec(family, network_depth=2, branch_factors=[2, 2])
    eager = build_model(spec)
    accelerated = build_model({**spec, "projection_backend": backend.upper()})
    eager_report, accelerated_report = model_report(eager), model_report(accelerated)
    assert accelerated_report["projection_backend"] == backend
    for field in (
        "total_parameters",
        "trainable_parameters",
        "parameter_bytes",
        "buffer_bytes",
        "parameter_shapes",
        "topology_sha256",
    ):
        assert accelerated_report[field] == eager_report[field]
    assert eager.state_dict().keys() == accelerated.state_dict().keys()
    for name, value in eager.state_dict().items():
        torch.testing.assert_close(
            value, accelerated.state_dict()[name], rtol=0, atol=0
        )
    projections = {
        name: module
        for name, module in accelerated.named_modules()
        if isinstance(module, IndexedSparseLinear)
    }
    assert projections
    assert (
        projections.keys() == accelerated_report["indexed_projection_backends"].keys()
    )
    for name, projection in projections.items():
        assert projection.projection_backend == backend
        assert accelerated_report["indexed_projection_backends"][name] == {
            "requested": backend,
            "resolved": None,
            "device": None,
        }
    if family == "grouped_gated":
        assert len(projections) == 4  # value and gate at each external layer
    if family.startswith("dendritic_"):
        for layer in accelerated_report["layer_receipts"]:
            assert (
                layer["production_config"]["sparsity"]["indexed"]["projection_backend"]
                == backend
            )
        if family != "dendritic_signed":
            assert any("exc" in name for name in projections)
            assert any("inh" in name for name in projections)


@pytest.mark.parametrize("backend", ["recompute", "auto"])
@pytest.mark.parametrize("family", [value for value in FAMILIES if value != "dense"])
def test_cpu_backend_preserves_whole_model_forward_input_and_parameter_gradients(
    family, backend
):
    spec = _spec(family, network_depth=2)
    eager = build_model(spec)
    alternative = build_model({**spec, "projection_backend": backend})
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(947)).requires_grad_()
    alternate_x = x.detach().clone().requires_grad_()
    expected, actual = eager(x), alternative(alternate_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn(expected.shape, generator=torch.Generator().manual_seed(948))
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(alternate_x.grad, x.grad, rtol=1e-5, atol=1e-6)
    alternative_parameters = dict(alternative.named_parameters())
    for name, parameter in eager.named_parameters():
        assert (
            parameter.grad is not None and alternative_parameters[name].grad is not None
        )
        torch.testing.assert_close(
            alternative_parameters[name].grad, parameter.grad, rtol=1e-5, atol=1e-6
        )
    observations = model_report(alternative)["indexed_projection_backends"]
    assert observations
    assert all(
        entry == {"requested": backend, "resolved": "recompute", "device": "cpu"}
        for entry in observations.values()
    )


@pytest.mark.parametrize("backend", ["unknown", "", None, False, 1, {}])
def test_projection_backend_schema_rejects_invalid_values(backend):
    with pytest.raises(ValueError, match="projection_backend"):
        build_model(_spec("sparse", projection_backend=backend))


@pytest.mark.parametrize(
    "backend", ["recompute", "auto", "triton_transposed", "triton_fused", "triton_ell"]
)
def test_dense_rejects_inapplicable_projection_backends(backend):
    with pytest.raises(ValueError, match="dense has no indexed projections"):
        build_model(_spec("dense", projection_backend=backend))


@pytest.mark.parametrize("backend", ["triton_transposed", "triton_fused", "triton_ell"])
def test_explicit_accelerator_backend_does_not_silently_fallback_on_cpu(backend):
    model = build_model(_spec("sparse", projection_backend=backend))
    with pytest.raises(RuntimeError, match="unavailable"):
        model(torch.randn(2, 8))


def test_budget_matching_preserves_backend_and_nearest_width():
    eager, eager_report = match_parameter_budget(_spec("sparse"), 2048)
    accelerated, report = match_parameter_budget(
        _spec("sparse", projection_backend="triton_transposed"), 2048
    )
    assert accelerated["width"] == eager["width"]
    assert accelerated["projection_backend"] == "triton_transposed"
    assert report["total_parameters"] == eager_report["total_parameters"]
    assert report["topology_sha256"] == eager_report["topology_sha256"]
