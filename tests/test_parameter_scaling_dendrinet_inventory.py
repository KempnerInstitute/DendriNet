"""Observed production anatomy, independent counts, and unchanged computation."""

from __future__ import annotations

from math import prod

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.benchmarks.parameter_budget import (
    build_model,
    model_report,
)


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _spec(**overrides):
    return {
        "family": "dendritic_signed",
        "input_dim": 12,
        "output_dim": 3,
        "width": 8,
        "branch_factors": [2, 2],
        "contacts_e": 4,
        "contacts_i": 0,
        "network_depth": 1,
        "seed": 19,
        **overrides,
    }


@pytest.mark.parametrize("branch_factors", [[], [6], [2, 2], [2, 2, 2]])
@pytest.mark.parametrize("network_depth", [1, 3])
@pytest.mark.parametrize(
    "family, contacts_i",
    [("dendritic_signed", 0), ("dendritic_additive", 2), ("dendritic_shunting", 2)],
)
def test_observed_anatomy_matches_independent_tree_counts(
    branch_factors, network_depth, family, contacts_i
):
    model = build_model(
        _spec(
            family=family,
            branch_factors=branch_factors,
            contacts_i=contacts_i,
            network_depth=network_depth,
        )
    )
    report = model_report(model)
    inventory = report["dendrinet_inventory"]
    nonsomatic_per_soma = sum(
        prod(branch_factors[: depth + 1]) for depth in range(len(branch_factors))
    )
    contacted_per_soma = nonsomatic_per_soma or 1
    all_somas = 8 * network_depth
    e_count = all_somas * contacted_per_soma * 4
    i_count = all_somas * contacted_per_soma * contacts_i
    coupling_count = all_somas * nonsomatic_per_soma
    readout_count = 8 * 3 + 3
    assert inventory["population_count"] == network_depth
    assert inventory["population_count"] == sum(
        isinstance(module, DendriNet) for module in model.modules()
    )
    assert inventory["total_somas"] == all_somas
    assert inventory["total_compartments"] == all_somas * (nonsomatic_per_soma + 1)
    assert inventory["total_nonsomatic_compartments"] == all_somas * nonsomatic_per_soma
    assert inventory["parameter_partition"] == {
        "excitatory_synapses": e_count,
        "inhibitory_synapses": i_count,
        "recurrent_synapses": 0,
        "child_couplings": coupling_count,
        "reactivation": 0,
        "readout": readout_count,
        "shared": 0,
        "other": 0,
    }
    assert (
        report["total_parameters"] == e_count + i_count + coupling_count + readout_count
    )
    assert sum(inventory["parameter_partition"].values()) == report["total_parameters"]
    assert (
        sum(inventory["trainable_parameter_partition"].values())
        == report["trainable_parameters"]
    )
    for population in inventory["populations"]:
        assert population["contact_placement"] == (
            "all_nonsomatic_depths" if branch_factors else "soma_only"
        )
        assert population["internal_branch_depth"] == len(branch_factors)
        assert [d["depth_from_soma"] for d in population["depths"]] == list(
            range(len(branch_factors), -1, -1)
        )
        for depth in population["depths"]:
            contacted = bool(depth["depth_from_soma"] or not branch_factors)
            assert depth["contacts_e_per_compartment"] == (4 if contacted else 0)
            assert depth["contacts_i_per_compartment"] == (
                contacts_i if contacted else 0
            )
            assert depth["gate_enabled"] is True
            assert depth["gate_type"] == "ReLU"
            assert depth["gate_parameters"] == 0
            assert depth["total_parameters"] == (
                depth["input_synapse_parameters"] + depth["child_coupling_parameters"]
            )


def test_same_compartment_depth_contrast_has_exactly_same_budget():
    shallow = model_report(build_model(_spec(branch_factors=[6])))
    deep = model_report(build_model(_spec(branch_factors=[2, 2])))
    assert shallow["total_parameters"] == deep["total_parameters"]
    assert (
        shallow["dendrinet_inventory"]["parameter_partition"]
        == deep["dendrinet_inventory"]["parameter_partition"]
    )
    assert (
        shallow["dendrinet_inventory"]["total_compartments"]
        == deep["dendrinet_inventory"]["total_compartments"]
    )
    shallow_depths = shallow["dendrinet_inventory"]["populations"][0]["depths"]
    deep_depths = deep["dendrinet_inventory"]["populations"][0]["depths"]
    assert [d["compartment_count"] for d in shallow_depths] == [48, 8]
    assert [d["compartment_count"] for d in deep_depths] == [32, 16, 8]
    assert sum(d["children_per_compartment"] for d in shallow_depths) == 6
    assert sum(d["children_per_compartment"] for d in deep_depths) == 4


def test_inventory_reports_clipping_and_rejects_implicit_clipping():
    spec = _spec(input_dim=1, width=2, contacts_e=4, network_depth=2)
    with pytest.raises(ValueError, match="sources are available"):
        build_model(spec)
    model = build_model({**spec, "clip_contacts": True})
    inventory = model_report(model)["dendrinet_inventory"]
    for population in inventory["populations"]:
        for depth in population["depths"]:
            assert depth["contacts_e_per_compartment"] == (0 if depth["is_soma"] else 2)


def test_inventory_is_read_only_and_each_reported_gate_is_executed():
    model = build_model(_spec(network_depth=2)).double()
    x = torch.randn(4, 12, dtype=torch.float64)
    expected = model(x)
    expected_grads = torch.autograd.grad(
        expected.square().sum(), tuple(model.parameters())
    )
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    report = model_report(model)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in state.items()
    )
    seen = []
    handles = []
    for population in report["dendrinet_inventory"]["populations"]:
        for depth in population["depths"]:
            branch = model.get_submodule(depth["module_name"])
            handles.append(
                branch.reactivation.register_forward_hook(
                    lambda module, args, output, name=depth["module_name"]: seen.append(
                        name
                    )
                )
            )
    try:
        actual = model(x)
        actual_grads = torch.autograd.grad(
            actual.square().sum(), tuple(model.parameters())
        )
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
    assert len(seen) == 6
    assert len(set(seen)) == 6
    assert report["buffer_bytes"] == sum(
        b["bytes"] for b in report["dendrinet_inventory"]["registered_buffer_inventory"]
    )


def test_partition_counts_frozen_and_shared_parameters_once():
    model = build_model(_spec())
    projection = (
        model.hidden[0].layers[0].excitatory_cells.branch_layers[0].branch_excitation
    )
    projection.pre_w.requires_grad_(False)
    report = model_report(model)
    inventory = report["dendrinet_inventory"]
    assert inventory["trainable_parameter_partition"]["excitatory_synapses"] == (
        inventory["parameter_partition"]["excitatory_synapses"]
        - projection.pre_w.numel()
    )
    # A deliberately tied alias must not inflate the whole-model inventory.
    model.readout.register_parameter("shared_synapse", projection.pre_w)
    tied = model_report(model)
    assert tied["total_parameters"] == report["total_parameters"]
    assert (
        tied["dendrinet_inventory"]["parameter_partition"]["shared"]
        == projection.pre_w.numel()
    )
    assert (
        sum(tied["dendrinet_inventory"]["parameter_partition"].values())
        == tied["total_parameters"]
    )


def test_point_controls_do_not_claim_dendrinet_inventory():
    for family in ("dense", "sparse", "grouped_gated"):
        report = model_report(build_model(_spec(family=family)))
        assert "dendrinet_inventory" not in report
