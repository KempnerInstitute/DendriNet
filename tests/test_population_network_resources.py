"""Resource-accounting tests for recurrent population networks."""

from __future__ import annotations

import torch

from dendritic_modeling.analysis.utils.model_resources import (
    model_resource_summary,
    static_dendritic_resource_summary,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def test_static_direct_tree_resources_match_the_declared_config() -> None:
    summary = static_dendritic_resource_summary(
        branch_factors=[3, 3, 3, 3],
        n_somas=20,
        pathway_input_dims={"excitatory": 3072, "inhibitory": 3072},
        synapses_per_compartment={"excitatory": 25, "inhibitory": 25},
        somatic_synapses=False,
    )

    assert summary.level_compartments_per_soma == (3, 9, 27, 81)
    assert summary.nonsomatic_compartments_per_soma == 120
    assert summary.nonsomatic_compartments_total == 2400
    assert summary.active_synapses_by_pathway == {
        "excitatory": 60_000,
        "inhibitory": 60_000,
    }
    assert summary.active_synapses_total == 120_000
    assert summary.candidate_synapse_slots_total == 14_745_600
    assert summary.child_couplings_total == 2400
    assert summary.reactivation_parameters_total == 4840


def _tiny_recurrent_population_network() -> PopulationNetwork:
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=5,
            input_projection_dims=[4],
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 2,
                        "rec_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "reactivate": True,
                        "dbl_init_method": "default",
                        "weight_transform": "exp",
                        "topk_type": "standard",
                        "allow_self_recurrence": False,
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 3,
                            "branch_factors": [2],
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [],
                        },
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "pv"},
                        {"source": "pyr", "target": "pyr", "timing": "delayed"},
                        {"source": "pv", "target": "pyr", "timing": "delayed"},
                        {"source": "pyr", "target": "pv", "timing": "delayed"},
                        {"source": "pv", "target": "pv", "timing": "delayed"},
                    ],
                )
            ],
        )
    )


def test_effective_params_replace_every_recurrent_candidate_bank_with_mask_count() -> (
    None
):
    network = _tiny_recurrent_population_network()
    total = sum(parameter.numel() for parameter in network.parameters())
    expected = total
    saw_recurrent = False

    for branch_layer in network.branch_layers:
        for attr in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        ):
            synapse = getattr(branch_layer, attr, None)
            if synapse is None:
                continue
            if attr in {"branch_recurrent", "branch_rec_inhibition"}:
                saw_recurrent = True
            expected -= sum(parameter.numel() for parameter in synapse.parameters())
            expected += int(synapse.weight_mask().ne(0).sum().item())

    assert saw_recurrent
    assert network.get_effective_params() == expected
    assert network.get_effective_params() < total


def test_resource_summary_reports_recurrent_paths_point_i_and_persistent_state() -> (
    None
):
    network = _tiny_recurrent_population_network()

    summary = model_resource_summary(network)

    assert summary["trainable_parameters"] == sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    assert summary["active_synapses"] > 0
    assert summary["candidate_synapse_slots"] >= summary["active_synapses"]
    assert summary["persistent_state_scalars_per_sample"] > 0
    assert {record["pathway"] for record in summary["synapses"]} >= {
        "ff_excitatory",
        "rec_excitatory",
        "rec_inhibitory",
    }
    pv = network.layers[0].populations["pv"]
    assert pv.n_levels == 1


def test_population_network_no_self_recurrence_masks_same_population_sources() -> None:
    network = _tiny_recurrent_population_network()
    layer = network.layers[0]

    for target_name, pathway_attr in (
        ("pyr", "branch_recurrent"),
        ("pv", "branch_rec_inhibition"),
    ):
        population = layer.populations[target_name]
        n_target = layer._population_dims[target_name]
        for branch_layer in population.branch_layers:
            synapse = getattr(branch_layer, pathway_attr)
            mask = synapse.weight_mask().bool()
            repeats = mask.shape[0] // n_target
            owner = torch.arange(mask.shape[0]) // repeats
            assert not bool(mask[torch.arange(mask.shape[0]), owner].any())


def _indexed_point_population_network(
    *, layer_name: str = "microcircuit", population_name: str = "pyr"
) -> PopulationNetwork:
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=4,
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name=layer_name,
                    recurrent=True,
                    population_defaults={
                        "ff_excitatory_synapses": 2,
                        "rec_excitatory_synapses": 2,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "weight_transform": "identity",
                        "topk_type": "indexed",
                        "indexed_seed": 17,
                    },
                    populations=[
                        {
                            "name": population_name,
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [],
                        }
                    ],
                    connections=[
                        {"source": "input", "target": population_name},
                        {
                            "source": population_name,
                            "target": population_name,
                            "timing": "delayed",
                        },
                    ],
                )
            ],
        )
    )


def test_indexed_topology_is_reproducible_and_namespaced_by_pathway() -> None:
    first = _indexed_point_population_network()
    second = _indexed_point_population_network()
    first_branch = first.layers[0].populations["pyr"].branch_layers[0]
    second_branch = second.layers[0].populations["pyr"].branch_layers[0]

    assert torch.equal(
        first_branch.branch_excitation.connection_indices,
        second_branch.branch_excitation.connection_indices,
    )
    assert torch.equal(
        first_branch.branch_recurrent.connection_indices,
        second_branch.branch_recurrent.connection_indices,
    )
    assert not torch.equal(
        first_branch.branch_excitation.connection_indices,
        first_branch.branch_recurrent.connection_indices,
    )


def test_indexed_resource_summary_uses_compact_fixed_indices(monkeypatch) -> None:
    network = _indexed_point_population_network()

    def reject_dense_snapshot(_synapse):
        raise AssertionError("indexed resource accounting must remain compact")

    monkeypatch.setattr(
        "dendritic_modeling.analysis.utils.model_resources."
        "effective_synapse_snapshot",
        reject_dense_snapshot,
    )
    summary = model_resource_summary(network)

    assert summary["active_synapses"] == 16
    assert summary["candidate_synapse_slots"] == 32
    assert {record["selection_policy"] for record in summary["synapses"]} == {
        "indexed_fixed"
    }
    assert {record["mask_source"] for record in summary["synapses"]} == {
        "fixed_indices"
    }


def test_indexed_effective_params_uses_exact_compact_counts(monkeypatch) -> None:
    network = _indexed_point_population_network()
    total = sum(parameter.numel() for parameter in network.parameters())
    expected = total
    synapses = []
    for branch_layer in network.branch_layers:
        for attr in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        ):
            synapse = getattr(branch_layer, attr, None)
            if synapse is None or synapse in synapses:
                continue
            synapses.append(synapse)
            expected -= sum(parameter.numel() for parameter in synapse.parameters())
            expected += int(synapse.connectivity_resource_counts()["active_synapses"])

    def reject_dense_mask():
        raise AssertionError("indexed active accounting must remain compact")

    for synapse in synapses:
        monkeypatch.setattr(synapse, "weight_mask", reject_dense_mask)

    assert network.get_effective_params() == expected


def test_indexed_topology_base_seed_is_namespaced_by_layer_and_population() -> None:
    base = _indexed_point_population_network()
    renamed_layer = _indexed_point_population_network(layer_name="renamed")
    renamed_population = _indexed_point_population_network(population_name="principal")

    base_seed = base.layers[0].population_configs["pyr"].indexed_seed
    assert base_seed != renamed_layer.layers[0].population_configs["pyr"].indexed_seed
    assert (
        base_seed
        != renamed_population.layers[0].population_configs["principal"].indexed_seed
    )
