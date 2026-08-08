"""Contracts for the progressively scaled large dendritic RNN configs."""

from pathlib import Path

import pytest

from dendritic_modeling.config import load_config

CONFIG_DIR = Path("configs/dendritic_rnn/large_scale_500k_5k")


def _level_multiplicities(branch_factors):
    multiplicities = []
    for level_idx in range(len(branch_factors) + 1):
        product = 1
        for factor in branch_factors[level_idx:]:
            product *= factor
        multiplicities.append(product)
    return multiplicities


def _contacts_per_target(population):
    multiplicities = _level_multiplicities(population["branch_factors"])
    excitatory_by_level = [
        feedforward + recurrent
        for feedforward, recurrent in zip(
            population["ff_excitatory_synapses_by_level"],
            population["rec_excitatory_synapses_by_level"],
            strict=True,
        )
    ]
    inhibitory_by_level = [
        feedforward + recurrent
        for feedforward, recurrent in zip(
            population["ff_inhibitory_synapses_by_level"],
            population["rec_inhibitory_synapses_by_level"],
            strict=True,
        )
    ]
    excitatory = sum(
        multiplicity * count
        for multiplicity, count in zip(multiplicities, excitatory_by_level, strict=True)
    )
    inhibitory = sum(
        multiplicity * count
        for multiplicity, count in zip(multiplicities, inhibitory_by_level, strict=True)
    )
    return excitatory, inhibitory


@pytest.mark.parametrize(
    ("filename", "n_excitatory", "n_inhibitory", "contacts_per_neuron"),
    [
        ("canary.yaml", 400, 100, 50),
        ("gate_5k_500.yaml", 4_000, 1_000, 500),
        ("gate_50k_5k.yaml", 40_000, 10_000, 5_000),
        ("base.yaml", 400_000, 100_000, 5_000),
    ],
)
def test_large_rnn_scale_configs_preserve_contact_contract(
    filename,
    n_excitatory,
    n_inhibitory,
    contacts_per_neuron,
):
    config = load_config(str(CONFIG_DIR / filename))
    layer = config.model.core.unified_ei["layers"][0]
    excitatory = layer["excitatory"]
    inhibitory = layer["inhibitory"]

    assert excitatory["n_neurons"] == n_excitatory
    assert inhibitory["n_neurons"] == n_inhibitory
    for population in (excitatory, inhibitory):
        excitatory_contacts, inhibitory_contacts = _contacts_per_target(population)
        assert excitatory_contacts == 4 * contacts_per_neuron // 5
        assert inhibitory_contacts == contacts_per_neuron // 5
        assert population["indexed_index_dtype"] == "auto"
        assert population["indexed_workspace_mb"] == 128
        assert population["indexed_cache_transformed_weights"] is True
        assert population["indexed_projection_backend"] == "recompute"
        assert population.get("indexed_recompute_backward", False) is False
        assert population["indexed_persistent_indices"] is False
        assert population["indexed_init_mode"] == "rank0_broadcast"
        assert population["allow_self_recurrence"] is False

    total_neurons = n_excitatory + n_inhibitory
    assert total_neurons * contacts_per_neuron in {
        25_000,
        2_500_000,
        250_000_000,
        2_500_000_000,
    }
