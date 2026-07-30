"""Contracts for the 1M-neuron / 10k-contact dendritic RNN configs."""

from pathlib import Path

import pytest

from dendritic_modeling.config import load_config

CONFIG_DIR = Path("configs/dendritic_rnn/large_scale_1m_10k")


def _level_multiplicities(branch_factors):
    return [
        _product(branch_factors[level_idx:])
        for level_idx in range(len(branch_factors) + 1)
    ]


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _contacts_per_target(population):
    multiplicities = _level_multiplicities(population["branch_factors"])
    excitatory = sum(
        multiplicity * (feedforward + recurrent)
        for multiplicity, feedforward, recurrent in zip(
            multiplicities,
            population["ff_excitatory_synapses_by_level"],
            population["rec_excitatory_synapses_by_level"],
            strict=True,
        )
    )
    inhibitory = sum(
        multiplicity * (feedforward + recurrent)
        for multiplicity, feedforward, recurrent in zip(
            multiplicities,
            population["ff_inhibitory_synapses_by_level"],
            population["rec_inhibitory_synapses_by_level"],
            strict=True,
        )
    )
    return excitatory, inhibitory


@pytest.mark.parametrize(
    ("filename", "n_excitatory", "n_inhibitory", "contacts_per_neuron"),
    [
        ("diagnostic_20k_1k.yaml", 16_000, 4_000, 1_000),
        ("diagnostic_20k_10k.yaml", 16_000, 4_000, 10_000),
        ("gate_100k_10k.yaml", 80_000, 20_000, 10_000),
        ("base.yaml", 800_000, 200_000, 10_000),
    ],
)
def test_large_1m_configs_preserve_population_and_contact_contract(
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
    assert excitatory["branch_factors"] == [3, 3, 3]
    assert inhibitory["branch_factors"] == []
    assert _contacts_per_target(excitatory) == (
        4 * contacts_per_neuron // 5,
        contacts_per_neuron // 5,
    )
    assert _contacts_per_target(inhibitory) == (
        4 * contacts_per_neuron // 5,
        contacts_per_neuron // 5,
    )

    for population in (excitatory, inhibitory):
        assert population["indexed_index_dtype"] == "auto"
        assert population["indexed_workspace_mb"] == 128
        assert population["indexed_cache_transformed_weights"] is True
        assert population["indexed_recompute_backward"] is True
        assert population["indexed_persistent_indices"] is False
        assert population["indexed_init_mode"] == "rank0_broadcast"
        assert population["allow_self_recurrence"] is False
        assert population["dbl_init_method"] == "analytical_expectation"
        assert population["reactivation_init_policy"] == "analytical"


def test_exact_1m_config_has_expected_compartment_and_contact_scale():
    config = load_config(str(CONFIG_DIR / "base.yaml"))
    layer = config.model.core.unified_ei["layers"][0]
    excitatory = layer["excitatory"]
    inhibitory = layer["inhibitory"]

    excitatory_compartments = excitatory["n_neurons"] * sum(
        _level_multiplicities(excitatory["branch_factors"])
    )
    total_compartments = excitatory_compartments + inhibitory["n_neurons"]
    total_neurons = excitatory["n_neurons"] + inhibitory["n_neurons"]

    assert total_compartments == 32_200_000
    assert total_neurons * sum(_contacts_per_target(excitatory)) == 10_000_000_000


def test_paper_scale_control_uses_indexed_fsdp_path():
    config = load_config(str(CONFIG_DIR / "diagnostic_paper_scale_fsdp.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert config.distributed.mode == "fsdp"
    assert config.distributed.fsdp.auto_wrap_policy == "indexed_synapses"
    assert config.training.main.common.epochs == 20
    for population_name in ("excitatory", "inhibitory"):
        population = layer[population_name]
        assert population["topk_type"] == "indexed"
        assert population["indexed_recompute_backward"] is True
        assert population["branch_factors"] == [3, 3]


def test_exact_target_uses_indexed_synapse_fsdp_wrapping():
    config = load_config(str(CONFIG_DIR / "base.yaml"))

    assert config.distributed.fsdp.auto_wrap_policy == "indexed_synapses"
