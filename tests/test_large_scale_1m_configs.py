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
        ("diagnostic_20k_10k_adaptive.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_adaptive_lr10x.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_corrected.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_lr10x.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_normalized.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_normalized_tc100.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_readout_only.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_scaled_lr.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_10k_zero_head.yaml", 16_000, 4_000, 10_000),
        ("diagnostic_20k_20k_frozen_readout.yaml", 16_000, 4_001, 20_000),
        ("gate_100k_10k.yaml", 80_000, 20_000, 10_000),
        ("gate_100k_10k_scaled_lr.yaml", 80_000, 20_000, 10_000),
        ("base.yaml", 800_000, 200_000, 10_000),
        ("learn_1m_10k.yaml", 800_000, 200_000, 10_000),
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
    assert max(excitatory["rec_excitatory_synapses_by_level"]) <= n_excitatory - 1
    assert max(excitatory["rec_inhibitory_synapses_by_level"]) <= n_inhibitory
    assert max(inhibitory["rec_excitatory_synapses_by_level"]) <= n_excitatory
    assert max(inhibitory["rec_inhibitory_synapses_by_level"]) <= n_inhibitory - 1

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


@pytest.mark.parametrize(
    ("filename", "decoder_lr", "core_lr", "epochs"),
    [
        ("diagnostic_20k_10k_scaled_lr.yaml", 4e-6, 1e-3, 30),
        ("gate_100k_10k_scaled_lr.yaml", 8e-7, 1e-3, 30),
        ("learn_1m_10k.yaml", 8e-8, 1e-5, 10),
    ],
)
def test_width_scaled_learning_configs_split_optimizer_groups(
    filename,
    decoder_lr,
    core_lr,
    epochs,
):
    config = load_config(str(CONFIG_DIR / filename))
    groups = config.training.main.common.param_groups

    assert groups.split_params is True
    assert groups.topk_lr == core_lr
    assert groups.decoder_input_lr == decoder_lr
    assert groups.decoder_lr == 1e-3
    assert config.training.main.common.epochs == epochs


def test_degree_aware_diagnostic_enables_adaptive_initialization():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_adaptive.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert layer["excitatory"]["adaptive_initialization"] is True
    assert layer["inhibitory"]["adaptive_initialization"] is True


def test_lr10x_diagnostic_uses_larger_batches():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_lr10x.yaml"))

    assert config.training.main.common.batch_size == 8
    assert config.training.main.common.param_groups.decoder_input_lr == 4e-5
    assert config.training.main.common.param_groups.decoder_lr == 1e-3


def test_adaptive_lr10x_combines_both_interventions():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_adaptive_lr10x.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert config.training.main.common.batch_size == 8
    assert config.training.main.common.param_groups.decoder_input_lr == 4e-5
    assert config.training.main.common.param_groups.decoder_lr == 1e-3
    assert layer["excitatory"]["adaptive_initialization"] is True
    assert layer["inhibitory"]["adaptive_initialization"] is True


def test_20k_contact_diagnostic_keeps_feedforward_pathways_valid():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_20k_frozen_readout.yaml"))
    inhibitory = config.model.core.unified_ei["layers"][0]["inhibitory"]
    groups = config.training.main.common.param_groups

    assert inhibitory["ff_excitatory_synapses_by_level"] == [32]
    assert inhibitory["ff_inhibitory_synapses_by_level"] == [0]
    assert inhibitory["rec_excitatory_synapses_by_level"] == [15_968]
    assert inhibitory["rec_inhibitory_synapses_by_level"] == [4_000]
    assert config.distributed.fsdp.freeze_core is True
    assert config.distributed.fsdp.gradient_checkpointing is False
    assert config.model.decoder.params.input_normalization == "layer_norm"
    assert config.model.decoder.params.output_init == "zero"
    assert groups.lr == 0.0
    assert groups.topk_lr == 0.0
    assert groups.blocklinear_lr == 0.0
    assert groups.reactivation_lr == 0.0
    assert groups.decoder_input_lr == 4e-6
    assert groups.decoder_lr == 1e-3


def test_normalized_diagnostic_uses_parameter_free_readout_normalization():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_normalized.yaml"))
    decoder = config.model.decoder.params

    assert decoder.input_normalization == "layer_norm"
    assert config.training.main.common.param_groups.decoder_input_lr == 4e-6
    assert config.training.main.common.param_groups.decoder_lr == 1e-3


def test_tc100_diagnostic_preserves_moderate_recurrent_conductance():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_normalized_tc100.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert layer["excitatory"]["adaptive_target_conductance"] == 100.0
    assert layer["inhibitory"]["adaptive_target_conductance"] == 100.0


def test_exact_learning_config_uses_validated_scale_interventions():
    config = load_config(str(CONFIG_DIR / "learn_1m_10k.yaml"))
    layer = config.model.core.unified_ei["layers"][0]
    groups = config.training.main.common.param_groups

    assert config.model.decoder.params.input_normalization == "layer_norm"
    assert config.model.decoder.params.output_init == "zero"
    assert groups.decoder_input_lr == 8e-8
    assert groups.decoder_lr == 1e-3
    assert groups.topk_lr == 1e-5
    for population_name in ("excitatory", "inhibitory"):
        population = layer[population_name]
        assert population["adaptive_initialization"] is True
        assert population["adaptive_target_conductance"] == 100.0


def test_corrected_diagnostic_scales_wide_weight_and_small_head_separately():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_corrected.yaml"))
    groups = config.training.main.common.param_groups

    assert config.training.main.common.batch_size == 8
    assert groups.decoder_input_lr == 4e-6
    assert groups.decoder_lr == 1e-4


def test_readout_only_diagnostic_freezes_core_before_fsdp_wrapping():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_readout_only.yaml"))

    assert config.distributed.fsdp.freeze_core is True
    assert config.distributed.fsdp.gradient_checkpointing is False


def test_exact_1m_20k_frozen_readout_contract():
    config = load_config(str(CONFIG_DIR / "test_1m_20k_frozen_readout.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert layer["excitatory"]["n_neurons"] == 800_000
    assert layer["inhibitory"]["n_neurons"] == 200_000
    for population in (layer["excitatory"], layer["inhibitory"]):
        assert _contacts_per_target(population) == (16_000, 4_000)
        assert population["indexed_workspace_mb"] == 32
        assert population["indexed_cache_transformed_weights"] is False
        assert population["indexed_recompute_backward"] is False
    assert config.distributed.fsdp.freeze_core is True
    assert config.distributed.fsdp.cache_frozen_core_outputs is True
    assert config.distributed.fsdp.cached_core_batch_size == 4
    assert config.distributed.fsdp.gradient_checkpointing is False
    assert config.model.decoder.params.hidden_dims == [32]
    assert config.model.decoder.params.output_init == "zero"
    assert config.training.main.common.epochs == 30
    dataset = config.data.dataset_params["romo_delay_comparison_positive"]
    assert dataset["train"]["n_samples"] == 128
    assert dataset["valid"]["n_samples"] == 32
    assert dataset["test"]["n_samples"] <= config.experiment.max_test_samples


def test_exact_1m_20k_5000_epoch_extension_preserves_model_contract():
    config = load_config(str(CONFIG_DIR / "test_1m_20k_frozen_readout_5000ep.yaml"))
    layer = config.model.core.unified_ei["layers"][0]

    assert layer["excitatory"]["n_neurons"] == 800_000
    assert layer["inhibitory"]["n_neurons"] == 200_000
    for population in (layer["excitatory"], layer["inhibitory"]):
        assert _contacts_per_target(population) == (16_000, 4_000)
    assert config.distributed.fsdp.freeze_core is True
    assert config.distributed.fsdp.cache_frozen_core_outputs is True
    assert config.distributed.fsdp.cached_core_batch_size == 4
    assert config.training.main.common.epochs == 5000
    assert config.training.main.common.print_every == 100
    assert config.outputs.exact_run_dir is True
    assert config.outputs.results_dir.endswith("test_1m_20k_frozen_readout_5000ep")


def test_zero_head_diagnostic_starts_with_balanced_logits():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_zero_head.yaml"))

    assert config.model.decoder.params.output_init == "zero"
    assert config.training.main.common.param_groups.decoder_input_lr == 4e-6
    assert config.training.main.common.param_groups.decoder_lr == 1e-3


def test_readout_only_diagnostic_freezes_all_core_optimizer_groups():
    config = load_config(str(CONFIG_DIR / "diagnostic_20k_10k_readout_only.yaml"))
    groups = config.training.main.common.param_groups

    assert groups.lr == 0.0
    assert groups.topk_lr == 0.0
    assert groups.blocklinear_lr == 0.0
    assert groups.reactivation_lr == 0.0
    assert groups.decoder_input_lr == 4e-6
    assert groups.decoder_lr == 1e-3
