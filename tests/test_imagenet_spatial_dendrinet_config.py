"""Contract tests for the raw ImageNet spatial DendriNet experiment."""

import math
from pathlib import Path

from dendritic_modeling.config import load_config

BP_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "vision"
    / "imagenet_raw_spatial_dendrinet_alexnet_matched_bp.yaml"
)
LOCALCA_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "vision"
    / "imagenet_raw_spatial_dendrinet_alexnet_matched_localca.yaml"
)
CONFIG_PATHS = (BP_CONFIG_PATH, LOCALCA_CONFIG_PATH)


def test_raw_imagenet_config_is_alexnet_parameter_matched() -> None:
    for config_path in CONFIG_PATHS:
        config = load_config(str(config_path))
        core = config.model.core

        n_soma = core.architecture.excitatory_layer_sizes[0]
        branch_factors = core.architecture.excitatory_branch_factors
        branches_per_soma = sum(
            math.prod(branch_factors[:depth])
            for depth in range(1, len(branch_factors) + 1)
        )
        compartments_per_soma = branches_per_soma + 1
        k_exc = core.connectivity.ee_synapses_per_branch_per_layer[0]
        k_inh = core.connectivity.ie_synapses_per_branch_per_layer[0]
        output_dim = config.model.decoder.params.output_dim

        branch_synapses = n_soma * branches_per_soma * (k_exc + k_inh)
        axial_couplings = n_soma * branches_per_soma
        reactivation_parameters = n_soma * compartments_per_soma * 2
        readout_parameters = n_soma * output_dim + output_dim
        total = (
            branch_synapses
            + axial_couplings
            + reactivation_parameters
            + readout_parameters
        )

        assert n_soma == 9916
        assert branches_per_soma == 120
        assert total == 61_103_392
        assert total - 61_100_840 == 2552


def test_raw_imagenet_config_has_no_learned_image_backbone() -> None:
    config = load_config(str(BP_CONFIG_PATH))
    structured = config.model.core.connectivity.structured

    assert config.model.encoder.type == "identity"
    assert config.model.core.type == "tensor_map"
    assert config.model.vision_replacement.enabled is False
    assert config.data.processing.flatten is False
    assert config.data.processing.normalize is False
    assert structured.method == "spatial_morphology"
    assert structured.spatial["input_shape"] == [3, 224, 224]
    assert config.model.core.sparsity.type == "indexed"
    assert config.model.core.sparsity.indexed["index_dtype"] == "int32"
    assert config.model.core.sparsity.indexed["persistent_indices"] is False
    assert config.training.main.common.batch_size == 64
    assert config.training.main.common.param_groups.decoder_lr == 3e-5
    assert config.distributed.mode == "ddp"
    assert config.analysis.performance_analysis.final_splits == ["valid"]


def test_localca_config_changes_credit_rule_not_model_or_data() -> None:
    bp = load_config(str(BP_CONFIG_PATH))
    local = load_config(str(LOCALCA_CONFIG_PATH))

    assert local.model == bp.model
    assert local.data == bp.data
    assert local.training.main.strategy == "local_ca"
    assert local.training.main.common.batch_size == bp.training.main.common.batch_size
    assert local.training.main.optimizer == bp.training.main.optimizer
    assert local.training.main.common.param_groups.split_params is True
    assert local.training.main.common.param_groups.topk_lr == 5e-5
    assert local.training.main.common.param_groups.blocklinear_lr == 2e-5
    assert local.training.main.common.param_groups.reactivation_lr == 1e-5
    assert local.training.main.common.param_groups.decoder_lr == 3e-5
    assert local.training.main.learning_strategy_config.rule_variant == "5f"
    assert (
        local.training.main.learning_strategy_config.error_broadcast_mode == "per_soma"
    )
    assert local.training.main.learning_strategy_config.decoder_update_mode == "local"
    assert local.distributed.mode == "ddp"
