"""Contract tests for the raw ImageNet spatial DendriNet experiment."""

import math
from pathlib import Path

from dendritic_modeling.config import load_config

CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "vision"
    / "imagenet_raw_spatial_dendrinet_alexnet_matched_bp.yaml"
)


def test_raw_imagenet_config_is_alexnet_parameter_matched() -> None:
    config = load_config(str(CONFIG_PATH))
    core = config.model.core

    n_soma = core.architecture.excitatory_layer_sizes[0]
    branch_factors = core.architecture.excitatory_branch_factors
    branches_per_soma = sum(
        math.prod(branch_factors[:depth]) for depth in range(1, len(branch_factors) + 1)
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
        branch_synapses + axial_couplings + reactivation_parameters + readout_parameters
    )

    assert n_soma == 9916
    assert branches_per_soma == 120
    assert total == 61_103_392
    assert total - 61_100_840 == 2552


def test_raw_imagenet_config_has_no_learned_image_backbone() -> None:
    config = load_config(str(CONFIG_PATH))
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
