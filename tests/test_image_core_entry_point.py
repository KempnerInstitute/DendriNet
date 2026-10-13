"""Image-input cores build and train through the encoder-core-decoder entry point.

Spatial cores (``hierarchical_dendritic_conv``, ``sparse_active_matched_point_conv``
and their aliases) and ``dendritic_alexnet`` read images. The model builder
gives them the channel count of the configured image shape, not the flattened
input size, and flattens the feature maps of spatial cores for the decoder.
"""

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.networks.architectures.factory_spatial import (
    FEATURE_MAP_CORE_TYPES,
    IMAGE_INPUT_CORE_TYPES,
)
from dendritic_modeling.networks.architectures.vision import FlattenedSpatialCore
from dendritic_modeling.scripts.script_utils import setup_utils
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.training.train_experiments import _build_main_trainer

_CELL = {
    "architecture": {
        "excitatory_layer_sizes": [8, 4],
        "inhibitory_layer_sizes": [2, 1],
        "excitatory_branch_factors": [2],
        "inhibitory_branch_factors": [1],
    },
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [4, 2],
        "ei_synapses_per_branch_per_layer": [2, 1],
        "ie_synapses_per_branch_per_layer": [1, 1],
        "ii_synapses_per_branch_per_layer": [0, 0],
    },
    "spatial": {"kernel_size": 3, "stride": 2, "padding": 1},
}
_ALEXNET = {
    "architecture": {
        "excitatory_branch_factors": [2],
        "inhibitory_branch_factors": [1],
    },
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [4],
        "ei_synapses_per_branch_per_layer": [3],
        "ie_synapses_per_branch_per_layer": [1],
        "ii_synapses_per_branch_per_layer": [0],
    },
    "transfer": {"output_activation": "relu"},
    "vision": {
        "input_transform": "signed_split",
        "channels": [4, 4, 4, 4, 4],
        "kernels": [3, 3, 3, 3, 3],
        "strides": [1, 1, 1, 1, 1],
        "paddings": [1, 1, 1, 1, 1],
        "pool_after": [0, 1, 4],
        "pool_kernel": 2,
        "pool_stride": 2,
        "adaptive_pool_shape": [1, 1],
        "classifier_widths": [6, 6],
        "inhibitory_ratio": 0.25,
        "num_classes": 3,
    },
}


def _sections(core_type: str) -> dict:
    if core_type == "dendritic_alexnet":
        return deepcopy(_ALEXNET)
    sections = deepcopy(_CELL)
    if core_type not in {"hierarchical_dendritic_conv", "spatial_dendritic_conv"}:
        sections["sparsity"] = {"indexed": {"seed": 3}}
    return sections


def _image_shape(core_type: str) -> list[int]:
    # Patch-point controls match the fan-in of a reference cell, which needs
    # more patch inputs than three channels give.
    if core_type == "dendritic_alexnet":
        return [3, 16, 16]
    if core_type in {"hierarchical_dendritic_conv", "spatial_dendritic_conv"}:
        return [3, 8, 8]
    return [8, 6, 6]


def _experiment(core_type: str, shape: list[int], *, input_dim=None) -> dict:
    return {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "classification",
            "encoder": {
                "type": "identity",
                "params": {
                    "input_dim": input_dim or math.prod(shape),
                    "input_shape": shape,
                },
            },
            "core": {"type": core_type, **_sections(core_type)},
            "decoder": {
                "type": "MLP",
                "params": {"hidden_dims": [], "output_dim": 3},
            },
        },
        "training": {
            "main": {
                "strategy": "standard",
                "common": {
                    "epochs": 1,
                    "batch_size": 4,
                    "loss_function": "ce",
                    "early_stopping": False,
                    "load_best_state_dict": False,
                },
                "optimizer": {"name": "adam"},
            }
        },
    }


def _build(core_type: str, shape: list[int], **options):
    config = config_from_mapping(_experiment(core_type, shape, **options))
    torch.manual_seed(0)
    model, _encoder = initialize_model(config.model)
    return model


@pytest.mark.parametrize("core_type", sorted(IMAGE_INPUT_CORE_TYPES))
def test_image_cores_train_on_images(core_type):
    shape = _image_shape(core_type)
    model = _build(core_type, shape)
    images = torch.rand(2, *shape)

    logits = model(images)
    logits.square().sum().backward()

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    core = model.core_network
    if core_type in FEATURE_MAP_CORE_TYPES:
        assert isinstance(core, FlattenedSpatialCore)
        assert core.core.in_channels == shape[0]
        channels, height, width = core.output_shape
        assert core.output_dim == channels * height * width
        assert core.output_shape[1:] == (shape[1] // 2, shape[2] // 2)
    else:
        assert core.input_channels == shape[0]
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("core_type", sorted(IMAGE_INPUT_CORE_TYPES))
def test_image_cores_accept_the_channel_count_as_input_dim(core_type):
    # Configurations built without data may give input_dim as C.
    shape = _image_shape(core_type)
    from_data = _build(core_type, shape)
    from_channels = _build(core_type, shape, input_dim=shape[0])

    reference = from_data.state_dict()
    assert reference.keys() == from_channels.state_dict().keys()
    for key, value in from_channels.state_dict().items():
        assert torch.equal(value, reference[key]), key


@pytest.mark.parametrize("core_type", sorted(IMAGE_INPUT_CORE_TYPES))
def test_image_cores_reject_flattened_inputs(core_type):
    with pytest.raises(ValueError, match="flatten: false"):
        _build(core_type, [math.prod(_image_shape(core_type))])


def test_image_cores_reject_a_non_identity_encoder():
    data = _experiment("hierarchical_dendritic_conv", [3, 8, 8])
    data["model"]["encoder"] = {
        "type": "mlpautoencoder",
        "params": {
            "input_dim": 192,
            "input_shape": [3, 8, 8],
            "encoder_hidden_dims": [16],
            "decoder_hidden_dims": [16],
            "latent_dim": 8,
        },
    }
    with pytest.raises(ValueError, match="identity encoder"):
        initialize_model(config_from_mapping(data).model)


def test_alexnet_input_channels_must_match_the_images():
    data = _experiment("dendritic_alexnet", [1, 16, 16])
    data["model"]["core"]["vision"]["input_channels"] = 3
    with pytest.raises(ValueError, match="input_channels=3"):
        initialize_model(config_from_mapping(data).model)


def test_spatial_core_rejects_other_image_sizes():
    model = _build("hierarchical_dendritic_conv", [3, 8, 8])
    with pytest.raises(ValueError, match=r"built for images of shape \(3, 8, 8\)"):
        model(torch.rand(2, 3, 10, 10))


@pytest.mark.parametrize(
    "core_type", ["hierarchical_dendritic_conv", "sparse_active_matched_point_conv"]
)
def test_spatial_core_trains_through_setup_environment(
    core_type, monkeypatch, tmp_path
):
    shape = _image_shape(core_type)
    generator = torch.Generator().manual_seed(1)
    images = TensorDataset(
        torch.rand(12, *shape, generator=generator),
        torch.arange(12) % 3,
    )
    monkeypatch.setattr(
        setup_utils,
        "get_unified_datasets",
        lambda *, task_cfg: (images, images, images),
    )
    monkeypatch.setattr(
        setup_utils.logger_manager, "set_log_directory", lambda *_: None
    )
    data = _experiment(core_type, shape)
    # Training sets the input size from the data.
    data["model"]["encoder"]["params"] = {"input_dim": 1, "input_shape": [1]}
    config = config_from_mapping(data)
    run_path, train_ds, valid_ds, _test, model, _encoder = (
        setup_utils.setup_environment(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data,
            outputs_config=SimpleNamespace(results_dir=str(tmp_path), run_name="image"),
            experiment_config=config.experiment,
            is_main=True,
        )
    )
    assert config.model.encoder.params.input_shape == shape
    before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    setup = _build_main_trainer(
        unwrapped_model=model,
        training_config=config.training,
        model_config=config.model,
        experiment_config=config.experiment,
        run_save_path=run_path,
        encoded_train_ds=train_ds,
        analysis_manager=None,
    )

    result = setup.trainer.train(model, train_ds, valid_ds)

    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["train_losses"])
    changed = [
        name
        for name, value in model.state_dict().items()
        if value.is_floating_point() and not torch.equal(value.cpu(), before[name])
    ]
    assert any(name.startswith("core_network.core.") for name in changed)
