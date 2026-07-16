"""Tests for config-driven stacked soma-population accessibility."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.managers.analysis_manager import AnalysisManager
from dendritic_modeling.analysis.tools.representation_accessibility import (
    RepresentationAccessibilityAnalyzer,
    apply_balanced_countsketch,
    balanced_countsketch_projection,
    population_layer_inventory,
)
from dendritic_modeling.config import (
    AnalysisConfig,
    AnalysisRuntimeConfig,
    EvaluationRuntimeConfig,
    RepresentationAccessibilityAnalysisConfig,
    RepresentationAccessibilityAnalysisParams,
    RepresentationInformationConfig,
    RepresentationProbeConfig,
    RepresentationProjectionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks.architectures.classical import Identity
from dendritic_modeling.networks.architectures.classical.mlp import MLP
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _layer(name: str, *, readout: str) -> PopulationLayerConfig:
    return PopulationLayerConfig(
        name=name,
        recurrent=False,
        readout_population=readout,
        population_defaults={
            "ff_excitatory_synapses": 2,
            "ff_inhibitory_synapses": 0,
            "somatic_synapses": True,
            "reactivate": False,
            "dbl_init_method": "default",
            "weight_transform": "identity",
            "topk_type": "standard",
        },
        populations=[
            {
                "name": f"e{name[-1]}",
                "polarity": "excitatory",
                "n_neurons": 3,
                "branch_factors": [],
            },
            {
                "name": f"i{name[-1]}",
                "polarity": "inhibitory",
                "n_neurons": 2,
                "branch_factors": [],
            },
        ],
        connections=[
            {"source": "input", "target": f"e{name[-1]}"},
            {"source": "input", "target": f"i{name[-1]}"},
        ],
    )


def _model() -> Classifier:
    torch.manual_seed(7)
    core = PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=4,
            layers=[
                _layer("layer0", readout="e0"),
                _layer("layer1", readout="e1"),
            ],
        )
    )
    return Classifier(
        Identity(4),
        core,
        MLP(input_dim=3, hidden_dims=[], output_dim=2, activation="relu"),
        learned_output_scale=False,
    )


def _dataset(seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(48, 4, generator=generator)
    labels = (inputs[:, 0] > 0).long()
    return TensorDataset(inputs, labels)


def test_balanced_countsketch_is_seeded_balanced_and_label_independent() -> None:
    buckets, signs, metadata = balanced_countsketch_projection(11, 4, seed=23)
    repeated = balanced_countsketch_projection(11, 4, seed=23)
    assert np.array_equal(buckets, repeated[0])
    assert np.array_equal(signs, repeated[1])
    assert (
        max(metadata["bucket_occupancies"]) - min(metadata["bucket_occupancies"]) <= 1
    )
    values = np.arange(55, dtype=np.float32).reshape(5, 11)
    projected, projected_metadata = apply_balanced_countsketch(
        values,
        4,
        seed=23,
    )
    assert projected.shape == (5, 4)
    assert projected_metadata == metadata


def test_analyzer_captures_each_network_layer_and_population_group() -> None:
    model = _model()
    inventory = population_layer_inventory(model)
    assert [layer["network_layer_name"] for layer in inventory] == [
        "layer0",
        "layer1",
    ]
    params = RepresentationAccessibilityAnalysisParams(
        representations=["excitatory", "inhibitory", "joint"],
        projections=[RepresentationProjectionConfig(name="native", type="identity")],
        probes=[
            RepresentationProbeConfig(
                name="linear",
                type="ridge",
                n_label_shuffles=0,
            )
        ],
        information=RepresentationInformationConfig(enabled=False),
        seed=31,
    )
    result = RepresentationAccessibilityAnalyzer(params).analyze(
        model,
        valid_ds=_dataset(1),
        test_ds=_dataset(2),
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="materialize",
            batch_size=13,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert result["fit_split"] == "valid"
    assert result["evaluation_split"] == "test"
    assert result["test_used_for_probe_selection"] is False
    assert len(result["probe_records"]) == 6
    widths = {
        (record["network_layer_index"], record["representation"]): record[
            "native_feature_count"
        ]
        for record in result["probe_records"]
    }
    assert widths == {
        (0, "excitatory"): 3,
        (0, "inhibitory"): 2,
        (0, "joint"): 5,
        (1, "excitatory"): 3,
        (1, "inhibitory"): 2,
        (1, "joint"): 5,
    }
    assert not result["missing_representations"]


def test_params_normalize_nested_yaml_mappings() -> None:
    params = RepresentationAccessibilityAnalysisParams(
        projections=[
            {
                "name": "matched2",
                "type": "balanced_countsketch",
                "output_dim": 2,
                "draws": 3,
            }
        ],
        probes=[
            {
                "name": "nonlinear",
                "type": "mlp",
                "hidden_dims": [8],
                "max_projection_draws": 1,
                "n_label_shuffles": 1,
            }
        ],
        information={
            "enabled": True,
            "projection_names": ["matched2"],
            "n_label_shuffles": 1,
        },
    )
    assert isinstance(params.projections[0], RepresentationProjectionConfig)
    assert params.projections[0].output_dim == 2
    assert isinstance(params.probes[0], RepresentationProbeConfig)
    assert params.probes[0].hidden_dims == [8]
    assert params.probes[0].max_projection_draws == 1
    assert isinstance(params.information, RepresentationInformationConfig)


def test_probe_can_cap_projection_repeats() -> None:
    params = RepresentationAccessibilityAnalysisParams(
        representations=["joint"],
        projections=[
            RepresentationProjectionConfig(
                name="matched2",
                type="balanced_countsketch",
                output_dim=2,
                draws=3,
            )
        ],
        probes=[
            RepresentationProbeConfig(
                name="first_draw",
                type="ridge",
                max_projection_draws=1,
                n_label_shuffles=0,
            ),
            RepresentationProbeConfig(
                name="first_two_draws",
                type="ridge",
                max_projection_draws=2,
                n_label_shuffles=0,
            ),
        ],
        information=RepresentationInformationConfig(enabled=False),
    )
    result = RepresentationAccessibilityAnalyzer(params).analyze(
        _model(),
        valid_ds=_dataset(1),
        test_ds=_dataset(2),
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="materialize",
            batch_size=32,
            num_workers=0,
            pin_memory=False,
        ),
    )

    counts = Counter(record["probe_name"] for record in result["probe_records"])
    assert counts == {"first_draw": 2, "first_two_draws": 4}


def test_information_outputs_keep_joint_and_scalar_estimands_separate() -> None:
    params = RepresentationAccessibilityAnalysisParams(
        representations=["excitatory"],
        projections=[
            RepresentationProjectionConfig(
                name="matched2",
                type="balanced_countsketch",
                output_dim=2,
            )
        ],
        probes=[
            RepresentationProbeConfig(
                name="linear",
                type="ridge",
                n_label_shuffles=0,
            )
        ],
        information=RepresentationInformationConfig(
            enabled=True,
            compute_joint=True,
            compute_per_soma=True,
            projection_names=["matched2"],
            per_soma_representations=["excitatory"],
            n_neighbors=2,
            n_label_shuffles=1,
        ),
        seed=37,
    )
    result = RepresentationAccessibilityAnalyzer(params).analyze(
        _model(),
        valid_ds=_dataset(3),
        test_ds=_dataset(4),
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=24,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert len(result["joint_information_records"]) == 2
    assert len(result["per_soma_information_records"]) == 6
    assert len(result["per_soma_information_summary"]) == 2
    assert {
        record["information_scope"] for record in result["joint_information_records"]
    } == {"joint projected population"}
    assert {
        record["information_scope"] for record in result["per_soma_information_records"]
    } == {"single soma scalar"}
    assert all(
        "not joint population MI" in record["interpretation"]
        for record in result["per_soma_information_summary"]
    )


def test_analysis_manager_runs_registered_analyzer_and_routes_probe_seed(
    tmp_path,
) -> None:
    params = RepresentationAccessibilityAnalysisParams(
        representations=["excitatory"],
        projections=[RepresentationProjectionConfig()],
        probes=[
            RepresentationProbeConfig(
                name="linear",
                type="ridge",
                n_label_shuffles=0,
            )
        ],
        information=RepresentationInformationConfig(enabled=False),
        seed=1,
    )
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=24,
        num_workers=0,
        pin_memory=False,
    )
    config = AnalysisConfig(
        runtime=AnalysisRuntimeConfig(training=runtime, final=runtime),
        representation_accessibility_analysis=(
            RepresentationAccessibilityAnalysisConfig(
                enabled=True,
                training=False,
                params=params,
            )
        ),
    )
    manager = AnalysisManager(
        _model(),
        {"train": _dataset(5), "valid": _dataset(6), "test": _dataset(7)},
        config,
        save_root=str(tmp_path),
        device="cpu",
        evaluation_seed=41,
        probe_seed=43,
    )
    manager.run_analysis(filename="final", training=False)

    assert params.seed == 43
    output = tmp_path / "representation_accessibility" / "final.json"
    assert output.is_file()


def test_analysis_manager_rejects_training_time_representation_probes(
    tmp_path,
) -> None:
    config = AnalysisConfig(
        representation_accessibility_analysis=(
            RepresentationAccessibilityAnalysisConfig(
                enabled=True,
                training=True,
            )
        )
    )
    with pytest.raises(ValueError, match="final-only"):
        AnalysisManager(
            _model(),
            {"train": _dataset(8), "valid": _dataset(9), "test": _dataset(10)},
            config,
            save_root=str(tmp_path),
            device="cpu",
        )
