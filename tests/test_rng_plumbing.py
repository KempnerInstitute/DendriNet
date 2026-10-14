"""Regression tests for independent experiment random streams."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.managers.analysis_manager import AnalysisManager
from dendritic_modeling.config import (
    AnalysisConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
    as_population_definition_config,
    population_config_from_definition,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.scripts.script_utils import setup_utils
from dendritic_modeling.training.dataloader_utils import seed_dataloader_worker
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils import resolve_experiment_seeds, set_seed


def _loader_order_after_architecture(
    architecture_factory,
    *,
    loader_seed: int,
) -> tuple[torch.Tensor, object]:
    torch.manual_seed(71)
    architecture = architecture_factory()
    optimizer = torch.optim.SGD(architecture.parameters(), lr=0.01)
    # Consume a deliberately architecture-dependent amount of global RNG after
    # model construction. The loader's private generator must isolate ordering.
    torch.rand(sum(parameter.numel() for parameter in architecture.parameters()))

    trainer = Trainer(
        optimizer=optimizer,
        epochs=1,
        batch_size=4,
        shuffle=True,
        suppress_prints=True,
        num_workers=0,
        seed=7,
        loader_seed=loader_seed,
    )
    indices = torch.arange(24)
    dataset = TensorDataset(indices, indices)
    train_loader, _ = trainer._dataloaders(dataset, dataset)
    order = torch.cat([batch[0] for batch in train_loader])
    return order, train_loader


def test_loader_seed_is_independent_of_architecture_rng_consumption():
    shallow_order, shallow_loader = _loader_order_after_architecture(
        lambda: torch.nn.Linear(3, 2),
        loader_seed=404,
    )
    deep_order, deep_loader = _loader_order_after_architecture(
        lambda: torch.nn.Sequential(
            torch.nn.Linear(3, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, 2),
        ),
        loader_seed=404,
    )

    torch.testing.assert_close(shallow_order, deep_order)
    assert shallow_loader.generator is not None
    assert deep_loader.generator is not None
    assert shallow_loader.worker_init_fn is seed_dataloader_worker
    assert deep_loader.worker_init_fn is seed_dataloader_worker


def test_setup_environment_separates_dataset_and_model_rng(monkeypatch, tmp_path):
    dataset_draws = []
    model_draws = []
    task_parameters = []
    propagated_initialization_seeds = []

    dataset = TensorDataset(torch.zeros(4, 3), torch.zeros(4, dtype=torch.long))

    def fake_get_unified_datasets(*, task_cfg):
        dataset_draws.append(torch.rand(5))
        task_parameters.append(dict(task_cfg.parameters))
        return dataset, dataset, dataset

    def fake_initialize_model(_model_config):
        model_draws.append(torch.rand(5))
        propagated_initialization_seeds.append(_model_config.core.initialization_seed)
        return torch.nn.Identity(), torch.nn.Identity()

    monkeypatch.setattr(
        setup_utils,
        "get_unified_datasets",
        fake_get_unified_datasets,
    )
    monkeypatch.setattr(setup_utils, "initialize_model", fake_initialize_model)

    for model_seed in (101, 202):
        # Training fills core.initialization_seed for cores that consume it.
        # Feedforward EINet keeps its historical global stream instead (see
        # test_configuration_consumption), so use a unified E/I core here.
        model_config = ModelConfig()
        model_config.core.type = "unified_ei"
        setup_utils.setup_environment(
            model_config=model_config,
            training_config=TrainingConfig(),
            data_config=DataConfig(dataset_name="dummy", base_dir=""),
            outputs_config=SimpleNamespace(
                results_dir=str(tmp_path),
                run_name="rng-test",
            ),
            experiment_config=ExperimentConfig(
                seed=5,
                dataset_seed=33,
                split_seed=34,
                model_seed=model_seed,
                topology_seed=44,
                loader_seed=55,
            ),
            is_main=False,
        )

    expected_dataset = torch.rand(5, generator=torch.Generator().manual_seed(33))
    expected_model_101 = torch.rand(5, generator=torch.Generator().manual_seed(101))
    expected_model_202 = torch.rand(5, generator=torch.Generator().manual_seed(202))
    torch.testing.assert_close(dataset_draws[0], expected_dataset)
    torch.testing.assert_close(dataset_draws[1], expected_dataset)
    torch.testing.assert_close(model_draws[0], expected_model_101)
    torch.testing.assert_close(model_draws[1], expected_model_202)
    assert task_parameters[0]["seed"] == 33
    assert task_parameters[0]["split_seed"] == 34
    assert task_parameters[0]["label_noise_seed"] == 33
    assert propagated_initialization_seeds == [101, 202]


def test_standard_setup_propagates_flat_recurrent_core_seed_defaults():
    for core_type, parameter_name in (
        ("heterogeneous_leak_ctrnn", "heterogeneous_leak_ctrnn"),
        ("legendre_memory", "legendre_memory"),
    ):
        model_config = ModelConfig()
        model_config.core.type = core_type
        setattr(model_config.core, parameter_name, {})

        setup_utils._apply_initialization_seed_defaults(model_config, model_seed=101)
        setup_utils._apply_topology_seed_defaults(model_config, topology_seed=202)

        parameters = getattr(model_config.core, parameter_name)
        assert parameters["initialization_seed"] == 101
        assert parameters["topology_seed"] == 202


def test_standard_setup_preserves_explicit_flat_recurrent_core_seeds():
    for core_type, parameter_name in (
        ("heterogeneous_ctrnn", "heterogeneous_leak_ctrnn"),
        ("legendre_memory", "legendre_memory"),
    ):
        model_config = ModelConfig()
        model_config.core.type = core_type
        setattr(
            model_config.core,
            parameter_name,
            {"initialization_seed": 17, "topology_seed": 19},
        )

        setup_utils._apply_initialization_seed_defaults(model_config, model_seed=101)
        setup_utils._apply_topology_seed_defaults(model_config, topology_seed=202)

        parameters = getattr(model_config.core, parameter_name)
        assert parameters["initialization_seed"] == 17
        assert parameters["topology_seed"] == 19


def test_legacy_seed_config_resolves_all_streams_compatibly(tmp_path):
    config_path = tmp_path / "legacy_seed.yaml"
    config_path.write_text("experiment:\n  seed: 19\n", encoding="utf-8")
    config = load_config(str(config_path))

    assert config.experiment.dataset_seed is None
    resolved = resolve_experiment_seeds(config.experiment, write_back=True)

    assert resolved.seed == 19
    assert resolved.strict_deterministic is False
    for field_name in (
        "dataset_seed",
        "split_seed",
        "model_seed",
        "topology_seed",
        "loader_seed",
        "evaluation_seed",
        "probe_seed",
    ):
        assert getattr(resolved, field_name) == 19
        assert getattr(config.experiment, field_name) == 19


def test_explicit_seed_fields_load_from_yaml(tmp_path):
    config_path = tmp_path / "independent_seeds.yaml"
    config_path.write_text(
        """
experiment:
  seed: 1
  dataset_seed: 2
  split_seed: 3
  model_seed: 4
  topology_seed: 5
  loader_seed: 6
  evaluation_seed: 7
  probe_seed: 8
  strict_deterministic: true
""",
        encoding="utf-8",
    )
    config = load_config(str(config_path))

    assert resolve_experiment_seeds(config.experiment).asdict() == {
        "seed": 1,
        "dataset_seed": 2,
        "split_seed": 3,
        "model_seed": 4,
        "topology_seed": 5,
        "loader_seed": 6,
        "evaluation_seed": 7,
        "probe_seed": 8,
        "strict_deterministic": True,
    }


def test_strict_deterministic_enables_algorithms_and_cublas(monkeypatch):
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        set_seed(
            13,
            deterministic=False,
            cudnn_benchmark=True,
            allow_tf32=True,
            strict_deterministic=True,
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


@pytest.mark.requires_git_checkout
def test_setup_environment_records_resolved_seeds(monkeypatch, tmp_path):
    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(
        setup_utils,
        "get_unified_datasets",
        lambda *, task_cfg: (dataset, dataset, dataset),
    )
    monkeypatch.setattr(
        setup_utils,
        "initialize_model",
        lambda _model_config: (torch.nn.Identity(), torch.nn.Identity()),
    )

    class DummyHandler:
        def setFormatter(self, _formatter):  # noqa: N802 - logging API name
            return None

        def set_name(self, _name):
            return None

    # Avoid depending on logger handler internals while exercising the actual
    # run-directory and seed-manifest write path.
    monkeypatch.setattr(
        setup_utils.logger_manager, "set_log_directory", lambda *_: None
    )
    monkeypatch.setattr(setup_utils.logging, "FileHandler", lambda *_: DummyHandler())
    monkeypatch.setattr(setup_utils.logger, "addHandler", lambda *_: None)
    experiment = ExperimentConfig(seed=29)
    run_path, *_ = setup_utils.setup_environment(
        model_config=ModelConfig(),
        training_config=TrainingConfig(),
        data_config=DataConfig(dataset_name="dummy", base_dir=""),
        outputs_config=SimpleNamespace(
            results_dir=str(tmp_path),
            run_name="record-seeds",
        ),
        experiment_config=experiment,
        is_main=True,
    )

    with open(f"{run_path}/resolved_seeds.json", encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert recorded["dataset_seed"] == 29
    assert recorded["split_seed"] == 29
    assert recorded["model_seed"] == 29
    assert recorded["loader_seed"] == 29
    with open(f"{run_path}/runtime_environment.json", encoding="utf-8") as handle:
        environment = json.load(handle)
    package = Path(setup_utils.__file__).resolve().parents[2]
    assert environment["library"]["package_path"] == str(package)
    assert environment["library"]["git"]["available"] is True
    assert environment["library"]["checkout"] == str(package.parents[1])
    assert environment["torch"]["version"] == torch.__version__
    assert environment["dependencies"]["numpy"] == np.__version__


def test_setup_environment_uses_exact_preclaimed_run_directory(monkeypatch, tmp_path):
    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(
        setup_utils,
        "get_unified_datasets",
        lambda *, task_cfg: (dataset, dataset, dataset),
    )
    monkeypatch.setattr(
        setup_utils,
        "initialize_model",
        lambda _model_config: (torch.nn.Identity(), torch.nn.Identity()),
    )

    class DummyHandler:
        def setFormatter(self, _formatter):  # noqa: N802 - logging API name
            return None

        def set_name(self, _name):
            return None

    exact_path = tmp_path / "manifest-run"
    exact_path.mkdir()
    monkeypatch.setattr(
        setup_utils.logger_manager, "set_log_directory", lambda *_: None
    )
    monkeypatch.setattr(setup_utils.logging, "FileHandler", lambda *_: DummyHandler())
    monkeypatch.setattr(setup_utils.logger, "addHandler", lambda *_: None)

    run_path, *_ = setup_utils.setup_environment(
        model_config=ModelConfig(),
        training_config=TrainingConfig(),
        data_config=DataConfig(dataset_name="dummy", base_dir=""),
        outputs_config=SimpleNamespace(
            results_dir=str(exact_path),
            run_name="exact",
            exact_run_dir=True,
        ),
        experiment_config=ExperimentConfig(
            seed=31,
            record_dataset_fingerprints=True,
        ),
        is_main=True,
    )

    assert run_path == str(exact_path)
    assert (exact_path / "resolved_seeds.json").is_file()
    assert (exact_path / "dataset_fingerprints.json").is_file()


def test_named_population_topology_seed_stays_in_nested_overrides():
    definition = {
        "name": "pyr",
        "polarity": "excitatory",
        "n_neurons": 8,
        "branch_factors": [2],
    }

    setup_utils._seed_named_population_topology(definition, topology_seed=41)

    assert "indexed_seed" not in definition
    assert definition["population"]["indexed_seed"] == 41
    typed = as_population_definition_config(definition)
    config = population_config_from_definition(typed, {}, layer_idx=0)
    assert config.indexed_seed == 41


def test_named_population_topology_seed_preserves_explicit_nested_seed():
    definition = PopulationDefinitionConfig(
        name="pyr",
        polarity="excitatory",
        n_neurons=8,
        branch_factors=[2],
        population={"indexed_seed": 17},
    )

    setup_utils._seed_named_population_topology(definition, topology_seed=41)

    assert definition.population["indexed_seed"] == 17


@pytest.mark.parametrize("layer_seed, expected", [(17, 17), (None, 41)])
def test_layer_population_defaults_seed_reaches_every_population(layer_seed, expected):
    from types import SimpleNamespace

    defaults = {} if layer_seed is None else {"indexed_seed": layer_seed}
    layer = {
        "name": "layer0",
        "population_defaults": defaults,
        "populations": [
            {"name": "pyr", "polarity": "excitatory", "n_neurons": 8},
            {"name": "pv", "polarity": "inhibitory", "n_neurons": 2},
        ],
    }
    model = SimpleNamespace(
        core={"type": "population_network", "population_network": {"layers": [layer]}}
    )

    setup_utils._apply_topology_seed_defaults(model, topology_seed=41)

    assert defaults["indexed_seed"] == expected
    for population in layer["populations"]:
        typed = as_population_definition_config(population)
        config = population_config_from_definition(typed, defaults, layer_idx=0)
        assert config.indexed_seed == expected


def _build_keyed_placement_population(
    ff_excitatory_by_level: list[int],
) -> StatefulDendriNet:
    pop_config = PopulationConfig(
        n_neurons=3,
        branch_factors=[2, 2],
        ff_excitatory_synapses=0,
        ff_excitatory_synapses_by_level=ff_excitatory_by_level,
        ff_inhibitory_synapses=0,
        rec_excitatory_synapses=0,
        rec_inhibitory_synapses=2,
        rec_inhibitory_synapses_by_level=[2, 2, 2],
        reactivate=False,
        adaptive_initialization=False,
        initialization_seed=808,
        initialization_namespace="matched.layer0.excitatory",
    )
    return StatefulDendriNet(
        pop_config=pop_config,
        excitatory_input_dim=7,
        inhibitory_input_dim=None,
        recurrent_excitatory_input_dim=None,
        recurrent_inhibitory_input_dim=5,
    )


@pytest.mark.requires_gpu
def test_keyed_initialization_leaves_the_cuda_generator_untouched():
    torch.manual_seed(5)
    before = torch.cuda.get_rng_state()
    _build_keyed_placement_population([1, 1, 1])
    assert torch.equal(torch.cuda.get_rng_state(), before)


def test_component_keyed_initialization_holds_unchanged_rec_i_weights_fixed():
    """Moving FF K across depths must not perturb the matched REC-I path."""
    torch.manual_seed(1)
    ff_proximal = _build_keyed_placement_population([0, 0, 3])
    torch.manual_seed(999)
    ff_distal = _build_keyed_placement_population([3, 0, 0])

    proximal_rec_i = [
        layer.branch_rec_inhibition.pre_w.detach()
        for layer in ff_proximal.branch_layers
    ]
    distal_rec_i = [
        layer.branch_rec_inhibition.pre_w.detach() for layer in ff_distal.branch_layers
    ]
    assert len(proximal_rec_i) == len(distal_rec_i) == 3
    for left, right in zip(proximal_rec_i, distal_rec_i, strict=True):
        assert torch.equal(left, right)


def test_named_population_initialization_seed_uses_nested_overrides():
    definition = {
        "name": "pyr",
        "polarity": "excitatory",
        "n_neurons": 8,
        "branch_factors": [2],
    }

    setup_utils._seed_named_population_initialization(
        definition,
        model_seed=53,
        namespace="population_network.layer0.pyr",
    )

    assert "initialization_seed" not in definition
    assert definition["population"]["initialization_seed"] == 53
    assert (
        definition["population"]["initialization_namespace"]
        == "population_network.layer0.pyr"
    )


def test_evaluation_and_probe_seeds_reach_runtime_consumers(tmp_path):
    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))
    config = AnalysisConfig()
    manager = AnalysisManager(
        model=torch.nn.Identity(),
        data={"train": dataset, "valid": dataset, "test": dataset},
        analysis_config=config,
        save_root=str(tmp_path),
        evaluation_seed=73,
        probe_seed=91,
    )

    from dendritic_modeling.config.seeds import derive_seed

    assert config.runtime.training.seed == 73
    assert config.runtime.final.seed == 73
    # Probe-stream consumers get seeds derived from the probe seed and their
    # config path; the flat decoder_seed mirror stays unset.
    assert config.information.params.decoder_seed is None
    assert config.information.params.estimator.decoder.seed == derive_seed(
        91, "analysis.information.params.estimator.decoder.seed"
    )
    assert config.ablation.params.methods.shuffle_seed == derive_seed(
        91, "analysis.ablation.params.methods.shuffle_seed"
    )

    evaluation_draws = []
    manager._run_ordered_analysis_sequence = lambda **_kwargs: evaluation_draws.append(
        (torch.rand(1).item(), np.random.random(), random.random())
    )
    torch_state = torch.random.get_rng_state().clone()
    manager.run_analysis()
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    manager.run_analysis()
    assert evaluation_draws[0] == evaluation_draws[1]

    probe_draws = []
    # This test isolates RNG plumbing, not analyzer construction. Information
    # analyses now correctly require both an enabled config and an initialized
    # analyzer, so provide the smallest sentinel satisfying that runtime gate.
    config.information.enabled = True
    manager.information_analyzer = SimpleNamespace(analysis_split="test")
    manager._run_logged_test_dataset_analysis = lambda **_kwargs: probe_draws.append(
        (torch.rand(1).item(), np.random.random(), random.random())
    )
    manager._run_information_analysis(filename="probe", training=False, runtime=None)
    manager._run_information_analysis(filename="probe", training=False, runtime=None)
    assert probe_draws[0] == probe_draws[1]
    assert probe_draws[0] != evaluation_draws[0]


def test_written_analysis_seeds_are_kept_and_unset_ones_filled():
    from dendritic_modeling.config.analysis import fill_analysis_seeds

    config = AnalysisConfig()
    config.runtime.final.seed = 5
    config.ablation.params.methods.shuffle_seed = 7
    filled = fill_analysis_seeds(config, evaluation_seed=73, probe_seed=91)

    assert config.runtime.final.seed == 5
    assert config.ablation.params.methods.shuffle_seed == 7
    assert config.runtime.training.seed == 73
    assert "runtime.final.seed" not in filled
    assert len(set(filled.values()) - {73}) == len(filled) - 1  # distinct seeds

    unfilled = AnalysisConfig()
    assert fill_analysis_seeds(unfilled, evaluation_seed=None, probe_seed=None) == {}
    assert unfilled.runtime.training.seed is None


def test_sealed_test_requires_disabled_access_and_bounded_dataset():
    dataset = TensorDataset(torch.zeros(8, 2), torch.zeros(8, dtype=torch.long))
    sealed = ExperimentConfig(
        sealed_test=True,
        allow_test_data=False,
        max_test_samples=8,
    )
    setup_utils._validate_sealed_test_dataset(sealed, dataset)

    with np.testing.assert_raises_regex(ValueError, "allow_test_data=false"):
        setup_utils._validate_sealed_test_dataset(
            ExperimentConfig(sealed_test=True, max_test_samples=8),
            dataset,
        )
    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        setup_utils._validate_sealed_test_dataset(
            ExperimentConfig(
                sealed_test=True,
                allow_test_data=False,
                max_test_samples=7,
            ),
            dataset,
        )


def test_sealed_test_rejects_enabled_analysis():
    from dendritic_modeling.scripts.training.train_experiments import (
        _validate_sealed_test_analysis,
    )

    analysis = AnalysisConfig()
    _validate_sealed_test_analysis(
        ExperimentConfig(
            sealed_test=True,
            allow_test_data=False,
            max_test_samples=8,
        ),
        analysis,
    )
    analysis.performance.enabled = True
    with np.testing.assert_raises_regex(ValueError, "performance"):
        _validate_sealed_test_analysis(
            ExperimentConfig(
                sealed_test=True,
                allow_test_data=False,
                max_test_samples=8,
            ),
            analysis,
        )
