"""Standard-pipeline support for the context-cued EMNIST task (presentation_stream)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets.contextual_remapping import (
    ContextualRemappingConfig,
    ContextualRemappingDataConfig,
    ContextualRemappingStream,
    contextual_remapping_batch,
    contextual_remapping_data_config,
    fixed_evaluation_set,
    partition_emnist_balanced,
)
from dendritic_modeling.datasets.registry import get_dataset_spec
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    build_population_impact_rewiring,
    population_rewiring_groups,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.presentation_stream import (
    PresentationStreamTrainer,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "examples"
TASK = ContextualRemappingConfig(num_classes=5, num_contexts=3)


def _labels(per_class=7, classes=5):
    return torch.arange(classes).repeat_interleave(per_class)[
        torch.randperm(per_class * classes, generator=torch.Generator().manual_seed(1))
    ]


def test_partition_reproduces_per_class_permutation_protocol():
    labels = _labels()
    train, validation, pool = partition_emnist_balanced(
        labels, pool_per_class=3, validation_per_class=2, seed=31003
    )
    rng = torch.Generator().manual_seed(31003)
    expected = {"train": [], "validation": [], "pool": []}
    for label in range(5):
        ids = labels.eq(label).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=rng)]
        expected["pool"].append(ids[:3])
        expected["validation"].append(ids[:2])
        expected["train"].append(ids[3:])
    assert torch.equal(train, torch.cat(expected["train"]))
    assert torch.equal(validation, torch.cat(expected["validation"]))
    assert torch.equal(pool, torch.cat(expected["pool"]))
    assert not set(train.tolist()) & set(pool.tolist())


def test_stream_sampling_uses_index_then_corruption_generators():
    images, labels = torch.rand(20, 6), torch.randint(5, (20,))
    stream = ContextualRemappingStream(images, labels, TASK)
    gi, gn = torch.Generator().manual_seed(7), torch.Generator().manual_seed(8)
    x, y = stream.sample(4, index_generator=gi, noise_generator=gn)
    ri, rn = torch.Generator().manual_seed(7), torch.Generator().manual_seed(8)
    ids = torch.randint(20, (4,), generator=ri)
    ex, ey, _ = contextual_remapping_batch(images[ids], labels[ids], TASK, generator=rn)
    assert torch.equal(x, ex) and torch.equal(y, ey)
    assert x.shape == (4, 6 + TASK.num_contexts)


def test_fixed_evaluation_set_pairs_noisy_and_clean_inputs():
    images, labels = torch.rand(10, 6), torch.randint(5, (10,))
    evaluation = fixed_evaluation_set(images, labels, TASK, seed=78019)
    noisy, targets, contexts = contextual_remapping_batch(
        images, labels, TASK, generator=torch.Generator().manual_seed(78019)
    )
    clean_inputs, clean_targets = evaluation.variants["clean"].tensors
    assert torch.equal(evaluation.variants["noisy"].tensors[0], noisy)
    assert torch.equal(clean_targets, targets) and torch.equal(
        evaluation.contexts, contexts
    )
    assert torch.equal(clean_inputs[:, :6], images)
    assert evaluation[3][0].shape == (6 + TASK.num_contexts,)


def test_data_config_rejects_unknown_keys_and_uses_base_dir():
    task_cfg = SimpleNamespace(
        data_path="/data", parameters={"seed": 1, "split_seed": 2, "flatten": True}
    )
    assert contextual_remapping_data_config(task_cfg).root == "/data"
    task_cfg.parameters["partition_sed"] = 3
    with pytest.raises(ValueError, match="partition_sed"):
        contextual_remapping_data_config(task_cfg)
    with pytest.raises(ValueError):
        ContextualRemappingDataConfig(validation_per_class=300)
    assert get_dataset_spec("contextual_remapping") is not None


class _Stream:
    def __init__(self, n=64, dim=4, classes=3):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, dim, generator=generator)
        self.y = (self.x[:, 0] > 0).long() + (self.x[:, 1] > 0).long()

    def to(self, device):
        return self

    def sample(self, count, *, index_generator, noise_generator):
        ids = torch.randint(len(self.y), (count,), generator=index_generator)
        return self.x[ids], self.y[ids]


class _Counter:
    def __init__(self):
        self.calls = 0

    def step(self, optimizer):
        assert isinstance(optimizer, torch.optim.Adam)
        self.calls += 1
        return 1


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

    def decay_weights(self, weight_decay, weight_boosting=False):
        pass

    def apply_rewiring(self):
        pass


def _trainer(tmp_path, **stream):
    torch.manual_seed(0)
    model = _Model()
    counter = _Counter()
    optimizer = CustomWeightDecayOptimizer(
        model,
        torch.optim.Adam(model.parameters(), lr=0.05),
        0.0,
        False,
        topology_controllers=[counter],
    )
    trainer = PresentationStreamTrainer(
        optimizer,
        loss_function="ce",
        epochs=4,
        batch_size=16,
        grad_clip_value=0,
        save_path=str(tmp_path),
        lr_schedule="cosine",
        presentation_stream_config=dict(
            presentations_per_epoch=40, lr_terminal_multiplier=0.1, **stream
        ),
    )
    trainer.device = "cpu"
    return trainer, model, counter


def test_presentation_stream_trains_schedules_and_selects(tmp_path):
    trainer, model, counter = _trainer(tmp_path)
    stream = _Stream()
    valid = torch.utils.data.TensorDataset(stream.x[:20], stream.y[:20])
    result = trainer.train(model, SimpleNamespace(stream=stream), valid)
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and result["steps_per_epoch"] == 3
    assert counter.calls == 12 and all(row["replaced"] == 3 for row in rows)
    scores = [
        (
            r["validation"]["validation"]["accuracy"],
            -r["validation"]["validation"]["loss"],
        )
        for r in rows
    ]
    assert result["best_epoch"] == 1 + scores.index(max(scores))
    total = 12
    assert trainer._lr_factor(0, total, 3) == pytest.approx(1.0)
    assert trainer._lr_factor(total - 1, total, 3) == pytest.approx(0.1)
    trainer.lr_warmup_epochs = 1
    assert trainer._lr_factor(0, total, 3) == pytest.approx(1 / 3)


def test_presentation_stream_stops_early_and_rejects_bad_options(tmp_path):
    trainer, model, _ = _trainer(tmp_path, stop_after_epochs=2)
    stream = _Stream()
    trainer.train(
        model,
        SimpleNamespace(stream=stream),
        torch.utils.data.TensorDataset(stream.x, stream.y),
    )
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    model = _Model()
    optimizer = CustomWeightDecayOptimizer(
        model, torch.optim.Adam(model.parameters()), 0.0
    )
    with pytest.raises(ValueError, match="unknown presentation_stream"):
        PresentationStreamTrainer(
            optimizer,
            loss_function="ce",
            presentation_stream_config={"presentations": 1},
        )
    with pytest.raises(ValueError, match="loss_function"):
        PresentationStreamTrainer(optimizer, loss_function="mse")
    PresentationStreamTrainer(optimizer, loss_function="cross_entropy")  # an alias
    with pytest.raises(TypeError, match="sampling stream"):
        PresentationStreamTrainer(optimizer, loss_function="ce", epochs=1).train(
            model, [], []
        )


@pytest.mark.parametrize(
    "options, stream, message",
    [
        ({"lr_warmup_epochs": 1}, {}, "part of the cosine schedule"),
        (
            {"lr_schedule": "cosine", "lr_warmup_epochs": -1},
            {"lr_terminal_multiplier": 0.1},
            "must be >= 0",
        ),
        ({"lr_schedule": "cosine"}, {}, "rate never changes"),
    ],
)
def test_presentation_stream_schedule_follows_the_common_rules(
    options, stream, message
):
    from dendritic_modeling.training.factory import get_trainer

    model = _Model()
    optimizer = CustomWeightDecayOptimizer(
        model, torch.optim.Adam(model.parameters()), 0.0
    )
    with pytest.raises(ValueError, match=message):
        get_trainer(
            "presentation_stream",
            optimizer,
            {
                "loss_function": "ce",
                "epochs": 2,
                "presentation_stream_config": stream,
                **options,
            },
        )


def _yaml_model(name):
    config = load_config(str(CONFIGS / f"{name}.yaml"))
    config.model.encoder.params.input_dim = 816
    config.model.encoder.params.input_shape = [816]
    torch.manual_seed(0)
    model, _ = initialize_model(config.model)
    return config, model


@pytest.mark.parametrize(
    "name,stored",
    [
        ("population_network_contextual_remapping", 238367),
        ("contextual_remapping_affine_mlp", 238367),
    ],
)
def test_example_configs_build_their_models(name, stored):
    config, model = _yaml_model(name)
    assert sum(p.numel() for p in model.parameters()) == stored
    assert config.training.main.strategy == "presentation_stream"
    rounding = [
        m._signal_round_dtype
        for m in model.modules()
        if hasattr(m, "_signal_round_dtype")
    ]
    if name.startswith("population_network"):
        assert rounding and all(dtype is None for dtype in rounding)


def test_impact_rewiring_builder_groups_banks_and_sets_freeze():
    _, model = _yaml_model("population_network_contextual_remapping")
    groups = population_rewiring_groups(model.core_network)
    assert [n for n, _ in groups] == [72, 192, 192, 72, 72, 192, 192]
    assert all(
        isinstance(bank, IndexedReleaseLinear) for _, banks in groups for bank in banks
    )
    controller = build_population_impact_rewiring(
        model.core_network,
        {"frequency": 100, "freeze_last_epochs": 20},
        steps_per_epoch=128,
        epochs=400,
        seed=2,
    )
    assert controller.config.until_step == 380 * 128 and controller.config.seed == 2
    with pytest.raises(ValueError, match="derived"):
        build_population_impact_rewiring(
            model.core_network, {"until_step": 5}, steps_per_epoch=1, epochs=1, seed=0
        )
    _, topk = _yaml_model("population_network_target_levels")
    with pytest.raises(ValueError, match="indexed_release"):
        build_population_impact_rewiring(
            topk.core_network, {}, steps_per_epoch=1, epochs=1, seed=0
        )


@pytest.mark.parametrize("fault", ["loss", "gradient"])
def test_presentation_stream_stops_on_non_finite_values_before_the_step(
    tmp_path, fault
):
    trainer, model, _ = _trainer(tmp_path)
    stream = _Stream()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if fault == "loss":
        forward = model.forward
        model.forward = lambda x: forward(x) * float("nan")
    else:
        next(model.parameters()).register_hook(lambda grad: grad * float("nan"))

    with pytest.raises(RuntimeError, match=f"non-finite {fault} at epoch 1, step 0"):
        trainer.train(
            model,
            SimpleNamespace(stream=stream),
            torch.utils.data.TensorDataset(stream.x[:20], stream.y[:20]),
        )
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )
