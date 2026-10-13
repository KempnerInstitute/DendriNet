"""Context-cued modular labels and brightness-dependent image corruption.

Inputs are nonnegative, flattened images; context cues are appended after
corruption and never corrupted themselves. Besides the batch operations, the
module provides the registered ``contextual_remapping`` dataset: the EMNIST
Balanced partition, fixed corrupted evaluation sets and the presentation stream
used by the ``presentation_stream`` trainer.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset

from dendritic_modeling.config.base import BaseConfig


@dataclass
class ContextualRemappingConfig(BaseConfig):
    num_classes: int = 47
    num_contexts: int = 32
    context_scale: float = 1.0
    gain_min: float = 0.05
    gain_max: float = 20.0
    noise_divisor_min: float = 1.0
    noise_divisor_max: float = 4.0

    def __post_init__(self):
        if not 1 <= self.num_contexts <= self.num_classes:
            raise ValueError("require 1 <= num_contexts <= num_classes")
        for lo, hi in [
            (self.gain_min, self.gain_max),
            (self.noise_divisor_min, self.noise_divisor_max),
        ]:
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(
                    "gain and noise-divisor bounds must be finite, positive and ordered"
                )
        if not math.isfinite(self.context_scale) or self.context_scale <= 0:
            raise ValueError("context_scale must be finite and positive")


def brightness_dependent_noise(images, gain, noise_divisor, *, generator=None):
    """Return max(g*x + sqrt(mean(g*x))/X * z, 0), with no upper clip.

    gain and noise_divisor contain one value per image (or broadcast scalars).
    Randomness uses only the caller's generator when one is provided.
    """
    if images.ndim != 2:
        raise ValueError("images must have shape [batch, pixels]")
    gain = torch.as_tensor(gain, device=images.device, dtype=images.dtype).reshape(
        -1, 1
    )
    divisor = torch.as_tensor(
        noise_divisor, device=images.device, dtype=images.dtype
    ).reshape(-1, 1)
    scaled = images * gain
    sigma = scaled.mean(-1, keepdim=True).sqrt() / divisor
    noise = torch.randn(
        images.shape, device=images.device, dtype=images.dtype, generator=generator
    )
    return (scaled + sigma * noise).clamp_min(0)


def contextual_remapping_batch(
    images, labels, config, *, generator=None, corrupt=True, contexts=None
):
    """Sample log-uniform gain/noise and append an uncorrupted one-hot cue.

    Returns inputs, shifted labels and the sampled context indices. Supplying
    contexts permits paired clean/noisy evaluations with identical labels.
    """
    if images.ndim != 2 or labels.shape != (len(images),):
        raise ValueError("expected flattened images and one label per image")
    if contexts is None:
        contexts = torch.randint(
            config.num_contexts, labels.shape, device=labels.device, generator=generator
        )
    if contexts.shape != labels.shape:
        raise ValueError("contexts and labels must have identical shapes")
    if corrupt:
        draws = torch.rand(
            (2, len(images)),
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        )
        gain = (
            math.log(config.gain_min)
            + draws[0] * math.log(config.gain_max / config.gain_min)
        ).exp()
        divisor = (
            math.log(config.noise_divisor_min)
            + draws[1] * math.log(config.noise_divisor_max / config.noise_divisor_min)
        ).exp()
        images = brightness_dependent_noise(images, gain, divisor, generator=generator)
    cue = (
        F.one_hot(contexts, config.num_contexts).to(images.dtype) * config.context_scale
    )
    return (
        torch.cat((images, cue), -1),
        (labels + contexts) % config.num_classes,
        contexts,
    )


def load_emnist_balanced(
    root, *, train, download=False, upright=True, peak_normalize=True
):
    """Load EMNIST Balanced as flattened float images and integer labels.

    ``upright`` transposes the stored EMNIST image axes. Peak normalization is
    per image, before corruption. No split or scientific protocol is implied.
    """
    from torchvision.datasets import EMNIST

    dataset = EMNIST(root=root, split="balanced", train=train, download=download)
    images = dataset.data.transpose(-1, -2) if upright else dataset.data
    images = images.reshape(len(images), -1).float() / 255
    if peak_normalize:
        images = images / images.amax(-1, keepdim=True).clamp_min(1 / 255)
    return images, dataset.targets.clone()


EMNIST_BALANCED_CLASSES = 47


@dataclass
class ContextualRemappingDataConfig(BaseConfig):
    """Data protocol of the context-cued EMNIST Balanced task.

    ``pool_per_class`` training images per class are held out as a validation
    pool; the first ``validation_per_class`` of them form the fixed validation
    set. The partition uses one CPU permutation per class from
    ``partition_seed``. Validation and test inputs carry one fixed context and
    corruption per image, drawn on ``generator_device`` ("cuda" reproduces
    evaluation sets generated on a GPU). ``root`` defaults to ``data.base_dir``.
    """

    root: str = ""
    download: bool = False
    upright: bool = True
    peak_normalize: bool = True
    pool_per_class: int = 200
    validation_per_class: int = 50
    partition_seed: int = 31003
    num_contexts: int = 32
    context_scale: float = 1.0
    gain_min: float = 0.05
    gain_max: float = 20.0
    noise_divisor_min: float = 1.0
    noise_divisor_max: float = 4.0
    validation_seed: int = 78019
    test_seed: int = 99173
    generator_device: str = "cpu"

    def __post_init__(self):
        if not 0 < self.validation_per_class <= self.pool_per_class:
            raise ValueError("require 0 < validation_per_class <= pool_per_class")
        if self.generator_device not in {"cpu", "cuda"}:
            raise ValueError("generator_device must be 'cpu' or 'cuda'")
        self.task_config()

    def task_config(self) -> ContextualRemappingConfig:
        return ContextualRemappingConfig(
            num_classes=EMNIST_BALANCED_CLASSES,
            num_contexts=self.num_contexts,
            context_scale=self.context_scale,
            gain_min=self.gain_min,
            gain_max=self.gain_max,
            noise_divisor_min=self.noise_divisor_min,
            noise_divisor_max=self.noise_divisor_max,
        )


def partition_emnist_balanced(labels, *, pool_per_class, validation_per_class, seed):
    """Per-class random partition: (training, validation, validation pool) indices.

    Each class is permuted with one shared CPU generator, in class order; the
    first ``pool_per_class`` images form the pool, whose first
    ``validation_per_class`` images are the fixed validation set.
    """
    generator = torch.Generator().manual_seed(int(seed))
    training, validation, pool = [], [], []
    for label in range(int(labels.max()) + 1):
        ids = labels.eq(label).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=generator)]
        pool.append(ids[:pool_per_class])
        validation.append(ids[:validation_per_class])
        training.append(ids[pool_per_class:])
    return torch.cat(training), torch.cat(validation), torch.cat(pool)


class ContextualRemappingStream:
    """Training presentations: images sampled uniformly with replacement.

    Every presentation draws a fresh context, gain and noise level. Index and
    corruption draws use separate caller-owned generators on the data device.
    """

    def __init__(self, images, labels, task: ContextualRemappingConfig):
        self.images, self.labels, self.task = images, labels, task

    def to(self, device):
        return ContextualRemappingStream(
            self.images.to(device), self.labels.to(device), self.task
        )

    @property
    def device(self):
        return self.images.device

    def sample(self, count, *, index_generator, noise_generator):
        ids = torch.randint(
            len(self.labels), (count,), device=self.device, generator=index_generator
        )
        inputs, targets, _ = contextual_remapping_batch(
            self.images[ids], self.labels[ids], self.task, generator=noise_generator
        )
        return inputs, targets


class ContextualRemappingTrainingSet(Dataset):
    """Clean training images; train through ``stream``.

    Items show each image uncorrupted with context 0, which fixes the input
    shape for model construction. The ``presentation_stream`` trainer samples
    corrupted presentations from ``stream`` instead of iterating items.
    """

    def __init__(self, images, labels, task: ContextualRemappingConfig):
        self.stream = ContextualRemappingStream(images, labels, task)

    def __len__(self):
        return len(self.stream.labels)

    def __getitem__(self, index):
        images = self.stream.images[index : index + 1]
        labels = self.stream.labels[index : index + 1]
        inputs, targets, _ = contextual_remapping_batch(
            images,
            labels,
            self.stream.task,
            contexts=torch.zeros_like(labels),
            corrupt=False,
        )
        return inputs[0], targets[0]


class ContextualRemappingEvaluationSet(Dataset):
    """Fixed evaluation inputs with one context and corruption per image.

    Items are the corrupted ("noisy") inputs; ``variants`` also holds the same
    images and contexts without corruption ("clean").
    """

    def __init__(self, noisy, clean, targets, contexts):
        self.contexts = contexts
        self.variants = {
            "noisy": TensorDataset(noisy, targets),
            "clean": TensorDataset(clean, targets),
        }

    def __len__(self):
        return len(self.variants["noisy"])

    def __getitem__(self, index):
        return self.variants["noisy"][index]


def fixed_evaluation_set(images, labels, task, *, seed, device="cpu"):
    """Corrupt each image once, from a seeded generator on ``device``."""
    images, labels = images.to(device), labels.to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noisy, targets, contexts = contextual_remapping_batch(
        images, labels, task, generator=generator
    )
    clean, _, _ = contextual_remapping_batch(
        images, labels, task, contexts=contexts, corrupt=False
    )
    return ContextualRemappingEvaluationSet(
        noisy.cpu(), clean.cpu(), targets.cpu(), contexts.cpu()
    )


def contextual_remapping_data_config(task_cfg) -> ContextualRemappingDataConfig:
    """Typed data config from ``data.dataset_params.contextual_remapping``.

    Unknown keys fail. Generic processing options and pipeline seeds, which the
    loader merges into the same mapping, are ignored here.
    """
    from dendritic_modeling.datasets.parameters import PIPELINE_PARAMETER_KEYS

    parameters = dict(getattr(task_cfg, "parameters", None) or {})
    known = {item.name for item in fields(ContextualRemappingDataConfig)}
    unknown = sorted(set(parameters) - known - PIPELINE_PARAMETER_KEYS)
    if unknown:
        raise ValueError(f"unknown contextual_remapping data parameters: {unknown}")
    config = ContextualRemappingDataConfig(
        **{key: value for key, value in parameters.items() if key in known}
    )
    if not config.root:
        config.root = str(getattr(task_cfg, "data_path", "") or "")
    if not config.root:
        raise ValueError("contextual_remapping requires data.base_dir or a root")
    if config.generator_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("generator_device='cuda' requires a visible GPU")
    return config


def build_contextual_remapping_datasets(task_cfg):
    """(training, validation, official test) sets of the context-cued task."""
    config = contextual_remapping_data_config(task_cfg)
    task = config.task_config()
    raw = str(Path(config.root) / "raw")
    images, labels = load_emnist_balanced(
        raw,
        train=True,
        download=config.download,
        upright=config.upright,
        peak_normalize=config.peak_normalize,
    )
    training, validation, _ = partition_emnist_balanced(
        labels,
        pool_per_class=config.pool_per_class,
        validation_per_class=config.validation_per_class,
        seed=config.partition_seed,
    )
    test_images, test_labels = load_emnist_balanced(
        raw,
        train=False,
        download=config.download,
        upright=config.upright,
        peak_normalize=config.peak_normalize,
    )
    return (
        ContextualRemappingTrainingSet(images[training], labels[training], task),
        fixed_evaluation_set(
            images[validation],
            labels[validation],
            task,
            seed=config.validation_seed,
            device=config.generator_device,
        ),
        fixed_evaluation_set(
            test_images,
            test_labels,
            task,
            seed=config.test_seed,
            device=config.generator_device,
        ),
    )
