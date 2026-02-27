"""
Sequence and specialized datasets for dendritic network training.

Sequence datasets return (input_seq, target) where input_seq: [seq_len, input_dim].
For many-to-one tasks, target is a scalar or vector.
For many-to-many tasks, target is [seq_len, ...].

Also includes real neural data datasets (e.g. StringerV1Dataset) that return
(response_vector, label) for feedforward classification.
"""

import logging
import math
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

logger = logging.getLogger(__name__)


class SequentialMNISTDataset(Dataset):
    """Sequential MNIST: processes image pixel-by-pixel or row-by-row.

    Args:
        mode: "pixel" for pixel-by-pixel (seq_len=784, dim=1) or
              "row" for row-by-row (seq_len=28, dim=28).
        train: If True, training set; else test set.
        data_path: Path to MNIST data directory.
        normalize: Whether to normalize to [0, 1].
    """

    def __init__(
        self,
        mode: str = "pixel",
        train: bool = True,
        data_path: Optional[str] = None,
        normalize: bool = True,
    ):
        from torchvision import datasets, transforms

        transform = (
            transforms.ToTensor()
            if normalize
            else transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
            )
        )
        root = data_path or "./data"
        mnist = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        self.mode = mode
        images = mnist.data.float() / 255.0 if normalize else mnist.data.float()
        self.labels = mnist.targets

        if mode == "pixel":
            # [N, 28, 28] -> [N, 784, 1]
            self.sequences = images.reshape(-1, 784, 1)
        elif mode == "row":
            # [N, 28, 28] -> [N, 28, 28]
            self.sequences = images
        else:
            raise ValueError(f"mode must be 'pixel' or 'row', got '{mode}'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class CopyTaskDataset(Dataset):
    """Copy task: memorize a sequence and reproduce it after a delay.

    Input: [total_len, n_symbols + 2] where total_len = seq_len + delay + seq_len.
    Target: [total_len] with ignore_index=-100 for non-output positions,
            symbol indices for the output phase (last seq_len timesteps).

    Compatible with both output_mode="all" (many-to-many with ignore_index)
    and output_mode="last" (use only final timestep, though seq_len>1 needs "all").

    For training with output_mode="all", use CrossEntropyLoss(ignore_index=-100).

    Args:
        n_samples: Number of samples to generate.
        seq_len: Length of the sequence to memorize.
        delay: Number of blank timesteps between input and output.
        n_symbols: Number of distinct symbols (excluding blank and trigger).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 10,
        delay: int = 50,
        n_symbols: int = 8,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.delay = delay
        self.n_symbols = n_symbols
        self.total_len = seq_len + delay + seq_len
        self.input_dim = n_symbols + 2  # symbols + blank + trigger

        # Pre-generate all samples
        self.inputs = torch.zeros(n_samples, self.total_len, self.input_dim)
        # Full-length targets: -100 (ignore) for non-output, symbols for output phase
        self.targets = torch.full(
            (n_samples, self.total_len), self.IGNORE_INDEX, dtype=torch.long
        )

        for i in range(n_samples):
            symbols = torch.randint(0, n_symbols, (seq_len,))
            # Input phase: one-hot symbols
            for t, s in enumerate(symbols):
                self.inputs[i, t, s] = 1.0
            # Delay phase: blank (all zeros, could add blank channel)
            # Trigger marker at start of output phase
            self.inputs[i, seq_len + delay, n_symbols + 1] = 1.0
            # Output phase targets: symbol indices
            output_start = seq_len + delay
            self.targets[i, output_start : output_start + seq_len] = symbols

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class AddingProblemDataset(Dataset):
    """Adding problem: sum two marked entries in a long sequence.

    Input: [seq_len, 2] — first channel is random uniform, second is mask.
    Target: scalar (sum of the two marked entries).

    Args:
        n_samples: Number of samples.
        seq_len: Sequence length.
    """

    def __init__(self, n_samples: int = 10000, seq_len: int = 200):
        self.n_samples = n_samples
        self.seq_len = seq_len

        self.inputs = torch.zeros(n_samples, seq_len, 2)
        self.targets = torch.zeros(n_samples, 1)

        for i in range(n_samples):
            # Random values in [0, 1)
            self.inputs[i, :, 0] = torch.rand(seq_len)
            # Mark two random positions
            marks = torch.randperm(seq_len)[:2]
            self.inputs[i, marks[0], 1] = 1.0
            self.inputs[i, marks[1], 1] = 1.0
            # Target is sum of the marked values
            self.targets[i] = self.inputs[i, marks[0], 0] + self.inputs[i, marks[1], 0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class VariableDelayDMSDataset(Dataset):
    """Variable-delay delayed match-to-sample task.

    Stimulus presented, delay period, then match/non-match probe.
    Target: binary (match=1, non-match=0).

    Args:
        n_samples: Number of samples.
        stimulus_dim: Dimensionality of the stimulus.
        stimulus_duration: Number of timesteps for stimulus presentation.
        min_delay: Minimum delay in timesteps.
        max_delay: Maximum delay in timesteps.
        probe_duration: Number of timesteps for probe presentation.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        stimulus_dim: int = 16,
        stimulus_duration: int = 10,
        min_delay: int = 10,
        max_delay: int = 100,
        probe_duration: int = 10,
    ):
        self.n_samples = n_samples
        self.stimulus_dim = stimulus_dim
        self.stimulus_duration = stimulus_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.probe_duration = probe_duration
        self.max_total_len = stimulus_duration + max_delay + probe_duration
        self.input_dim = stimulus_dim + 1  # stimulus + timing channel

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
            self.delays[i] = delay
            total_len = stimulus_duration + delay + probe_duration

            # Generate stimulus
            stimulus = torch.randn(stimulus_dim)
            stimulus = stimulus / stimulus.norm()

            # Stimulus phase
            for t in range(stimulus_duration):
                self.inputs[i, t, :stimulus_dim] = stimulus

            # Timing channel: ramp during delay
            for t in range(stimulus_duration, stimulus_duration + delay):
                self.inputs[i, t, stimulus_dim] = (t - stimulus_duration) / delay

            # Probe phase: match or non-match (50/50)
            is_match = torch.rand(1).item() > 0.5
            if is_match:
                probe = stimulus
                self.targets[i] = 1
            else:
                probe = torch.randn(stimulus_dim)
                probe = probe / probe.norm()
                self.targets[i] = 0

            for t in range(
                stimulus_duration + delay, min(total_len, self.max_total_len)
            ):
                self.inputs[i, t, :stimulus_dim] = probe

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        delays = self.delays[idx]
        if delays.dim() == 0:
            # Single index: return scalar seq_len
            seq_len = self.stimulus_duration + delays.item() + self.probe_duration
        else:
            # Slice/list index: return tensor of seq_lengths
            seq_len = self.stimulus_duration + delays + self.probe_duration
        return self.inputs[idx], self.targets[idx], seq_len


class MultiFrequencyDataset(Dataset):
    """Multi-frequency classification task.

    Input is a mixture of sinusoids at different frequencies.
    Target: class index based on dominant frequency.

    Args:
        n_samples: Number of samples.
        seq_len: Number of timesteps.
        n_frequencies: Number of frequency classes.
        base_freq: Base frequency (Hz).
        noise_std: Standard deviation of additive noise.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 200,
        n_frequencies: int = 5,
        base_freq: float = 0.01,
        noise_std: float = 0.1,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_frequencies = n_frequencies

        t = torch.linspace(0, 1, seq_len).unsqueeze(0)  # [1, seq_len]
        freqs = [(i + 1) * base_freq for i in range(n_frequencies)]

        self.inputs = torch.zeros(n_samples, seq_len, 1)
        self.targets = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_samples):
            # Choose dominant frequency
            dominant = torch.randint(0, n_frequencies, (1,)).item()
            self.targets[i] = dominant

            # Build signal: dominant + weaker harmonics + noise
            signal = torch.zeros(1, seq_len)
            for f_idx, freq in enumerate(freqs):
                amplitude = 1.0 if f_idx == dominant else 0.2
                phase = torch.rand(1) * 2 * math.pi
                signal += amplitude * torch.sin(
                    2 * math.pi * freq * t * seq_len + phase
                )

            signal += noise_std * torch.randn(1, seq_len)
            self.inputs[i, :, 0] = signal.squeeze()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LorenzSequenceDataset(Dataset):
    """Lorenz attractor dynamics prediction (regression benchmark).

    Input: [seq_len, 3] (x, y, z coordinates).
    Target: [3] (next-step prediction of the final timestep).

    Args:
        n_samples: Number of trajectory segments.
        seq_len: Length of each segment.
        dt: Integration timestep.
        sigma, rho, beta: Lorenz system parameters.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 50,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len

        # Generate a long trajectory then split into segments
        total_steps = n_samples * seq_len + seq_len
        trajectory = torch.zeros(total_steps + 1, 3)
        # Random initial condition near attractor
        trajectory[0] = torch.tensor([1.0, 1.0, 1.0]) + 0.1 * torch.randn(3)

        for t in range(total_steps):
            x, y, z = trajectory[t]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            trajectory[t + 1] = trajectory[t] + dt * torch.tensor([dx, dy, dz])

        # Normalize
        mean = trajectory.mean(dim=0)
        std = trajectory.std(dim=0) + 1e-8
        trajectory = (trajectory - mean) / std

        # Extract segments
        self.inputs = torch.zeros(n_samples, seq_len, 3)
        self.targets = torch.zeros(n_samples, 3)

        for i in range(n_samples):
            start = i * seq_len
            self.inputs[i] = trajectory[start : start + seq_len]
            self.targets[i] = trajectory[start + seq_len]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class StringerV1Dataset(Dataset):
    """Stringer et al. (2019) V1 population responses to drifting gratings.

    Loads real calcium imaging data from mouse V1 (11,311 neurons, 4,442 trials).
    Selects top neurons by response variance and bins orientations into discrete
    classes for feedforward classification.

    Returns (response_vector, label) where response_vector: [n_neurons] and
    label: int in [0, n_classes).

    Running speed metadata is stored for post-hoc cross-context analysis
    (e.g., comparing representations during locomotion vs. stillness).

    Data source: Stringer et al., "Spontaneous behaviors drive multidimensional,
    brainwide activity", Science 364, 2019. CC0 license on Figshare.

    Args:
        data_dir: Directory containing gratings_drifting_GT3_2019_04_05_1.npy
                  and all_running.npy.
        n_neurons: Number of neurons to select by response variance.
        n_classes: Number of orientation bins (orientations span [0, 2*pi)).
        split: "train" or "test".
        train_fraction: Fraction of trials for training (rest = test).
        normalize: Z-score normalize each neuron's responses.
        seed: Random seed for reproducible train/test split.
        session_idx: Index into all_running.npy for this session (default 20 = GT3).
    """

    _GRATING_FILE = "gratings_drifting_GT3_2019_04_05_1.npy"
    _RUNNING_FILE = "all_running.npy"

    def __init__(
        self,
        data_dir: str,
        n_neurons: int = 500,
        n_classes: int = 8,
        split: str = "train",
        train_fraction: float = 0.9,
        normalize: bool = True,
        seed: int = 42,
        session_idx: int = 20,
    ):
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        # --- Load raw data ---
        grating_path = os.path.join(data_dir, self._GRATING_FILE)
        running_path = os.path.join(data_dir, self._RUNNING_FILE)

        if not os.path.exists(grating_path):
            raise FileNotFoundError(
                f"Stringer V1 data not found at {grating_path}. "
                f"Download from https://figshare.com/articles/dataset/9598406"
            )

        gdata = np.load(grating_path, allow_pickle=True).item()
        sresp = gdata["sresp"]  # (n_neurons_total, n_trials)
        istim = gdata["istim"]  # (n_trials,) orientations in radians [0, 2*pi)

        n_neurons_total, n_trials = sresp.shape
        logger.info(f"Loaded Stringer V1: {n_neurons_total} neurons, {n_trials} trials")

        # --- Load running speed ---
        if os.path.exists(running_path):
            all_running = np.load(running_path, allow_pickle=True)
            running = all_running[session_idx]  # (n_trials,)
            self._running_speed = running.astype(np.float32)
        else:
            logger.warning(
                f"Running speed file not found at {running_path}, "
                f"setting all running speeds to 0"
            )
            self._running_speed = np.zeros(n_trials, dtype=np.float32)

        # --- Select top neurons by variance ---
        neuron_var = sresp.var(axis=1)  # variance across trials per neuron
        n_select = min(n_neurons, n_neurons_total)
        top_idx = np.argsort(neuron_var)[-n_select:][::-1]  # descending
        self._neuron_indices = top_idx
        responses = sresp[top_idx, :].T  # (n_trials, n_select)

        # --- Normalize ---
        if normalize:
            mu = responses.mean(axis=0, keepdims=True)
            std = responses.std(axis=0, keepdims=True) + 1e-8
            responses = (responses - mu) / std

        # --- Bin orientations into classes ---
        # istim is in [0, 2*pi), bin into n_classes equal bins
        bin_edges = np.linspace(0, 2 * np.pi, n_classes + 1)
        labels = np.digitize(istim, bin_edges) - 1
        labels = np.clip(labels, 0, n_classes - 1)  # handle edge case at 2*pi

        # --- Train/test split ---
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_trials)
        n_train = int(n_trials * train_fraction)

        if split == "train":
            idx = perm[:n_train]
        else:
            idx = perm[n_train:]

        self.responses = torch.tensor(responses[idx], dtype=torch.float32)
        self.labels = torch.tensor(labels[idx], dtype=torch.long)
        self._running_mask = torch.tensor(
            self._running_speed[idx] > np.median(self._running_speed),
            dtype=torch.bool,
        )
        self._split_indices = idx

        logger.info(
            f"StringerV1Dataset({split}): {len(self)} trials, "
            f"{n_select} neurons, {n_classes} classes"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.responses[idx], self.labels[idx]

    @property
    def running_mask(self):
        """Boolean mask: True for high-running trials (above median speed)."""
        return self._running_mask

    @property
    def neuron_indices(self):
        """Indices of selected neurons in original sresp array."""
        return self._neuron_indices


def get_sequence_datasets(
    dataset_name: str,
    data_path: Optional[str] = None,
    train_valid_split: float = 0.9,
    **kwargs,
):
    """Factory function for sequence datasets.

    Args:
        dataset_name: Name of the dataset.
        data_path: Path for datasets that need file storage.
        train_valid_split: Fraction of training data to use for training.
        **kwargs: Dataset-specific parameters.

    Returns:
        Tuple of (train_ds, valid_ds, test_ds).
    """
    if dataset_name == "sequential_mnist":
        mode = kwargs.get("mode", "pixel")
        normalize = kwargs.get("normalize", True)
        train_full = SequentialMNISTDataset(
            mode=mode, train=True, data_path=data_path, normalize=normalize
        )
        test_ds = SequentialMNISTDataset(
            mode=mode, train=False, data_path=data_path, normalize=normalize
        )
    elif dataset_name == "copy_task":
        n_train = kwargs.get("n_samples", 10000)
        train_full = CopyTaskDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 10),
            delay=kwargs.get("delay", 50),
            n_symbols=kwargs.get("n_symbols", 8),
        )
        test_ds = CopyTaskDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 10),
            delay=kwargs.get("delay", 50),
            n_symbols=kwargs.get("n_symbols", 8),
        )
    elif dataset_name == "adding_problem":
        n_train = kwargs.get("n_samples", 10000)
        train_full = AddingProblemDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 200),
        )
        test_ds = AddingProblemDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 200),
        )
    elif dataset_name == "variable_delay_dms":
        n_train = kwargs.get("n_samples", 10000)
        train_full = VariableDelayDMSDataset(
            n_samples=n_train,
            stimulus_dim=kwargs.get("stimulus_dim", 16),
            stimulus_duration=kwargs.get("stimulus_duration", 10),
            min_delay=kwargs.get("min_delay", 10),
            max_delay=kwargs.get("max_delay", 100),
            probe_duration=kwargs.get("probe_duration", 10),
        )
        test_ds = VariableDelayDMSDataset(
            n_samples=max(n_train // 5, 1000),
            stimulus_dim=kwargs.get("stimulus_dim", 16),
            stimulus_duration=kwargs.get("stimulus_duration", 10),
            min_delay=kwargs.get("min_delay", 10),
            max_delay=kwargs.get("max_delay", 100),
            probe_duration=kwargs.get("probe_duration", 10),
        )
    elif dataset_name == "multi_frequency":
        n_train = kwargs.get("n_samples", 10000)
        train_full = MultiFrequencyDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 200),
            n_frequencies=kwargs.get("n_frequencies", 5),
            base_freq=kwargs.get("base_freq", 0.01),
            noise_std=kwargs.get("noise_std", 0.1),
        )
        test_ds = MultiFrequencyDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 200),
            n_frequencies=kwargs.get("n_frequencies", 5),
            base_freq=kwargs.get("base_freq", 0.01),
            noise_std=kwargs.get("noise_std", 0.1),
        )
    elif dataset_name == "lorenz":
        n_train = kwargs.get("n_samples", 10000)
        train_full = LorenzSequenceDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 50),
            dt=kwargs.get("dt", 0.01),
        )
        test_ds = LorenzSequenceDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 50),
            dt=kwargs.get("dt", 0.01),
        )
    elif dataset_name == "stringer_v1":
        data_dir = kwargs.get("data_dir", data_path)
        if data_dir is None:
            raise ValueError(
                "stringer_v1 requires data_dir (path to directory with "
                "gratings_drifting_GT3_2019_04_05_1.npy)"
            )
        n_neurons = kwargs.get("n_neurons", 500)
        n_classes = kwargs.get("n_classes", 8)
        normalize = kwargs.get("normalize", True)
        seed = kwargs.get("label_noise_seed", 42)
        # Fixed 90/10 train/test split; train_valid_split handled by factory below
        test_fraction = 0.1

        train_full = StringerV1Dataset(
            data_dir=data_dir,
            n_neurons=n_neurons,
            n_classes=n_classes,
            split="train",
            train_fraction=1.0 - test_fraction,
            normalize=normalize,
            seed=seed,
        )
        test_ds = StringerV1Dataset(
            data_dir=data_dir,
            n_neurons=n_neurons,
            n_classes=n_classes,
            split="test",
            train_fraction=1.0 - test_fraction,
            normalize=normalize,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown sequence dataset: {dataset_name}")

    # Split training into train/valid
    n_train = int(len(train_full) * train_valid_split)
    n_valid = len(train_full) - n_train
    if n_valid > 0:
        train_ds, valid_ds = random_split(
            train_full,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = train_full
        valid_ds = test_ds  # use test set as validation if no split

    return train_ds, valid_ds, test_ds
