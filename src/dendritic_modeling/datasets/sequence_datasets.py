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

    Input: [seq_len, 2] -- first channel is random uniform, second is mask.
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


class OculomotorDelayedResponseDataset(Dataset):
    """Delayed-response task with a remembered angular cue.

    A brief cue indicates one of ``n_targets`` angular locations. The network
    must retain that location over a silent delay and report it after a go cue.
    This is a simple delayed-response benchmark motivated by oculomotor memory
    tasks used in systems neuroscience.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_targets: int = 8,
        input_tuning_dim: int = 16,
        fixation_duration: int = 6,
        cue_duration: int = 8,
        min_delay: int = 10,
        max_delay: int = 100,
        go_duration: int = 8,
        tuning_kappa: float = 2.0,
        cue_noise_std: float = 0.05,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_targets = n_targets
        self.input_tuning_dim = input_tuning_dim
        self.fixation_duration = fixation_duration
        self.cue_duration = cue_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.go_duration = go_duration
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = input_tuning_dim + 2
        self.max_total_len = fixation_duration + cue_duration + max_delay + go_duration

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)

        pref_angles = torch.linspace(
            0.0, 2.0 * math.pi, steps=input_tuning_dim + 1, dtype=torch.float32
        )[:-1]
        target_angles = torch.linspace(
            0.0, 2.0 * math.pi, steps=n_targets + 1, dtype=torch.float32
        )[:-1]

        for i in range(n_samples):
            target_idx = torch.randint(0, n_targets, (1,)).item()
            delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
            total_len = fixation_duration + cue_duration + delay + go_duration

            angle = target_angles[target_idx]
            cue_pattern = torch.exp(tuning_kappa * torch.cos(pref_angles - angle))
            cue_pattern = cue_pattern / cue_pattern.max().clamp(min=1e-8)

            self.targets[i] = target_idx
            self.delays[i] = delay

            # Fixation stays on through the silent memory period.
            self.inputs[
                i, : fixation_duration + cue_duration + delay, input_tuning_dim
            ] = 1.0

            cue_start = fixation_duration
            cue_end = fixation_duration + cue_duration
            self.inputs[i, cue_start:cue_end, :input_tuning_dim] = (
                cue_pattern.unsqueeze(0)
            )
            self.inputs[
                i, cue_start:cue_end, :input_tuning_dim
            ] += cue_noise_std * torch.randn(cue_duration, input_tuning_dim)
            if self.positive_input_encoding:
                self.inputs[i, cue_start:cue_end, :input_tuning_dim].clamp_(min=0.0)

            go_start = fixation_duration + cue_duration + delay
            self.inputs[i, go_start:total_len, input_tuning_dim + 1] = 1.0

        self.seq_lengths = (
            self.fixation_duration + self.cue_duration + self.delays + self.go_duration
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths


class RomoDelayComparisonDataset(Dataset):
    """Delayed comparison of two vibrotactile frequencies.

    The first stimulus frequency must be retained over a silent delay and then
    compared against a second frequency. The target is binary: whether the
    second frequency is higher than the first.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        frequency_values: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        stimulus_duration: int = 20,
        min_delay: int = 10,
        max_delay: int = 100,
        response_duration: int = 8,
        noise_std: float = 0.08,
        positive_input_encoding: bool = False,
        tuning_dim: int = 8,
        tuning_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.frequency_values = tuple(float(v) for v in frequency_values)
        self.stimulus_duration = stimulus_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.response_duration = response_duration
        self.positive_input_encoding = bool(positive_input_encoding)
        self.tuning_dim = int(tuning_dim)
        self.input_dim = (self.tuning_dim if self.positive_input_encoding else 1) + 3
        self.max_total_len = 2 * stimulus_duration + max_delay + response_duration

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)

        phase_grid = torch.arange(stimulus_duration, dtype=torch.float32)
        if self.positive_input_encoding:
            freq_min = min(self.frequency_values)
            freq_max = max(self.frequency_values)
            self.frequency_pref = torch.linspace(
                float(freq_min), float(freq_max), steps=self.tuning_dim
            )
            spacing = (
                float(freq_max - freq_min) / max(self.tuning_dim - 1, 1)
                if self.tuning_dim > 1
                else 1.0
            )
            self.tuning_sigma = (
                float(tuning_sigma) if tuning_sigma is not None else spacing
            )

        for i in range(n_samples):
            f1_idx = torch.randint(0, len(self.frequency_values), (1,)).item()
            f2_idx = torch.randint(0, len(self.frequency_values), (1,)).item()
            while f2_idx == f1_idx:
                f2_idx = torch.randint(0, len(self.frequency_values), (1,)).item()

            f1 = self.frequency_values[f1_idx]
            f2 = self.frequency_values[f2_idx]
            delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
            total_len = 2 * stimulus_duration + delay + response_duration

            if self.positive_input_encoding:
                sig1 = self._positive_frequency_code(f1, noise_std)
                sig2 = self._positive_frequency_code(f2, noise_std)
            else:
                sig1 = torch.sin(2.0 * math.pi * f1 * phase_grid / stimulus_duration)
                sig2 = torch.sin(2.0 * math.pi * f2 * phase_grid / stimulus_duration)
                sig1 = sig1 + noise_std * torch.randn_like(sig1)
                sig2 = sig2 + noise_std * torch.randn_like(sig2)

            self.targets[i] = int(f2 > f1)
            self.delays[i] = delay

            s1_start = 0
            s1_end = stimulus_duration
            s2_start = stimulus_duration + delay
            s2_end = s2_start + stimulus_duration
            rsp_start = s2_end

            stim_dim = self.tuning_dim if self.positive_input_encoding else 1
            self.inputs[i, s1_start:s1_end, :stim_dim] = sig1
            self.inputs[i, s1_start:s1_end, stim_dim] = 1.0
            self.inputs[i, s2_start:s2_end, :stim_dim] = sig2
            self.inputs[i, s2_start:s2_end, stim_dim + 1] = 1.0
            self.inputs[i, rsp_start:total_len, stim_dim + 2] = 1.0

        self.seq_lengths = (
            2 * self.stimulus_duration + self.delays + self.response_duration
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths

    def _positive_frequency_code(
        self, frequency: float, noise_std: float
    ) -> torch.Tensor:
        tuning = torch.exp(
            -0.5
            * ((self.frequency_pref - float(frequency)) / max(self.tuning_sigma, 1e-6))
            ** 2
        )
        tuning = tuning / tuning.max().clamp(min=1e-8)
        signal = tuning.unsqueeze(0).repeat(self.stimulus_duration, 1)
        if noise_std > 0:
            signal = signal + noise_std * torch.randn_like(signal)
        return signal.clamp_min(0.0)


class GainModulatedContextualComparisonDataset(Dataset):
    """Context-cued delayed comparison under multiplicative nuisance gain.

    Each trial contains:
    1. a brief cue selecting one relevant pathway,
    2. a first short stimulus on every pathway,
    3. a silent delay with optional distractor bursts,
    4. a second short stimulus on every pathway,
    5. a response window.

    The target is binary: whether the relevant pathway's second latent value is
    larger than its first. All pathways are corrupted by a trial-global gain and
    pathway-specific gains, making the task a sequence-level nuisance-robust
    delayed comparison problem rather than a pure memory benchmark.

    Inputs are arranged as:
        [context one-hot (n_pathways) |
         pathway features (n_pathways * pathway_dim) |
         phase markers (sample1, sample2, response)]

    The contiguous pathway blocks are intentional so configs can test explicit
    clustered vs. dispersed pathway routing with the ``PathwayRouter`` encoder.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_pathways: int = 4,
        pathway_dim: int = 4,
        cue_duration: int = 8,
        stimulus_duration: int = 16,
        min_delay: int = 20,
        max_delay: int = 80,
        response_duration: int = 8,
        amplitude_values: tuple[float, ...] = (0.45, 0.6, 0.75, 0.9, 1.05, 1.2),
        comparison_margin: float = 0.15,
        global_gain_min: float = 0.85,
        global_gain_max: float = 1.15,
        pathway_gain_min: float = 0.8,
        pathway_gain_max: float = 1.2,
        gain_drift_std: float = 0.03,
        noise_std: float = 0.08,
        cue_noise_std: float = 0.02,
        distractor_burst_prob: float = 0.35,
        distractor_burst_scale: float = 1.5,
        distractor_burst_duration: int = 4,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        if n_pathways < 2:
            raise ValueError("n_pathways must be >= 2")
        if pathway_dim < 1:
            raise ValueError("pathway_dim must be >= 1")
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError(
                f"Expected 0 <= min_delay <= max_delay, got {min_delay}, {max_delay}"
            )
        if len(amplitude_values) < 2:
            raise ValueError("amplitude_values must contain at least two entries")
        if global_gain_min <= 0 or pathway_gain_min <= 0:
            raise ValueError("Gain ranges must be positive")

        self.n_samples = int(n_samples)
        self.n_pathways = int(n_pathways)
        self.pathway_dim = int(pathway_dim)
        self.cue_duration = int(cue_duration)
        self.stimulus_duration = int(stimulus_duration)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.response_duration = int(response_duration)
        self.comparison_margin = float(comparison_margin)
        self.global_gain_min = float(global_gain_min)
        self.global_gain_max = float(global_gain_max)
        self.pathway_gain_min = float(pathway_gain_min)
        self.pathway_gain_max = float(pathway_gain_max)
        self.gain_drift_std = float(gain_drift_std)
        self.noise_std = float(noise_std)
        self.cue_noise_std = float(cue_noise_std)
        self.distractor_burst_prob = float(distractor_burst_prob)
        self.distractor_burst_scale = float(distractor_burst_scale)
        self.distractor_burst_duration = max(1, int(distractor_burst_duration))
        self.positive_input_encoding = bool(positive_input_encoding)

        self.context_dim = self.n_pathways
        self.phase_marker_dim = 3
        self.pathway_feature_start = self.context_dim
        self.pathway_feature_end = self.context_dim + self.n_pathways * self.pathway_dim
        self.phase_marker_start = self.pathway_feature_end
        self.input_dim = self.phase_marker_start + self.phase_marker_dim
        self.max_total_len = (
            self.cue_duration
            + 2 * self.stimulus_duration
            + self.max_delay
            + self.response_duration
        )

        amplitude_tensor = torch.tensor(amplitude_values, dtype=torch.float32)
        if amplitude_tensor.numel() < 2:
            raise ValueError("Need at least two amplitude values for comparison")
        self.amplitude_values = amplitude_tensor

        self.inputs = torch.zeros(self.n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(self.n_samples, dtype=torch.long)
        self.delays = torch.zeros(self.n_samples, dtype=torch.long)
        self.relevant_pathways = torch.zeros(self.n_samples, dtype=torch.long)
        self.seq_lengths = torch.zeros(self.n_samples, dtype=torch.long)
        self.sample1_values = torch.zeros(self.n_samples, self.n_pathways)
        self.sample2_values = torch.zeros(self.n_samples, self.n_pathways)
        self.global_gains = torch.zeros(self.n_samples, dtype=torch.float32)
        self.pathway_gain_sample1 = torch.zeros(self.n_samples, self.n_pathways)
        self.pathway_gain_sample2 = torch.zeros(self.n_samples, self.n_pathways)
        self.delay_burst_flags = torch.zeros(self.n_samples, dtype=torch.bool)

        self.pathway_templates = self._build_pathway_templates()
        self.pathway_slices = [
            slice(
                self.pathway_feature_start + pathway_idx * self.pathway_dim,
                self.pathway_feature_start + (pathway_idx + 1) * self.pathway_dim,
            )
            for pathway_idx in range(self.n_pathways)
        ]

        for i in range(self.n_samples):
            relevant = torch.randint(0, self.n_pathways, (1,)).item()
            delay = torch.randint(self.min_delay, self.max_delay + 1, (1,)).item()
            total_len = (
                self.cue_duration
                + 2 * self.stimulus_duration
                + delay
                + self.response_duration
            )
            cue_end = self.cue_duration
            s1_end = cue_end + self.stimulus_duration
            s2_start = s1_end + delay
            s2_end = s2_start + self.stimulus_duration
            response_start = s2_end

            global_gain = torch.empty(1).uniform_(
                self.global_gain_min, self.global_gain_max
            )[0]
            pathway_gain_base = torch.empty(self.n_pathways).uniform_(
                self.pathway_gain_min, self.pathway_gain_max
            )
            sample2_drift = torch.exp(
                0.25 * torch.randn(self.n_pathways, dtype=torch.float32)
            )
            pathway_gain_s1 = pathway_gain_base
            pathway_gain_s2 = torch.clamp(
                pathway_gain_base * sample2_drift,
                min=self.pathway_gain_min,
                max=self.pathway_gain_max,
            )

            target = int(torch.rand(1).item() > 0.5)
            relevant_v1, relevant_v2 = self._sample_relevant_pair(target)

            values1 = self._sample_pathway_values()
            values2 = self._sample_pathway_values()
            values1[relevant] = relevant_v1
            values2[relevant] = relevant_v2

            self.targets[i] = target
            self.delays[i] = delay
            self.relevant_pathways[i] = relevant
            self.seq_lengths[i] = total_len
            self.sample1_values[i] = values1
            self.sample2_values[i] = values2
            self.global_gains[i] = global_gain
            self.pathway_gain_sample1[i] = pathway_gain_s1
            self.pathway_gain_sample2[i] = pathway_gain_s2

            cue = torch.zeros(self.cue_duration, self.context_dim)
            cue[:, relevant] = 1.0
            cue = cue + self.cue_noise_std * torch.randn_like(cue)
            if self.positive_input_encoding:
                cue = cue.clamp_min(0.0)
            self.inputs[i, :cue_end, : self.context_dim] = cue

            self.inputs[i, cue_end:s1_end, self.phase_marker_start] = 1.0
            self.inputs[i, s2_start:s2_end, self.phase_marker_start + 1] = 1.0
            self.inputs[i, response_start:total_len, self.phase_marker_start + 2] = 1.0

            for pathway_idx in range(self.n_pathways):
                template = self.pathway_templates[pathway_idx]
                gain_trace1 = self._build_gain_trace(pathway_gain_s1[pathway_idx])
                gain_trace2 = self._build_gain_trace(pathway_gain_s2[pathway_idx])

                seg1 = (
                    global_gain
                    * gain_trace1.unsqueeze(-1)
                    * values1[pathway_idx]
                    * template
                )
                seg2 = (
                    global_gain
                    * gain_trace2.unsqueeze(-1)
                    * values2[pathway_idx]
                    * template
                )
                seg1 = seg1 + self.noise_std * torch.randn_like(seg1)
                seg2 = seg2 + self.noise_std * torch.randn_like(seg2)
                if self.positive_input_encoding:
                    seg1 = seg1.clamp_min(0.0)
                    seg2 = seg2.clamp_min(0.0)

                block = self.pathway_slices[pathway_idx]
                self.inputs[i, cue_end:s1_end, block] = seg1
                self.inputs[i, s2_start:s2_end, block] = seg2

            if (
                delay >= self.distractor_burst_duration
                and torch.rand(1).item() < self.distractor_burst_prob
            ):
                burst_rel = torch.randint(0, self.n_pathways - 1, (1,)).item()
                burst_pathway = burst_rel if burst_rel < relevant else burst_rel + 1
                burst_start = torch.randint(
                    0, delay - self.distractor_burst_duration + 1, (1,)
                ).item()
                burst_end = burst_start + self.distractor_burst_duration
                delay_start = s1_end + burst_start
                delay_end = s1_end + burst_end
                burst_template = self.pathway_templates[burst_pathway][
                    : self.distractor_burst_duration
                ]
                burst_gain = self._build_gain_trace(
                    pathway_gain_s2[burst_pathway],
                    length=self.distractor_burst_duration,
                )
                burst = (
                    global_gain
                    * burst_gain.unsqueeze(-1)
                    * self.distractor_burst_scale
                    * burst_template
                )
                burst = burst + self.noise_std * torch.randn_like(burst)
                if self.positive_input_encoding:
                    burst = burst.clamp_min(0.0)
                self.inputs[
                    i, delay_start:delay_end, self.pathway_slices[burst_pathway]
                ] += burst
                self.delay_burst_flags[i] = True

    def _build_pathway_templates(self) -> torch.Tensor:
        time = torch.linspace(0.0, 1.0, self.stimulus_duration)
        templates = torch.zeros(
            self.n_pathways, self.stimulus_duration, self.pathway_dim
        )
        for pathway_idx in range(self.n_pathways):
            base_phase = 2.0 * math.pi * pathway_idx / self.n_pathways
            for feature_idx in range(self.pathway_dim):
                freq = 1.0 + 0.35 * feature_idx
                waveform = torch.sin(2.0 * math.pi * freq * time + base_phase)
                waveform = waveform + 0.35 * torch.cos(
                    2.0 * math.pi * (0.5 + 0.2 * pathway_idx) * time + 0.7 * feature_idx
                )
                if self.positive_input_encoding:
                    waveform = waveform - waveform.min()
                    waveform = waveform / waveform.max().clamp(min=1e-6)
                else:
                    waveform = waveform / waveform.std().clamp(min=1e-6)
                templates[pathway_idx, :, feature_idx] = waveform
        return templates

    def _sample_pathway_values(self) -> torch.Tensor:
        indices = torch.randint(
            0, len(self.amplitude_values), (self.n_pathways,), dtype=torch.long
        )
        return self.amplitude_values.index_select(0, indices).clone()

    def _sample_relevant_pair(self, target: int) -> tuple[float, float]:
        valid_pairs: list[tuple[float, float]] = []
        values = self.amplitude_values.tolist()
        for v1 in values:
            for v2 in values:
                if abs(v2 - v1) < self.comparison_margin:
                    continue
                if int(v2 > v1) == int(target):
                    valid_pairs.append((float(v1), float(v2)))
        if not valid_pairs:
            raise ValueError(
                "No valid amplitude pairs satisfy the requested comparison_margin"
            )
        pair_idx = torch.randint(0, len(valid_pairs), (1,)).item()
        return valid_pairs[pair_idx]

    def _build_gain_trace(self, base_gain: torch.Tensor, length: Optional[int] = None):
        length = self.stimulus_duration if length is None else int(length)
        if self.gain_drift_std <= 0.0 or length <= 1:
            return torch.full((length,), float(base_gain), dtype=torch.float32)

        log_drift = self.gain_drift_std * torch.randn(length, dtype=torch.float32)
        trace = float(base_gain) * torch.exp(torch.cumsum(log_drift, dim=0) / length)
        return trace.clamp(
            min=0.5 * self.pathway_gain_min,
            max=1.5 * self.pathway_gain_max,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths


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


class MultiSineForecastDataset(Dataset):
    """Forecast a quasi-periodic 1D signal built from multiple sinusoids.

    Each sample is a smooth mixture of frequencies drawn from a shared bank,
    optionally modulated by a slow envelope. The task is next-step prediction
    from a context window, with autonomous rollout evaluated by feeding model
    predictions back as the next observation.

    Args:
        n_samples: Number of sequences.
        seq_len: Length of the context window given to the model.
        rollout_horizon: Number of future steps reserved for autonomous rollout.
        prediction_mode: "all" for per-timestep next-step targets or "last"
            for only the next sample after the full context.
        n_components: Number of sinusoidal components in each sequence.
        frequency_bank: Frequencies (cycles over the full normalized interval).
        amplitude_min: Minimum component amplitude.
        amplitude_max: Maximum component amplitude.
        envelope_scale: Strength of a slow multiplicative modulation.
        envelope_freq_min: Minimum envelope frequency.
        envelope_freq_max: Maximum envelope frequency.
        observation_noise_std: Noise level added to observations.
        seed: RNG seed for sample generation.
        normalize_signal: Whether to z-score signals using dataset statistics.
    """

    def __init__(
        self,
        n_samples: int = 12000,
        seq_len: int = 160,
        rollout_horizon: int = 160,
        prediction_mode: str = "all",
        n_components: int = 3,
        frequency_bank: Optional[list[float]] = None,
        amplitude_min: float = 0.3,
        amplitude_max: float = 1.0,
        envelope_scale: float = 0.15,
        envelope_freq_min: float = 0.15,
        envelope_freq_max: float = 0.6,
        observation_noise_std: float = 0.02,
        seed: int = 42,
        normalize_signal: bool = True,
    ):
        super().__init__()

        if prediction_mode not in {"all", "last"}:
            raise ValueError(
                f"prediction_mode must be 'all' or 'last', got {prediction_mode}"
            )
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_noise_std = observation_noise_std
        self.state_dim = 1
        self.observation_dim = 1

        if frequency_bank is None:
            frequency_bank = [1.0, 2.5, 5.5, 9.0, 14.0]
        if len(frequency_bank) < n_components:
            raise ValueError(
                "frequency_bank must contain at least n_components entries"
            )
        self.frequency_bank = tuple(float(f) for f in frequency_bank)

        total_steps = seq_len + rollout_horizon + 1
        self.total_steps = total_steps
        time = np.linspace(0.0, 1.0, total_steps, dtype=np.float32)
        rng = np.random.default_rng(seed)

        signals = np.zeros((n_samples, total_steps), dtype=np.float32)
        for sample_idx in range(n_samples):
            chosen = rng.choice(
                np.asarray(self.frequency_bank),
                size=n_components,
                replace=False,
            )
            amplitudes = rng.uniform(amplitude_min, amplitude_max, size=n_components)
            phases = rng.uniform(0.0, 2.0 * math.pi, size=n_components)

            signal = np.zeros_like(time)
            for freq, amp, phase in zip(chosen, amplitudes, phases):
                signal += amp * np.sin(2.0 * math.pi * freq * time + phase)

            if envelope_scale > 0.0:
                env_freq = rng.uniform(envelope_freq_min, envelope_freq_max)
                env_phase = rng.uniform(0.0, 2.0 * math.pi)
                envelope = 1.0 + envelope_scale * np.sin(
                    2.0 * math.pi * env_freq * time + env_phase
                )
                signal *= envelope

            # Small smooth bias term broadens the family without making the task
            # chaotic or unstable.
            bias = rng.uniform(-0.15, 0.15)
            signal += bias
            signals[sample_idx] = signal.astype(np.float32)

        states_raw = torch.from_numpy(signals).unsqueeze(-1)  # [N, T, 1]
        if normalize_signal:
            mean = states_raw.mean(dim=(0, 1), keepdim=True)
            std = states_raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        else:
            mean = torch.zeros(1, 1, 1, dtype=states_raw.dtype)
            std = torch.ones(1, 1, 1, dtype=states_raw.dtype)

        states = (states_raw - mean) / std
        observations = states.clone()
        if observation_noise_std > 0.0:
            observations = observations + observation_noise_std * torch.randn_like(
                observations
            )

        self.state_mean = mean
        self.state_std = std
        self.full_states_raw = states_raw
        self.full_states = states
        self.full_observations = observations
        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = states[:, 1 : seq_len + 1]
        else:
            self.targets = states[:, seq_len]

    def denormalize_state_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.state_std.to(value.device) + self.state_mean.to(
            value.device
        )

    def states_to_observations(
        self, states_raw: torch.Tensor, add_noise: bool = False
    ) -> torch.Tensor:
        observations = (
            states_raw - self.state_mean.to(states_raw.device)
        ) / self.state_std.to(states_raw.device)
        if add_noise and self.observation_noise_std > 0.0:
            observations = observations + self.observation_noise_std * torch.randn_like(
                observations
            )
        return observations

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LorenzSequenceDataset(Dataset):
    """Lorenz attractor dynamics prediction (regression benchmark).

    By default this behaves like the original one-step benchmark:
    input = [seq_len, 3] state history and target = [3] next-step state.
    It also supports a stronger setting for recurrent regression:
    - partial or projected observations
    - per-timestep next-state targets for ``output_mode="all"``
    - stored future trajectory for post-hoc autonomous rollout evaluation

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
        rollout_horizon: int = 50,
        prediction_mode: str = "last",
        observation_mode: str = "full",
        observation_dim: int = 3,
        observation_noise_std: float = 0.0,
        burn_in_steps: int = 200,
        seed: int = 42,
        test_seed: Optional[int] = None,
        integration_method: str = "euler",
        normalize_states: bool = True,
        normalize_observations: bool = True,
        normalization_steps: int = 10000,
        initial_state_scale: float = 0.1,
    ):
        if prediction_mode not in {"last", "all"}:
            raise ValueError(
                f"prediction_mode must be 'last' or 'all', got '{prediction_mode}'"
            )
        if observation_mode not in {"full", "partial", "linear"}:
            raise ValueError(
                "observation_mode must be one of {'full', 'partial', 'linear'}, "
                f"got '{observation_mode}'"
            )
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if integration_method not in {"euler", "rk4"}:
            raise ValueError(
                f"integration_method must be 'euler' or 'rk4', got '{integration_method}'"
            )

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_mode = observation_mode
        self.observation_dim = 3 if observation_mode == "full" else observation_dim
        self.observation_noise_std = observation_noise_std
        self.burn_in_steps = burn_in_steps
        self.seed = seed if test_seed is None else test_seed
        self.integration_method = integration_method
        self.normalize_states = normalize_states
        self.normalize_observations = normalize_observations
        self.normalization_steps = normalization_steps
        self.initial_state_scale = initial_state_scale
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        total_future_steps = seq_len + rollout_horizon
        generator = torch.Generator().manual_seed(self.seed)

        self.observation_matrix = self._build_observation_matrix(
            mode=observation_mode,
            observation_dim=self.observation_dim,
            seed=self.seed,
        )
        self.state_mean, self.state_std = self._compute_state_stats(
            n_steps=normalization_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )
        self.obs_mean, self.obs_std = self._compute_observation_stats(
            n_steps=normalization_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )

        self.full_states_raw = self._generate_trajectories(
            n_samples=n_samples,
            total_steps=total_future_steps,
            burn_in_steps=burn_in_steps,
            generator=generator,
        )
        self.full_states = self.normalize_state_tensor(self.full_states_raw)
        observations = self.states_to_observations(
            self.full_states_raw,
            add_noise=observation_noise_std > 0.0,
            generator=generator,
        )

        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = self.full_states[:, 1 : seq_len + 1]
        else:
            self.targets = self.full_states[:, seq_len]

    def _lorenz_derivative(self, state: torch.Tensor) -> torch.Tensor:
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack((dx, dy, dz), dim=-1)

    def _integrate_step(self, state: torch.Tensor) -> torch.Tensor:
        if self.integration_method == "euler":
            return state + self.dt * self._lorenz_derivative(state)

        k1 = self._lorenz_derivative(state)
        k2 = self._lorenz_derivative(state + 0.5 * self.dt * k1)
        k3 = self._lorenz_derivative(state + 0.5 * self.dt * k2)
        k4 = self._lorenz_derivative(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _sample_initial_states(
        self, n_samples: int, generator: torch.Generator
    ) -> torch.Tensor:
        center = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        noise = (
            torch.randn(n_samples, 3, generator=generator) * self.initial_state_scale
        )
        return center.unsqueeze(0) + noise

    def _generate_trajectories(
        self,
        n_samples: int,
        total_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        states = self._sample_initial_states(n_samples=n_samples, generator=generator)
        for _ in range(burn_in_steps):
            states = self._integrate_step(states)

        trajectories = torch.zeros(n_samples, total_steps + 1, 3, dtype=torch.float32)
        trajectories[:, 0] = states
        for step_idx in range(total_steps):
            states = self._integrate_step(states)
            trajectories[:, step_idx + 1] = states
        return trajectories

    def _reference_trajectory(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        state = self._sample_initial_states(n_samples=1, generator=generator).squeeze(0)
        for _ in range(burn_in_steps):
            state = self._integrate_step(state)
        trajectory = torch.zeros(n_steps + 1, 3, dtype=torch.float32)
        trajectory[0] = state
        for step_idx in range(n_steps):
            state = self._integrate_step(state)
            trajectory[step_idx + 1] = state
        return trajectory

    def _build_observation_matrix(
        self, mode: str, observation_dim: int, seed: int
    ) -> torch.Tensor:
        if mode == "full":
            return torch.eye(3, dtype=torch.float32)

        if mode == "partial":
            if observation_dim > 3:
                raise ValueError(
                    f"partial observation_dim must be <= 3, got {observation_dim}"
                )
            basis = torch.eye(3, dtype=torch.float32)
            return basis[:, :observation_dim]

        generator = torch.Generator().manual_seed(seed + 17)
        matrix = torch.randn(3, observation_dim, generator=generator)
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return matrix.to(dtype=torch.float32)

    def _compute_state_stats(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self._reference_trajectory(
            n_steps=n_steps, burn_in_steps=burn_in_steps, generator=generator
        )
        mean = reference.mean(dim=0)
        std = reference.std(dim=0).clamp(min=1e-8)
        return mean, std

    def _compute_observation_stats(
        self,
        n_steps: int,
        burn_in_steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self._reference_trajectory(
            n_steps=n_steps, burn_in_steps=burn_in_steps, generator=generator
        )
        projected = torch.matmul(reference, self.observation_matrix)
        mean = projected.mean(dim=0)
        std = projected.std(dim=0).clamp(min=1e-8)
        return mean, std

    def normalize_state_tensor(self, states_raw: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states_raw
        return (states_raw - self.state_mean) / self.state_std

    def denormalize_state_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states
        return states * self.state_std + self.state_mean

    def normalize_observation_tensor(
        self, observations_raw: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations_raw
        return (observations_raw - self.obs_mean) / self.obs_std

    def denormalize_observation_tensor(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations
        return observations * self.obs_std + self.obs_mean

    def states_to_observations(
        self,
        states_raw: torch.Tensor,
        add_noise: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        observations_raw = torch.matmul(states_raw, self.observation_matrix)
        observations = self.normalize_observation_tensor(observations_raw)
        if add_noise and self.observation_noise_std > 0.0:
            noise = torch.randn(
                observations.shape,
                generator=generator,
                dtype=observations.dtype,
                device=observations.device,
            )
            observations = observations + self.observation_noise_std * noise
        return observations

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SwitchingLDSDataset(Dataset):
    """Partially observed switching linear dynamical system benchmark.

    The latent dynamics follow a mode-dependent linear transition:
        x_{t+1} = A_{m_t} x_t + eps_t
        y_t = C x_t + eta_t

    Modes switch only every ``min_mode_duration`` to ``max_mode_duration`` steps,
    making this a standard switching-LDS setting with a slow hidden regime and
    faster within-mode linear state evolution. The benchmark is intended to test
    whether models can simultaneously track a slow latent mode and a fast latent
    state under partial observation.
    """

    def __init__(
        self,
        n_samples: int = 12000,
        seq_len: int = 96,
        latent_dim: int = 4,
        observation_dim: int = 2,
        n_modes: int = 3,
        rollout_horizon: int = 80,
        prediction_mode: str = "all",
        observation_noise_std: float = 0.03,
        transition_noise_std: float = 0.02,
        min_mode_duration: int = 36,
        max_mode_duration: int = 96,
        burn_in_steps: int = 24,
        seed: int = 42,
        test_seed: Optional[int] = None,
        normalize_states: bool = True,
        normalize_observations: bool = True,
        normalization_samples: int = 4096,
        stats_seed: int = 1234,
        initial_state_scale: float = 0.35,
    ):
        super().__init__()
        if prediction_mode not in {"last", "all"}:
            raise ValueError(
                f"prediction_mode must be 'last' or 'all', got '{prediction_mode}'"
            )
        if latent_dim != 4:
            raise ValueError(
                f"SwitchingLDSDataset currently expects latent_dim=4, got {latent_dim}"
            )
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if n_modes < 2:
            raise ValueError(f"n_modes must be >= 2, got {n_modes}")

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.n_modes = n_modes
        self.rollout_horizon = rollout_horizon
        self.prediction_mode = prediction_mode
        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std
        self.min_mode_duration = max(1, min_mode_duration)
        self.max_mode_duration = max(self.min_mode_duration, max_mode_duration)
        self.burn_in_steps = max(0, burn_in_steps)
        self.seed = seed if test_seed is None else test_seed
        self.normalize_states = normalize_states
        self.normalize_observations = normalize_observations
        self.normalization_samples = normalization_samples
        self.stats_seed = stats_seed
        self.initial_state_scale = initial_state_scale

        self.transition_matrices = self._build_transition_matrices(n_modes)
        self.observation_matrix = self._build_observation_matrix(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            seed=stats_seed + 17,
        )

        stats_generator = torch.Generator().manual_seed(stats_seed)
        reference_states, _ = self._simulate(
            n_samples=normalization_samples,
            total_steps=seq_len + rollout_horizon,
            generator=stats_generator,
        )
        reference_observations = self._states_to_observations(
            reference_states, add_noise=False
        )
        self.state_mean = reference_states.mean(dim=(0, 1))
        self.state_std = reference_states.std(dim=(0, 1)).clamp(min=1e-8)
        self.obs_mean = reference_observations.mean(dim=(0, 1))
        self.obs_std = reference_observations.std(dim=(0, 1)).clamp(min=1e-8)

        generator = torch.Generator().manual_seed(self.seed)
        total_steps = seq_len + rollout_horizon
        self.full_states_raw, self.full_modes = self._simulate(
            n_samples=n_samples,
            total_steps=total_steps,
            generator=generator,
        )
        self.full_states = self.normalize_state_tensor(self.full_states_raw)
        observations = self.states_to_observations(
            self.full_states_raw,
            add_noise=observation_noise_std > 0.0,
            generator=generator,
        )

        self.inputs = observations[:, :seq_len]
        if prediction_mode == "all":
            self.targets = self.full_states[:, 1 : seq_len + 1]
        else:
            self.targets = self.full_states[:, seq_len]

        self.switch_mask = self.full_modes[:, 1:] != self.full_modes[:, :-1]
        self.time_since_switch = self._compute_time_since_switch(self.full_modes)

    def _rotation_block(self, radius: float, angle: float) -> torch.Tensor:
        return torch.tensor(
            [
                [radius * math.cos(angle), -radius * math.sin(angle)],
                [radius * math.sin(angle), radius * math.cos(angle)],
            ],
            dtype=torch.float32,
        )

    def _build_transition_matrices(self, n_modes: int) -> torch.Tensor:
        fast_blocks = [
            self._rotation_block(radius=0.985, angle=0.33),
            self._rotation_block(radius=0.975, angle=-0.24),
            self._rotation_block(radius=0.992, angle=0.11),
        ]
        slow_blocks = [
            torch.tensor([[0.996, 0.018], [0.0, 0.986]], dtype=torch.float32),
            torch.tensor([[0.989, -0.021], [0.012, 0.993]], dtype=torch.float32),
            torch.tensor([[0.995, 0.0], [0.024, 0.984]], dtype=torch.float32),
        ]
        slow_to_fast = [
            torch.tensor([[0.09, -0.03], [0.02, 0.05]], dtype=torch.float32),
            torch.tensor([[-0.07, 0.05], [0.06, -0.02]], dtype=torch.float32),
            torch.tensor([[0.03, 0.08], [-0.05, 0.04]], dtype=torch.float32),
        ]
        fast_to_slow = [
            torch.tensor([[0.012, 0.0], [0.0, 0.015]], dtype=torch.float32),
            torch.tensor([[0.008, -0.006], [0.005, 0.01]], dtype=torch.float32),
            torch.tensor([[0.004, 0.009], [-0.008, 0.006]], dtype=torch.float32),
        ]

        matrices = []
        for mode_idx in range(n_modes):
            fast = fast_blocks[mode_idx % len(fast_blocks)]
            slow = slow_blocks[mode_idx % len(slow_blocks)]
            sf = slow_to_fast[mode_idx % len(slow_to_fast)]
            fs = fast_to_slow[mode_idx % len(fast_to_slow)]
            top = torch.cat((fast, sf), dim=1)
            bottom = torch.cat((fs, slow), dim=1)
            matrix = torch.cat((top, bottom), dim=0)
            eigvals = torch.linalg.eigvals(matrix)
            spectral_radius = eigvals.abs().max().real.item()
            if spectral_radius >= 0.998:
                matrix = matrix * (0.998 / spectral_radius)
            matrices.append(matrix)
        return torch.stack(matrices, dim=0)

    def _build_observation_matrix(
        self, latent_dim: int, observation_dim: int, seed: int
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(latent_dim, observation_dim, generator=generator)
        matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return matrix.to(dtype=torch.float32)

    def _sample_initial_states(
        self, n_samples: int, generator: torch.Generator
    ) -> torch.Tensor:
        return (
            torch.randn(n_samples, self.latent_dim, generator=generator)
            * self.initial_state_scale
        )

    def _generate_mode_sequences(
        self, n_samples: int, total_steps: int, generator: torch.Generator
    ) -> torch.Tensor:
        modes = torch.zeros(n_samples, total_steps + 1, dtype=torch.long)
        for sample_idx in range(n_samples):
            current_mode = torch.randint(
                0, self.n_modes, (1,), generator=generator
            ).item()
            t = 0
            while t <= total_steps:
                block = torch.randint(
                    self.min_mode_duration,
                    self.max_mode_duration + 1,
                    (1,),
                    generator=generator,
                ).item()
                end = min(total_steps + 1, t + block)
                modes[sample_idx, t:end] = current_mode
                if end >= total_steps + 1:
                    break
                next_mode = torch.randint(
                    0, self.n_modes - 1, (1,), generator=generator
                ).item()
                current_mode = next_mode if next_mode < current_mode else next_mode + 1
                t = end
        return modes

    def _simulate(
        self, n_samples: int, total_steps: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode_sequences = self._generate_mode_sequences(
            n_samples=n_samples,
            total_steps=total_steps + self.burn_in_steps,
            generator=generator,
        )

        states = self._sample_initial_states(n_samples=n_samples, generator=generator)
        trajectories = torch.zeros(
            n_samples,
            total_steps + self.burn_in_steps + 1,
            self.latent_dim,
            dtype=torch.float32,
        )
        trajectories[:, 0] = states

        for step_idx in range(total_steps + self.burn_in_steps):
            active_modes = mode_sequences[:, step_idx]
            transition = self.transition_matrices[active_modes]
            process_noise = (
                torch.randn(n_samples, self.latent_dim, generator=generator)
                * self.transition_noise_std
            )
            states = torch.einsum("nij,nj->ni", transition, states) + process_noise
            trajectories[:, step_idx + 1] = states

        start = self.burn_in_steps
        end = start + total_steps + 1
        return trajectories[:, start:end], mode_sequences[:, start:end]

    def _states_to_observations(
        self,
        states_raw: torch.Tensor,
        *,
        add_noise: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        observations = states_raw @ self.observation_matrix
        if add_noise and self.observation_noise_std > 0.0:
            noise = torch.randn(
                observations.shape, generator=generator, dtype=observations.dtype
            )
            observations = observations + self.observation_noise_std * noise
        return observations

    def _compute_time_since_switch(self, modes: torch.Tensor) -> torch.Tensor:
        steps = torch.zeros_like(modes, dtype=torch.long)
        for sample_idx in range(modes.shape[0]):
            current = 0
            for step_idx in range(1, modes.shape[1]):
                if modes[sample_idx, step_idx] != modes[sample_idx, step_idx - 1]:
                    current = 0
                else:
                    current += 1
                steps[sample_idx, step_idx] = current
        return steps

    def normalize_state_tensor(self, states_raw: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states_raw
        return (states_raw - self.state_mean) / self.state_std

    def denormalize_state_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states
        return states * self.state_std + self.state_mean

    def normalize_observation_tensor(
        self, observations_raw: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations_raw
        return (observations_raw - self.obs_mean) / self.obs_std

    def denormalize_observation_tensor(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        if not self.normalize_observations:
            return observations
        return observations * self.obs_std + self.obs_mean

    def states_to_observations(
        self,
        states_raw: torch.Tensor,
        *,
        add_noise: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        observations_raw = self._states_to_observations(
            states_raw, add_noise=add_noise, generator=generator
        )
        return self.normalize_observation_tensor(observations_raw)

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
        split: "train", "test", or "all".
        train_fraction: Fraction of trials for training (rest = test).
        normalize: Z-score normalize each neuron's responses.
        seed: Random seed for reproducible train/test split.
        session_idx: Index into all_running.npy for this session (default 20 = GT3).
        running_state: Keep all trials, or only the low/high running subset.
        running_quantile: Quantile used to define low/high states. `0.5` gives
            a median split, `1/3` gives lower/upper tertiles.
        gain_group_count: Number of pathway-specific multiplicative gain factors.
        gain_grouping: How neurons are grouped for pathway-specific gains.
            Supported: "preferred_class", "random", "contiguous".
        global_gain_min/global_gain_max: Range for a per-trial global positive gain.
        group_gain_min/group_gain_max: Range for per-group positive gains.
        gain_seed: Seed for the deterministic gain augmentation.
        poisson_sampling: If True, apply Poisson shot noise after multiplicative gain.
        poisson_rate_scale: Scale factor before Poisson sampling.
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
        running_state: str = "all",
        running_quantile: float = 0.5,
        gain_group_count: int = 1,
        gain_grouping: str = "preferred_class",
        global_gain_min: float = 1.0,
        global_gain_max: float = 1.0,
        group_gain_min: float = 1.0,
        group_gain_max: float = 1.0,
        gain_seed: Optional[int] = None,
        poisson_sampling: bool = False,
        poisson_rate_scale: float = 1.0,
    ):
        super().__init__()
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be 'train', 'test', or 'all', got '{split}'")
        if running_state not in ("all", "low", "high"):
            raise ValueError(
                f"running_state must be 'all', 'low', or 'high', got '{running_state}'"
            )
        if not (0.0 < running_quantile <= 0.5):
            raise ValueError(
                f"running_quantile must be in (0, 0.5], got {running_quantile}"
            )
        if gain_group_count < 1:
            raise ValueError(f"gain_group_count must be >= 1, got {gain_group_count}")
        if poisson_rate_scale <= 0:
            raise ValueError(
                f"poisson_rate_scale must be > 0, got {poisson_rate_scale}"
            )

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
            running_speed_full = running.astype(np.float32)
        else:
            logger.warning(
                f"Running speed file not found at {running_path}, "
                f"setting all running speeds to 0"
            )
            running_speed_full = np.zeros(n_trials, dtype=np.float32)

        # --- Select top neurons by variance ---
        neuron_var = sresp.var(axis=1)  # variance across trials per neuron
        n_select = min(n_neurons, n_neurons_total)
        top_idx = np.argsort(neuron_var)[-n_select:][::-1]  # descending
        self._neuron_indices = top_idx
        raw_responses = sresp[top_idx, :].T.astype(np.float32)  # (n_trials, n_select)

        # --- Normalize ---
        responses = raw_responses.copy()
        if normalize:
            mu = responses.mean(axis=0, keepdims=True)
            std = responses.std(axis=0, keepdims=True) + 1e-8
            responses = (responses - mu) / std

        # --- Bin orientations into classes ---
        # istim is in [0, 2*pi), bin into n_classes equal bins
        bin_edges = np.linspace(0, 2 * np.pi, n_classes + 1)
        labels = np.digitize(istim, bin_edges) - 1
        labels = np.clip(labels, 0, n_classes - 1)  # handle edge case at 2*pi
        full_labels = labels.copy()

        if normalize and (
            gain_group_count > 1
            or global_gain_min != 1.0
            or global_gain_max != 1.0
            or group_gain_min != 1.0
            or group_gain_max != 1.0
            or poisson_sampling
        ):
            raise ValueError(
                "Stringer gain augmentation requires normalize=False so the inputs "
                "remain nonnegative and rate-like."
            )

        # --- Running-state filter ---
        low_thresh = np.quantile(running_speed_full, running_quantile)
        high_thresh = np.quantile(running_speed_full, 1.0 - running_quantile)
        if running_state == "low":
            keep_mask = running_speed_full <= low_thresh
        elif running_state == "high":
            keep_mask = running_speed_full >= high_thresh
        else:
            keep_mask = np.ones(n_trials, dtype=bool)

        available_idx = np.flatnonzero(keep_mask)
        responses = responses[available_idx]
        labels = labels[available_idx]
        running_speed = running_speed_full[available_idx]

        # --- Train/test split ---
        if split == "all":
            idx = np.arange(len(labels))
        else:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(labels))
            n_train = int(len(labels) * train_fraction)
            if split == "train":
                idx = perm[:n_train]
            else:
                idx = perm[n_train:]

        selected_global_idx = available_idx[idx]
        selected_responses = responses[idx]
        selected_labels = labels[idx]
        selected_running = running_speed[idx]

        gain_groups = self._build_gain_groups(
            raw_responses=raw_responses,
            full_labels=full_labels,
            n_classes=n_classes,
            gain_group_count=gain_group_count,
            gain_grouping=gain_grouping,
            seed=(gain_seed if gain_seed is not None else seed),
        )
        selected_responses = self._apply_gain_augmentation(
            responses=selected_responses,
            sample_indices=selected_global_idx,
            gain_groups=gain_groups,
            gain_group_count=gain_group_count,
            global_gain_min=global_gain_min,
            global_gain_max=global_gain_max,
            group_gain_min=group_gain_min,
            group_gain_max=group_gain_max,
            gain_seed=(gain_seed if gain_seed is not None else seed),
            poisson_sampling=poisson_sampling,
            poisson_rate_scale=poisson_rate_scale,
        )

        self.responses = torch.tensor(selected_responses, dtype=torch.float32)
        self.labels = torch.tensor(selected_labels, dtype=torch.long)
        self._running_speed = selected_running.astype(np.float32)
        self._running_mask = torch.tensor(
            self._running_speed > np.median(running_speed_full),
            dtype=torch.bool,
        )
        self._split_indices = selected_global_idx

        logger.info(
            f"StringerV1Dataset({split}, state={running_state}): {len(self)} trials, "
            f"{n_select} neurons, {n_classes} classes"
        )

    @staticmethod
    def _build_gain_groups(
        raw_responses: np.ndarray,
        full_labels: np.ndarray,
        n_classes: int,
        gain_group_count: int,
        gain_grouping: str,
        seed: int,
    ) -> np.ndarray:
        n_neurons = raw_responses.shape[1]
        if gain_group_count <= 1:
            return np.zeros(n_neurons, dtype=np.int64)

        if gain_grouping == "random":
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_neurons)
            group_ids = np.zeros(n_neurons, dtype=np.int64)
            group_ids[perm] = np.arange(n_neurons) % gain_group_count
            return group_ids

        if gain_grouping == "contiguous":
            return np.floor(
                np.arange(n_neurons, dtype=np.float32) * gain_group_count / n_neurons
            ).astype(np.int64)

        if gain_grouping != "preferred_class":
            raise ValueError(
                f"Unsupported gain_grouping={gain_grouping!r}. "
                "Use 'preferred_class', 'random', or 'contiguous'."
            )

        class_means = np.zeros((n_classes, n_neurons), dtype=np.float32)
        for c in range(n_classes):
            mask = full_labels == c
            if not np.any(mask):
                continue
            class_means[c] = raw_responses[mask].mean(axis=0)

        preferred_class = class_means.argmax(axis=0)
        return np.floor(preferred_class * gain_group_count / n_classes).astype(np.int64)

    @staticmethod
    def _apply_gain_augmentation(
        responses: np.ndarray,
        sample_indices: np.ndarray,
        gain_groups: np.ndarray,
        gain_group_count: int,
        global_gain_min: float,
        global_gain_max: float,
        group_gain_min: float,
        group_gain_max: float,
        gain_seed: int,
        poisson_sampling: bool,
        poisson_rate_scale: float,
    ) -> np.ndarray:
        if (
            gain_group_count <= 1
            and global_gain_min == 1.0
            and global_gain_max == 1.0
            and group_gain_min == 1.0
            and group_gain_max == 1.0
            and not poisson_sampling
        ):
            return responses

        if min(global_gain_min, global_gain_max, group_gain_min, group_gain_max) <= 0:
            raise ValueError("All gain ranges must be strictly positive.")

        rng = np.random.RandomState(gain_seed)
        max_trial_idx = int(sample_indices.max()) + 1
        global_gains = rng.uniform(global_gain_min, global_gain_max, size=max_trial_idx)
        group_gains = rng.uniform(
            group_gain_min,
            group_gain_max,
            size=(max_trial_idx, gain_group_count),
        )

        neuron_gains = group_gains[sample_indices][:, gain_groups]
        total_gain = global_gains[sample_indices, None] * neuron_gains
        augmented = np.clip(
            responses * total_gain.astype(np.float32), a_min=0.0, a_max=None
        )

        if poisson_sampling:
            rate = np.clip(augmented * poisson_rate_scale, a_min=0.0, a_max=None)
            augmented = rng.poisson(rate).astype(np.float32) / float(poisson_rate_scale)

        return augmented.astype(np.float32)

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


class HierarchicalTemporalDataset(Dataset):
    """Context-gated multi-timescale classification task.

    Tests whether a model can jointly perform:
    1. **Slow memory** -- maintain a context cue over the full sequence (τ~125)
    2. **Medium integration** -- identify block-level patterns (τ~25)
    3. **Fast filtering** -- denoise individual timesteps (τ~5)
    4. **Multiplicative gating** -- context determines which blocks matter

    Structure per sample:
    - *Context phase* (``context_duration`` steps): present one of ``n_contexts``
      context cues as a one-hot vector.
    - *Integration phase* (``n_blocks x block_duration`` steps): sequence of
      blocks.  Each block has a type drawn uniformly from ``n_block_types``.
      The per-timestep input is the block's prototype vector + Gaussian noise.
    - Classification: each context ``c`` maps to a target block type ``t_c``.
      Class is determined by whether the target type count exceeds the
      expected count (``n_blocks / n_block_types``).

    Samples are generated with balanced classes (50/50) by first choosing
    the target class then constructing a block sequence consistent with it.

    The task is designed so that DendriNet's geometric tau spacing
    (125, 25, 5) aligns with the three timescales of the task.

    Args:
        n_samples: Number of samples.
        n_contexts: Number of context cues.
        n_block_types: Number of distinct block types.
        n_blocks: Number of blocks in the integration phase.
        context_duration: Steps for which the context cue is shown.
        block_duration: Steps per block.
        signal_dim: Dimensionality of the input signal.
        noise_std: Standard deviation of fast Gaussian noise.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_contexts: int = 2,
        n_block_types: int = 4,
        n_blocks: int = 8,
        context_duration: int = 20,
        block_duration: int = 25,
        signal_dim: int = 16,
        noise_std: float = 0.5,
    ):
        self.n_samples = n_samples
        self.n_contexts = n_contexts
        self.n_block_types = n_block_types
        self.n_blocks = n_blocks
        self.context_duration = context_duration
        self.block_duration = block_duration
        self.signal_dim = signal_dim
        self.noise_std = noise_std
        self.total_len = context_duration + n_blocks * block_duration
        self.input_dim = signal_dim

        # Build fixed block prototypes: [n_block_types, signal_dim]
        rng = np.random.RandomState(0)
        prototypes = rng.randn(n_block_types, signal_dim).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.prototypes = torch.from_numpy(prototypes)

        # Build context cue vectors: [n_contexts, signal_dim]
        ctx_vecs = rng.randn(n_contexts, signal_dim).astype(np.float32)
        ctx_vecs /= np.linalg.norm(ctx_vecs, axis=1, keepdims=True)
        self.context_vectors = torch.from_numpy(ctx_vecs)

        # Map each context to a target block type
        self.context_to_target = torch.arange(n_contexts) % n_block_types
        # Threshold: target type must appear more than expected by chance
        self.threshold = n_blocks // n_block_types + 1

        # Generate all samples with balanced classes
        self.inputs = torch.zeros(n_samples, self.total_len, signal_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, dtype=torch.long)
        self.block_types = torch.zeros(n_samples, n_blocks, dtype=torch.long)

        for i in range(n_samples):
            ctx = torch.randint(0, n_contexts, (1,)).item()
            self.contexts[i] = ctx
            target_type = self.context_to_target[ctx].item()

            # Decide class first (balanced 50/50)
            label = i % 2
            self.targets[i] = label

            # Build block sequence consistent with the label
            block_seq = self._sample_block_seq(
                target_type, label, n_blocks, n_block_types, self.threshold
            )
            self.block_types[i] = block_seq

            # Context phase
            for t in range(context_duration):
                self.inputs[i, t] = self.context_vectors[ctx]

            # Integration phase
            for b in range(n_blocks):
                bt = block_seq[b].item()
                t_start = context_duration + b * block_duration
                for t in range(t_start, t_start + block_duration):
                    self.inputs[i, t] = self.prototypes[bt]

            # Add fast noise to entire sequence
            self.inputs[i] += torch.randn(self.total_len, signal_dim) * noise_std

    @staticmethod
    def _sample_block_seq(
        target_type: int,
        label: int,
        n_blocks: int,
        n_block_types: int,
        threshold: int,
    ) -> torch.Tensor:
        """Sample a block sequence with the correct number of target blocks."""
        non_target_types = [t for t in range(n_block_types) if t != target_type]
        for _ in range(1000):
            if label == 1:
                # Target type must appear >= threshold times
                n_target = torch.randint(threshold, n_blocks + 1, (1,)).item()
            else:
                # Target type must appear < threshold times
                n_target = torch.randint(0, threshold, (1,)).item()
            seq = torch.full((n_blocks,), target_type, dtype=torch.long)
            # Fill remaining positions with non-target types
            other_indices = torch.randperm(n_blocks)[n_target:]
            for idx in other_indices:
                seq[idx] = non_target_types[
                    torch.randint(0, len(non_target_types), (1,)).item()
                ]
            # Shuffle
            seq = seq[torch.randperm(n_blocks)]
            return seq
        # Fallback
        return torch.randint(0, n_block_types, (n_blocks,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ContextDependentDecisionDataset(Dataset):
    """Context-dependent integration task inspired by Mante et al. (2013).

    Two sensory channels (e.g. "color" and "motion") are presented simultaneously
    as noisy streams.  A brief context cue at sequence onset indicates which
    channel is task-relevant.  The model must integrate the relevant channel
    over a variable delay and report the sign of the integrated evidence.

    This tests:
    1. **Context gating** -- selectively route one channel through dendritic
       pathways while ignoring the other.
    2. **Noisy integration** -- accumulate evidence over time using slow traces.
    3. **Interference resistance** -- the irrelevant channel provides a strong
       distractor that must be suppressed.

    Input structure per timestep:
        [context_cue (2d) | channel_A (signal_dim) | channel_B (signal_dim)]
    Total input_dim = 2 + 2 * signal_dim

    Target: binary (0/1) -- sign of integrated evidence in the relevant channel.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        signal_dim: int = 8,
        context_duration: int = 10,
        integration_duration: int = 50,
        coherence_levels: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
        noise_std: float = 1.0,
        variable_delay: bool = True,
        min_integration: int = 20,
        max_integration: int = 80,
        delay_duration: int = 0,
        min_delay: Optional[int] = None,
        max_delay: Optional[int] = None,
        conflict_fraction: float = 0.0,
        distractor_burst_prob: float = 0.0,
        distractor_burst_scale: float = 2.0,
        distractor_burst_duration: int = 5,
        cue_dropout_prob: float = 0.0,
        cue_keep_duration: int = 1,
        late_context_switch_prob: float = 0.0,
        context_signal_scale: float = 1.0,
        return_seq_lengths: bool = False,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.signal_dim = signal_dim
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 2 + (4 if self.positive_input_encoding else 2) * signal_dim
        self.context_duration = context_duration
        self.integration_duration = integration_duration
        self.variable_integration = bool(variable_delay)
        self.min_integration = min_integration
        self.max_integration = max_integration
        self.delay_duration = delay_duration
        self.min_delay = delay_duration if min_delay is None else min_delay
        self.max_delay = delay_duration if max_delay is None else max_delay
        self.return_seq_lengths = bool(
            return_seq_lengths
            or self.variable_integration
            or self.min_delay != self.max_delay
        )

        max_integration_len = (
            max_integration if self.variable_integration else integration_duration
        )
        max_delay_len = self.max_delay
        self.seq_len = context_duration + max_integration_len + max_delay_len

        self.inputs = torch.zeros(n_samples, self.seq_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.seq_lengths = torch.full((n_samples,), self.seq_len, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, dtype=torch.long)
        self.final_contexts = torch.zeros(n_samples, dtype=torch.long)
        self.integration_lengths = torch.zeros(n_samples, dtype=torch.long)
        self.delay_lengths = torch.zeros(n_samples, dtype=torch.long)
        self.switch_steps = torch.full((n_samples,), -1, dtype=torch.long)
        self.conflict_flags = torch.zeros(n_samples, dtype=torch.bool)
        self.coherence_a = torch.zeros(n_samples, dtype=torch.float32)
        self.coherence_b = torch.zeros(n_samples, dtype=torch.float32)
        self.sign_a = torch.zeros(n_samples, dtype=torch.float32)
        self.sign_b = torch.zeros(n_samples, dtype=torch.float32)
        self.relevant_scores = torch.zeros(n_samples, dtype=torch.float32)

        coherence_tensor = torch.tensor(coherence_levels, dtype=torch.float32)
        cue_keep_duration = max(1, min(cue_keep_duration, context_duration))
        distractor_burst_duration = max(1, distractor_burst_duration)

        for i in range(n_samples):
            initial_context = torch.randint(0, 2, (1,)).item()
            integration_len = (
                torch.randint(min_integration, max_integration + 1, (1,)).item()
                if self.variable_integration
                else integration_duration
            )
            delay_len = (
                torch.randint(self.min_delay, self.max_delay + 1, (1,)).item()
                if self.max_delay > self.min_delay
                else self.max_delay
            )
            seq_len = context_duration + integration_len + delay_len

            conflict = torch.rand(1).item() < conflict_fraction
            switch_trial = (
                integration_len >= 4 and torch.rand(1).item() < late_context_switch_prob
            )
            switch_step = -1
            if switch_trial:
                switch_low = max(1, integration_len // 3)
                switch_high = max(switch_low + 1, integration_len - 1)
                switch_step = torch.randint(switch_low, switch_high, (1,)).item()

            sign_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            if conflict:
                sign_b = -sign_a
            else:
                sign_b = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            coh_a = coherence_tensor[
                torch.randint(0, len(coherence_levels), (1,))
            ].item()
            coh_b = coherence_tensor[
                torch.randint(0, len(coherence_levels), (1,))
            ].item()

            self.contexts[i] = initial_context
            self.final_contexts[i] = (
                1 - initial_context if switch_step >= 0 else initial_context
            )
            self.integration_lengths[i] = integration_len
            self.delay_lengths[i] = delay_len
            self.seq_lengths[i] = seq_len
            self.switch_steps[i] = switch_step
            self.conflict_flags[i] = conflict
            self.coherence_a[i] = coh_a
            self.coherence_b[i] = coh_b
            self.sign_a[i] = sign_a
            self.sign_b[i] = sign_b

            cue = torch.zeros(context_duration, 2)
            cue[:, initial_context] = context_signal_scale
            if torch.rand(1).item() < cue_dropout_prob:
                cue[cue_keep_duration:] = 0.0
            self.inputs[i, :context_duration, :2] = cue

            burst_start = None
            if (
                integration_len >= distractor_burst_duration
                and torch.rand(1).item() < distractor_burst_prob
            ):
                burst_start = torch.randint(
                    0,
                    integration_len - distractor_burst_duration + 1,
                    (1,),
                ).item()

            relevant_score = 0.0
            for t in range(integration_len):
                active_context = initial_context
                if switch_step >= 0 and t >= switch_step:
                    active_context = 1 - initial_context
                    abs_t = context_duration + t
                    self.inputs[i, abs_t, :2] = 0.0
                    self.inputs[i, abs_t, active_context] = context_signal_scale

                a_signal = sign_a * coh_a + noise_std * torch.randn(signal_dim)
                b_signal = sign_b * coh_b + noise_std * torch.randn(signal_dim)

                if (
                    burst_start is not None
                    and burst_start <= t < burst_start + distractor_burst_duration
                ):
                    burst_sign = -sign_a if active_context == 0 else -sign_b
                    burst = distractor_burst_scale * max(coh_a, coh_b) * burst_sign
                    if active_context == 0:
                        b_signal = b_signal + burst
                    else:
                        a_signal = a_signal + burst

                abs_t = context_duration + t
                if self.positive_input_encoding:
                    a_pos = torch.relu(a_signal)
                    a_neg = torch.relu(-a_signal)
                    b_pos = torch.relu(b_signal)
                    b_neg = torch.relu(-b_signal)
                    self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_pos
                    self.inputs[i, abs_t, 2 + signal_dim : 2 + 2 * signal_dim] = a_neg
                    self.inputs[
                        i,
                        abs_t,
                        2 + 2 * signal_dim : 2 + 3 * signal_dim,
                    ] = b_pos
                    self.inputs[i, abs_t, 2 + 3 * signal_dim :] = b_neg
                else:
                    self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_signal
                    self.inputs[i, abs_t, 2 + signal_dim :] = b_signal

                if active_context == 0:
                    relevant_score += sign_a * coh_a
                else:
                    relevant_score += sign_b * coh_b

            self.relevant_scores[i] = relevant_score
            self.targets[i] = int(relevant_score > 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.return_seq_lengths:
            seq_lengths = self.seq_lengths[idx]
            if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
                seq_lengths = int(seq_lengths.item())
        else:
            seq_lengths = None
        return self.inputs[idx], self.targets[idx], seq_lengths


class SwitchingContextDecisionDataset(Dataset):
    """Continuous evidence integration under slowly switching context.

    This task is designed to separate models that can simultaneously maintain
    a slow latent context and perform fast evidence integration. Each sample is
    a fixed-length sequence partitioned into long context blocks. Within a
    block, the relevant context stays constant while the local evidence target
    changes every ``trial_duration`` steps. The model must emit the correct
    binary decision at every timestep.

    Input structure per timestep:
        [context_hint (2d) | channel_A (signal_dim) | channel_B (signal_dim)]

    Target per timestep:
        binary decision (0/1) given by the sign of the context-relevant
        evidence segment active at that timestep.
    """

    def __init__(
        self,
        n_samples: int = 16000,
        signal_dim: int = 8,
        seq_len: int = 360,
        trial_duration: int = 12,
        min_context_duration: int = 90,
        max_context_duration: int = 150,
        coherence_levels: tuple[float, ...] = (0.03, 0.06, 0.12, 0.24),
        noise_std: float = 1.1,
        conflict_fraction: float = 0.75,
        distractor_burst_prob: float = 0.35,
        distractor_burst_scale: float = 2.0,
        distractor_burst_duration: int = 3,
        context_cue_duration: int = 10,
        switch_cue_duration: int = 4,
        context_signal_scale: float = 1.0,
        switch_signal_scale: float = 0.35,
        min_blocks: int = 2,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.signal_dim = signal_dim
        self.seq_len = seq_len
        self.trial_duration = max(1, trial_duration)
        self.min_context_duration = max(self.trial_duration, min_context_duration)
        self.max_context_duration = max(self.min_context_duration, max_context_duration)
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = 2 + (4 if self.positive_input_encoding else 2) * signal_dim

        coherence_tensor = torch.tensor(coherence_levels, dtype=torch.float32)
        context_cue_duration = max(1, context_cue_duration)
        switch_cue_duration = max(1, switch_cue_duration)
        distractor_burst_duration = max(1, distractor_burst_duration)
        min_blocks = max(2, min_blocks)

        self.inputs = torch.zeros(n_samples, seq_len, self.input_dim)
        self.targets = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.contexts = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.time_since_switch = torch.zeros(n_samples, seq_len, dtype=torch.long)
        self.switch_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.cue_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.conflict_mask = torch.zeros(n_samples, seq_len, dtype=torch.bool)
        self.relevant_signal = torch.zeros(n_samples, seq_len, dtype=torch.float32)
        self.irrelevant_signal = torch.zeros(n_samples, seq_len, dtype=torch.float32)

        for i in range(n_samples):
            block_lengths = []
            remaining = seq_len
            while remaining > 0:
                min_block = min(self.min_context_duration, remaining)
                max_block = min(self.max_context_duration, remaining)
                if len(block_lengths) + 1 < min_blocks:
                    # Reserve enough room for the remaining required blocks.
                    reserve = self.min_context_duration * (
                        min_blocks - len(block_lengths) - 1
                    )
                    max_block = max(min_block, min(max_block, remaining - reserve))
                block_len = (
                    torch.randint(min_block, max_block + 1, (1,)).item()
                    if max_block > min_block
                    else min_block
                )
                block_lengths.append(block_len)
                remaining -= block_len
            if len(block_lengths) < min_blocks:
                # Fallback to equal partitions when clipping produced too few blocks.
                base = seq_len // min_blocks
                block_lengths = [base] * min_blocks
                block_lengths[-1] += seq_len - sum(block_lengths)

            current_context = torch.randint(0, 2, (1,)).item()
            t_start = 0

            for block_idx, block_len in enumerate(block_lengths):
                t_end = min(seq_len, t_start + block_len)
                cue_duration = min(
                    context_cue_duration if block_idx == 0 else switch_cue_duration,
                    t_end - t_start,
                )
                cue_scale = (
                    context_signal_scale if block_idx == 0 else switch_signal_scale
                )

                self.contexts[i, t_start:t_end] = current_context
                self.time_since_switch[i, t_start:t_end] = torch.arange(
                    0, t_end - t_start, dtype=torch.long
                )
                self.switch_mask[i, t_start] = True
                self.cue_mask[i, t_start : t_start + cue_duration] = True
                self.inputs[i, t_start : t_start + cue_duration, current_context] = (
                    cue_scale
                )

                trial_t = t_start
                while trial_t < t_end:
                    seg_end = min(t_end, trial_t + self.trial_duration)
                    conflict = torch.rand(1).item() < conflict_fraction
                    sign_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    sign_b = (
                        -sign_a
                        if conflict
                        else (1.0 if torch.rand(1).item() > 0.5 else -1.0)
                    )
                    coh_a = coherence_tensor[
                        torch.randint(0, len(coherence_tensor), (1,))
                    ].item()
                    coh_b = coherence_tensor[
                        torch.randint(0, len(coherence_tensor), (1,))
                    ].item()

                    burst_start = None
                    seg_len = seg_end - trial_t
                    if (
                        seg_len >= distractor_burst_duration
                        and torch.rand(1).item() < distractor_burst_prob
                    ):
                        burst_start = torch.randint(
                            0, seg_len - distractor_burst_duration + 1, (1,)
                        ).item()

                    for local_t, abs_t in enumerate(range(trial_t, seg_end)):
                        a_signal = sign_a * coh_a + noise_std * torch.randn(signal_dim)
                        b_signal = sign_b * coh_b + noise_std * torch.randn(signal_dim)

                        if (
                            burst_start is not None
                            and burst_start
                            <= local_t
                            < burst_start + distractor_burst_duration
                        ):
                            burst_sign = -sign_a if current_context == 0 else -sign_b
                            burst = (
                                distractor_burst_scale * max(coh_a, coh_b) * burst_sign
                            )
                            if current_context == 0:
                                b_signal = b_signal + burst
                            else:
                                a_signal = a_signal + burst

                        if self.positive_input_encoding:
                            a_pos = torch.relu(a_signal)
                            a_neg = torch.relu(-a_signal)
                            b_pos = torch.relu(b_signal)
                            b_neg = torch.relu(-b_signal)
                            self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_pos
                            self.inputs[
                                i, abs_t, 2 + signal_dim : 2 + 2 * signal_dim
                            ] = a_neg
                            self.inputs[
                                i, abs_t, 2 + 2 * signal_dim : 2 + 3 * signal_dim
                            ] = b_pos
                            self.inputs[i, abs_t, 2 + 3 * signal_dim :] = b_neg
                        else:
                            self.inputs[i, abs_t, 2 : 2 + signal_dim] = a_signal
                            self.inputs[i, abs_t, 2 + signal_dim :] = b_signal
                        self.conflict_mask[i, abs_t] = conflict

                        if current_context == 0:
                            self.targets[i, abs_t] = int(sign_a > 0)
                            self.relevant_signal[i, abs_t] = sign_a * coh_a
                            self.irrelevant_signal[i, abs_t] = sign_b * coh_b
                        else:
                            self.targets[i, abs_t] = int(sign_b > 0)
                            self.relevant_signal[i, abs_t] = sign_b * coh_b
                            self.irrelevant_signal[i, abs_t] = sign_a * coh_a

                    trial_t = seg_end

                t_start = t_end
                current_context = 1 - current_context

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TimescaleGeneralizationDataset(Dataset):
    """Train on one temporal regime, test on another.

    A simple delayed match-to-sample task where the delay duration is drawn
    from a specified range.  By creating separate train/test datasets with
    non-overlapping delay ranges, we test timescale *generalization* rather
    than interpolation.

    This directly tests whether learnable tau / hierarchical timescales
    provide compositional temporal generalization.

    Input structure:
        [stimulus phase | delay (zeros) | probe phase]
    Target: binary match/non-match.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        stimulus_dim: int = 16,
        stimulus_duration: int = 10,
        min_delay: int = 10,
        max_delay: int = 100,
        probe_duration: int = 10,
        positive_input_encoding: bool = False,
        noise_std: float = 0.3,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.positive_input_encoding = bool(positive_input_encoding)
        max_seq_len = stimulus_duration + max_delay + probe_duration
        self.max_seq_len = max_seq_len

        inputs = torch.zeros(n_samples, max_seq_len, stimulus_dim)
        seq_lengths = torch.zeros(n_samples, dtype=torch.long)

        delays = torch.randint(min_delay, max_delay + 1, (n_samples,))
        matches = torch.randint(0, 2, (n_samples,))

        # Generate all templates and probes at once
        if self.positive_input_encoding:
            templates = torch.rand(n_samples, stimulus_dim)
            templates = (
                templates / templates.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
            )
            alt_probes = torch.rand(n_samples, stimulus_dim)
            alt_probes = (
                alt_probes / alt_probes.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0
            )
        else:
            templates = torch.randn(n_samples, stimulus_dim)
            templates = templates / templates.norm(dim=1, keepdim=True) * 2.0
            alt_probes = torch.randn(n_samples, stimulus_dim)
            alt_probes = alt_probes / alt_probes.norm(dim=1, keepdim=True) * 2.0
        probes = torch.where(matches.unsqueeze(1).bool(), templates, alt_probes)

        # Stimulus noise and probe noise
        stim_noise = noise_std * torch.randn(n_samples, stimulus_duration, stimulus_dim)
        probe_noise = noise_std * torch.randn(n_samples, probe_duration, stimulus_dim)

        # Fill stimulus phase (same for all samples)
        inputs[:, :stimulus_duration] = templates.unsqueeze(1) + stim_noise
        if self.positive_input_encoding:
            inputs[:, :stimulus_duration].clamp_(min=0.0)

        # Fill probe phase (varies by delay)
        for i in range(n_samples):
            d = delays[i].item()
            total_len = stimulus_duration + d + probe_duration
            probe_start = stimulus_duration + d
            inputs[i, probe_start:total_len] = probes[i].unsqueeze(0) + probe_noise[i]
            if self.positive_input_encoding:
                inputs[i, probe_start:total_len].clamp_(min=0.0)
            seq_lengths[i] = total_len

        targets = matches
        self.inputs = inputs
        self.targets = targets
        self.seq_lengths = seq_lengths

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.seq_lengths[idx]


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
    elif dataset_name == "hierarchical_temporal":
        n_train = kwargs.get("n_samples", 20000)
        common = {
            "n_contexts": kwargs.get("n_contexts", 2),
            "n_block_types": kwargs.get("n_block_types", 4),
            "n_blocks": kwargs.get("n_blocks", 8),
            "context_duration": kwargs.get("context_duration", 20),
            "block_duration": kwargs.get("block_duration", 25),
            "signal_dim": kwargs.get("signal_dim", 16),
            "noise_std": kwargs.get("noise_std", 0.5),
        }
        train_full = HierarchicalTemporalDataset(n_samples=n_train, **common)
        test_ds = HierarchicalTemporalDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "context_dependent_decision",
        "context_dependent_decision_hard",
        "context_dependent_decision_positive",
        "context_dependent_decision_hard_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common = {
            "signal_dim": kwargs.get("signal_dim", 8),
            "context_duration": kwargs.get("context_duration", 10),
            "integration_duration": kwargs.get("integration_duration", 50),
            "coherence_levels": tuple(
                kwargs.get("coherence_levels", (0.05, 0.1, 0.2, 0.4))
            ),
            "noise_std": kwargs.get("noise_std", 1.0),
            "variable_delay": kwargs.get("variable_delay", True),
            "min_integration": kwargs.get("min_integration", 20),
            "max_integration": kwargs.get("max_integration", 80),
            "delay_duration": kwargs.get("delay_duration", 0),
            "min_delay": kwargs.get("min_delay", None),
            "max_delay": kwargs.get("max_delay", None),
            "conflict_fraction": kwargs.get("conflict_fraction", 0.0),
            "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.0),
            "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
            "distractor_burst_duration": kwargs.get("distractor_burst_duration", 5),
            "cue_dropout_prob": kwargs.get("cue_dropout_prob", 0.0),
            "cue_keep_duration": kwargs.get("cue_keep_duration", 1),
            "late_context_switch_prob": kwargs.get("late_context_switch_prob", 0.0),
            "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
            "return_seq_lengths": kwargs.get("return_seq_lengths", False),
            "positive_input_encoding": positive_input_encoding,
        }
        train_full = ContextDependentDecisionDataset(n_samples=n_train, **common)
        test_ds = ContextDependentDecisionDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "switching_context_decision",
        "switching_context_decision_positive",
    }:
        n_train = kwargs.get("n_samples", 16000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common = {
            "signal_dim": kwargs.get("signal_dim", 8),
            "seq_len": kwargs.get("seq_len", 360),
            "trial_duration": kwargs.get("trial_duration", 12),
            "min_context_duration": kwargs.get("min_context_duration", 90),
            "max_context_duration": kwargs.get("max_context_duration", 150),
            "coherence_levels": tuple(
                kwargs.get("coherence_levels", (0.03, 0.06, 0.12, 0.24))
            ),
            "noise_std": kwargs.get("noise_std", 1.1),
            "conflict_fraction": kwargs.get("conflict_fraction", 0.75),
            "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
            "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
            "distractor_burst_duration": kwargs.get("distractor_burst_duration", 3),
            "context_cue_duration": kwargs.get("context_cue_duration", 10),
            "switch_cue_duration": kwargs.get("switch_cue_duration", 4),
            "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
            "switch_signal_scale": kwargs.get("switch_signal_scale", 0.35),
            "min_blocks": kwargs.get("min_blocks", 2),
            "positive_input_encoding": positive_input_encoding,
        }
        train_full = SwitchingContextDecisionDataset(n_samples=n_train, **common)
        test_ds = SwitchingContextDecisionDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "timescale_generalization",
        "timescale_generalization_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "stimulus_dim": kwargs.get("stimulus_dim", 16),
            "stimulus_duration": kwargs.get("stimulus_duration", 10),
            "min_delay": kwargs.get("min_delay", 10),
            "max_delay": kwargs.get("max_delay", 100),
            "probe_duration": kwargs.get("probe_duration", 10),
            "positive_input_encoding": positive_input_encoding,
            "noise_std": kwargs.get("noise_std", 0.3),
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
                split_kwargs = dict(kwargs.get(split_name, {}))
                split_params = {
                    **common_defaults,
                    **{
                        key: value
                        for key, value in split_kwargs.items()
                        if key != "n_samples"
                    },
                }
                split_n = split_kwargs.get("n_samples", default_n_samples)
                return TimescaleGeneralizationDataset(n_samples=split_n, **split_params)

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = TimescaleGeneralizationDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = TimescaleGeneralizationDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "odr_generalization",
        "odr_generalization_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "n_targets": kwargs.get("n_targets", 8),
            "input_tuning_dim": kwargs.get("input_tuning_dim", 16),
            "fixation_duration": kwargs.get("fixation_duration", 6),
            "cue_duration": kwargs.get("cue_duration", 8),
            "min_delay": kwargs.get("min_delay", 10),
            "max_delay": kwargs.get("max_delay", 100),
            "go_duration": kwargs.get("go_duration", 8),
            "tuning_kappa": kwargs.get("tuning_kappa", 2.0),
            "cue_noise_std": kwargs.get("cue_noise_std", 0.05),
            "positive_input_encoding": positive_input_encoding,
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
                split_kwargs = dict(kwargs.get(split_name, {}))
                split_params = {
                    **common_defaults,
                    **{
                        key: value
                        for key, value in split_kwargs.items()
                        if key != "n_samples"
                    },
                }
                split_n = split_kwargs.get("n_samples", default_n_samples)
                return OculomotorDelayedResponseDataset(
                    n_samples=split_n, **split_params
                )

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = OculomotorDelayedResponseDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = OculomotorDelayedResponseDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "romo_delay_comparison",
        "romo_delay_comparison_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "frequency_values": tuple(
                kwargs.get("frequency_values", (2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
            ),
            "stimulus_duration": kwargs.get("stimulus_duration", 20),
            "min_delay": kwargs.get("min_delay", 10),
            "max_delay": kwargs.get("max_delay", 100),
            "response_duration": kwargs.get("response_duration", 8),
            "noise_std": kwargs.get("noise_std", 0.08),
            "positive_input_encoding": positive_input_encoding,
            "tuning_dim": kwargs.get("tuning_dim", 8),
            "tuning_sigma": kwargs.get("tuning_sigma", None),
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
                split_kwargs = dict(kwargs.get(split_name, {}))
                split_params = {
                    **common_defaults,
                    **{
                        key: value
                        for key, value in split_kwargs.items()
                        if key != "n_samples"
                    },
                }
                split_n = split_kwargs.get("n_samples", default_n_samples)
                return RomoDelayComparisonDataset(n_samples=split_n, **split_params)

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = RomoDelayComparisonDataset(n_samples=n_train, **common_defaults)
        test_ds = RomoDelayComparisonDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "gain_modulated_contextual_comparison",
        "gain_modulated_contextual_comparison_positive",
    }:
        n_train = kwargs.get("n_samples", 24000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "n_pathways": kwargs.get("n_pathways", 4),
            "pathway_dim": kwargs.get("pathway_dim", 4),
            "cue_duration": kwargs.get("cue_duration", 8),
            "stimulus_duration": kwargs.get("stimulus_duration", 16),
            "min_delay": kwargs.get("min_delay", 20),
            "max_delay": kwargs.get("max_delay", 80),
            "response_duration": kwargs.get("response_duration", 8),
            "amplitude_values": tuple(
                kwargs.get(
                    "amplitude_values",
                    (0.45, 0.6, 0.75, 0.9, 1.05, 1.2),
                )
            ),
            "comparison_margin": kwargs.get("comparison_margin", 0.15),
            "global_gain_min": kwargs.get("global_gain_min", 0.85),
            "global_gain_max": kwargs.get("global_gain_max", 1.15),
            "pathway_gain_min": kwargs.get("pathway_gain_min", 0.8),
            "pathway_gain_max": kwargs.get("pathway_gain_max", 1.2),
            "gain_drift_std": kwargs.get("gain_drift_std", 0.03),
            "noise_std": kwargs.get("noise_std", 0.08),
            "cue_noise_std": kwargs.get("cue_noise_std", 0.02),
            "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
            "distractor_burst_scale": kwargs.get("distractor_burst_scale", 1.5),
            "distractor_burst_duration": kwargs.get("distractor_burst_duration", 4),
            "positive_input_encoding": positive_input_encoding,
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
                split_kwargs = dict(kwargs.get(split_name, {}))
                split_params = {
                    **common_defaults,
                    **{
                        key: value
                        for key, value in split_kwargs.items()
                        if key != "n_samples"
                    },
                }
                split_n = split_kwargs.get("n_samples", default_n_samples)
                return GainModulatedContextualComparisonDataset(
                    n_samples=split_n, **split_params
                )

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = GainModulatedContextualComparisonDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = GainModulatedContextualComparisonDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name == "multi_sine_forecast":
        n_train = kwargs.get("n_samples", 12000)
        common = {
            "seq_len": kwargs.get("seq_len", 160),
            "rollout_horizon": kwargs.get("rollout_horizon", 160),
            "prediction_mode": kwargs.get("prediction_mode", "all"),
            "n_components": kwargs.get("n_components", 3),
            "frequency_bank": kwargs.get("frequency_bank", [1.0, 2.5, 5.5, 9.0, 14.0]),
            "amplitude_min": kwargs.get("amplitude_min", 0.3),
            "amplitude_max": kwargs.get("amplitude_max", 1.0),
            "envelope_scale": kwargs.get("envelope_scale", 0.15),
            "envelope_freq_min": kwargs.get("envelope_freq_min", 0.15),
            "envelope_freq_max": kwargs.get("envelope_freq_max", 0.6),
            "observation_noise_std": kwargs.get("observation_noise_std", 0.02),
            "seed": kwargs.get("seed", 42),
            "normalize_signal": kwargs.get("normalize_signal", True),
        }
        train_full = MultiSineForecastDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = MultiSineForecastDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        )
    elif dataset_name in {"lorenz", "lorenz_sequence"}:
        n_train = kwargs.get("n_samples", 10000)
        common = {
            "seq_len": kwargs.get("seq_len", 50),
            "dt": kwargs.get("dt", 0.01),
            "sigma": kwargs.get("sigma", 10.0),
            "rho": kwargs.get("rho", 28.0),
            "beta": kwargs.get("beta", 8.0 / 3.0),
            "rollout_horizon": kwargs.get("rollout_horizon", 50),
            "prediction_mode": kwargs.get("prediction_mode", "last"),
            "observation_mode": kwargs.get("observation_mode", "full"),
            "observation_dim": kwargs.get("observation_dim", 3),
            "observation_noise_std": kwargs.get("observation_noise_std", 0.0),
            "burn_in_steps": kwargs.get("burn_in_steps", 200),
            "seed": kwargs.get("seed", 42),
            "integration_method": kwargs.get("integration_method", "euler"),
            "normalize_states": kwargs.get("normalize_states", True),
            "normalize_observations": kwargs.get("normalize_observations", True),
            "normalization_steps": kwargs.get("normalization_steps", 10000),
            "initial_state_scale": kwargs.get("initial_state_scale", 0.1),
        }
        train_full = LorenzSequenceDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = LorenzSequenceDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        )
    elif dataset_name == "switching_lds":
        n_train = kwargs.get("n_samples", 12000)
        common = {
            "seq_len": kwargs.get("seq_len", 96),
            "latent_dim": kwargs.get("latent_dim", 4),
            "observation_dim": kwargs.get("observation_dim", 2),
            "n_modes": kwargs.get("n_modes", 3),
            "rollout_horizon": kwargs.get("rollout_horizon", 80),
            "prediction_mode": kwargs.get("prediction_mode", "all"),
            "observation_noise_std": kwargs.get("observation_noise_std", 0.03),
            "transition_noise_std": kwargs.get("transition_noise_std", 0.02),
            "min_mode_duration": kwargs.get("min_mode_duration", 36),
            "max_mode_duration": kwargs.get("max_mode_duration", 96),
            "burn_in_steps": kwargs.get("burn_in_steps", 24),
            "seed": kwargs.get("seed", 42),
            "normalize_states": kwargs.get("normalize_states", True),
            "normalize_observations": kwargs.get("normalize_observations", True),
            "normalization_samples": kwargs.get("normalization_samples", 4096),
            "stats_seed": kwargs.get("stats_seed", 1234),
            "initial_state_scale": kwargs.get("initial_state_scale", 0.35),
        }
        train_full = SwitchingLDSDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = SwitchingLDSDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
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
        session_idx = kwargs.get("session_idx", 20)
        running_quantile = kwargs.get("running_quantile", 0.5)
        running_state = kwargs.get("running_state", "all")
        train_running_state = kwargs.get("train_running_state", running_state)
        test_running_state = kwargs.get("test_running_state", running_state)
        gain_group_count = kwargs.get("gain_group_count", 1)
        gain_grouping = kwargs.get("gain_grouping", "preferred_class")
        poisson_sampling = kwargs.get("poisson_sampling", False)
        poisson_rate_scale = kwargs.get("poisson_rate_scale", 1.0)
        gain_seed = kwargs.get("gain_seed", seed)

        def _state_kwargs(prefix: str, state: str) -> dict:
            return {
                "running_state": state,
                "running_quantile": running_quantile,
                "gain_group_count": kwargs.get(
                    f"{prefix}gain_group_count", gain_group_count
                ),
                "gain_grouping": kwargs.get(f"{prefix}gain_grouping", gain_grouping),
                "global_gain_min": kwargs.get(
                    f"{prefix}global_gain_min", kwargs.get("global_gain_min", 1.0)
                ),
                "global_gain_max": kwargs.get(
                    f"{prefix}global_gain_max", kwargs.get("global_gain_max", 1.0)
                ),
                "group_gain_min": kwargs.get(
                    f"{prefix}group_gain_min", kwargs.get("group_gain_min", 1.0)
                ),
                "group_gain_max": kwargs.get(
                    f"{prefix}group_gain_max", kwargs.get("group_gain_max", 1.0)
                ),
                "gain_seed": kwargs.get(f"{prefix}gain_seed", gain_seed),
                "poisson_sampling": kwargs.get(
                    f"{prefix}poisson_sampling", poisson_sampling
                ),
                "poisson_rate_scale": kwargs.get(
                    f"{prefix}poisson_rate_scale", poisson_rate_scale
                ),
            }

        common = {
            "data_dir": data_dir,
            "n_neurons": n_neurons,
            "n_classes": n_classes,
            "normalize": normalize,
            "seed": seed,
            "session_idx": session_idx,
        }

        use_state_specific_split = (
            train_running_state != "all" or test_running_state != "all"
        )
        if use_state_specific_split:
            train_full = StringerV1Dataset(
                split="all",
                **common,
                **_state_kwargs("train_", train_running_state),
            )
            test_ds = StringerV1Dataset(
                split="all",
                **common,
                **_state_kwargs("test_", test_running_state),
            )
        else:
            # Fixed 90/10 train/test split; train_valid_split handled by factory below
            test_fraction = 0.1
            train_full = StringerV1Dataset(
                split="train",
                train_fraction=1.0 - test_fraction,
                **common,
                **_state_kwargs("train_", train_running_state),
            )
            test_ds = StringerV1Dataset(
                split="test",
                train_fraction=1.0 - test_fraction,
                **common,
                **_state_kwargs("test_", test_running_state),
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
