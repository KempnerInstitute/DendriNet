"""Shared sparse projections composed with existing native DendriNet banks.

No branch integration rule is reimplemented here. A group uses a flat native
population, followed by a signed BlockLinear pool. This is a composition of
native compartments and linear operators, not a nonlinear-soma tree.
"""

import math
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn

from dendritic_modeling.config.shared_projection import SharedProjectionShuntingConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
    DendriNetConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
    IndexedSynapseConfig,
    MorphologyConfig,
    ReactivationConfig,
    TopKConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)


class SharedProjectionShunting(nn.Module):
    """Compute b + C sum_k a[j,k] K(v[j], z[j,k]) using native branches.

    v[j] is a normalized positive sparse projection. Initially it is a uniform
    average, a[j,k]=1/W, C[o,j]=1/q, and b=0. Conductance g=(1+epsilon)/z
    compensates the native denominator epsilon exactly in real arithmetic.
    Parameters are stored even when frozen. Eager indexed projections and
    backpropagation are the qualified execution/training paths.
    """

    def __init__(self, config: SharedProjectionShuntingConfig):
        super().__init__()
        config.validate()
        self.config = deepcopy(config)
        self.input_dim, self.output_dim = config.input_dim, config.output_dim
        self.groups, self.width = len(config.supports), len(config.nodes)
        q, w = self.groups, self.width
        # Explicit support and complete reinitialization make this deterministic.
        # Native constructors may consume RNG; do not advance the caller's stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.projection = IndexedSparseLinear(
                config.input_dim,
                q,
                len(config.supports[0]),
                connection_indices=torch.tensor(config.supports, dtype=torch.int64),
                weight_transform="softplus",
                weight_norm_order=1,
                gamma=1.0,
                projection_backend="eager",
            )
            synapse = DendriticSynapseConfig(
                topk=TopKConfig(type="indexed"),
                indexed=IndexedSynapseConfig(seed=0, projection_backend="eager"),
                morphology=MorphologyConfig(
                    use_shunting=True,
                    leak_conductance=1.0,
                    learnable_leak=False,
                    epsilon=config.epsilon,
                    weight_transform="exp",
                ),
                reactivation=ReactivationConfig(enabled=False),
            )
            self.banks = nn.ModuleList(
                [
                    DendriNet(
                        config=DendriNetConfig(
                            n_soma=w,
                            branch_factors=(),
                            excitatory_input_dim=1,
                            excitatory_synapses_per_branch=1,
                            inhibitory_input_dim=None,
                            inhibitory_synapses_per_branch=0,
                            somatic_synapses=True,
                            synapse_config=synapse,
                        )
                    )
                    for _ in range(q)
                ]
            )
            self.pool = BlockLinear(q * w, q, weight_transform="identity")
            self.readout = nn.Linear(q, config.output_dim)
        self.to(dtype=getattr(torch, config.dtype))
        with torch.no_grad():
            self.projection.pre_w.zero_()
            raw = (
                math.log1p(config.epsilon)
                - torch.tensor(config.nodes, dtype=self.readout.weight.dtype).log()
            )
            max_log = math.log(torch.finfo(raw.dtype).max) - 1
            if (
                not torch.isfinite(raw).all()
                or (raw > max_log).any()
                or (raw.exp() == 0).any()
            ):
                raise ValueError("nodes cannot be represented by the selected dtype")
            for bank in self.banks:
                bank.branch_layers[0].branch_excitation.pre_w.copy_(raw[:, None])
            self.pool.log_weight.fill_(1 / w)
            self.readout.weight.fill_(1 / q)
            self.readout.bias.zero_()
        self.projection.requires_grad_(config.train_projection)
        self.banks.requires_grad_(config.train_conductance)
        self.readout.weight.requires_grad_(config.train_output)
        # The affine intercept remains learnable in fixed-feature regression.

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return qW executed native features in group-major order."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("inputs must have shape (batch, input_dim)")
        if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
            raise ValueError("shared shunting inputs must be finite and nonnegative")
        v = self.projection(x)
        return torch.cat(
            [bank(v[:, j : j + 1]) for j, bank in enumerate(self.banks)], dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.pool(self.features(x)))

    def effective_nodes(self) -> torch.Tensor:
        """Return current transition scales, including native leak and epsilon."""
        return torch.stack(
            [
                (1 + self.config.epsilon)
                / bank.branch_layers[0].branch_excitation.sparse_weight().flatten()
                for bank in self.banks
            ]
        )

    def resource_report(self) -> dict:
        """Count actual parameters/buffers separately from arithmetic estimates."""
        from dendritic_modeling.networks.benchmarks.parameter_budget import model_report

        report = model_report(self)
        q, w, s, o = self.groups, self.width, self.projection.K, self.output_dim
        report.update(
            resolved_config=asdict(self.config),
            native_synapse_config=asdict(self.banks[0].synapse_config),
            native_geometry={
                "n_soma_per_bank": w,
                "bank_count": q,
                "branch_factors": [],
                "excitatory_input_dim": 1,
                "excitatory_synapses_per_branch": 1,
                "inhibitory_synapses_per_branch": 0,
                "somatic_synapses": True,
            },
            implementation="shared sparse projection + native flat DendriNet banks + signed block pool",
            parameter_roles={
                "projection": q * s,
                "conductance": q * w,
                "pool": q * w,
                "output": q * o + o,
            },
            expected_stored_parameters=q * s + 2 * q * w + q * o + o,
            shunting_divisions_per_example=q * w,
            sparse_contact_macs_per_example=q * s + q * w,
            pooling_and_output_macs_per_example=q * w + q * o,
            support_index_count=self.projection.connection_indices.numel(),
            training_algorithm="backpropagation",
        )
        return report


def build_shared_projection_shunting(
    config: SharedProjectionShuntingConfig,
) -> SharedProjectionShunting:
    """Construct the explicit native composition without changing old models."""
    return SharedProjectionShunting(config)
