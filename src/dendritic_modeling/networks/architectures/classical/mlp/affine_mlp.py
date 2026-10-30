"""Dense and sparse affine MLPs with explicit initialization and layout."""

import math
from dataclasses import asdict

import torch
from torch import nn

from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.sparse_maintenance import forward_maintenance


class SparseAffine(nn.Module):
    """Signed fixed-index or learned Top-K projection followed by a bias.

    Top-K candidate rows are normalized so their initially retained squared
    weights sum to two. Fixed weights have variance 2 / fanin. Initialization
    consumes the caller's Torch RNG; seed controls only fixed connectivity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fanin: int,
        *,
        learned: bool,
        seed: int | None = None,
        projection_backend: str = "eager",
        workspace_mb: float = 128.0,
        index_dtype: str = "int32",
    ):
        super().__init__()
        if learned:
            self.linear = TopKLinear(
                in_features,
                out_features,
                fanin,
                weight_transform="identity",
                param_space="linear",
            )
            nn.init.normal_(self.linear.pre_w)
            with torch.no_grad():
                values = self.linear.pre_w.abs().topk(fanin, dim=1).values
                self.linear.pre_w.mul_(
                    math.sqrt(2)
                    / values.square().sum(1).sqrt().clamp_min(1e-12)[:, None]
                )
        else:
            self.linear = IndexedSparseLinear(
                in_features,
                out_features,
                fanin,
                weight_transform="identity",
                param_space="linear",
                seed=seed,
                projection_backend=projection_backend,
                workspace_mb=workspace_mb,
                index_dtype=index_dtype,
            )
            nn.init.normal_(self.linear.pre_w, std=math.sqrt(2 / fanin))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.bias


class AffineMLP(nn.Sequential):
    """``nn.Sequential`` affine MLP that forwards sparse maintenance.

    The layout, module order and checkpoint keys are those of a plain
    ``nn.Sequential``; the subclass only adds the optimizer-step maintenance
    API so the learned Top-K and fixed-index layers receive weight decay,
    boosting and rewiring like every other sparse core.
    """

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
        """Forward weight decay to every sparse projection once."""
        forward_maintenance(self, "decay_weights", weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        """Forward the rewiring step to every sparse projection once."""
        forward_maintenance(self, "apply_rewiring")


def build_affine_mlp(config: AffineMLPConfig) -> nn.Sequential:
    """Build from a validated typed config, without reseeding global RNGs.

    The flat Sequential layout preserves ordinary numeric checkpoint keys.
    No enclosing layers prefix is introduced. Dense construction precedes
    explicit initialization of all dense layers, unless ``pytorch_default``
    retains each Linear constructor's weights and biases. Sparse initialization
    occurs inside each affine constructor. Neither path changes the existing
    MLP API.
    """

    config.validate()
    modules: list[nn.Module] = []
    affine_layers: list[nn.Module] = []
    for i, (nin, nout) in enumerate(zip(config.dimensions, config.dimensions[1:])):
        if config.mode == "dense":
            layer = nn.Linear(nin, nout)
        else:
            layer = SparseAffine(
                nin,
                nout,
                config.fanin[i],
                learned=config.mode == "topk",
                seed=config.connectivity_seeds[i] if config.mode == "fixed" else None,
                projection_backend=config.projection_backend,
                workspace_mb=config.workspace_mb,
                index_dtype=config.index_dtype,
            )
        affine_layers.append(layer)
        modules.append(layer)
        if i < len(config.dimensions) - 2 or config.activate_output:
            modules.append(
                nn.ReLU()
                if config.activation == "relu"
                else nn.LeakyReLU(config.negative_slope)
            )
            if config.dropout is not None:
                modules.append(nn.Dropout(config.dropout))
    model = AffineMLP(*modules)
    if config.mode == "dense":
        for layer in affine_layers:
            if config.initialization == "pytorch_default":
                continue
            if config.initialization == "kaiming_normal":
                nn.init.kaiming_normal_(layer.weight)
            elif layer is affine_layers[-1]:
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        final_weight = affine_layers[-1].weight
    else:
        final_weight = affine_layers[-1].linear.pre_w
    # Keep existing policies' operation order, but do not mutate a native
    # Linear tensor's version counter for an explicitly unscaled replay.
    if config.initialization != "pytorch_default" or config.output_weight_scale != 1.0:
        with torch.no_grad():
            final_weight.mul_(config.output_weight_scale)
    model.input_dim = config.dimensions[0]
    model.output_dim = config.dimensions[-1]
    model.affine_mlp_config = asdict(config)
    return model
