"""
Base model class for dendritic modeling.

This module contains the abstract BaseModel class that provides common functionality
for all model wrappers including parameter grouping and weight decay.
"""

from itertools import chain

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.networks import (
    BaseNetwork,
    BlockLinear,
    CreditGatedDeepstLinear,
    DeepstLinear,
    ExcitationInhibitionLayer,
    Identity,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    ParametricActivation,
    TopKLinear,
)
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.utils.sparse_maintenance import forward_maintenance

# Gate modules whose parameters form the ``reactivation`` optimizer group.
_REACTIVATION_GATE_TYPES = (ParametricActivation, EndpointNormalizedTanh)
# Learned classifier output scales; they train with the decoder.
_OUTPUT_SCALE_PARAMETERS = ("log_output_scale", "log_global_output_scale")


def _is_split_param_group_module(module: nn.Module) -> bool:
    """Return whether a core module participates in split optimizer groups."""
    return isinstance(
        module,
        (
            TopKLinear,
            IndexedSparseLinear,
            IndexedDynamicTopKLinear,
            DeepstLinear,
            CreditGatedDeepstLinear,
            BlockLinear,
            *_REACTIVATION_GATE_TYPES,
            ExcitationInhibitionLayer,
        ),
    )


class BaseModel(nn.Module):
    def __init__(
        self,
        encoder_network: BaseNetwork,
        core_network: BaseNetwork,
        decoder_network: BaseNetwork,
    ):
        super().__init__()

        self.encoder_network = encoder_network
        self.core_network = core_network
        self.decoder_network = decoder_network
        # Cache once: non-recurrent models avoid per-forward branching checks.
        # EINetwork.is_recurrent property handles this.
        recurrent_flag = getattr(self.core_network, "is_recurrent", False)
        if callable(recurrent_flag):
            recurrent_flag = recurrent_flag()
        self._is_recurrent_core = bool(recurrent_flag)
        self._freeze_core_forward = False

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
        *,
        precomputed_core: bool = False,
    ) -> torch.Tensor:
        if not precomputed_core:
            # Recurrent cores expect [batch, seq_len, feature_dim].
            # Encoder may need per-timestep application.
            if self._is_recurrent_core:
                if not isinstance(self.encoder_network, Identity) and x.dim() == 3:
                    # Apply encoder per timestep: [B, T, D] -> [B, T, D']
                    B, T, D = x.shape
                    x = x.reshape(B * T, D)
                    x = self.encoder_network(x)
                    x = x.reshape(B, T, -1)
                else:
                    x = self.encoder_network(x)
                if self._freeze_core_forward and not x.requires_grad:
                    with torch.no_grad():
                        x = self.core_network(x, seq_lengths=seq_lengths)
                else:
                    x = self.core_network(x, seq_lengths=seq_lengths)
            else:
                x = self.encoder_network(x)
                if self._freeze_core_forward and not x.requires_grad:
                    with torch.no_grad():
                        x = self.core_network(x)
                else:
                    x = self.core_network(x)
        x = self.decoder_network(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Predict method must be implemented in subclass.")

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
        """Decay every sparse synapse leaf of the encoder, core and decoder once.

        Sub-networks that implement ``decay_weights`` forward it to their own
        leaves; any other sub-network is searched for sparse leaves (for
        example an affine Top-K decoder built as a plain ``nn.Sequential``).
        """
        forward_maintenance(self, "decay_weights", weight_decay, weight_boosting)

    def apply_rewiring(self):
        """Advance rewiring once for every sparse leaf of encoder, core and decoder."""
        forward_maintenance(self, "apply_rewiring")

    def get_param_groups(self, param_groups_config: ParamGroupsConfig) -> list[dict]:
        """Return the optimizer parameter groups (see :func:`model_param_groups`)."""
        return model_param_groups(self, param_groups_config)


def model_param_groups(
    model: nn.Module, param_groups_config: ParamGroupsConfig
) -> list[dict]:
    """Return the optimizer parameter groups of ``model``.

    Every training path (single process, DDP and FSDP) builds its optimizer
    from these groups. With ``split_params`` the groups are, in order:

    - ``topk``, ``blocklinear``, ``reactivation`` (``topk_lr``,
      ``blocklinear_lr``, ``reactivation_lr``): the sparse synapses, the
      BlockLinear branch couplings and the reactivation gates of
      ``model.core_network``;
    - ``mlp_inh_net`` (``lr``): MLP inhibitory cells of E/I layers in the core;
    - ``decoder_input`` (``decoder_input_lr``, when set): the first ``Linear``
      weight of ``model.decoder_network``;
    - ``decoder`` (``decoder_lr``): the rest of the decoder, including any
      synapse or gate modules it contains, and the learned output scales;
    - ``encoder`` (``encoder_lr``, default ``lr``): trainable encoder
      parameters;
    - ``catch_all`` (``lr``): every other trainable parameter.

    A model without ``core_network`` is treated as its own core, excluding
    its ``encoder_network`` and ``decoder_network``. Without ``split_params``
    there is one group ``all`` at ``lr``. Lists are materialized because
    one-shot iterators cannot be reused by the optimizer.
    """
    split_params = getattr(param_groups_config, "split_params", True)
    if not split_params:
        return [
            {
                "name": "all",
                "params": model.parameters(),
                "lr": getattr(param_groups_config, "lr", 0.001),
            }
        ]

    encoder_network = getattr(model, "encoder_network", None)
    core_network = getattr(model, "core_network", None)
    decoder_network = getattr(model, "decoder_network", None)
    if core_network is not None:
        core_modules = core_network.modules()
    else:
        excluded = {
            id(module)
            for network in (encoder_network, decoder_network)
            if network is not None
            for module in network.modules()
        }
        core_modules = (
            module for module in model.modules() if id(module) not in excluded
        )

    topk_params = []
    blocklinear_params = []
    reactivation_params = []
    mlp_inh_net_params = []
    decoder_input_params = []
    decoder_params = []

    for module in core_modules:
        if not _is_split_param_group_module(module):
            continue
        if isinstance(
            module,
            (
                TopKLinear,
                IndexedSparseLinear,
                IndexedDynamicTopKLinear,
                DeepstLinear,
                CreditGatedDeepstLinear,
            ),
        ):
            topk_params.append(module.parameters())
        elif isinstance(module, BlockLinear):
            blocklinear_params.append(module.parameters())
        elif isinstance(module, _REACTIVATION_GATE_TYPES):
            reactivation_params.append(module.parameters())
        elif isinstance(module, ExcitationInhibitionLayer):
            if isinstance(module.inhibitory_cells, nn.Sequential):
                mlp_inh_net_params.append(module.inhibitory_cells.parameters())

    decoder_input_lr = getattr(param_groups_config, "decoder_input_lr", None)
    first_decoder_weight = None
    if decoder_network is not None:
        if decoder_input_lr is not None:
            for module in decoder_network.modules():
                if isinstance(module, nn.Linear):
                    first_decoder_weight = module.weight
                    decoder_input_params.append([first_decoder_weight])
                    break
        decoder_params.append(
            param
            for param in decoder_network.parameters()
            if param is not first_decoder_weight
        )
    for name in _OUTPUT_SCALE_PARAMETERS:
        scale = getattr(model, name, None)
        if isinstance(scale, nn.Parameter):
            decoder_params.append([scale])

    # Trainable encoder parameters (not a frozen pretrained prefix).
    encoder_params = []
    if encoder_network is not None:
        encoder_trainable = [p for p in encoder_network.parameters() if p.requires_grad]
        if encoder_trainable:
            encoder_params.append(iter(encoder_trainable))

    param_groups = []
    categorized_ids: set[int] = set()

    global_lr = getattr(param_groups_config, "lr", 0.001)

    for group_name, params_list, lr_key, default_lr in [
        ("topk", topk_params, "topk_lr", 0.001),
        ("blocklinear", blocklinear_params, "blocklinear_lr", 0.001),
        ("reactivation", reactivation_params, "reactivation_lr", 0.001),
        ("mlp_inh_net", mlp_inh_net_params, "lr", 0.001),
        ("decoder_input", decoder_input_params, "decoder_input_lr", global_lr),
        ("decoder", decoder_params, "decoder_lr", 0.001),
        ("encoder", encoder_params, "encoder_lr", global_lr),
    ]:
        if params_list:
            materialized = list(chain(*params_list))
            categorized_ids.update(id(p) for p in materialized)
            lr = getattr(param_groups_config, lr_key, None)
            if lr is None:
                lr = default_lr
            param_groups.append({"name": group_name, "params": materialized, "lr": lr})

    # Catch-all: any params not in the above groups
    uncategorized = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in categorized_ids
    ]
    if uncategorized:
        param_groups.append(
            {
                "name": "catch_all",
                "params": uncategorized,
                "lr": getattr(param_groups_config, "lr", 0.001),
            }
        )

    return param_groups


__all__ = ["BaseModel", "model_param_groups"]
