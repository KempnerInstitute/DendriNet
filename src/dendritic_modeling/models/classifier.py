"""
Standard classification models.

This module contains standard classifiers using cross-entropy loss.
"""

import logging
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.config.model import (
    fold_learned_output_scale,
    resolve_output_scale_mode,
)
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.base import BaseNetwork

logger = logging.getLogger(__name__)


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(
        self,
        encoder_network: Optional[BaseNetwork],
        core_network: BaseNetwork,
        decoder_network: Optional[BaseNetwork],
        *,
        learned_output_scale: Optional[bool] = None,
        fixed_output_scale: float = 1.0,
        output_scale_mode: Optional[str] = None,
    ):
        super().__init__(encoder_network, core_network, decoder_network)

        self.fixed_output_scale = float(fixed_output_scale)
        if not math.isfinite(self.fixed_output_scale) or self.fixed_output_scale <= 0:
            raise ValueError(
                "fixed_output_scale must be finite and positive, "
                f"got {self.fixed_output_scale!r}"
            )
        if learned_output_scale is not None:
            warnings.warn(
                "Classifier(learned_output_scale=...) was retired on 2026-10-08; "
                "pass output_scale_mode instead (False is 'fixed').",
                DeprecationWarning,
                stacklevel=2,
            )
            output_scale_mode = fold_learned_output_scale(
                learned_output_scale, output_scale_mode
            )
        # The mode defaults to a learned per-class scale.
        self.output_scale_mode = resolve_output_scale_mode(output_scale_mode)
        learned = self.output_scale_mode != "fixed"
        if (
            learned
            and output_scale_mode is not None
            and not isinstance(self.decoder_network, Identity)
        ):
            # Only an identity decoder gets a learned scale; with any other
            # decoder the default mode simply learns none.
            raise ValueError(
                f"model.output_scale_mode={self.output_scale_mode!r} learns an "
                "output scale only for an identity decoder, and the "
                f"{type(self.decoder_network).__name__} decoder gets none. Use "
                "output_scale_mode: fixed or an identity decoder."
            )
        if isinstance(self.decoder_network, Identity):
            if self.output_scale_mode == "per_class":
                self.log_output_scale = nn.Parameter(
                    torch.zeros((self.core_network.output_dim,)), requires_grad=True
                )
            elif self.output_scale_mode == "global":
                # One element rather than 0-d, which FSDP cannot shard.
                self.log_global_output_scale = nn.Parameter(
                    torch.zeros((1,)), requires_grad=True
                )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Files written before 2026-10-10 hold the global scale 0-d.
        key = prefix + "log_global_output_scale"
        value = state_dict.get(key)
        if torch.is_tensor(value) and value.dim() == 0:
            state_dict[key] = value.reshape(1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = super().forward(x, **kwargs)
        x = self.fixed_output_scale * x

        if hasattr(self, "log_output_scale"):
            x = self.log_output_scale.exp() * x
        elif hasattr(self, "log_global_output_scale"):
            x = self.log_global_output_scale.exp() * x

        return x

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, **kwargs)
            return logits.argmax(dim=-1)


__all__ = ["Classifier"]
