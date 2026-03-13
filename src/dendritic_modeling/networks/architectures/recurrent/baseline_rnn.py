"""
Baseline RNN wrappers (GRU, LSTM, Vanilla RNN) with same interface as EINetwork.
"""

from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn

from dendritic_modeling.networks.base import BaseNetwork


@dataclass
class BaselineRNNConfig:
    """Configuration for baseline RNN models."""

    cell_type: str = "gru"  # "gru", "lstm", "vanilla"
    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    output_mode: str = "last"  # "last", "mean", "all"
    input_projection_dims: list[int] = field(default_factory=list)


class BaselineRNN(BaseNetwork):
    """Wraps nn.GRU, nn.LSTM, or nn.RNN for baseline comparison.

    Provides the same interface as EINetwork:
    - ``is_recurrent`` property
    - ``output_dim`` attribute
    - ``forward(x, hidden=None, return_hidden=False)``
    """

    def __init__(self, config: BaselineRNNConfig):
        super().__init__()
        self.config = config
        self._output_mode = config.output_mode

        proj_layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.input_projection_dims:
            proj_layers.append(nn.Linear(prev_dim, hidden_dim))
            proj_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )

        rnn_cls = {
            "gru": nn.GRU,
            "lstm": nn.LSTM,
            "vanilla": nn.RNN,
        }
        cell_type = config.cell_type.lower()
        if cell_type not in rnn_cls:
            raise ValueError(
                f"Unknown cell_type '{config.cell_type}'. "
                f"Choose from: gru, lstm, vanilla"
            )

        self.rnn = rnn_cls[cell_type](
            input_size=prev_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_dim = config.hidden_dim

    @property
    def is_recurrent(self) -> bool:
        return True

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._output_mode == "last":
            if seq_lengths is not None:
                B, T, _ = outputs.shape
                idx = (seq_lengths - 1).long().clamp(0, T - 1)
                return outputs[torch.arange(B, device=outputs.device), idx]
            return outputs[:, -1, :]
        if self._output_mode == "mean":
            return outputs.mean(dim=1)
        if self._output_mode == "all":
            return outputs
        raise ValueError(f"Unknown output_mode: {self._output_mode}")

    def forward(
        self,
        x: torch.Tensor,
        hidden=None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, any]]:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, input_dim].
            hidden: Optional hidden state from previous call.
            return_hidden: If True, return (output, hidden_state) tuple.
            seq_lengths: Optional per-sample sequence lengths [batch].
                When provided and output_mode="last", reads output at position
                seq_lengths[i]-1 instead of the last timestep.

        Returns:
            output: Reduced output based on output_mode.
        """
        # Apply per-timestep projection
        if not isinstance(self.input_projection, nn.Identity):
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            x = self.input_projection(x)
            x = x.reshape(B, T, -1)

        rnn_out, hidden_out = self.rnn(x, hidden)
        out = self._reduce_outputs(rnn_out, seq_lengths=seq_lengths)

        if return_hidden:
            return out, hidden_out
        return out
