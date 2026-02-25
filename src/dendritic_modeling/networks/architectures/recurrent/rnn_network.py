"""
DendriticRNNNetwork: sequence-level wrapper around DendriticRNNCell.

Processes [batch, seq_len, input_dim] sequences through an optional input
projection MLP, then unrolls the cell over timesteps. Returns a plain tensor
whose shape depends on output_mode ("last", "all", "mean").

Implements BaseNetwork interface so decoder sees no difference from feedforward.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.config.recurrent import RecurrentConfig
from dendritic_modeling.networks.architectures.recurrent.rnn_cell import (
    DendriticRNNCell,
)
from dendritic_modeling.networks.architectures.recurrent.state import RecurrentState
from dendritic_modeling.networks.base import BaseNetwork


class DendriticRNNNetwork(BaseNetwork):
    """Sequence-level recurrent network with dendritic branch dynamics.

    Wraps one or more DendriticRNNCell layers with input projection and
    output_mode-based temporal reduction. Stores _routing_info only when
    store_routing=True (off by default to avoid memory balloon).

    Args:
        config: RecurrentConfig with all cell/branch/output parameters.
        input_dim: Dimension of input features (after encoder).
    """

    def __init__(self, config: RecurrentConfig, input_dim: int):
        super().__init__()

        self.config = config
        self._input_dim = input_dim
        self.n_exc = config.n_excitatory
        self.n_inh = config.n_inhibitory
        self.output_mode = config.output_mode
        self._store_routing = config.store_routing
        self.dropout_rate = config.dropout

        # Input projection MLP (per-timestep)
        proj_layers = []
        prev_dim = input_dim
        for hidden_dim in config.input_projection_layers:
            proj_layers.append(nn.Linear(prev_dim, hidden_dim))
            proj_layers.append(nn.ReLU())
            if config.dropout > 0:
                proj_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )
        self.proj_output_dim = prev_dim

        # Recurrent cell(s)
        self.cells = nn.ModuleList()
        cell_input_dim = self.proj_output_dim
        for _layer_idx in range(config.num_layers):
            cell = DendriticRNNCell(config, cell_input_dim)
            self.cells.append(cell)
            cell_input_dim = config.n_excitatory

        # Routing info storage (only populated when store_routing=True)
        self._routing_info: list[dict] = []

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.n_exc

    @property
    def store_routing(self) -> bool:
        return self._store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._store_routing = value
        for cell in self.cells:
            cell.store_routing = value

    def forward(
        self, x: torch.Tensor, hidden: Optional[list[RecurrentState]] = None
    ) -> torch.Tensor:
        """Process a sequence through the recurrent network.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            hidden: Optional list of RecurrentState (one per layer).
                    If None, zero-initialized.

        Returns:
            Output tensor whose shape depends on output_mode:
            - "last":  [batch, n_excitatory]
            - "all":   [batch, seq_len, n_excitatory]
            - "mean":  [batch, n_excitatory]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = [
                RecurrentState.init(
                    batch_size,
                    self.n_exc,
                    self.n_inh,
                    len(self.config.branches),
                    device,
                    dtype,
                )
                for _ in range(len(self.cells))
            ]

        # Clear routing info
        self._routing_info = []

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_dim]

            # Input projection
            x_t = self.input_projection(x_t)

            # Pass through cell stack
            timestep_info = {}
            for layer_idx, cell in enumerate(self.cells):
                x_t, hidden[layer_idx], info = cell(x_t, hidden[layer_idx], t_step=t)
                if info:
                    timestep_info[f"layer_{layer_idx}"] = info

            outputs.append(x_t)

            if self.store_routing and timestep_info:
                self._routing_info.append(timestep_info)

        # Stack outputs: [batch, seq_len, n_exc]
        outputs = torch.stack(outputs, dim=1)

        if self.output_mode == "last":
            return outputs[:, -1, :]
        elif self.output_mode == "mean":
            return outputs.mean(dim=1)
        elif self.output_mode == "all":
            return outputs
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
