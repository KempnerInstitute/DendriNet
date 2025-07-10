"""
Dendritic Convolution Layer - CNN-like feature extraction with biological constraints.

This module implements a dendritic convolution layer that mimics cortical simple cells
with push-pull excitation/inhibition and divisive normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, 

class DendriteConv2d(nn.Module):
    """
    Dendritic Convolution Layer with E/I balance and divisive normalization.

    This layer implements the transfer function:
        v_out = E / (E + I + ε)

    where E and I are excitatory and inhibitory convolutions respectively.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (dendritic units)
    kernel_size : int or tuple
        Size of the convolution kernel
    stride : int or tuple, optional
        Stride of the convolution (default: 1)
    padding : int or tuple, optional
        Padding added to input (default: 0)
    dilation : int or tuple, optional
        Spacing between kernel elements (default: 1)
    groups : int, optional
        Number of blocked connections from input to output (default: 1)
    bias : bool, optional
        If True, adds a learnable bias to the output (default: True)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-4)
    init_method : str, optional
        Weight initialization method (default: 'softplus_normal')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        epsilon: float = 1e-4,
        init_method: str = 'softplus_normal',
        v_in_channels: Optional[int] = None,
    ):
        super(DendriteConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.epsilon = epsilon
        self.init_method = init_method

        # V_in channels (for apical/contextual input)
        self.v_in_channels = v_in_channels or out_channels

        # Excitatory weights (positive only)
        self.weight_E_raw = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        # Inhibitory weights (positive only)
        self.weight_I_raw = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        # Optional bias terms
        if bias:
            self.bias_E = nn.Parameter(torch.empty(out_channels))
            self.bias_I = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_E', None)
            self.register_parameter('bias_I', None)

        # Apical modulation weights (for v_in integration)
        self.apical_E = nn.Parameter(torch.empty(out_channels, self.v_in_channels))
        self.apical_I = nn.Parameter(torch.empty(out_channels, self.v_in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        if self.init_method == 'softplus_normal':
            # Initialize in pre-softplus space
            fan_in = self.in_channels * np.prod(self.kernel_size)
            std = 0.01 / np.sqrt(fan_in)

            # Softplus inverse: log(exp(x) - 1)
            mean_pre = np.log(np.exp(0.1) - 1)  # Small positive mean

            nn.init.normal_(self.weight_E_raw, mean=mean_pre, std=std)
            nn.init.normal_(self.weight_I_raw, mean=mean_pre, std=std)

            # Apical weights
            nn.init.normal_(self.apical_E, mean=0, std=0.01)
            nn.init.normal_(self.apical_I, mean=0, std=0.01)

        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(self.weight_E_raw, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_I_raw, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.apical_E, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.apical_I, mode='fan_out', nonlinearity='relu')

        if self.bias_E is not None:
            nn.init.zeros_(self.bias_E)
            nn.init.zeros_(self.bias_I)

    @property
    def weight_E(self):
        """Excitatory weights (constrained positive)."""
        return F.softplus(self.weight_E_raw)

    @property
    def weight_I(self):
        """Inhibitory weights (constrained positive)."""
        return F.softplus(self.weight_I_raw)

    def forward(self, x: torch.Tensor, v_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through dendritic convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width)
        v_in : torch.Tensor, optional
            Apical/contextual input of shape (batch, v_in_channels) or
            (batch, v_in_channels, height, width)

        Returns
        -------
        v_out : torch.Tensor
            Output tensor of shape (batch, out_channels, height_out, width_out)
        """
        # Excitatory convolution
        E = F.conv2d(
            x, self.weight_E, self.bias_E,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )

        # Inhibitory convolution
        I_var = F.conv2d(
            x, self.weight_I, self.bias_I,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )

        # Apply apical modulation if v_in is provided
        if v_in is not None:
            # If v_in is global (batch, channels), expand to spatial
            if v_in.dim() == 2:
                v_in = v_in.unsqueeze(-1).unsqueeze(-1)
                v_in = v_in.expand(-1, -1, E.shape[2], E.shape[3])

            # Modulate E and I with apical input
            # Shape: (batch, out_channels, H, W) = (batch, v_in_channels, H, W) @ (v_in_channels, out_channels)
            apical_mod_E = F.conv2d(
                v_in.permute(0, 2, 3, 1).reshape(-1, self.v_in_channels, 1, 1),
                self.apical_E.t().reshape(self.out_channels, self.v_in_channels, 1, 1)
            ).reshape(x.shape[0], E.shape[2], E.shape[3], self.out_channels).permute(0, 3, 1, 2)

            apical_mod_I = F.conv2d(
                v_in.permute(0, 2, 3, 1).reshape(-1, self.v_in_channels, 1, 1),
                self.apical_I.t().reshape(self.out_channels, self.v_in_channels, 1, 1)
            ).reshape(x.shape[0], I.shape[2], I.shape[3], self.out_channels).permute(0, 3, 1, 2)

            E = E * torch.sigmoid(apical_mod_E)
            I_var = I * torch.sigmoid(apical_mod_I)

        # Divisive normalization
        v_out = E / (E + I + self.epsilon)

        return v_out

    def extra_repr(self) -> str:
        """Extra representation for print."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias_E is None:
            s += ', bias=False'
        s += f', epsilon={self.epsilon}'
        return s.format(**self.__dict__)

class DendriteAttention(nn.Module):
    """
    Dendritic Attention mechanism with E/I balance.

    Replaces softmax attention with divisive normalization:
        α_ij = E_ij / (Σ_k E_ik + I_i + ε)

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model
    num_heads : int
        Number of parallel attention heads
    dropout : float, optional
        Dropout probability (default: 0.0)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-4)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        epsilon: float = 1e-4,
    ):
        super(DendriteAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Inhibitory gating (per query)
        self.i_proj = nn.Linear(embed_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize inhibition to be slightly weaker
        nn.init.normal_(self.i_proj.weight, std=0.01)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.zeros_(self.i_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        v_in: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dendritic attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch, seq_len, embed_dim)
        key : torch.Tensor
            Key tensor of shape (batch, seq_len, embed_dim)
        value : torch.Tensor
            Value tensor of shape (batch, seq_len, embed_dim)
        attn_mask : torch.Tensor, optional
            Attention mask of shape (batch, seq_len, seq_len)
        v_in : torch.Tensor, optional
            Apical input for modulation

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim)
        attn_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (excitatory)
        E = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attn_mask is not None:
            E = E.masked_fill(attn_mask == 0, -1e9)

        # Compute inhibition (per query, per head)
        I_var = F.softplus(self.i_proj(query))  # (batch, seq_len, num_heads)
        I_var = I.transpose(1, 2).unsqueeze(-1)  # (batch, num_heads, seq_len, 1)

        # Apply v_in modulation if provided
        if v_in is not None:
            # v_in modulates the inhibition strength
            v_in_mod = torch.sigmoid(v_in).mean(dim=-1, keepdim=True)  # Global modulation
            I_var = I * v_in_mod.unsqueeze(1)

        # Divisive normalization attention
        E_positive = F.softplus(E)
        E_sum = E_positive.sum(dim=-1, keepdim=True)
        attn_weights = E_positive / (E_sum + I + self.epsilon)

        # Apply attention to values
        attn_output = torch.matmul(self.dropout(attn_weights), V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attn_output)

        return output, attn_weights
