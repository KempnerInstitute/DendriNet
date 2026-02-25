"""
Typed recurrent state for dendritic RNN cells.

RecurrentState holds per-branch E/I/V traces and inhibitory network state.
Different compartment modes use different fields:
- topk: branch_E, branch_I (branch_V unused)
- dendritic_branch: branch_E, branch_E_rec, branch_I, branch_I_rec, branch_V
- dendrinet: branch_V (branch_E, branch_I unused)
- linear: branch_E only
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class RecurrentState:
    """Typed state container for DendriticRNNCell.

    Fields:
        branch_E: Per-branch FF excitatory traces [batch, n_exc], one per branch.
        branch_E_rec: Per-branch recurrent excitatory traces [batch, n_exc].
        branch_I: Per-branch FF inhibitory traces [batch, n_exc], one per branch.
        branch_I_rec: Per-branch recurrent inhibitory traces [batch, n_exc].
        branch_V: Per-branch post-shunting voltage [batch, n_exc], one per branch.
                   Used by compartment="dendritic_branch" and "dendrinet".
        h_inh: Inhibitory network state [batch, n_inh]. None when n_inhibitory=0.
    """

    branch_E: list[Tensor] = field(default_factory=list)
    branch_E_rec: list[Tensor] = field(default_factory=list)
    branch_I: list[Tensor] = field(default_factory=list)
    branch_I_rec: list[Tensor] = field(default_factory=list)
    branch_V: list[Tensor] = field(default_factory=list)
    h_inh: Optional[Tensor] = None

    @staticmethod
    def init(
        batch_size: int,
        n_exc: int,
        n_inh: int,
        n_branches: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "RecurrentState":
        """Create zero-initialized state."""

        def _zeros():
            return [
                torch.zeros(batch_size, n_exc, device=device, dtype=dtype)
                for _ in range(n_branches)
            ]

        h_inh = (
            torch.zeros(batch_size, n_inh, device=device, dtype=dtype)
            if n_inh > 0
            else None
        )
        return RecurrentState(
            branch_E=_zeros(),
            branch_E_rec=_zeros(),
            branch_I=_zeros(),
            branch_I_rec=_zeros(),
            branch_V=_zeros(),
            h_inh=h_inh,
        )

    def detach(self) -> "RecurrentState":
        """Detach all tensors from the computation graph (for TBPTT)."""
        return RecurrentState(
            branch_E=[e.detach() for e in self.branch_E],
            branch_E_rec=[e.detach() for e in self.branch_E_rec],
            branch_I=[i.detach() for i in self.branch_I],
            branch_I_rec=[i.detach() for i in self.branch_I_rec],
            branch_V=[v.detach() for v in self.branch_V],
            h_inh=self.h_inh.detach() if self.h_inh is not None else None,
        )

    def reset(self) -> None:
        """Zero all tensors in-place."""
        for e in self.branch_E:
            e.zero_()
        for e in self.branch_E_rec:
            e.zero_()
        for i in self.branch_I:
            i.zero_()
        for i in self.branch_I_rec:
            i.zero_()
        for v in self.branch_V:
            v.zero_()
        if self.h_inh is not None:
            self.h_inh.zero_()
