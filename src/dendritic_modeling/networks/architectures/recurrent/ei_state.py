"""
State containers for unified E-I recurrent layers.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class DendriNetState:
    """Per-level traces for one population."""

    trace_E: list[Tensor]
    trace_E_rec: list[Tensor]
    trace_I: list[Tensor]
    trace_I_rec: list[Tensor]
    trace_branch: list[Tensor]

    @staticmethod
    def zeros(
        level_dims: list[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "DendriNetState":
        def _mk():
            return [
                torch.zeros(batch_size, dim, device=device, dtype=dtype)
                for dim in level_dims
            ]

        return DendriNetState(
            trace_E=_mk(),
            trace_E_rec=_mk(),
            trace_I=_mk(),
            trace_I_rec=_mk(),
            trace_branch=_mk(),
        )

    def detach(self) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.detach() for t in self.trace_E],
            trace_E_rec=[t.detach() for t in self.trace_E_rec],
            trace_I=[t.detach() for t in self.trace_I],
            trace_I_rec=[t.detach() for t in self.trace_I_rec],
            trace_branch=[t.detach() for t in self.trace_branch],
        )

    def clone(self) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.clone() for t in self.trace_E],
            trace_E_rec=[t.clone() for t in self.trace_E_rec],
            trace_I=[t.clone() for t in self.trace_I],
            trace_I_rec=[t.clone() for t in self.trace_I_rec],
            trace_branch=[t.clone() for t in self.trace_branch],
        )

    def to(self, device: torch.device) -> "DendriNetState":
        return DendriNetState(
            trace_E=[t.to(device) for t in self.trace_E],
            trace_E_rec=[t.to(device) for t in self.trace_E_rec],
            trace_I=[t.to(device) for t in self.trace_I],
            trace_I_rec=[t.to(device) for t in self.trace_I_rec],
            trace_branch=[t.to(device) for t in self.trace_branch],
        )


@dataclass
class EIState:
    """State for one unified E-I layer."""

    e_state: DendriNetState
    i_state: Optional[DendriNetState]
    s_E: Tensor
    h_I: Optional[Tensor]

    def detach(self) -> "EIState":
        return EIState(
            e_state=self.e_state.detach(),
            i_state=self.i_state.detach() if self.i_state is not None else None,
            s_E=self.s_E.detach(),
            h_I=self.h_I.detach() if self.h_I is not None else None,
        )

    def clone(self) -> "EIState":
        return EIState(
            e_state=self.e_state.clone(),
            i_state=self.i_state.clone() if self.i_state is not None else None,
            s_E=self.s_E.clone(),
            h_I=self.h_I.clone() if self.h_I is not None else None,
        )

    def to(self, device: torch.device) -> "EIState":
        return EIState(
            e_state=self.e_state.to(device),
            i_state=self.i_state.to(device) if self.i_state is not None else None,
            s_E=self.s_E.to(device),
            h_I=self.h_I.to(device) if self.h_I is not None else None,
        )
