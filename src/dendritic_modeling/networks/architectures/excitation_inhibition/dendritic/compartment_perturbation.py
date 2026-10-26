"""Additive perturbations of branch-layer compartments for causal analyses.

A :class:`CompartmentPerturbation` set on a
:class:`~dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer`
(``set_compartment_perturbation`` or ``perturb_compartments``) adds its
offset to the layer's compartments during the layer's own forward pass, in one
coordinate, so the perturbed layer runs once:

``output``
    added to the layer's output after its activation: the signal its parent,
    or the readout, receives.
``voltage``
    added to the pre-activation voltage, so the activation's slope applies.
``current``
    an injected current added to the numerator of the shunting voltage
    equation, ``V = (N + J) / (D + epsilon)``, so the layer's total
    conductance ``D`` attenuates it. Shunting layers only.

The offset must broadcast to the layer's output, ``[batch, compartments]``,
without changing its shape, and must match its device and dtype; compartments
left unperturbed get zeros. A perturbation changes no parameter or buffer.
Gradients of a loss with respect to a zero offset that requires gradients are
the exact errors of the compartments in that coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

COORDINATES = ("output", "voltage", "current")


@dataclass(frozen=True, eq=False)
class CompartmentPerturbation:
    """An additive offset for one branch layer's compartments in one coordinate."""

    offset: torch.Tensor
    coordinate: str = "output"

    def __post_init__(self) -> None:
        if not isinstance(self.offset, torch.Tensor):
            raise TypeError(
                f"the perturbation offset must be a tensor, got {type(self.offset).__name__}"
            )
        if self.coordinate not in COORDINATES:
            raise ValueError(
                f"unknown perturbation coordinate {self.coordinate!r}; "
                f"use one of {', '.join(COORDINATES)}"
            )

    def add_to(self, value: torch.Tensor) -> torch.Tensor:
        """``value`` plus the offset, after checking that the offset fits it."""
        offset = self.offset
        if offset.device != value.device:
            raise ValueError(
                f"the {self.coordinate} perturbation is on {offset.device}, the "
                f"layer computes on {value.device}"
            )
        if offset.dtype != value.dtype:
            raise ValueError(
                f"the {self.coordinate} perturbation has dtype {offset.dtype}, the "
                f"layer computes in {value.dtype}"
            )
        try:
            shape = torch.broadcast_shapes(offset.shape, value.shape)
        except RuntimeError:
            shape = None
        if shape != value.shape:
            raise ValueError(
                f"a {self.coordinate} perturbation of shape {tuple(offset.shape)} "
                f"does not broadcast to the layer's {tuple(value.shape)}"
            )
        return value + offset


__all__ = ["COORDINATES", "CompartmentPerturbation"]
