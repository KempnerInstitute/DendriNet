"""Paired E/I routing with positive, branch-normalized weights.

Routing assigns each source coordinate to a branch. It is distinct from the
within-row Top-K contact selection of a dendritic layer. This reusable control
uses native current integration but does not claim a biological affine readout.
"""

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.config.routed_ei import RoutedEIConfig
from dendritic_modeling.networks.functional import additive_voltage, shunting_voltage
from dendritic_modeling.networks.utils.routing import (
    assignment_statistics,
    routing_assignments,
)
from dendritic_modeling.networks.utils.weight_normalization import normalize_weight_rows
from dendritic_modeling.networks.utils.weight_transforms import apply_weight_transform


class RoutedEIReadout(nn.Module):
    """Routed branch currents, integration and an affine decoder.

    A fixed assignment keeps the same initialization draws and parameter names
    as learned routing. Its unused route logits (and the local I weights for
    global-mean inhibition) receive no gradients. Empty hard-routed branches
    carry zero drive; no undisclosed balancing/reassignment step is applied.
    """

    def __init__(self, config: RoutedEIConfig):
        super().__init__()
        config = RoutedEIConfig(**asdict(config))
        config.validate()  # Fail before drawing or registering parameters.
        self.config = config
        self.features = config.features
        self.branches = config.branches
        self.input_dim = 2 * config.features
        self.output_dim = config.output_dim
        self.route_logits = nn.Parameter(torch.zeros(self.branches, self.features))
        nn.init.normal_(self.route_logits, mean=0.0, std=config.route_init_std)
        self.e_weight_raw = nn.Parameter(torch.zeros(self.branches, self.features))
        self.i_weight_raw = nn.Parameter(torch.zeros(self.branches, self.features))
        nn.init.normal_(self.e_weight_raw, mean=0.0, std=config.weight_init_std)
        nn.init.normal_(self.i_weight_raw, mean=0.0, std=config.weight_init_std)
        self.decoder = nn.Linear(self.branches, self.output_dim)
        self.register_buffer(
            "fixed_assignment", torch.tensor(config.fixed_assignment, dtype=torch.long)
        )

    @property
    def temperature(self):
        return self.config.temperature

    @temperature.setter
    def temperature(self, value):
        updated = RoutedEIConfig(**{**asdict(self.config), "temperature": value})
        updated.validate()
        self.config = updated

    def assignment(self, *, hard=None):
        hard = self._hard(hard)
        if self.fixed_assignment.numel():
            return F.one_hot(self.fixed_assignment, num_classes=self.branches).T.to(
                self.route_logits.dtype
            )
        return routing_assignments(
            self.route_logits,
            self.temperature,
            dim=0,
            hard=hard,
            training=self.training,
        )

    def _hard(self, hard):
        if hard is not None and type(hard) is not bool:
            raise ValueError("hard must be a bool or None")
        return self.config.hard_routing if hard is None else hard

    def routing_statistics(self):
        """Mean assignment entropy and squared deviation from equal branch mass."""
        if self.fixed_assignment.numel():
            zero = self.route_logits.new_zeros(())
            return zero, zero
        soft = routing_assignments(self.route_logits, self.temperature, dim=0)
        return assignment_statistics(soft, destination_dim=0)

    def routing_regularizer(self, *, entropy_weight=0.002, balance_weight=0.25):
        """Explicit objective term; forward never adds or backpropagates a loss."""
        entropy, balance = self.routing_statistics()
        return entropy_weight * entropy + balance_weight * balance

    def inferred_assignment(self):
        """Branch per source as a detached tensor on the model's device."""
        if self.fixed_assignment.numel():
            return self.fixed_assignment.detach().clone()
        return self.route_logits.argmax(dim=0).detach()

    def _inputs(self, excitation, inhibition):
        if inhibition is None:
            if excitation.ndim != 2 or excitation.shape[1] != self.input_dim:
                raise ValueError(
                    f"concatenated [E, I] needs shape [batch, {self.input_dim}]"
                )
            excitation, inhibition = excitation.split(self.features, dim=1)
        if (
            excitation.ndim != 2
            or excitation.shape[1] != self.features
            or inhibition.shape != excitation.shape
        ):
            raise ValueError("E and I must have the same [batch, features] shape")
        for value in (excitation, inhibition):
            if (
                value.dtype != self.route_logits.dtype
                or value.device != self.route_logits.device
            ):
                raise ValueError("inputs must match the model dtype and device")
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError("routed_ei inputs must be finite and nonnegative")
        return excitation, inhibition

    def drives(self, excitation, inhibition=None, *, hard=None):
        excitation, inhibition = self._inputs(excitation, inhibition)
        return self._drives(excitation, inhibition, hard=hard)

    def _drives(self, excitation, inhibition, *, hard):
        assignment = self.assignment(hard=self._hard(hard))
        floor = self.config.weight_normalization_floor
        ew, iw = (
            normalize_weight_rows(
                apply_weight_transform(raw, "softplus") * assignment,
                1,
                1.0,
                sum_floor=floor,
            )
            for raw in (self.e_weight_raw, self.i_weight_raw)
        )
        return excitation @ ew.T, inhibition @ iw.T

    def branch_values(self, excitation, inhibition=None, *, hard=None):
        excitation, inhibition = self._inputs(excitation, inhibition)
        e, i = self._drives(excitation, inhibition, hard=hard)
        if self.config.inhibitory_support == "global_mean":
            i = inhibition.mean(dim=1, keepdim=True)
        if self.config.mechanism == "additive":
            voltage = additive_voltage(e, i)
        else:
            voltage = shunting_voltage(
                e, i, leak=self.config.leak_conductance, epsilon=self.config.epsilon
            )
        return (
            self.config.branch_scale
            * (voltage - self.config.branch_center)
            / self.config.branch_divisor
        )

    def forward(self, excitation, inhibition=None, *, hard=None):
        return self.decoder(self.branch_values(excitation, inhibition, hard=hard))


def build_routed_ei(config: RoutedEIConfig) -> RoutedEIReadout:
    return RoutedEIReadout(config)


__all__ = ["RoutedEIReadout", "build_routed_ei"]
