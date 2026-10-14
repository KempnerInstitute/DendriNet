"""Analysis-only replacements of a shunting branch layer's voltage rule.

A trained shunting layer computes its branch voltage as ``N / (1 + T)``, with
numerator ``N = E + C`` (excitatory conductance plus child-branch current) and
conductance ``T = E + I + G`` (excitation, inhibition and child coupling,
without the unit leak). An override keeps every synapse, coupling and gate of
the trained layer and replaces only that algebra, so an analysis can ask which
part of the rule a trained network relies on. Overrides hold no parameters and
leave the state dict unchanged; they apply while :func:`override_branch_rules`
(or :meth:`DendriticBranchLayer.override_branch_rule`) is active.

Rules (``n0`` and ``t0`` are a layer's anchor numerator and conductance,
``d0 = 1 + t0`` and ``v0 = n0 / d0``; rolls shift by one sample along the
batch):

* ``anchor_tangent``: ``v0 + ((N - n0) - v0 (T - t0)) / d0``, the tangent of
  the shunting rule at the anchor.
* ``sample_rolled_tangent``: the anchor tangent with ``T`` taken from the
  previous sample of the batch.
* ``frozen_denominator``: ``N / d0``.
* ``sample_rolled_denominator``: ``N / (1 + T)`` with ``T`` taken from the
  previous sample of the batch.
* ``external_denominator``: ``N / (1 + T_ext)`` for a given conductance
  ``T_ext`` that broadcasts against ``T``.
* ``permuted_current_denominator``: ``N / (1 + T[p])`` for a batch-local
  permutation ``p``; a conductance without a batch dimension is used as is.
* ``raw_additive``: ``E - I + C``, the additive rule (with the layer's
  across-feature z-score when it has one).
* ``raw_additive_moment_matched``: the additive drive standardized with the
  given raw mean and standard deviation, then rescaled to the given voltage
  mean and standard deviation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

# Tensor fields each rule reads; every other field must stay unset.
_RULE_FIELDS: dict[str, tuple[str, ...]] = {
    "anchor_tangent": ("anchor_numerator", "anchor_conductance"),
    "sample_rolled_tangent": ("anchor_numerator", "anchor_conductance"),
    "frozen_denominator": ("anchor_conductance",),
    "sample_rolled_denominator": (),
    "external_denominator": ("external_conductance",),
    "permuted_current_denominator": ("denominator_permutation",),
    "raw_additive": (),
    "raw_additive_moment_matched": (
        "raw_mean",
        "raw_std",
        "voltage_mean",
        "voltage_std",
    ),
}

BRANCH_RULE_OVERRIDES = frozenset(_RULE_FIELDS)


@dataclass(frozen=True)
class BranchRuleOverride:
    """One analysis-only voltage rule for a shunting branch layer.

    ``rule`` selects the algebra (see the module docstring); each rule takes
    exactly the tensors it reads, and the others must be left unset.
    """

    rule: str
    anchor_numerator: Optional[torch.Tensor] = None
    anchor_conductance: Optional[torch.Tensor] = None
    external_conductance: Optional[torch.Tensor] = None
    denominator_permutation: Optional[torch.Tensor] = None
    raw_mean: Optional[torch.Tensor] = None
    raw_std: Optional[torch.Tensor] = None
    voltage_mean: Optional[torch.Tensor] = None
    voltage_std: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.rule not in _RULE_FIELDS:
            choices = ", ".join(sorted(_RULE_FIELDS))
            raise ValueError(
                f"unknown branch rule override {self.rule!r} (expected one of "
                f"{choices})"
            )
        needed = _RULE_FIELDS[self.rule]
        tensors = [field.name for field in fields(self) if field.name != "rule"]
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            raise ValueError(f"rule {self.rule!r} needs {', '.join(missing)}")
        unused = [
            name
            for name in tensors
            if name not in needed and getattr(self, name) is not None
        ]
        if unused:
            raise ValueError(f"rule {self.rule!r} does not use {', '.join(unused)}")
        for name in needed:
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
            if name == "denominator_permutation":
                if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                    raise ValueError(
                        "denominator_permutation must be a one-dimensional integer "
                        "tensor with one index per sample"
                    )
            elif not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")

    def voltage(
        self,
        owner: Any,
        numerator: torch.Tensor,
        conductance: torch.Tensor,
        *,
        excitation: Any,
        inhibition: Any,
        child_current: Any,
    ) -> torch.Tensor:
        """Branch voltage of this rule for the layer ``owner``."""
        rule = self.rule
        epsilon = owner.epsilon

        def tensor(name: str, like: torch.Tensor) -> torch.Tensor:
            return getattr(self, name).to(device=like.device, dtype=like.dtype)

        if rule == "raw_additive":
            return owner.normalize_additive_voltage(
                excitation - inhibition + child_current
            )
        if rule == "raw_additive_moment_matched":
            raw = excitation - inhibition + child_current
            scale = tensor("voltage_std", raw) / tensor("raw_std", raw).clamp_min(1e-8)
            return (raw - tensor("raw_mean", raw)) * scale + tensor("voltage_mean", raw)
        if rule == "external_denominator":
            return numerator / (
                1 + tensor("external_conductance", conductance) + epsilon
            )
        if rule == "permuted_current_denominator":
            # A conductance without a batch dimension (for example one fixed
            # value per branch) has no sample pairing to permute.
            if (
                conductance.ndim != numerator.ndim
                or conductance.ndim == 0
                or conductance.shape[0] != numerator.shape[0]
            ):
                return numerator / (1 + conductance + epsilon)
            permutation = self.denominator_permutation
            if permutation.numel() != conductance.shape[0]:
                raise ValueError(
                    "denominator_permutation has "
                    f"{permutation.numel()} indices for a batch of "
                    f"{conductance.shape[0]} samples"
                )
            paired = conductance.index_select(
                0, permutation.to(device=conductance.device, dtype=torch.long)
            )
            return numerator / (1 + paired + epsilon)
        if rule == "sample_rolled_denominator":
            return numerator / (1 + torch.roll(conductance, 1, dims=0) + epsilon)

        t0 = tensor("anchor_conductance", conductance)
        d0 = 1 + t0 + epsilon
        if rule == "frozen_denominator":
            return numerator / d0
        n0 = tensor("anchor_numerator", numerator)
        v0 = n0 / d0
        if rule == "sample_rolled_tangent":
            conductance = torch.roll(conductance, 1, dims=0)
        return v0 + ((numerator - n0) - v0 * (conductance - t0)) / d0


def check_branch_rule_override_target(layer: Any) -> None:
    """Raise unless ``layer`` computes the rule an override replaces."""
    if not getattr(layer, "use_shunting", False):
        raise ValueError(
            "branch rule overrides replace the shunting branch algebra; an "
            "additive layer would ignore them"
        )
    if layer.learnable_leak or layer.leak_conductance != 1:
        raise ValueError("branch rule overrides require a fixed unit leak")


@contextmanager
def override_branch_rules(
    overrides: Mapping[Any, BranchRuleOverride],
) -> Iterator[None]:
    """Apply analysis-only voltage rules to branch layers within the block.

    ``overrides`` maps each :class:`DendriticBranchLayer` to its
    :class:`BranchRuleOverride` (anchors are per layer). The layers' previous
    rules are restored on exit, also after an exception. The feedforward
    forward pass applies the overrides; computing voltages from currents
    (recurrent and stateful paths) raises while one is active.
    """
    for layer, override in overrides.items():
        if not isinstance(override, BranchRuleOverride):
            raise TypeError(
                f"expected a BranchRuleOverride, got {type(override).__name__}"
            )
        check_branch_rule_override_target(layer)
    previous = [
        (layer, layer.__dict__.get("_analysis_rule_override")) for layer in overrides
    ]
    try:
        for layer, override in overrides.items():
            layer._analysis_rule_override = override
        yield
    finally:
        for layer, rule in previous:
            layer._analysis_rule_override = rule


def active_branch_rule_override(owner: Any) -> Optional[BranchRuleOverride]:
    """The override a forward pass of ``owner`` applies, or ``None``."""
    override = getattr(owner, "_analysis_rule_override", None)
    if override is None or isinstance(override, BranchRuleOverride):
        return override
    raise TypeError(
        "branch rule overrides are applied with override_branch_rules(...) or "
        "DendriticBranchLayer.override_branch_rule(...), not by setting "
        "_analysis_rule_override"
    )


__all__ = [
    "BRANCH_RULE_OVERRIDES",
    "BranchRuleOverride",
    "active_branch_rule_override",
    "check_branch_rule_override_target",
    "override_branch_rules",
]
