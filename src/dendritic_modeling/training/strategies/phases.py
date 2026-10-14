"""Helpers for trainers that run several phases (warm-ups, two-step training)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from torch import nn

HISTORY_KEYS = ("train_losses", "valid_losses", "train_losses_base", "train_losses_reg")


@contextmanager
def evaluation_mode(module: nn.Module) -> Iterator[nn.Module]:
    """Run ``module`` in eval mode and restore its previous mode afterwards."""
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)


def merge_phase_results(
    first: Mapping[str, Any],
    final: dict[str, Any],
    *,
    first_phase: str,
    final_phase: str,
    first_loss: str | None = None,
) -> dict[str, Any]:
    """Prepend the first phase's loss histories to ``final`` and name the phases.

    A first phase without auxiliary terms may omit ``train_losses_base`` and
    ``train_losses_reg``; they are its total loss and zero, so the four
    histories stay aligned. ``phases`` gives each phase's epochs and the loss
    the first phase's values measure.
    """
    first_train = list(first.get("train_losses", []))
    final_epochs = len(final.get("train_losses", []))
    missing = {
        "train_losses_base": first_train,
        "train_losses_reg": [0.0] * len(first_train),
    }
    for key in HISTORY_KEYS:
        earlier = first.get(key, missing.get(key, []))
        final[key] = list(earlier) + list(final.get(key, []))
    final["phases"] = [
        {"name": first_phase, "epochs": len(first_train), "loss": first_loss},
        {"name": final_phase, "epochs": final_epochs},
    ]
    return final


__all__ = ["HISTORY_KEYS", "evaluation_mode", "merge_phase_results"]
