"""Runtime state containers for population-network layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState


@dataclass
class PopulationLayerState:
    """Runtime state for one population-network layer."""

    population_states: dict[str, DendriNetState]
    outputs: dict[str, torch.Tensor]
    # Supplied one-step-old source activities from earlier updates, newest first.
    # Each tensor has shape [batch, extra delay steps, source neurons].
    delay_history: dict[str, torch.Tensor] | None = None

    def detach(self) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.detach() for name, state in self.population_states.items()
            },
            outputs={name: output.detach() for name, output in self.outputs.items()},
            delay_history=(
                {name: v.detach() for name, v in self.delay_history.items()}
                if self.delay_history
                else None
            ),
        )

    def clone(self) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.clone() for name, state in self.population_states.items()
            },
            outputs={name: output.clone() for name, output in self.outputs.items()},
            delay_history=(
                {name: v.clone() for name, v in self.delay_history.items()}
                if self.delay_history
                else None
            ),
        )

    def to(self, device: torch.device) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.to(device) for name, state in self.population_states.items()
            },
            outputs={name: output.to(device) for name, output in self.outputs.items()},
            delay_history=(
                {name: v.to(device) for name, v in self.delay_history.items()}
                if self.delay_history
                else None
            ),
        )


__all__ = ["PopulationLayerState"]
