"""Deliver past population outputs to existing synapses without changing their weights."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from dendritic_modeling.config.population_routing import projection_delays
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    resolve_synapses_by_level,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationProjectionConfig,
)


@dataclass(frozen=True)
class DelayedStream:
    """One vector of source ages for each distinct timing pattern in a pathway."""

    sources: tuple[str, ...]
    patterns: tuple[tuple[int, ...], ...]
    level_indices: tuple[int, ...]

    def read(
        self, previous: list[torch.Tensor], history: dict[str, torch.Tensor] | None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if history is None:
            raise ValueError(
                "This population network requires its multi-step delay history"
            )
        streams = []
        for pattern in self.patterns:
            values = [
                value if age == 1 else history[source][:, age - 2, :]
                for source, value, age in zip(self.sources, previous, pattern)
            ]
            streams.append(values[0] if len(values) == 1 else torch.cat(values, dim=-1))
        if len(streams) == 1:
            return streams[0]
        return tuple(streams[index] for index in self.level_indices)


def plan_delayed_streams(
    connections: list[PopulationProjectionConfig],
    populations: dict[str, PopulationConfig],
) -> tuple[dict[tuple[str, str], DelayedStream], dict[str, int]]:
    """Group equal delays across levels and retain older outputs only where used."""
    by_stream = defaultdict(list)
    history_lengths: dict[str, int] = {}
    for conn in connections:
        if conn.timing != "delayed":
            continue
        pop = populations[conn.target]
        n = len(pop.branch_factors) + 1
        delays = projection_delays(
            conn.timing, conn.delay_steps, conn.delay_steps_by_level, n
        )
        if not conn.enabled:
            continue
        # A zero-probability source still occupies its input columns, but
        # its mask is empty at every depth, so no older samples are needed.
        if conn.probability == 0:
            delays = (1,) * n
        elif max(delays) > 1:
            depths = conn.target_depths(n)
            field = f"{conn.pathway}_synapses"
            contacts = resolve_synapses_by_level(
                per_level=getattr(pop, field + "_by_level"),
                fallback=getattr(pop, field),
                n_levels=n,
                field_name=field + "_by_level",
            )
            delays = tuple(
                (
                    age
                    if count
                    and (depth != 0 or pop.somatic_synapses)
                    and (depths is None or depth in depths)
                    else 1
                )
                for age, count, depth in zip(delays, contacts, reversed(range(n)))
            )
        if max(delays) > 1:
            history_lengths[conn.source] = max(
                history_lengths.get(conn.source, 0), max(delays) - 1
            )
        by_stream[conn.target, conn.pathway].append((conn.source, delays))

    streams = {}
    for key, routes in by_stream.items():
        sources, delays_by_route = zip(*routes)
        if all(max(delays) == 1 for delays in delays_by_route):
            continue
        patterns_by_level = tuple(zip(*delays_by_route))
        patterns = tuple(dict.fromkeys(patterns_by_level))
        indices = {pattern: i for i, pattern in enumerate(patterns)}
        streams[key] = DelayedStream(
            sources, patterns, tuple(indices[p] for p in patterns_by_level)
        )
    return streams, history_lengths
