"""Tests for shared population-network configuration helpers."""

from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
    population_config_from_definition,
)


def test_population_config_additive_default_uses_occupancy_quantile() -> None:
    pop = PopulationDefinitionConfig(
        name="e",
        polarity="excitatory",
        n_neurons=8,
        branch_factors=[2],
    )

    cfg = population_config_from_definition(
        pop,
        {"use_shunting": False},
        layer_idx=0,
    )

    assert cfg.reactivation_init_policy == "occupancy_quantile"


def test_population_config_shunting_default_stays_analytical() -> None:
    pop = PopulationDefinitionConfig(
        name="e",
        polarity="excitatory",
        n_neurons=8,
        branch_factors=[2],
    )

    cfg = population_config_from_definition(
        pop,
        {"use_shunting": True},
        layer_idx=0,
    )

    assert cfg.reactivation_init_policy == "analytical"


def test_population_config_preserves_explicit_reactivation_policy() -> None:
    pop = PopulationDefinitionConfig(
        name="e",
        polarity="excitatory",
        n_neurons=8,
        branch_factors=[2],
    )

    cfg = population_config_from_definition(
        pop,
        {"use_shunting": False, "reactivation_init_policy": "fixed"},
        layer_idx=0,
    )

    assert cfg.reactivation_init_policy == "fixed"
