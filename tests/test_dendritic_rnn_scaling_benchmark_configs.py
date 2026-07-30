"""Contracts for the matched dendritic-RNN scaling benchmark."""

import json
from pathlib import Path

from dendritic_modeling.config import load_config

CONFIG_DIR = Path("configs/dendritic_rnn/scaling_benchmark_v1")


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _multiplicities(branch_factors):
    return [
        _product(branch_factors[level:]) for level in range(len(branch_factors) + 1)
    ]


def _contacts(population):
    multiplicities = _multiplicities(population["branch_factors"])
    excitatory = sum(
        count * multiplicity
        for count, multiplicity in zip(
            (
                ff + recurrent
                for ff, recurrent in zip(
                    population["ff_excitatory_synapses_by_level"],
                    population["rec_excitatory_synapses_by_level"],
                    strict=True,
                )
            ),
            multiplicities,
            strict=True,
        )
    )
    inhibitory = sum(
        count * multiplicity
        for count, multiplicity in zip(
            (
                ff + recurrent
                for ff, recurrent in zip(
                    population["ff_inhibitory_synapses_by_level"],
                    population["rec_inhibitory_synapses_by_level"],
                    strict=True,
                )
            ),
            multiplicities,
            strict=True,
        )
    )
    return excitatory, inhibitory


def test_frozen_scaling_configs_change_only_declared_scale_contract():
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text())

    for scale in manifest["scales"]:
        config = load_config(str(CONFIG_DIR / scale["config"]))
        layer = config.model.core.unified_ei["layers"][0]
        excitatory = layer["excitatory"]
        inhibitory = layer["inhibitory"]

        assert excitatory["n_neurons"] == scale["excitatory_neurons"]
        assert inhibitory["n_neurons"] == scale["inhibitory_neurons"]
        assert excitatory["branch_factors"] == [3, 3, 3]
        assert inhibitory["branch_factors"] == []
        assert _contacts(excitatory) == (16_000, 4_000)
        assert _contacts(inhibitory) == (16_000, 4_000)
        assert config.distributed.fsdp.freeze_core is True
        assert config.distributed.fsdp.cache_frozen_core_outputs is True
        assert config.distributed.fsdp.cached_core_batch_size == 4
        assert config.training.main.common.epochs == 150
        assert (
            config.training.main.common.param_groups.decoder_input_lr
            == scale["decoder_input_lr"]
        )


def test_end_to_end_canary_trains_recurrent_core():
    config = load_config(str(CONFIG_DIR / "e2e_100k_20k_canary.yaml"))
    layer = config.model.core.unified_ei["layers"][0]
    groups = config.training.main.common.param_groups

    assert config.distributed.fsdp.freeze_core is False
    assert config.distributed.fsdp.cache_frozen_core_outputs is False
    assert config.distributed.fsdp.gradient_checkpointing is True
    assert config.training.main.common.epochs == 1
    assert groups.topk_lr == 1.0e-5
    assert groups.reactivation_lr == 1.0e-5
    for population in (layer["excitatory"], layer["inhibitory"]):
        assert population["indexed_cache_transformed_weights"] is True
        assert population["indexed_recompute_backward"] is True


def test_stable_end_to_end_variant_changes_only_learning_rate_allocation():
    baseline = load_config(str(CONFIG_DIR / "e2e_100k_20k_30ep.yaml"))
    stable = load_config(str(CONFIG_DIR / "e2e_100k_20k_stable_30ep.yaml"))
    baseline_groups = baseline.training.main.common.param_groups
    stable_groups = stable.training.main.common.param_groups

    assert baseline_groups.topk_lr == 1.0e-5
    assert baseline_groups.decoder_input_lr == 8.0e-7
    assert stable_groups.topk_lr == 1.0e-7
    assert stable_groups.blocklinear_lr == 1.0e-7
    assert stable_groups.reactivation_lr == 1.0e-7
    assert stable_groups.decoder_input_lr == 4.0e-6
    assert stable.model.core.unified_ei == baseline.model.core.unified_ei
    assert stable.data == baseline.data
