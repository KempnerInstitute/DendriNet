"""Contract for the current million-neuron dendritic RNN configuration."""

from pathlib import Path

from dendritic_modeling.config import load_config

CONFIG_PATH = Path(
    "configs/dendritic_rnn/large_scale_1m_10k/" "test_1m_20k_frozen_readout_5000ep.yaml"
)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _level_multiplicities(branch_factors):
    return [
        _product(branch_factors[level_idx:])
        for level_idx in range(len(branch_factors) + 1)
    ]


def _contacts_per_target(population):
    multiplicities = _level_multiplicities(population["branch_factors"])
    excitatory = sum(
        multiplicity * (feedforward + recurrent)
        for multiplicity, feedforward, recurrent in zip(
            multiplicities,
            population["ff_excitatory_synapses_by_level"],
            population["rec_excitatory_synapses_by_level"],
            strict=True,
        )
    )
    inhibitory = sum(
        multiplicity * (feedforward + recurrent)
        for multiplicity, feedforward, recurrent in zip(
            multiplicities,
            population["ff_inhibitory_synapses_by_level"],
            population["rec_inhibitory_synapses_by_level"],
            strict=True,
        )
    )
    return excitatory, inhibitory


def test_current_exact_1m_20k_5000_epoch_contract():
    config = load_config(str(CONFIG_PATH))
    layer = config.model.core.unified_ei["layers"][0]
    excitatory = layer["excitatory"]
    inhibitory = layer["inhibitory"]

    assert excitatory["n_neurons"] == 800_000
    assert inhibitory["n_neurons"] == 200_000
    assert excitatory["branch_factors"] == [3, 3, 3]
    assert inhibitory["branch_factors"] == []
    assert _contacts_per_target(excitatory) == (16_000, 4_000)
    assert _contacts_per_target(inhibitory) == (16_000, 4_000)
    assert config.distributed.fsdp.freeze_core is True
    assert config.distributed.fsdp.cache_frozen_core_outputs is True
    assert config.distributed.fsdp.cached_core_batch_size == 4
    assert config.training.main.common.epochs == 5000
    assert config.training.main.common.print_every == 100
    assert config.model.decoder.params.hidden_dims == [32]
    assert config.model.decoder.params.input_normalization == "layer_norm"
    assert config.model.decoder.params.output_init == "zero"
    assert config.outputs.exact_run_dir is True
    assert config.outputs.results_dir.endswith("test_1m_20k_frozen_readout_5000ep")
