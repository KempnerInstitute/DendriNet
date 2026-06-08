"""Tests for the shared recurrent-core protocol interfaces."""

from dendritic_modeling.networks.architectures.recurrent import (
    BaselineRNN,
    BaselineRNNConfig,
    PopulationConfig,
    RecurrentCore,
    StatefulDendriNet,
    SteppableRecurrentCore,
)


def test_baseline_rnn_satisfies_sequence_recurrent_core_protocol():
    core = BaselineRNN(BaselineRNNConfig(input_dim=3, hidden_dim=5))

    assert isinstance(core, RecurrentCore)
    assert core.output_dim == 5
    assert core.is_recurrent


def test_stateful_dendrinet_satisfies_steppable_protocol():
    population = StatefulDendriNet(
        pop_config=PopulationConfig(
            n_neurons=4,
            branch_factors=[1],
            ff_excitatory_synapses=1,
            reactivate=False,
            somatic_synapses=True,
        ),
        excitatory_input_dim=3,
    )

    assert isinstance(population, SteppableRecurrentCore)
    assert population.output_dim == 4
