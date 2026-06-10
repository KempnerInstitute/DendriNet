"""Analysis compatibility tests for named recurrent populations."""

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.dendritic_timetraces import (
    DendriticTimetracesAnalyzer,
)
from dendritic_modeling.analysis.tools.inhibitory_specialization import (
    InhibitorySpecializationAnalyzer,
)
from dendritic_modeling.analysis.tools.routing_analysis import RoutingAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    recurrent_population_summary,
)
from dendritic_modeling.config.analysis import (
    DendriticTimetracesAnalysisParams,
    InhibitorySpecializationAnalysisParams,
    SynapticActivationAnalysisParams,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.recurrent import (
    CellTypeConfig,
    PopulationGraphConfig,
    PopulationGraphNetwork,
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _population_graph():
    return PopulationGraphNetwork(
        PopulationGraphConfig(
            input_dim=5,
            readout_population="pyr",
            output_mode="last",
            population_defaults={
                "ff_excitatory_synapses": 2,
                "ff_inhibitory_synapses": 0,
                "rec_excitatory_synapses": 1,
                "rec_inhibitory_synapses": 1,
                "somatic_synapses": True,
                "reactivate": True,
                "dbl_init_method": "default",
                "weight_transform": "exp",
                "dynamics_mode": "spike",
                "spike_readout": "rate",
                "spike_threshold": -0.1,
            },
            cell_types=[
                CellTypeConfig(
                    name="pyr",
                    polarity="excitatory",
                    n_neurons=4,
                    branch_factors=[1],
                ),
                CellTypeConfig(
                    name="pv",
                    polarity="inhibitory",
                    n_neurons=2,
                    branch_factors=[1],
                ),
            ],
        )
    )


def _population_network():
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=5,
            output_mode="last",
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": True,
                        "dbl_init_method": "default",
                        "weight_transform": "exp",
                        "dynamics_mode": "spike",
                        "spike_readout": "rate",
                        "spike_threshold": -0.1,
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                            "population": {"rec_inhibitory_synapses": 1},
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "pv"},
                        {"source": "pv", "target": "pyr", "timing": "delayed"},
                    ],
                )
            ],
        )
    )


def test_recurrent_population_summary_reports_new_components():
    core = _population_graph()

    summary = recurrent_population_summary(core)

    assert summary["population_graph/pyr"]["polarity"] == "excitatory"
    assert summary["population_graph/pyr"]["spiking_enabled"]
    assert summary["population_graph/pyr"]["spike_readout"] == "rate"
    assert summary["population_graph/pv"]["polarity"] == "inhibitory"


def test_recurrent_population_summary_reports_population_network_components():
    core = _population_network()

    summary = recurrent_population_summary(core)

    assert summary["population_network/microcircuit.pyr"]["polarity"] == "excitatory"
    assert summary["population_network/microcircuit.pyr"]["spiking_enabled"]
    assert summary["population_network/microcircuit.pv"]["polarity"] == "inhibitory"


def test_routing_analyzer_collects_population_graph_entries():
    core = _population_graph()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    data = torch.utils.data.TensorDataset(
        torch.rand(6, 3, 5), torch.randint(0, 4, (6,))
    )

    results = RoutingAnalyzer().analyze(model, data, batch_size=3)

    assert "pyr" in results
    assert "pv" in results
    assert results["pyr"]["n_levels"] == 2


def test_dendritic_timetraces_collects_population_graph_excitatory_population():
    core = _population_graph()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    params = DendriticTimetracesAnalysisParams(n_timesteps=12, save_figures=False)

    results = DendriticTimetracesAnalyzer(params).analyze(model)

    assert "population_graph/pyr" in results["layers"]
    assert results["layers"]["population_graph/pyr"]["n_levels"] == 2
    assert "population_graph/pv" not in results["layers"]


def test_inhibitory_specialization_collects_population_graph_target():
    core = _population_graph()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    params = InhibitorySpecializationAnalysisParams(save_figures=False)

    results = InhibitorySpecializationAnalyzer(params).analyze(model)

    entry = results["layers"]["population_graph/pyr"]
    assert entry["n_E"] == 4
    assert entry["n_I"] == 2
    assert entry["source_populations"] == ["pv"]


def test_inhibitory_specialization_collects_population_network_target():
    core = _population_network()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    params = InhibitorySpecializationAnalysisParams(save_figures=False)

    results = InhibitorySpecializationAnalyzer(params).analyze(model)

    entry = results["layers"]["population_network/microcircuit.pyr"]
    assert entry["n_E"] == 4
    assert entry["n_I"] == 2
    assert entry["source_populations"] == ["pv"]


def test_synaptic_activation_supports_population_graph_topk_hooks():
    core = _population_graph()
    dataset = TensorDataset(torch.ones(4, 5, 5), torch.zeros(4, dtype=torch.long))
    analyzer = SynapticActivationAnalyzer(
        SynapticActivationAnalysisParams(raw_summary=False, downsample=False)
    )

    results = analyzer.analyze(core, dataset, device="cpu")

    assert results is not None
    assert "network_summary" in results
    assert "exc" in results["network_summary"]
    assert "rec_inh" in results["network_summary"]
