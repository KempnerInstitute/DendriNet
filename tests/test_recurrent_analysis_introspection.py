"""Analysis compatibility tests for named recurrent populations."""

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.dendritic_timetraces import (
    DendriticTimetracesAnalyzer,
)
from dendritic_modeling.analysis.tools.inhibitory_specialization import (
    InhibitorySpecializationAnalyzer,
)
from dendritic_modeling.analysis.tools.routing_analysis import (
    RoutingAnalyzer,
    _append_masked_routing_values,
    _extract_seq_lengths,
    _valid_timestep_mask,
)
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
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
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


def test_recurrent_population_summary_reports_population_network_components():
    core = _population_network()

    summary = recurrent_population_summary(core)

    assert summary["population_network/microcircuit.pyr"]["polarity"] == "excitatory"
    assert summary["population_network/microcircuit.pyr"]["spiking_enabled"]
    assert summary["population_network/microcircuit.pv"]["polarity"] == "inhibitory"


def test_routing_analyzer_collects_population_network_entries():
    core = _population_network()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    data = torch.utils.data.TensorDataset(
        torch.rand(6, 3, 5), torch.randint(0, 4, (6,))
    )

    results = RoutingAnalyzer().analyze(model, data, batch_size=3)

    assert "microcircuit/pyr" in results
    assert "microcircuit/pv" in results
    assert results["microcircuit/pyr"]["n_levels"] == 2
    assert results["microcircuit/pyr"]["n_samples"] > 0
    assert "mean_raw_voltage_norm" in results["microcircuit/pyr"]
    assert not core.store_routing


def test_routing_helpers_extract_and_mask_sequence_lengths():
    seq_lengths = torch.tensor([1, 3])
    batch = (torch.zeros(2, 3, 5), torch.zeros(2), seq_lengths)

    assert _extract_seq_lengths(batch) is seq_lengths
    assert _extract_seq_lengths(batch[:2]) is None

    valid_mask, should_process = _valid_timestep_mask(None, timestep_idx=5)
    assert valid_mask is None
    assert should_process

    valid_mask, should_process = _valid_timestep_mask(seq_lengths, timestep_idx=1)
    assert valid_mask.tolist() == [False, True]
    assert should_process

    valid_mask, should_process = _valid_timestep_mask(seq_lengths, timestep_idx=3)
    assert valid_mask.tolist() == [False, False]
    assert not should_process


def test_append_masked_routing_values_filters_detached_cpu_rows():
    values = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)
    accumulator: dict[str, list[torch.Tensor]] = {}

    _append_masked_routing_values(
        accumulator,
        "layer/pyr",
        values,
        torch.tensor([False, True]),
    )

    collected = accumulator["layer/pyr"][0]
    assert collected.device.type == "cpu"
    assert not collected.requires_grad
    assert torch.equal(collected, values.detach()[1:].cpu())

    _append_masked_routing_values(
        accumulator,
        "empty",
        values,
        torch.tensor([False, False]),
    )
    assert "empty" not in accumulator


def test_dendritic_timetraces_collects_population_network_excitatory_population():
    core = _population_network()
    model = Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    params = DendriticTimetracesAnalysisParams(n_timesteps=12, save_figures=False)

    results = DendriticTimetracesAnalyzer(params).analyze(model)

    assert "population_network/microcircuit.pyr" in results["layers"]
    assert results["layers"]["population_network/microcircuit.pyr"]["n_levels"] == 2
    assert "population_network/microcircuit.pv" not in results["layers"]


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


def test_synaptic_activation_supports_population_network_topk_hooks():
    core = _population_network()
    dataset = TensorDataset(torch.ones(4, 5, 5), torch.zeros(4, dtype=torch.long))
    analyzer = SynapticActivationAnalyzer(
        SynapticActivationAnalysisParams(raw_summary=True, downsample=False)
    )

    results = analyzer.analyze(core, dataset, device="cpu")

    assert results is not None
    assert "network_summary" in results
    assert "exc" in results["network_summary"]
    assert "rec_inh" in results["network_summary"]
    raw_synapse_types = {
        synapse_type
        for layer_data in results["raw_summary"].values()
        for synapse_type in layer_data
    }
    assert {"exc", "rec_inh"}.issubset(raw_synapse_types)
