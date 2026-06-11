"""Tests for spike-train analysis on opt-in LIF recurrent populations."""

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.spike_trains import SpikeTrainAnalyzer
from dendritic_modeling.config.analysis import SpikeTrainAnalysisParams
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
    PopulationProjectionConfig,
)


def _spiking_population_network() -> PopulationNetwork:
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=3,
            output_mode="all",
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    population_defaults={
                        "ff_excitatory_synapses": 1,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 1,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "dynamics_mode": "spike",
                        "spike_threshold": -0.1,
                        "spike_tau": 2.0,
                        "spike_readout": "spikes",
                    },
                    connections=[
                        PopulationProjectionConfig(source="input", target="pyr"),
                        PopulationProjectionConfig(
                            source="pyr",
                            target="pyr",
                            timing="delayed",
                            probability=1.0,
                        ),
                    ],
                )
            ],
        )
    )


def test_spike_train_analyzer_reports_population_statistics():
    torch.manual_seed(0)
    model = _spiking_population_network()
    dataset = TensorDataset(torch.ones(5, 6, 3), torch.zeros(5, dtype=torch.long))
    analyzer = SpikeTrainAnalyzer(
        SpikeTrainAnalysisParams(max_samples=5, include_raster=True)
    )

    results = analyzer.analyze(model, dataset, device="cpu")

    assert "populations" in results
    assert len(results["populations"]) == 1
    stats = next(iter(results["populations"].values()))
    assert stats["n_samples"] == 5
    assert stats["n_timesteps"] == 6
    assert stats["n_units"] == 4
    assert stats["total_spikes"] > 0
    assert stats["fano_factor_mean"] >= 0
    assert isinstance(stats["raster_events"], list)


def test_spike_train_summary_helpers_match_loop_semantics():
    seq_a = torch.zeros(2, 4, 3)
    seq_a[0, [0, 2], 0] = 1
    seq_a[0, [1, 3], 1] = 1
    seq_a[1, [0, 1, 3], 2] = 1
    seq_b = torch.zeros(1, 3, 3)
    seq_b[0, [0, 1, 2], 0] = 1
    sequences = [seq_a, seq_b]

    isi = SpikeTrainAnalyzer._collect_isis(sequences)
    assert sorted(isi.tolist()) == [1, 1, 1, 2, 2, 2]

    expected_rates = []
    for t_idx in range(4):
        vals = [seq[:, t_idx, :] for seq in sequences if t_idx < seq.shape[1]]
        expected_rates.append(torch.cat(vals, dim=0).float().mean())
    assert torch.allclose(
        SpikeTrainAnalyzer._per_timestep_rate(sequences),
        torch.stack(expected_rates),
    )

    analyzer = SpikeTrainAnalyzer(
        SpikeTrainAnalysisParams(include_raster=True, max_raster_samples=2)
    )
    assert analyzer._compact_raster_events(sequences) == [
        {"sample": 0, "time": 0, "unit": 0},
        {"sample": 0, "time": 1, "unit": 1},
        {"sample": 0, "time": 2, "unit": 0},
        {"sample": 0, "time": 3, "unit": 1},
        {"sample": 1, "time": 0, "unit": 2},
        {"sample": 1, "time": 1, "unit": 2},
        {"sample": 1, "time": 3, "unit": 2},
    ]
