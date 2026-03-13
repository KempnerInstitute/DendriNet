# Dendritic Modeling Framework

Code for dendritic excitation-inhibition networks, matched MLP baselines,
recurrent E-I models, training strategies, and analysis utilities.

<p align="center">
  <img src="docs/_static/dendritic_diagram.png" alt="Dendritic model diagram" width="600">
</p>

## Installation

```bash
bash install.sh
```

Manual setup is documented in [INSTALL.md](INSTALL.md).

## Common Commands

Train a standard experiment:

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/config_exp.yaml
```

Generate an FSDP SLURM script:

```bash
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py \
    configs/config_exp_fsdp.yaml --generate-only
```

Run a sweep from an example config:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/EI_sweeps_example.yaml --generate-only
```

Analyze an existing sweep:

```bash
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    /path/to/sweep/results
```

## Documentation

The maintained user docs live under [docs/README.md](docs/README.md).
Start with:

- [docs/network_architectures.rst](docs/network_architectures.rst)
- [docs/training_strategies.rst](docs/training_strategies.rst)
- [docs/datasets.rst](docs/datasets.rst)
- [docs/analysis.rst](docs/analysis.rst)
- [docs/sweeps.rst](docs/sweeps.rst)
- [docs/fsdp_troubleshooting.rst](docs/fsdp_troubleshooting.rst)
- [docs/config_reference.rst](docs/config_reference.rst)

## Core Type Strings

Feedforward cores:

- `einet`
- `dendritic_shunting`
- `dendritic_additive`
- `flat_shunting`
- `flat_additive`
- `dendritic_mlp`
- `mlp`

Unified recurrent-capable E-I core:

- `ei_unified`
- aliases: `unified_ei`, `ei_net`, `unified_einet`

Baseline recurrent cores:

- `gru`
- `lstm`
- `vanilla_rnn`
- alias: `rnn`

## Repository Layout

- `src/dendritic_modeling/networks/`: architectures and layers
- `src/dendritic_modeling/training/`: trainer implementations and utilities
- `src/dendritic_modeling/analysis/`: analyzers and metric code
- `src/dendritic_modeling/scripts/`: CLI entry points
- `configs/`: experiment, sweep, and comparison configs
- `tests/`: pytest-based test suite

## License

MIT. See [LICENSE](LICENSE).
