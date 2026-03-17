# Dendritic Modeling Framework

[![CI Pipeline](https://github.com/KempnerInstitute/dendritic-modeling/actions/workflows/ci-simplified.yml/badge.svg?branch=develop)](https://github.com/KempnerInstitute/dendritic-modeling/actions/workflows/ci-simplified.yml)
[![codecov](https://codecov.io/gh/KempnerInstitute/dendritic-modeling/branch/develop/graph/badge.svg)](https://codecov.io/gh/KempnerInstitute/dendritic-modeling)
[![Documentation](https://github.com/KempnerInstitute/dendritic-modeling/actions/workflows/deploy-docs.yml/badge.svg?branch=develop)](https://github.com/KempnerInstitute/dendritic-modeling/actions/workflows/deploy-docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/KempnerInstitute/dendritic-modeling/blob/develop/pyproject.toml)

A framework for biologically-inspired dendritic excitation-inhibition networks:
feedforward and recurrent architectures with shunting inhibition, sparse
connectivity, and hierarchical dendritic trees.

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

Distributed training (DDP or FSDP) — set `distributed.mode` in your YAML
config and use the unified launcher:

```bash
# Auto-selects entrypoint based on distributed.mode in YAML
sbatch scripts/train.sh configs/my_config.yaml

# Multi-node
sbatch --nodes=2 scripts/train.sh configs/my_config.yaml

# Or use dedicated launchers directly
sbatch scripts/train_ddp.sh configs/my_config.yaml
sbatch scripts/train_fsdp.sh configs/my_config.yaml
```

Run a sweep from an example config:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/sweep_config.yaml --generate-only
```

Analyze an existing sweep:

```bash
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    /path/to/sweep/results
```

Analysis runtime is configured centrally under `analysis.runtime` in YAML.
Use `mode: "stream"` there for large datasets, and set `max_samples` for
expensive analyses that should only inspect a bounded subset. Keep
`mode: "auto"` for the normal default, and use `mode: "materialize"` when you
want the closest match to the old exact small-dataset single-GPU analysis path.
If `analysis.runtime` is omitted entirely, non-distributed runs default to that
backward-compatible materialized behavior automatically.
Analyzers may temporarily move the model to the requested analysis device, but
they restore the original device and train/eval mode before returning.

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

## Core Network Types

Feedforward dendritic cores:

- `einet` / `dendritic_shunting` — shunting (divisive) inhibition
- `dendritic_additive` — additive inhibition
- `flat_shunting` / `flat_additive` — single-compartment variants

Recurrent dendritic cores:

- `unified_ei` — hierarchical recurrent E-I with learnable time constants
- aliases: `ei_unified`, `ei_net`, `unified_einet`

Pretrained-replacement cores (vision):

- `hierarchical_dendritic_tensor_map` — replace FC layers in pretrained CNNs
- `hierarchical_dendritic_conv` — spatially local dendritic convolution

## Repository Layout

- `src/dendritic_modeling/networks/`: architectures and layers
- `src/dendritic_modeling/training/`: trainer implementations and utilities
- `src/dendritic_modeling/analysis/`: analyzers and metric code
- `src/dendritic_modeling/scripts/`: CLI entry points
- `configs/`: experiment, sweep, and comparison configs
- `tests/`: pytest-based test suite

## License

MIT. See [LICENSE](LICENSE).
