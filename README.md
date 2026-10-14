# DendriNet

[![CI Pipeline](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml/badge.svg?branch=develop)](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml?query=branch%3Adevelop)
[![Coverage](https://raw.githubusercontent.com/KempnerInstitute/DendriNet/develop/docs/_static/badges/coverage.svg)](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml?query=branch%3Adevelop)
[![Documentation](https://github.com/KempnerInstitute/DendriNet/actions/workflows/deploy-docs.yml/badge.svg?branch=develop)](https://github.com/KempnerInstitute/DendriNet/actions/workflows/deploy-docs.yml?query=branch%3Adevelop)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

DendriNet (Python package `dendritic_modeling`) is a PyTorch library for
neural networks built from dendritic neurons. Each neuron is a tree of branch
compartments with sparse excitatory and inhibitory synapses, combined by
shunting or additive integration and passed through reactivation gates.
Population networks arrange such neurons in named excitatory and inhibitory
populations, feedforward or recurrent, with optional receptor synapse types,
spiking somas, dendritic spikes and soma-to-dendrite feedback. The library
also provides matched point-neuron and recurrent baselines, training by
backpropagation, feedback alignment and local credit-assignment rules,
distributed training, parameter sweeps and analysis tools.

<p align="center">
  <img src="docs/_static/dendritic_diagram.png" alt="Diagram of a dendritic neuron: a soma with basal and apical dendrite trees and an axon; excitatory synapses are drawn as triangles and inhibitory synapses as squares." width="600">
</p>

## Installation

```bash
git clone --branch develop https://github.com/KempnerInstitute/DendriNet.git
cd DendriNet
pip install -e .
```

Python 3.10 or newer and PyTorch 2.8 or newer are required.
[docs/installation.rst](docs/installation.rst) describes the tested
environment, the pinned dependency versions and the optional extras.

## First steps

Run from the repository root:

```bash
# Train a population network on MNIST for one epoch
dendritic-train configs/examples/population_network_feedforward.yaml --epochs 1

# Check a configuration without training
dendritic-train configs/examples/population_network_two_layer.yaml --validate-only

# Generate the configurations of a parameter sweep without submitting jobs
dendritic-sweep --config configs/sweep_config.yaml --generate-only
```

The installed commands include `dendritic-train` (training),
`dendritic-sweep` and `dendritic-analyze` (sweeps and their results),
`dendritic-analyze-checkpoint` (analysis of a trained checkpoint) and
`dendritic-compact-checkpoint` (compact export of a trained model);
[docs/cli_reference.rst](docs/cli_reference.rst) lists every command and its
options. Distributed training runs under `torchrun`; see
[docs/distributed_training.rst](docs/distributed_training.rst).

## Documentation

The documentation is built with Sphinx from `docs/`:

```bash
pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Start with:

- [Installation](docs/installation.rst) and [Quickstart](docs/quickstart.rst)
- [Example configurations](docs/examples.rst)
- [Model overview](docs/network_architectures.rst) and the
  [component catalogue](docs/available_components.rst)
- [Training strategies](docs/training_strategies.rst),
  [datasets](docs/datasets.rst) and [analysis](docs/analysis.rst)
- [Runs and outputs](docs/runs_and_outputs.rst) and
  [configuration files](docs/configuration_guide.rst)

## Repository layout

- `src/dendritic_modeling/`: the library (configuration, datasets, networks,
  training, analysis, deployment, command-line programs);
  [docs/engineering_architecture.rst](docs/engineering_architecture.rst) maps
  it.
- `configs/`: reference and example configurations.
- `docs/`: documentation sources.
- `tests/`: the pytest suite.
- `setup/`: conda environment file and pinned constraints.
- `scripts/train_fsdp.sh`: a Slurm launcher for FSDP training.

## Contributing

See [docs/contributing.rst](docs/contributing.rst) and
[CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
