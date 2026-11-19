# DendriNet

[![CI Pipeline](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml/badge.svg?branch=develop)](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml?query=branch%3Adevelop)
[![Coverage](docs/_static/badges/coverage.svg)](https://github.com/KempnerInstitute/DendriNet/actions/workflows/ci-simplified.yml?query=branch%3Adevelop)
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
  <img src="docs/_static/dendritic_diagram.png" alt="Three panels, left to right: a pyramidal neuron with basal and apical dendrites, excitatory synapses as dots on spines and inhibitory synapses as bars across the dendrite; the same neuron as a DendriNet tree of compartments on a triangular soma, each distal compartment receiving sparse excitatory and inhibitory inputs; and a two-layer network in which inputs drive excitatory neurons (triangular somata) and inhibitory neurons (round somata), inhibitory neurons inhibit the dendrites of neighbouring excitatory neurons, and excitatory neurons project to the next layer and the readout." width="100%">
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

After installation, export a small configuration into any working directory:

```bash
# Generate data locally and train a small population network
dendritic-example synthetic_population --output demo.yaml
dendritic-train demo.yaml --epochs 1

# Check a configuration without training
dendritic-train demo.yaml --validate-only

# From a repository checkout, generate a parameter sweep without submitting jobs
dendritic-sweep --config configs/sweep_config.yaml --generate-only
```

The installed commands include `dendritic-train` (training),
`dendritic-sweep` and `dendritic-analyze` (sweeps and their results),
`dendritic-analyze-checkpoint` (analysis of a trained checkpoint) and
`dendritic-compact-checkpoint` (compact export of a trained model);
[docs/cli_reference.rst](docs/cli_reference.rst) lists every command and its
options. Distributed training runs under `torchrun`; see
[docs/distributed_training.rst](docs/distributed_training.rst).

## DendriNet Studio

[DendriNet Studio](docs/studio.rst) is an optional local browser application.
Draw a feedforward or recurrent population network, select a neuron population
to see where its excitatory and inhibitory contacts come from, then train and
compare runs. Studio provides forms for other model families and a YAML editor
for the full configuration. The exported YAML also runs with `dendritic-train`.
To launch it from this checkout, run

```bash
python -m pip install -e . -e ./apps/studio
dendrinet-studio --workspace my-runs
```

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
