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

For the graphical editor, go straight to [DendriNet Studio](#dendrinet-studio).
For the Python library and command-line tools, install the core from PyPI
in a Python 3.10 or later environment.

```bash
python -m pip install --upgrade dendritic-modeling
```

The import name is `dendritic_modeling`. The examples run on a CPU.
The installation guide covers virtual environments, CPU and CUDA builds
of PyTorch, optional dependencies and editable installs for contributors.
[Core on PyPI](https://pypi.org/project/dendritic-modeling/) ·
[Installation guide](docs/installation.rst)

## DendriNet Studio

Studio lets you draw feedforward and recurrent networks, inspect a neuron's
dendrites and compare runs in your browser.

**[Open Studio online](https://mybinder.org/v2/gh/KempnerInstitute/DendriNet/9d1500ec580981ad47953abf259950cfd893634a?urlpath=open-studio)**
to try the editor in a personal CPU session. No account or local installation is
required. In **Training**, set **Epochs** to `1` and select **Train** to try the
two-class example. Download your YAML and loss curves before the session expires.

With Studio 0.2.0 and DendriNet 0.6.x, select a population to add an
incoming source, or select a delayed connection to set its travel time at
each dendritic depth. The laptop commands below install the latest
compatible pair published on PyPI. Check your installed versions with
`python -m pip show dendrinet-studio dendritic-modeling`.

To keep runs on your own computer, open a terminal on your laptop.
On an Apple silicon Mac or a Linux laptop with Python 3.10 or later, run
these commands the first time. Installing Studio also installs DendriNet and
PyTorch. On a Mac, use the arm64 build of Python.

```bash
python3 -m venv ~/dendrinet-env
source ~/dendrinet-env/bin/activate
python -m pip install --upgrade dendrinet-studio
dendrinet-studio --workspace ~/dendrinet-runs
```

If you use Conda on macOS or Linux, replace the first two lines with
`conda create -n dendrinet python=3.11 -y` and `conda activate dendrinet`.
[Windows PowerShell instructions](apps/studio/README.md#windows-powershell)

Keep the terminal open. If a browser tab doesn’t open, paste the entire
printed address, including the part after `#`, into your browser.

1. In **Model → Start from**, open **Two-class population network**. Select
   population **e** to see which inputs provide excitation and inhibition.
2. Choose **CPU** and click **Check model**. In **Training**, set **Epochs**
   to `1`. Click **Train**, name the run and select **Start training**.
3. Open **Runs** to see the loss and training log. The example generates
   its own data and saves the run in `~/dendrinet-runs`.

Press **Ctrl+C** in the terminal to stop Studio. Next time, activate the
same environment and repeat the `dendrinet-studio --workspace` command above
to see your saved runs. The editor also exports YAML for `dendritic-train`.
[Illustrated Studio guide](https://kempnerinstitute.github.io/DendriNet/studio.html#studio-first-run) ·
[Studio on PyPI](https://pypi.org/project/dendrinet-studio/)

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
