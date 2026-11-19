# DendriNet Studio

Design, check, train and compare [DendriNet](https://github.com/KempnerInstitute/DendriNet)
models on your computer. Select a neuron population to trace the external
and local sources of its excitatory and inhibitory contacts. Inspect the
contact count at each dendritic level, build the network, or train and compare
two designs. Each design exports as an ordinary DendriNet experiment YAML.

From a repository checkout, run `python -m pip install -e . -e ./apps/studio`
at the repository root. Once the packages are on PyPI, install and launch with

```bash
python -m pip install dendrinet-studio
dendrinet-studio --workspace my-runs
```

Installing Studio installs the `dendritic_modeling` library and PyTorch. The
command prints a local address including a session token and opens it in your
browser. Keep the command running while you work. The first two-class example
generates its data locally and runs on a CPU.

## What you can do

- **Design population networks visually**: layers, excitatory and inhibitory
  populations, neuron counts, dendritic branching, contacts per compartment
  and per level, shunting or additive integration, dendritic gates, time
  constants, and connections that act in the same step or one step later
  (recurrent networks).
- **Configure other model families** using typed forms and the YAML editor,
  including structured dendritic trees, GRU, LSTM and vanilla RNN baselines,
  and multi-timescale CTRNNs.
- **Set the task and training**: classification or regression, datasets and
  their options, losses, learning strategies (backpropagation, recurrent
  training, local learning rules), optimizers, learning rates per parameter
  group, schedules, early stopping, initialization and seeds. A searchable
  view lists every option; a YAML view edits the file directly.
- **Check and train**: build the model to count its parameters and synapses,
  train it in the background, follow its progress, and compare runs with
  overlaid loss curves, final measures and the settings that differ.

Five examples install with the library: a two-class population network that
trains in seconds with no download, the same task with structured dendritic
trees, a recurrent decision network, a GRU regression baseline and an MNIST
network.

## Remote machines

```bash
dendrinet-studio --workspace ~/studio-runs --no-browser   # on the remote machine
ssh -L 8765:127.0.0.1:8765 user@remote-host                # on your computer
```

Open the printed address on your computer. `--no-execution` turns a machine
into design-and-export only.

## Privacy

Studio listens on `127.0.0.1` only, and its interface requires the session
token in the printed address. It sends nothing anywhere and loads nothing from
the internet; a dataset is downloaded only when your configuration names a
public dataset that needs one.

## Documentation

The [illustrated Studio guide](https://kempnerinstitute.github.io/DendriNet/studio.html)
covers installation, first training, E/I wiring, configuration exchange and
run records. The [guide source](https://github.com/KempnerInstitute/DendriNet/blob/develop/docs/studio.rst)
is also in the repository. Studio is
released under the MIT license.
