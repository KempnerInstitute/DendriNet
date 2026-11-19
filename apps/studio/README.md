# DendriNet Studio

Design, check, train and compare [DendriNet](https://github.com/KempnerInstitute/DendriNet)
models in your browser. Studio draws networks of excitatory and inhibitory
dendritic neurons, edits every option of an experiment, builds and trains the
model on your computer, and compares runs. It writes ordinary experiment YAML
and runs it with DendriNet's own loader and training command, so everything
you design also runs without Studio.

```bash
pip install dendrinet-studio
dendrinet-studio --workspace my-runs
```

Installing Studio installs the `dendritic_modeling` library and PyTorch. The
command opens `http://127.0.0.1:8765/#<token>` in your browser; keep it running
while you work.

## What you can do

- **Design population networks visually**: layers, excitatory and inhibitory
  populations, neuron counts, dendritic branching, contacts per compartment
  and per level, shunting or additive integration, dendritic gates, time
  constants, and connections that act in the same step or one step later
  (recurrent networks).
- **Edit every other model family with forms** generated from the library's
  typed configuration: structured dendritic trees, GRU, LSTM and vanilla RNN
  baselines, multi-timescale CTRNNs and the other built-in cores.
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

The [Studio guide](https://github.com/KempnerInstitute/DendriNet/blob/develop/docs/studio.rst)
describes every view, the run records and the development setup. Studio is
released under the MIT license.
