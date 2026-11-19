# DendriNet Studio

Design, check, train and compare [DendriNet](https://github.com/KempnerInstitute/DendriNet)
models on your computer. Select a neuron population to trace the external
and local sources of its excitatory and inhibitory contacts. Inspect the
contact count at each dendritic level, build the network, or train and compare
two designs. Each design exports as an ordinary DendriNet experiment YAML.

## Install and open Studio on your laptop

Open a local terminal. Studio runs on your computer and opens in your browser.
Installing it from PyPI also installs DendriNet and PyTorch. Python 3.10 or
later is required.

### macOS or Linux

On an Apple silicon Mac, use an arm64 Python environment. The PyTorch
wheels required by Studio are published for that architecture on macOS.
Run these commands the first time on macOS or Linux.

```bash
python3 -m venv ~/dendrinet-env
source ~/dendrinet-env/bin/activate
python -m pip install dendrinet-studio==0.1.0
dendrinet-studio --workspace ~/dendrinet-runs
```

If you use Conda, replace the first two lines with
`conda create -n dendrinet python=3.11 -y` and `conda activate dendrinet`.
On later visits, activate the same environment and repeat the last command.

### Windows PowerShell

With Python 3.11 installed, run the following in PowerShell. The last line
also starts Studio on later visits.

```powershell
py -3.11 -m venv "$HOME\dendrinet-env"
& "$HOME\dendrinet-env\Scripts\python.exe" -m pip install dendrinet-studio==0.1.0
& "$HOME\dendrinet-env\Scripts\dendrinet-studio.exe" --workspace "$HOME\dendrinet-runs"
```

### Train a first network

If the browser tab doesn’t open, paste the entire local address printed by
Studio into your browser, including the session token after `#`. Keep the
terminal open while you work.

1. In **Model → Start from**, choose **Two-class population network**. Select
   population **e**. Under **Synapse sources per neuron**, its excitatory
   inputs come from the external features and its inhibitory inputs come from
   the local population **i**.
2. Choose **CPU** and click **Check model** to build the network and count
   its parameters and synapses. The example generates its data locally.
3. In **Training**, set **Epochs** to `1`. Click **Train**, name the run and
   select **Start training**. Open **Runs** to see the loss and training log.

Press **Ctrl+C** in the terminal when finished. The commands above keep
your runs in the `dendrinet-runs` folder in your home directory. Launch
Studio with that workspace again to reopen them.

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

For development, install both packages from the repository root with
`python -m pip install -e . -e ./apps/studio`.

The [illustrated Studio guide](https://kempnerinstitute.github.io/DendriNet/studio.html)
covers installation, first training, E/I wiring, configuration exchange and
run records. The [guide source](https://github.com/KempnerInstitute/DendriNet/blob/develop/docs/studio.rst)
is also in the repository. Studio is
released under the MIT license.
