# DendriNet Studio

A local graphical editor and run browser for native DendriNet population
networks. It generates ordinary YAML and delegates construction and training
to the installed library. It contains no model equations or training loop.

From this checkout, in a Python environment suitable for DendriNet:

```bash
pip install -e . -e apps/studio
dendrinet-studio --workspace /path/to/your/runs
```

The launcher opens a browser on `127.0.0.1`. The printed URL contains a session
token. Use `--no-browser` on a compute node and forward the port through SSH;
open the printed URL on your workstation. This is a local application, not a
multi-user hosted service. It does not upload your data.

## First session

1. Load `synthetic_population` (no download) or `mnist_population`.
2. Select a population. Change its neuron count, branching factors, contacts,
   integration, leak, gate or sparse-layer type. An empty branching list means
   a soma-only neuron. Edit connections with source/target selectors.
3. Inspect the population graph and the exact branching-factor tree. Edges
   show the resolved source population and pathway. In later layers,
   `input` follows the preceding readout. When a population has per-level
   contact counts, edit its vector from the distal compartments to the soma.
   Population overrides leave neighboring populations' defaults intact. Large trees retain
   exact counts but suppress individual-compartment drawing above 1,500 nodes.
4. Choose **Check model &amp; count resources** to build the model with the
   trainer's dataset, input shape and seeds. The selected dataset may require
   a download. The check reports resource counts before gate calibration.
5. Train: a worker runs the normal training CLI. The synthetic tutorial is
   deliberately small. The training path performs its normal dataset-dependent
   setup and validation. CPU and automatic device selection are available.
6. Read the worker log, real training curves and saved summary. Every execution
   keeps its YAML, command, package identity and output-directory overrides.

The library's normal run artifacts additionally record seeds, environment and
source provenance. A Studio experiment is not a qualified paper reproduction
just because it uses a paper's architecture.

## YAML and scope

The visual editor supports `population_network` configurations. Other model
families remain accessible through the library CLI. Advanced population,
initialization and training fields can be edited in the Configuration tab.
Values outside the visual controls survive import and export; comments and
formatting do not. Import a standalone composed experiment, not a sweep or an
`extends` file. Exported YAML also works without Studio.

The preview checks configuration structure, not every constructor combination.
The separate model check validates construction; training performs calibration
and dataset checks. Invalid configurations are reported, never rewritten to
fit the editor. Biological mode is on in the tutorial examples, with an
explicit opt-out. Changing the fallback learning rate leaves explicit component
rates unchanged, just as in the core CLI.

One local worker runs at a time. Stop it in the run browser or exit the server.
Existing records can be viewed after restarting. A run that was active when the
server disappeared has unknown status until separately verified; it is never
reported as completed. Construction and training on shared login nodes are
refused. Export distributed jobs for the existing `torchrun` or sweep tools.

This first version does not provide arbitrary cable morphologies, drag-to-wire
synapse editing, a cluster scheduler, checkpoint activity playback, or an
automatic search for the best configuration. Those require separate extensions
and qualification. No core defaults or paper source pins are changed.

## Development checks

On a compute node or workstation:

```bash
PYTHONPATH=src:apps/studio/src pytest apps/studio/tests -q
```

For the optional real-browser check, install `apps/studio[test]` and run
`python -m playwright install chromium`. The browser test edits a morphology,
exports YAML, constructs a two-layer model and completes a real one-epoch
synthetic training run. It is skipped when Playwright is not installed.

The browser uses local HTML, CSS and SVG/JavaScript assets, with no CDN or
JavaScript build step. E/I populations and curves differ by shape or line style
as well as color. Browser editing state is separate from the experiment YAML.

For the wheel installation and publication procedure, see the repository's
[release guide](../../docs/releases.rst).
