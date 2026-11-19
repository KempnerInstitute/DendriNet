# Dendritic integration in trained networks

This project accompanies the empirical DendriNet paper. It contains eight
training configurations, two analyses of published measurements and the
renderer for Supplementary Figure S28. The examples cover Figures 3D and 8D.
The full set of seed sweeps, checkpoints and source tables for the other
panels is outside this example package.

## Build a network

The models in these examples trained with DendriNet v0.3.0. In an environment
with that release, run a representative configuration from the repository root:

```bash
dendritic-train projects/empirical/configs/cifar_primary_shunting.yaml
```

The matched additive configuration uses the same CIFAR-10 split and training
seed. Both feed separate external E and I afferents with matched tuning to
the first-layer neurons. Each of the 120 dendritic compartments receives
40 E and 10 I contacts. The soma receives none. The MNIST pair uses seed 100,
the primary CIFAR-10 pair seed 0, the three-layer CIFAR-10 pair seed 10, and
the contextual pair seed 62000. Each YAML preserves its submitted settings
apart from the dataset and output directories. Put the datasets under
`data/` or set `data.base_dir` for your installation. The submitted and
portable configuration hashes are listed in [configs.json](reproducibility/configs.json).

For the contextual models, first calibrate the gates on the seeded training
presentation stream. The calibration writes an initial checkpoint and a YAML
that names it. Run from the repository root in the v0.3.0 environment:

```bash
python projects/empirical/analysis/calibrate_context.py \
  --config projects/empirical/configs/context_shunting.yaml --output calibration
dendritic-train calibration/effective_config.yaml
```

The contextual training and test presentations use the EMNIST Balanced split.
Changing the YAML alone cannot substitute for the training-stream calibration.

## Replay the numerical examples

Use the study source-data asset `empirical-example-data.zip` with its recorded
[checksum and table inventory](reproducibility/source_data.json). Run from
the repository root with Python 3.11, NumPy 2.2.6, pandas 2.3.3 and DendriNet.
The analyses run on a CPU and require neither datasets nor trained weights.

```bash
python projects/empirical/analysis/reproduce.py empirical-example-data.zip \
  --output figure3d-results
```

- **Figure 3D.** `transmission/seed_units.csv` contains one row per rule, trained
  seed and dendritic depth. `summary.csv` gives the mean across networks and
  the pointwise seed-bootstrap interval for three measurements. The script
  compares the resulting bytes with the reference table.
- **Figure 8D.** `context/routing.csv` contains the rates of selecting the
  correct target and the next cue's target for each trained seed. It covers
  intact networks and transplants in either layer. `context_summary.csv`
  gives both percentages and their pointwise seed-bootstrap intervals for four
  models. The source labels `main`, `additive`, `matched_additive` and
  `bernardo` mean primary shunting, separately tuned additive, additive with
  the shunting training settings, and endpoint-gate shunting, respectively.

To draw Supplementary Figure S28, use Matplotlib, PyYAML and the tutorial
configuration distributed with the Studio-capable checkout. The tutorial is
a separate, generated-data demonstration.

```bash
python projects/empirical/analysis/plot_software_guide.py \
  --tutorial src/dendritic_modeling/examples/synthetic_population.yaml \
  --summaries figure3d-results --output software-guide.pdf
```

The downloadable examples also work as a standalone directory. They include the
tutorial YAML and the source-data asset. From the unpacked directory, run:

```bash
python analysis/reproduce.py empirical-example-data.zip --output figure3d-results
python analysis/plot_software_guide.py --tutorial tutorial.yaml \
  --summaries figure3d-results --output software-guide.pdf
```

The experiment uses the model checkpoint to obtain the derivatives and donor
responses. The replay above starts at the resulting per-seed measurements.
The [figure index](reproducibility/figures.md) describes the scope of the
examples and identifies which comparisons have separate historical sources.
