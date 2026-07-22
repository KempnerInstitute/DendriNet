# Dendritic-RNN extreme-placement cube

This repository-owned matrix completes the two same-side, reversed-timescale
conditions that were missing from the ten-block confirmatory design. Training
uses the canonical `dendritic_modeling.scripts.training.train_experiments`
entrypoint and the current `src/dendritic_modeling` implementation. Nothing in
`drafts/` is imported or executed.

- `base.yaml` contains the full task, model, optimizer, and training contract.
- `matrix.yaml` contains the two placement cells, all eight RNG streams for
  each paired replicate, output root, and supported scheduler routes.
- `generated/` contains the 20 standalone YAML configs and `manifest.json`
  produced by the experiment launcher.

Materialize and validate:

```bash
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube materialize
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube validate
```

The H100-priority route is primary. Regular H100 is a scientifically
like-for-like fallback. The engineering partition is H200-backed and therefore
requires explicit opt-in rather than being silently mixed with the H100 matrix.

The complete repository-owned workflow is exposed by the same launcher:

```bash
# One construction-only H100 canary, then the 20 standard training runs.
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube preflight
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube submit

# One frozen-bank evaluation per successful training run.
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube \
  submit-evaluations --training-receipt <TRAINING_RECEIPT_JSON>

# CPU-only aggregation, paired-seed statistics, and four-panel figure after
# all 20 evaluation dependencies succeed.
PYTHONPATH=src python -m dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube \
  submit-analysis --evaluation-receipt <EVALUATION_RECEIPT_JSON>
```

Training and evaluation artifacts live below the `output_root` in
`matrix.yaml`. The final collector writes `collection.json`, `statistics.json`,
`figure_source.json`, and PDF/PNG figures below `<output_root>/analysis/`.
Only those completed artifacts should be copied into a paper directory;
training, inference, statistics, and rendering never execute from `drafts/`.
At evaluation submission, the launcher also freezes the standard Python
package below `<output_root>/_runtime/` and uses that read-only snapshot for
both evaluation and aggregation. This prevents unrelated edits to the live
working tree from changing queued scientific jobs.
