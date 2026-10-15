# Configurations

Reference and example YAML configurations for DendriNet. The documentation
page [docs/examples.rst](../docs/examples.rst) lists every file here, what it
shows and the command that uses it.

- `config_exp.yaml`: annotated single-process reference configuration for a
  population network on CIFAR-10, runnable on CPU or GPU.
- `config_exp_fsdp.yaml`: the same population architecture for FSDP training,
  using fixed indexed contacts and data-calibrated gates. Its contact policy
  and gate initialization differ from the single-process reference.
- `sweep_config.yaml`: annotated sweep template, for `dendritic-sweep`.
- `examples/`: training examples (`population_network_*.yaml` and others),
  analysis profiles for `dendritic-analyze-checkpoint` (`*_analysis.yaml`),
  sweep templates for the indexed projection backends (`indexed_backends/`),
  structured-core building blocks (`operator_presets/`), a compact-export
  policy (`deployment/`) and an epoch-selection specification.
- `templates/`: blocks to copy into a configuration (all analyzers; indexed
  synapse runtime settings).

Train an example from the repository root:

```bash
dendritic-train configs/examples/population_network_feedforward.yaml
```

Experiment files use `config_version: 4`. Values in an example are a chosen
recipe, not a declaration of the library defaults. Analysis profiles,
deployment policies, operator presets and sweep files have their own formats;
they are not standalone training configurations. Each sweep's `base_config`
uses the experiment format. Use `dendritic-upgrade-config` to migrate older
experiment files; the loader rejects unsupported options.

Keep this directory to general, runnable examples. Configurations and sweeps
of one study belong in that study's own repository; generated sweep
configurations belong with the run outputs. Personal, untracked
configurations can go in `configs/local/`, which git ignores.
