# Indexed-backend recipes

Two large sweep files that train the same model with different indexed
projection backends:

| Recipe | Model | Base backend |
| --- | --- | --- |
| `feedforward_large.yaml` | Three-layer feedforward E/I network (`einet`) on CIFAR-10: 8,192 E and 1,024 I neurons per layer, branch factors `[3, 3]` | `auto` |
| `rnn_large.yaml` | One-layer recurrent `unified_ei` network on `context_dependent_decision`: 4,096 E neurons (branch factors `[4, 4]`) and 1,024 I neurons | `recompute` |

Both use fixed `indexed` synapses with `index_dtype: auto` and
memory-efficient `param_tanh` reactivation. Each file defines four named
variants that differ only in the backend: `eager`, `recompute`,
`triton_transposed` and `auto`. Generate the four training configs with:

```bash
dendritic-sweep \
  --config configs/examples/indexed_backends/feedforward_large.yaml \
  --generate-only \
  --output-dir outputs/indexed_backends
```

The configs are written to
`outputs/indexed_backends/<run_name>_<timestamp>/configs/unified_config_<N>.yaml`,
and each records its variant name in the top-level key `_sweep_variant`. Use
the same command with `rnn_large.yaml` for the recurrent model. Submitting
the jobs needs a Slurm account and partition (`--slurm-account`,
`--slurm-partition`); see [`docs/sweeps.rst`](../../../docs/sweeps.rst).

`triton_transposed` needs Triton and a CUDA device. See
[`docs/performance_kernels.rst`](../../../docs/performance_kernels.rst) for
every backend, how `auto` chooses, and how to benchmark the variants;
[`configs/templates/indexed_runtime_template.yaml`](../../templates/indexed_runtime_template.yaml)
has the recommended runtime settings.
