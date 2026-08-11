# Large indexed-kernel comparison examples

This folder contains complete large-model recipes for the shared indexed
projection selector:

| Recipe | Model | Size |
| --- | --- | --- |
| `feedforward_large.yaml` | Three-layer feedforward E/I network | 8,192 E + 1,024 I neurons per layer |
| `rnn_large.yaml` | Recurrent unified E/I network | 4,096 E + 1,024 I neurons |
| `alexnet_fc6_large.yaml` | Configured replacement for AlexNet fc6 + ReLU | 4,096 E + 820 I neurons |
| `transformer_olmo2_1b_large.yaml` | Configured replacement for one OLMo-2 1B FFN | 8,192 gated dendritic units |

Each recipe defines ten named variants. Generate the complete training configs
with:

```bash
dendritic-sweep \
  --config configs/examples/indexed_backends/feedforward_large.yaml \
  --generate-only \
  --output-dir outputs/indexed_backends
```

Use the same command with `rnn_large.yaml` or `alexnet_fc6_large.yaml` for the
other model families, including `transformer_olmo2_1b_large.yaml`. The
generated config records its selected backend in `_sweep_variant`.

The variants compare implementations, rather than using ambiguous backend
names:

- `control_eager` is the standard portable PyTorch implementation.
- `iss_recompute`, `iss_triton`, `iss_cuda`, and `iss_auto` are the isolated
  `iss373_hs` controls and kernels.
- `ryan_fused`, `ryan_triton`, `ryan_cuda`, `ryan_hybrid`, and `ryan_auto` are
  Ryanconfig implementations.

An explicit accelerated condition fails if its selected implementation is not
available. Only an `*_auto` condition is allowed to select its documented
fallback, so a benchmark result cannot silently measure another implementation.

The RNN recipe is the primary workload for this comparison: its short,
recurrent batches can favor a different implementation than the feed-forward
and transformer examples. The AlexNet recipe uses a configured indexed core;
backend selection does not apply to `core_source: pretrained_span` because that
mode retains the original AlexNet modules.

The examples deliberately leave reactivation behavior unchanged. This prevents
an unrelated activation-memory optimization from affecting a kernel result.

The examples use fixed indexed connectivity rather than a dense candidate
bank, 32-bit topology indices, bounded gather workspaces, and efficient branch
aggregation. The transformer example additionally regenerates its immutable
graph from a seed and writes bitmask-encoded replacement checkpoints. These
choices reduce training and checkpoint memory; report them separately from
measured wall-clock speed, which depends on the selected backend and hardware.
