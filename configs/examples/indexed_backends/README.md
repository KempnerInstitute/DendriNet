# Large indexed-backend examples

This folder contains complete large-model recipes for the shared indexed
projection selector:

| Recipe | Model | Size |
| --- | --- | --- |
| `feedforward_large.yaml` | Three-layer feedforward E/I network | 8,192 E + 1,024 I neurons per layer |
| `rnn_large.yaml` | Recurrent unified E/I network | 4,096 E + 1,024 I neurons |
| `alexnet_fc6_large.yaml` | Configured replacement for AlexNet fc6 + ReLU | 4,096 E + 820 I neurons |
| `transformer_olmo2_1b_large.yaml` | Configured replacement for one OLMo-2 1B FFN | 8,192 gated dendritic units |

Each recipe defines five named variants: `eager`, `recompute`, `triton`,
`cuda`, and `auto`. Generate the five complete training configs with:

```bash
dendritic-sweep \
  --config configs/examples/indexed_backends/feedforward_large.yaml \
  --generate-only \
  --output-dir outputs/indexed_backends
```

Use the same command with `rnn_large.yaml` or `alexnet_fc6_large.yaml` for the
other model families, including `transformer_olmo2_1b_large.yaml`. The
generated config records its selected backend in `_sweep_variant`.

Backend selection is functional, not architecture-specific:

- `eager` is the standard portable PyTorch implementation.
- `recompute` is portable and retains fewer contact-sized tensors for backward.
- `triton` requires Triton and a CUDA device.
- `cuda` requires a CUDA device and a working extension build toolchain.
- `auto` selects CUDA, then Triton, on supported CUDA systems and otherwise
  uses `recompute`.

The native backends are valid for recurrent networks, but their repeated launch
cost can outweigh their benefit for short sequences or small batches. The RNN
recipe therefore uses `recompute` as its base setting while still providing all
five explicit variants. For the feedforward, AlexNet, and transformer examples,
benchmark `auto` against `recompute` on the intended GPU and batch shape:
`auto` selects an available native implementation but does not guarantee that
it is faster at every shape. The AlexNet recipe uses a configured indexed core;
backend selection does not apply to `core_source: pretrained_span` because that
mode retains the original AlexNet modules.

All four recipes also enable memory-efficient `param_tanh`. Diagnostic paths
that require intermediate voltages automatically retain the standard eager
activation behavior.

The examples use fixed indexed connectivity rather than a dense candidate
bank, 32-bit topology indices, bounded gather workspaces, and efficient branch
aggregation. The transformer example additionally regenerates its immutable
graph from a seed and writes bitmask-encoded replacement checkpoints. These
choices reduce training and checkpoint memory; report them separately from
measured wall-clock speed, which depends on the selected backend and hardware.
