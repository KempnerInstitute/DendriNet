# Scaling toggles — the indexed-gather optimization flags

When scaling training, turn these on. **All default OFF** so the repo is unchanged until you
flip them. There are **three**, all set per `population_defaults` block in the config YAML.

## The three toggles

| # | YAML key (in `population_defaults`) | Python kwarg | What it enables | Speed | Status |
|---|---|---|---|---|---|
| 1 | `indexed_use_fused_gather: true`  | `use_fused_gather`  | Custom-autograd fused backward (`index_add_` scatter + recompute) | ~5× whole-step vs original | ✅ proven, bit-identical under AMP |
| 2 | `indexed_use_triton_gather: true` | `use_triton_gather` | Triton forward + backward kernels | +18–32% whole-step vs fused | ✅ proven, fp64-unbiased + training-equivalent |
| 3 | `indexed_use_cuda_gather: true`   | `use_cuda_gather`   | CUDA forward + backward kernels (fastest) | beats Triton both directions (fwd 2.65×/1.74×, bwd 1.54×/1.82×) | ⏳ needs its GPU gate before turning on (see below) |

`indexed_output_chunk_size: 16384` is already set in the config — a tuned value, **not** a toggle.

## How they interact — PRECEDENCE, not additive

In `IndexedSparseLinear.forward` the order is:

```
cuda  ->  triton  ->  fused  ->  original loop
```

The highest **available** enabled path wins; if its toolchain is missing it **falls through** to
the next (never breaks a run). So:

- **Turn all three on** → uses CUDA where the toolchain exists, else Triton, else fused. Robust
  across heterogeneous nodes. This is the recommended "flip everything on" state.
- To get max speed you technically only need #3 on (CUDA); #1/#2 are the safety net beneath it.

## What to paste into the config

Add these three lines to **each** `population_defaults` block (there are 5 in
`configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml`), next to
`indexed_output_chunk_size`:

```yaml
          indexed_output_chunk_size: 16384      # already present
          indexed_use_fused_gather: true        # toggle 1
          indexed_use_triton_gather: true       # toggle 2
          indexed_use_cuda_gather: true         # toggle 3 (gate first, see below)
```

## Before turning on CUDA (#3) at scale

1. `source profilings/weight_transform_opt/setup_cuda_env.sh` on the GPU node
   (loads cuda/12.9.1 + gcc/12.2.0, sets CUDA_HOME). CUDA needs `nvcc` at runtime;
   the extension lazy-compiles on first use (~80s, then cached).
2. Run the correctness gate: `python profilings/weight_transform_opt/verify_cuda_gather.py`
   → expect `VERDICT: UNBIASED` (worst ratio ~1 across all 25 synapses).
3. Run a training-equivalence check (adapt `train_equivalence_triton.py` to the cuda flag).
4. Only then set `indexed_use_cuda_gather: true`.

Fused (#1) and Triton (#2) are already proven — safe to turn on without further gating.
Triton needs the `triton` pip package (present); fused is pure PyTorch (no extra deps).

## Verify a toggle actually reached the synapses

```python
sum(1 for m in model.modules()
    if isinstance(m, IndexedSparseLinear) and m.use_cuda_gather)   # -> 25 when on
```
