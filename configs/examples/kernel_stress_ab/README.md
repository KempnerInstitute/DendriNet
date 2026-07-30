# Indexed-gather kernel A/B stress test (feedforward)

Four configs that are **byte-identical except for the `indexed_use_*_gather` flags**,
for benchmarking the sparse-gather kernels on a wide + deep feedforward network.

| file | gather path |
|------|-------------|
| `kernel_stress_12L_16k_off.yaml`    | loop (baseline) |
| `kernel_stress_12L_16k_fused.yaml`  | `_ChunkedSparseGatherMulSum` (pure-torch fused autograd) |
| `kernel_stress_12L_16k_triton.yaml` | Triton fwd+bwd kernel |
| `kernel_stress_12L_16k_cuda.yaml`   | CUDA fwd+bwd kernel |

Runtime precedence is `cuda -> triton -> fused -> loop`; each variant sets exactly
one path on so precedence never masks the one under test.

## Model

Wider + deeper extension of `../sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml`:

|            | sglimit5      | this          | factor |
|------------|---------------|---------------|--------|
| depth      | 5 layers      | **12 layers** | 2.4x   |
| width      | 10,000E/1,250I| **16,000E/2,000I** | 1.6x |
| neurons    | ~56k          | ~216k         | 3.8x   |
| gather idx | 30.2M         | **116.2M**    | 3.9x   |
| params     | 37M           | 132M          | 3.6x   |

`topk_type: indexed_rewire` -> `IndexedRewireLinear` (subclass of `IndexedSparseLinear`),
so the gather flags engage. Verified: all 10 rewire synapses pick up the flag from config.

## Width is at the construction ceiling (why not wider)

The feedforward `population_network` path materializes a **dense
`[out_compartments x in_features]` connection mask** per inter-population connection
(`population_layer.py::_sample_connection_block`), which is O(width^2 * branch).
Measured cliff: per-layer **E<=18k builds, E>=20k OOM-kills a 784 GB host** (looks like
an int overflow past ~18k producing a garbage-huge allocation). 16k is the safe
ceiling. Depth is cheap and linear -- raise `n_layers` to stress harder. The recurrent
`ei_unified` path avoids this dense mask; making the feedforward path sparse at
construction is the change needed to go much wider.

## Expected results

The kernels are **equivalence-preserving**, so the **loss curve should match across all
four** (an equivalence panel). Speed and peak memory are what differ. Unlike the batch-1
RNN regime, this is the wide + batched feedforward regime where the CUDA/Triton gather
historically win big -- so expect off < fused < triton ~ cuda on throughput.

## Sizing (measured)

- 131.9M params, 0.43 GB int32 `connection_indices` buffer (replicated, FSDP does not shard).
- Params/buffers are trivial; the binding term is **activation memory**, set by
  `batch_size` (data-parallel splits it across GPUs).
- On **RTX PRO 6000 Blackwell (96 GB)**: 4 GPUs @ batch 256 ~= 39 GB/GPU (room to raise
  batch to 512 ~= 77 GB/GPU); 2 GPUs @ batch 256 fits; 1 GPU needs batch <= 128.
- The repo GPU estimator over-counts activations badly (ignores grad-ckpt/fused) -- do a
  one-step smoke run to confirm the real peak before a long sweep.

Construction takes ~80 s / 6.5 GB CPU RSS per run.
