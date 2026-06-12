# DendriNet Profiling — Handoff Brief for Claude

> **Purpose of this file:** a self-contained context document so a fresh Claude
> session can continue a performance-profiling effort on this repo without
> re-investigating from scratch. It captures (a) verified facts about the
> codebase, (b) the agreed profiling plan, (c) decisions already made, and
> (d) the exact next action. All file:line references were verified by reading
> the source on branch `ryanconfig`.

---

## 0. Operating constraints (do not violate)

- **Git:** Work ONLY on branch `ryanconfig`. **NEVER** push to `main`/`develop`.
  Default/main branch is `develop`. Commit/push only if the user explicitly asks.
- **No behavior/numerics changes.** All profiling must be **opt-in**, gated by an
  env var (`DENDRINET_PROFILE`) and/or a `--profile` CLI flag, **off by default**.
  When off, behavior must be byte-for-byte identical to current.
- **Keep traces small.** Profile only a few steps after warmup
  (`schedule(wait=1, warmup=1, active=3)`), then stop — never the whole run.
- **Minimal diffs.** Show the diff before applying. **Pause after each rung**
  so the user can read results before moving on.
- User is in **learning/exploring mode** — explain *why* each step, not just *what*.

---

## 1. Framework & GPU usage (verified)

- **PyTorch only** in the core training path. Pinned `torch>=2.0.0,!=2.6.0,<3.0.0`
  + `torchvision` (`pyproject.toml:28`). TensorFlow and `transformers` exist only as
  **optional extras** (`[tf]`, `[transformer]` in `pyproject.toml:94,106`) and are NOT
  on the training path to be profiled. **No JAX.**
- Conventional **eager-mode** PyTorch: `.to(device)`, one async H2D copy per batch.
- Backend knobs set in `src/dendritic_modeling/utils/general.py:117` (`set_seed`):
  `cudnn.benchmark`, `cuda.matmul.allow_tf32`, `set_float32_matmul_precision`.
  **Defaults are reproducible-but-slow:** `experiment.deterministic=True`,
  `cudnn_benchmark=False`, `allow_tf32=False`, `float32_matmul_precision="highest"`
  (`config.py:94-97`). Some apparent "slowness" is intentional determinism.

## 2. Entrypoint & training loop (verified — THE place to instrument)

- **CLI entrypoint:** `src/dendritic_modeling/scripts/training/train_experiments.py`
  - `cli_main()` at line 913 → `main(config_path, args)` at line 430.
  - Console script: `dendritic-train` (`pyproject.toml:54`).
- **Core per-batch loop:** `Trainer._epoch_train()` in
  `src/dendritic_modeling/training/strategies/standard.py:679`.
  Per-batch body is **lines 691–775**. Key lines:
  - H2D copy: `691–692` (`batch[0].to(self.device), batch[1].to(self.device)`)
  - forward (`self.loss_function(...)`): `697` (AMP path) / `739` (fp32 path)
  - `zero_grad` + `backward`: `715–716` (AMP) / `755–756` (fp32)
  - grad clip + `optimizer.step()`: `727–736` (AMP) / `764–770` (fp32)
  - **per-batch `.item()` x3** at `772–774` → 3 GPU→CPU syncs **every step**.
    PRIME SUSPECT for GPU stalls; Rung 1 will confirm. (Do NOT change it — observe it.)
- 10 trainer subclasses exist (recurrent, FSDP, feedback-alignment, local-learning,
  multi-stage, etc.) but all inherit this contract. **`standard.py` is the target.**

## 3. Data loading (verified)

- DataLoaders built in `standard.py:287` (train) / `:297` (valid). Honors
  `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` from config
  (`config/training.py:45-48`). Defaults: `num_workers=None`→auto `min(4, cpus)`,
  `pin_memory` on with CUDA+workers, `batch_size=512`.
- **Datasets are in-memory `TensorDataset`s** (MNIST/CIFAR/Fashion materialized up
  front). **No per-`__getitem__` disk/h5py reads → disk I/O is a non-issue.**
- Transforms are stock torchvision, CPU-side in workers.
- Only notable host-side hotspot: `PoissonGeneratorDataset.__getitem__`
  (`datasets/standard_datasets.py:989`) — un-vectorized per-sample `torch.poisson()`
  + gain sampling on CPU. **Only relevant if the chosen config uses that dataset
  (the chosen MNIST config does NOT).**

## 4. Custom kernels — Nsight Compute decision (verified)

**Verdict: NO custom kernels worth Nsight Compute. Plan to SKIP Rung 3.**
- ❌ No CUDA C++ extensions (no `.cu/.cuh`, no `CUDAExtension`/`load_inline`;
  `setup.py` is an empty shim).
- ❌ No Triton.
- ⚠️ Exactly ONE `torch.autograd.Function`: `SurrogateSpike`
  (`networks/architectures/recurrent/spiking_soma.py:11`). Its backward is pure
  element-wise PyTorch (abs/pow/div) — not a custom kernel.
- ⚙️ OPTIONAL `torch.compile` on one model forward
  (`networks/architectures/excitation_inhibition/ei_network.py:447`), gated by a
  `compile_forward` flag. Inductor-generated → Nsight Compute can't attribute it;
  torch.profiler is the right tool there.
- "Interesting" layers (`BlockLinear`, `TopKLinear`, `DendriteConv2d`, hierarchical
  dendritic conv) are **pure-PyTorch composition** → dispatch to stock cuBLAS/cuDNN.

**Working hypothesis:** the perf story is **kernel-launch overhead / many small ops /
Python-side syncs (the `.item()` calls)**, NOT a single slow kernel. That's diagnosed
by Rungs 1–2, not Rung 3.

## 5. Mixed precision / grad accum / distributed (verified)

- **AMP:** opt-in via `use_amp` (default **False**), `GradScaler`+`autocast`
  (`standard.py:694`). Uses the legacy `torch.cuda.amp` import (works on pinned Torch;
  `transformer_replacement.py` uses newer `torch.amp`).
- **Gradient accumulation: NOT implemented.** Optimizer steps every batch.
- **Distributed:** selected by `distributed.mode ∈ {none,ddp,fsdp}` (`config.py:282`).
  DDP via `DistributedSampler`+`all_reduce` in `standard.py`. FSDP via separate
  entrypoint `train_experiments_fsdp.py` + `FSDPTrainer`. Launched with
  `torchrun --nproc_per_node=N`.

---

## 6. The profiling ladder (agreed plan)

| Rung | Tool | Status | Answers |
|---|---|---|---|
| 1 | `torch.profiler` | **Start here** | Is the GPU busy or idling? Which ops dominate? Do the `.item()` syncs hurt? Exports TensorBoard + Chrome trace. |
| 2 | Nsight Systems (`nsys`) | Next | System timeline: dataloader/compute overlap, launch-overhead gaps, H2D copies, AMP. Needs NVTX ranges (added in Rung 1, flag-gated). |
| 3 | Nsight Compute (`ncu`) | **Expected SKIP** | Only if Rungs 1–2 point at a specific influenceable kernel. Phase 1 found none. |

## 7. Decisions already made

- **Target config:** `configs/examples/standard_mnist.yaml`
  (MNIST, one hidden layer `[32]`, `strategy: standard`, `batch_size: 256`).
  Chosen as smallest/fastest while still exercising the exact `Trainer._epoch_train`
  loop. No Poisson dataset hotspot. AMP off by default. (Epochs=50 is irrelevant —
  we profile ~3 steps and stop.)
- **Hardware:** GPU on a **remote Slurm cluster (Kempner Institute)**. Rung 1 runs
  inside an `srun`/`sbatch` GPU allocation. Rung 2 wraps `nsys` in `srun`; expect to
  `module load` Nsight Systems from the CUDA/HPC modules.

## 8. Exact next action (Rung 1 implementation — NOT yet applied)

Awaiting user "go". When approved, make a single minimal, opt-in change:
1. Add `DENDRINET_PROFILE` env var (+ optional `--profile` CLI flag in
   `train_experiments.py`) gating everything. Off → unchanged behavior.
2. In `_epoch_train` (`standard.py`), when enabled: wrap the batch loop in
   `torch.profiler.profile(schedule=schedule(wait=1, warmup=1, active=3),
   on_trace_ready=tensorboard_trace_handler(...))`, call `prof.step()` per batch,
   and **break after ~5 steps** so it profiles a handful of steps and exits without
   running the full epoch.
3. Export TensorBoard trace + Chrome trace (`export_chrome_trace`).
4. Add flag-gated NVTX `range_push/pop` around dataload/forward/backward/optimizer
   so the same annotations make Rung 2's timeline readable. No-ops when off.

Estimated diff: ~30–40 lines in `standard.py` + a few lines of arg/env plumbing in
the entrypoint. Show the diff before applying. Then provide the exact `srun` command
to produce the trace, and STOP to review together.

---

## 9. Quick-start commands (reference)

```bash
# Single-process train (the loop being profiled)
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/examples/standard_mnist.yaml

# DDP / FSDP (not the initial profiling target)
torchrun --nproc_per_node=N \
    src/dendritic_modeling/scripts/training/train_experiments.py <cfg>
```
