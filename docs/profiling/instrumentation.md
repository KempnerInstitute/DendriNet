# Instrumentation: design and code

## Design principles

1. **Opt-in, off by default.** Everything is gated by the `DENDRINET_PROFILE`
   env var. When unset, every helper is a no-op and training is byte-for-byte
   unchanged — no change to numerics or control flow.
2. **Small, contained.** The logic lives in one module
   (`training/utils/profiling.py`); trainers call into it.
3. **Both profilers from one annotation.** A single `record_section(name)` emits
   *both* a `torch.profiler.record_function` range (Rung 1) and an NVTX range
   (Rung 2), so the same labels serve both tools.
4. **Profiling runs stay short.** They capture a few steps (or N epochs) and then
   exit before post-training validation/analysis, keeping traces small.

## Files changed

| File | What |
|------|------|
| `src/dendritic_modeling/training/utils/profiling.py` | **New.** All the profiling helpers. |
| `src/dendritic_modeling/training/strategies/standard.py` | Instrumented `Trainer._epoch_train`; early-stop hook in `_run_training_loop`. |
| `src/dendritic_modeling/training/strategies/recurrent_trainer.py` | Instrumented `RecurrentTrainer._epoch_train` (the sequence path). |
| `src/dendritic_modeling/scripts/training/train_experiments.py` | `--profile`, `--profile_dir`, `--profile_backend` flags → env vars. |
| `jobs/profile_nsys.sbatch`, `jobs/profile_ncu.sbatch` | Batch jobs for Rungs 2/3. |
| `setup/setup_cuda129.sh` | CUDA-12.9-compatible env builder. |
| `.gitignore` | Ignore trace artifacts (`*.nsys-rep`, `*.ncu-rep`, `*.sqlite`, `*.pt.trace.json`, …). |

## `profiling.py` API

| Function | Role |
|----------|------|
| `profiling_enabled()` | True iff `DENDRINET_PROFILE` is truthy. |
| `profile_backend()` | `"torch"` (default) or `"nsys"`. |
| `record_section(name)` | Context manager: NVTX + `record_function` range (no-op when off). |
| `make_profiler()` | Returns a started `torch.profiler` (torch backend), or `None`. |
| `finish_profiler(prof)` | Stops it, logs the per-op CPU/GPU table. |
| `finalize_profiling_run()` | Synchronizes, tears down any dist group, exits 0. |
| `total_profiled_steps()` | `wait + warmup + active + 1` (step-mode count). |
| `profile_epochs()` | N for epoch mode, else 0. |

## How a trainer is instrumented

The pattern (see `Trainer._epoch_train` in `standard.py`):

```python
profile_active = profiling_enabled() and not getattr(self, "_profiling_done", False)
epoch_mode = profile_active and profile_epochs() > 0
prof = make_profiler() if (profile_active and not epoch_mode) else None

for step, batch in enumerate(train_loader):
    with record_section("data_h2d"):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
    with record_section("forward"):
        loss = self.loss_function(model, x, y)
    self.optimizer.zero_grad()
    with record_section("backward"):
        loss.backward()
    with record_section("optimizer"):
        self.optimizer.step()
    # ... existing bookkeeping ...

    if profile_active:
        if prof is not None:
            prof.step()
        if not epoch_mode and step + 1 >= total_profiled_steps():
            break

if profile_active:
    done = True
    if epoch_mode:
        self._profiled_epochs = getattr(self, "_profiled_epochs", 0) + 1
        done = self._profiled_epochs >= profile_epochs()
    if done:
        if prof is not None:
            finish_profiler(prof)
        self._profiling_done = True
        self._profiling_stop = True
        finalize_profiling_run()   # synchronize + exit 0
```

Key points:
- `record_section` returns `contextlib.nullcontext()` when profiling is off, so
  the `with` blocks are free.
- **Step mode** breaks after the step count; **epoch mode** runs full epochs and
  only finalizes after the Nth.
- `finalize_profiling_run()` does `sys.exit(0)` after `torch.cuda.synchronize()`,
  so the profiling run skips post-training analysis. (This only fires when
  profiling is enabled.)
- `make_profiler()` returns `None` for the `nsys` backend (no torch.profiler →
  no CUPTI clash with Nsight Systems) and in epoch mode.

## Which trainers are instrumented

- ✅ `Trainer` (`standard.py`) — used by `strategy: standard` and most subclasses.
- ✅ `RecurrentTrainer` (`recurrent_trainer.py`) — `strategy: recurrent`. This one
  overrides `_epoch_train` for sequence/temporal loss, so it needed the hooks
  ported in separately.
- ❌ Other specialized strategies (feedback-alignment, FSDP, multi-stage, etc.)
  that override `_epoch_train` are **not yet instrumented**.

### Instrumenting another trainer
If a strategy overrides `_epoch_train`, copy the pattern above into it:
1. Import the helpers from `training.utils.profiling`.
2. Wrap `data_h2d`/`forward`/`backward`/`optimizer` in `record_section(...)`.
3. Add the `profile_active`/`epoch_mode`/`prof` setup before the loop and the
   step/epoch finalize logic after it.
The inherited `_run_training_loop` already honors `_profiling_stop`, so no change
is needed there.
