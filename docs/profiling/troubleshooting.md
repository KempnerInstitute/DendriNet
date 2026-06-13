# Troubleshooting & lessons learned

Every problem we actually hit, and the fix. Roughly in the order they bit us.

## Environment / cluster

### `conda env create -f setup/environment.yml` fails with `pytorch-cuda=12.1 not found`
That file targets an old CUDA via a deprecated conda channel, and on Apple
Silicon there's no CUDA at all. **Don't use it.** Use `bash setup/setup_cuda129.sh`
on the cluster. See [setup.md](setup.md).

### `ModuleNotFoundError: No module named 'torch'`
The `dendritic_env` conda env isn't active. Your prompt should start with
`(dendritic_env)`. Run `module load python && conda activate dendritic_env`, and
confirm with `which python` (should point into `.../envs/dendritic_env/bin`).

### `ERROR: neither 'mamba' nor 'conda' found on PATH` (from a script)
A `bash script.sh` starts a non-interactive shell that doesn't auto-load modules.
Run `module load python` first, then the script. (The batch jobs do this
internally.)

### `which nsys` / `which ncu` prints nothing
They ship with the CUDA toolkit module, which isn't loaded by default:
`module load cuda/12.9.1-fasrc01`. For batch jobs, pass it as
`NSYS_MODULE=cuda/12.9.1-fasrc01` (or `NCU_MODULE=...`).

### `torch.cuda.is_available()` is `False`
Expected on a **login node** (no GPU). Only trust it from inside a GPU allocation
(`salloc ... --gres=gpu:1` or a batch job on a `holygpu*` node).

### I'm on a Mac and CUDA won't install
Apple Silicon has no NVIDIA CUDA. Profiling runs on the cluster. The Mac is for
editing/git and (optionally) CPU-only Rung 1. See [setup.md](setup.md).

## Slurm / running jobs

### `CANCELLED ... DUE TO PREEMPTION` (over and over)
`kempner_requeue` is **preemptible** — higher-priority jobs reclaim the GPU. This
killed interactive sessions repeatedly, sometimes mid-setup.
**Fix:** use the **batch jobs** (`jobs/*.sbatch`). They have `#SBATCH --requeue`,
so preemption just re-queues them and they re-run the (idempotent) capture until
one attempt completes — unattended. A `restarts=N` in the log is normal.
Keep job wall-time *short* (the captures take ~1–5 min); a large `--time` can
queue slower on a backfill scheduler.

### The job runs but downloads a dataset slowly, then times out
CIFAR-10 (~170 MB) wasn't cached and the compute node downloaded it at ~128 kB/s
(>20 min) — exceeding the job's wall time, and wasting GPU time on a download.
**Fix:** pre-download on a **login node** (better bandwidth, no GPU wasted):
```bash
cd ~/DendriNet && module load python && conda activate dendritic_env
python -c "import torchvision; torchvision.datasets.CIFAR10(root='outputs/datasets/cifar', download=True)"
ls outputs/datasets/cifar/cifar-10-batches-py/   # confirm
```
Then the GPU job loads from cache. (MNIST "just worked" because it was already
cached.) Synthetic-data configs — e.g. the recurrent `context_dependent_decision`
— need no download.

## Profiling output

### The `.nsys-rep` is 0 bytes / won't open ("created with a newer version")
The GUI's "newer version" message is misleading — the real error
(`VersionTagReader ... Cannot read from stream`) means the file is empty/truncated.
In our case a stray shell redirect destroyed it: pasting the `nsys stats` *output*
back into the terminal ran a line like `> profilings/.../report.nsys-rep`, which
**truncates the file to 0 bytes**.
**Fixes:**
- Never paste large command output back into a terminal. Send `nsys stats` to a
  file: `nsys stats ... > stats.txt`.
- Verify size before copying: `ls -l ...nsys-rep` (should be MBs, not 0).
- After `scp`/`rsync`, check the size matches on both ends (`shasum` both;
  `da39a3ee...` is the hash of an *empty* file).

### The profiling run "won't end" / keeps running after the profiler table
Without the fast-exit, a profiling run continued into post-training validation +
analysis (which can take many minutes). This is fixed: profiling runs now
`finalize_profiling_run()` (synchronize + `sys.exit(0)`) right after the capture.
If you see this, you're on old code — `git pull`.

### `--profile_backend nsys` did nothing for a config
That config uses a trainer whose `_epoch_train` isn't instrumented. The
`standard` and `recurrent` strategies are instrumented; others aren't yet. See
[instrumentation.md](instrumentation.md) → "Instrumenting another trainer".

### The `[job] DONE` line lists the wrong filename
Cosmetic only — an early version of `profile_nsys.sbatch` hard-coded the old
filename in the final `ls`. The **`Generated: ...`** line from `nsys` is the
source of truth for what was written. Fixed in the current job.

### `WARNING: Device-side CUDA Event completion trace is currently enabled`
Benign `nsys` notice (event tracing adds minor overhead). Ignore it, or add
`--cuda-event-trace=false` to the `nsys` command. It does not affect the analysis.

## Nsight Compute (Rung 3)

### `ERR_NVGPUCTRPERM: ... permission to access NVIDIA GPU Performance Counters`
**The Rung-3 gate.** `ncu` needs GPU performance-counter access, which is locked
down on shared clusters. Not your fault and not fixable from your side — email
`rchelp@rc.fas.harvard.edu`: *"Please enable GPU performance counters for user
profiling (Nsight Compute / `NVreg_RestrictProfilingToAdminUsers=0`) on the
Kempner GPU nodes."* Until then, `ncu` can't collect.

### `ncu` seems to hang
`ncu` replays each profiled kernel many times to read counters — it's slow even
with our `--launch-count 3` filter. A few minutes is expected, not a hang.

## Git / transferring reports

### `git add .` says "nothing to commit" for my reports
Trace artifacts are gitignored by **extension** (`*.nsys-rep`, `*.sqlite`,
`*.pt.trace.json`, …). Moving them to another folder doesn't help — the extension
matches everywhere.

### "untracked working tree files would be overwritten by merge"
A trace file got committed (via `git add -A` after an `rsync` pulled it local),
and another machine has its own untracked copy at the same path. Fixed by
gitignoring the trace types and `git rm --cached`-ing the committed blob. If you
hit it: `git stash` or move the untracked file aside, then pull — don't force.

### Transferring reports — the recommended way
Use `rsync` (no repo bloat), **on your Mac**:
```bash
rsync -avz <user>@login.rc.fas.harvard.edu:DendriNet/profilings/ ~/DendriNet/profilings/
```
If you *must* version a report in git, add a gitignore exception for a dedicated
folder (`!profilings-notignored/`, `!profilings-notignored/**`) and only commit
the `.nsys-rep` (never the regenerable `.sqlite`). Be selective — git keeps
binaries in history forever and GitHub rejects files >100 MB. For routine
large-file versioning, use `git-lfs`.

## General rule of thumb
For long-running cluster work, run inside `tmux` (`tmux new -s work`,
`Ctrl-b d` to detach, `tmux attach -t work` to return) so a dropped SSH
connection doesn't kill it. Better yet, use the batch jobs and walk away.
