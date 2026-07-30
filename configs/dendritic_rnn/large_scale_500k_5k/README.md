# Large dendritic RNN: 500k neurons / 5k contacts per neuron

The editable target is [`base.yaml`](base.yaml).  It uses only the standard
`src/dendritic_modeling` configuration and training stack; nothing under
`drafts/` is imported.  The first runnable check is
[`canary.yaml`](canary.yaml). The next measured step is
[`gate_5k_500.yaml`](gate_5k_500.yaml), with 5,000 neurons and 500 contacts per
neuron. The final logarithmic gate is
[`gate_50k_5k.yaml`](gate_50k_5k.yaml), with 50,000 neurons and the target
5,000 contacts per neuron.

## What the target specifies

- 400,000 excitatory and 100,000 inhibitory neurons.
- Excitatory morphology `[5, 5]`, giving 25 distal branches, 5 intermediate
  branches, and one soma per E neuron.
- Point inhibitory cells (`branch_factors: []`).
- Exactly 5,000 contacts onto every neuron and 2.5 billion contacts overall.
- Exactly 80% excitatory-source and 20% inhibitory-source contacts.
- Indexed sparse connectivity, efficient block-linear execution, and a
  recurrent Romo comparison task from the paper.
- Compact 32-bit topology buffers, bounded per-projection gather workspaces,
  deterministic rank-0 topology generation, and one transformed-weight graph
  per recurrent unroll.
- Backward gather recomputation, which retains the indexed topology and layer
  inputs instead of one contact-sized gathered-input tensor per recurrent
  step.
- No recurrent autapses. Self-indices are excluded directly during sparse
  sampling, without constructing a dense connectivity mask.

For an E target, the contact count is

```text
25 * (64 + 16) + 5 * (240 + 60) + (32 + 1168 + 300) = 5000.
```

For an I target it is `32 + 3968 + 1000 = 5000`.

The 5k/500 gate keeps the same 80/20 source ratio. For an E target,

```text
25 * (6 + 2) + 5 * (24 + 6) + (3 + 127 + 20) = 500.
```

For an I target it is `3 + 397 + 100 = 500`.

## Safe commands now

Set the repository and working Python once:

```bash
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
export PYTHONPATH=src
export PYTHON_BIN=/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python
mkdir -p logs
```

Load and inspect the full YAML without constructing the 500k model:

```bash
"$PYTHON_BIN" -c "from dendritic_modeling.config import load_config; c=load_config('configs/dendritic_rnn/large_scale_500k_5k/base.yaml'); print(c.model.core.type, c.distributed.mode)"
```

Construct the canary, its datasets, and its real indexed model without
training:

```bash
"$PYTHON_BIN" src/dendritic_modeling/scripts/training/train_experiments.py \
  configs/dendritic_rnn/large_scale_500k_5k/canary.yaml --validate-only
```

Run the one-epoch FSDP canary on two H100s:

```bash
sbatch \
  --job-name=dendrnn-canary \
  --partition=kempner_h100_priority \
  --account=kempner_dev \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h100_80gb_hbm3:2 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --output=logs/dendrnn_canary_%j.out \
  --error=logs/dendrnn_canary_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN" \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_500k_5k/canary.yaml 2
```

Run the measured 5k/500 gate with the same allocation:

```bash
sbatch \
  --job-name=dendrnn-5k-500 \
  --partition=kempner_h100_priority \
  --account=kempner_dev \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h100_80gb_hbm3:2 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --output=logs/dendrnn_5k_500_%j.out \
  --error=logs/dendrnn_5k_500_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN" \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_500k_5k/gate_5k_500.yaml 2
```

Each FSDP run reports the maximum CUDA allocated and reserved memory across
ranks after training and includes those byte counts in its results.

Run the 50k/5k gate on two H200s:

```bash
sbatch \
  --job-name=dendrnn-50k-5k \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:2 \
  --cpus-per-task=12 \
  --mem=128G \
  --time=02:00:00 \
  --output=logs/dendrnn_50k_5k_%j.out \
  --error=logs/dendrnn_50k_5k_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN" \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_500k_5k/gate_50k_5k.yaml 2
```

## Measured scaling results

`base.yaml` is schema-valid, and the principal implementation-level memory
controls are exposed in the typed configuration. For 2.5 billion contacts,
`indexed_index_dtype: auto` reduces the replicated topology buffer from 20 GB
to 10 GB per rank. `indexed_workspace_mb: 128` bounds each projection's
gathered-input workspace. Fixed transformed weights are built once per
recurrent unroll, topology is generated only on rank 0 and synchronized by
FSDP, deterministic indices are regenerated rather than included in every
checkpoint, and `indexed_recompute_backward: true` avoids retaining
contact-sized gathers for every recurrent step.

The indexed initializer, implicit self-exclusion, recurrent checkpoint
boundary, Romo sequence-length handling, and optional final state-dict gather
are covered by CPU tests. The FSDP entry point also honors the configured
11-dimensional Romo input.

The two-H100 canary has completed a full training and validation epoch,
including int32 CUDA indexing and rank-0 FSDP topology synchronization. Its
peak was 0.111 GiB allocated and 0.117 GiB reserved per rank.

The two-H100 5k/500 gate also completed. Its 2.5 million contacts used 0.855
GiB allocated and 0.939 GiB reserved at peak; the training/validation epoch
took 18.06 seconds.

The four-H200 50k/5k baseline, before backward recomputation, used 32.945 GiB
allocated and 35.605 GiB reserved per rank. Increasing the baseline from two
to four GPUs did not materially reduce this peak because recurrent
activations are replicated rather than FSDP-sharded. With
`indexed_recompute_backward: true`, the same four-H200 gate used 5.013 GiB
allocated and 6.869 GiB reserved. Its epoch took 1.88 seconds. The optimized
gate also completed on four H100s with the same memory counts and a 3.96
second epoch.

The exact `base.yaml` target then completed end to end on one node with four
H200 GPUs:

- 500,000 neurons, 5,000 contacts per neuron, and 2.5 billion contacts.
- FSDP setup peak: 30.785 GiB allocated / 35.586 GiB reserved per GPU.
- Training, validation, and final-checkpoint peak: 49.874 GiB allocated /
  73.121 GiB reserved per GPU.
- One epoch with 64 training and 32 validation examples: 200.67 seconds,
  excluding model construction and checkpoint time.
- End-to-end Slurm elapsed time: 8 minutes 35 seconds.
- Final full checkpoint size: 30.6 GB.

These are systems-validation measurements, not task-learning results. One
small pilot epoch is sufficient to exercise construction, FSDP
synchronization, forward, backward, optimizer state, validation, and
checkpointing, but not to evaluate Romo-task performance.

`--validate-only` on `base.yaml` still constructs the full model and therefore
requires a scale-appropriate allocation. A literal 5k-neuron/5k-contact
point-I gate cannot simultaneously use unique indices, exclude the cell's own
inhibitory index, and retain exactly 1,000 inhibitory-source contacts. The
50k/5k gate avoids that finite-size artifact while matching the target contact
allocation exactly.

The verified full-target allocation is one node with four H200 GPUs:

```bash
sbatch \
  --job-name=dendrnn-500k \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=32 \
  --mem=256G \
  --time=02:00:00 \
  --output=logs/dendrnn_500k_%j.out \
  --error=logs/dendrnn_500k_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN" \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_500k_5k/base.yaml 4
```

The current pilot writes a roughly 30.6 GB checkpoint. Disable checkpointing
for pure memory/timing gates, or budget storage before launching repeated
runs.
