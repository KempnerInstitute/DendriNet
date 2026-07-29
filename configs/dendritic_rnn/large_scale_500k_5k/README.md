# Large dendritic RNN: 500k neurons / 5k contacts per neuron

The editable target is [`base.yaml`](base.yaml).  It uses only the standard
`src/dendritic_modeling` configuration and training stack; nothing under
`drafts/` is imported.  The first runnable check is
[`canary.yaml`](canary.yaml).

## What the target specifies

- 400,000 excitatory and 100,000 inhibitory neurons.
- Excitatory morphology `[5, 5]`, giving 25 distal branches, 5 intermediate
  branches, and one soma per E neuron.
- Point inhibitory cells (`branch_factors: []`).
- Exactly 5,000 contacts onto every neuron and 2.5 billion contacts overall.
- Exactly 80% excitatory-source and 20% inhibitory-source contacts.
- Indexed sparse connectivity, efficient block-linear execution, and a
  recurrent Romo comparison task from the paper.

For an E target, the contact count is

```text
25 * (64 + 16) + 5 * (240 + 60) + (32 + 1168 + 300) = 5000.
```

For an I target it is `32 + 3968 + 1000 = 5000`.

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
  --wrap="cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling && PYTHONPATH=src $PYTHON_BIN -m torch.distributed.run --standalone --nproc_per_node=2 src/dendritic_modeling/scripts/training/train_experiments_fsdp.py configs/dendritic_rnn/large_scale_500k_5k/canary.yaml"
```

## Why the full job must wait

`base.yaml` is schema-valid, but a config alone cannot make the current
implementation scale safely. The indexed initializer, recurrent checkpoint
boundary, Romo sequence-length handling, and optional final state-dict gather
are now implemented and covered by CPU tests. The FSDP entry point also honors
the configured 11-dimensional Romo input.

The remaining gates are:

1. Pass the two-GPU CUDA/distributed canary.
2. Measure peak recurrent memory at intermediate neuron/contact scales.
3. Avoid or compress replicated connectivity-index buffers if those
   measurements show that they are limiting.

The exact 500k/5k target is still not a safe production run. FSDP shards
parameters, gradients, and optimizer state, but recurrent activations and
integer connectivity buffers need scale-gate measurements before extrapolating
to 2.5 billion contacts. Use measured peak memory from progressively larger
gates; do not infer feasibility from parameter count alone.

Do **not** run `--validate-only` on `base.yaml`: validation constructs the
model.  The next scale gates after the canary should be 5,000 neurons / 500
contacts, then 5,000 / 5,000 as an initializer benchmark, and only then the
full target.

After the scale gates pass, the intended maximum allocation is two nodes with
four H200 GPUs per node:

```bash
sbatch \
  --job-name=dendrnn-500k \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=48 \
  --mem=512G \
  --time=72:00:00 \
  --output=logs/dendrnn_500k_%j.out \
  --error=logs/dendrnn_500k_%j.err \
  --wrap="cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling && PYTHONPATH=src scripts/train_fsdp.sh configs/dendritic_rnn/large_scale_500k_5k/base.yaml 4"
```

That last command is the intended endpoint, not a command to submit yet.
