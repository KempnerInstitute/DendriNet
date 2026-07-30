# Million-neuron dendritic RNN

This directory intentionally contains one current experiment configuration:

- [`test_1m_20k_frozen_readout_5000ep.yaml`](test_1m_20k_frozen_readout_5000ep.yaml)

It defines the exact four-H200 experiment with:

- 800,000 excitatory and 200,000 inhibitory neurons;
- excitatory morphology `[3,3,3]` and point inhibitory neurons;
- 32.2 million dynamical compartments;
- 20,000 active contacts per neuron and 20 billion contacts overall;
- the full recurrent E/I, dendritic, shunting, and multiscale forward model;
- a frozen recurrent core and cached 800,000-dimensional excitatory state;
- a trainable `800000 -> 32 -> 2` normalized readout; and
- 5,000 readout-training epochs on the delayed frequency-comparison task.

The result directory is:

```text
results/dendritic_rnn/large_scale_1m_10k/test_1m_20k_frozen_readout_5000ep/
```

## Run

```bash
export PYTHON_BIN=/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python

sbatch \
  --job-name=dendrnn-1m-20k-5kep \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=48 \
  --mem=768G \
  --time=01:40:00 \
  --output=logs/dendrnn_1m_20k_5000ep_%j.out \
  --error=logs/dendrnn_1m_20k_5000ep_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN",DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS=3600,DENDRITIC_FSDP_CPU_THREADS_PER_RANK=12,PYTORCH_ALLOC_CONF=expandable_segments:True,NCCL_MAX_NCHANNELS=8,NCCL_BUFFSIZE=2097152 \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_1m_10k/test_1m_20k_frozen_readout_5000ep.yaml 4
```

Four H200s are the minimum validated GPU allocation. The measured topology
construction peak is approximately 602 GiB of host memory, so 768 GiB leaves
reasonable headroom.
