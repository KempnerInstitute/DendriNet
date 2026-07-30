# Million-neuron dendritic RNNs

This directory contains the standard YAML configurations used to test
million-neuron dendritic RNNs through the main `dendritic_modeling` model,
dataset, FSDP, and trainer paths.

## Configurations

| Configuration | Neurons | Contacts per neuron | Purpose |
| --- | ---: | ---: | --- |
| [`base.yaml`](base.yaml) | 1,000,000 | 10,000 | One-step full-core systems test on 8 H200s |
| [`learn_1m_10k.yaml`](learn_1m_10k.yaml) | 1,000,000 | 10,000 | Experimental full-core learning configuration |
| [`test_1m_20k_frozen_readout.yaml`](test_1m_20k_frozen_readout.yaml) | 1,000,000 | 20,000 | Exact 20-billion-contact recurrent system with a trainable cached readout on 4 H200s |
| [`test_1m_20k_frozen_readout_5000ep.yaml`](test_1m_20k_frozen_readout_5000ep.yaml) | 1,000,000 | 20,000 | Five-thousand-epoch extension of the exact cached-readout experiment |
| [`diagnostic_20k_10k_readout_only.yaml`](diagnostic_20k_10k_readout_only.yaml) | 20,000 | 10,000 | Frozen-core readout learning control |
| [`diagnostic_20k_20k_frozen_readout.yaml`](diagnostic_20k_20k_frozen_readout.yaml) | 20,001 | 20,000 | Degree-matched control for the exact 20k-contact model |
| [`diagnostic_paper_scale_fsdp.yaml`](diagnostic_paper_scale_fsdp.yaml) | 80 | paper scale | Learning check for the indexed-synapse FSDP path |

The other `diagnostic_*` and `gate_*` files preserve the initialization,
normalization, learning-rate, and scale interventions used to isolate the
large-model training problem.

## Exact 1M-neuron / 20k-contact geometry

[`test_1m_20k_frozen_readout.yaml`](test_1m_20k_frozen_readout.yaml) has
800,000 excitatory neurons with morphology `[3,3,3]` and 200,000 inhibitory
point neurons. The excitatory branch multiplicities are `[27,9,3,1]`.

Every excitatory target receives 16,000 excitatory-source contacts:

```text
27 * 128 + 9 * 384 + 3 * 1152 + (5600 + 32) = 16000.
```

It also receives 4,000 inhibitory-source contacts:

```text
27 * 32 + 9 * 96 + 3 * 288 + 1408 = 4000.
```

Every inhibitory target receives the same E/I total:

```text
(32 feedforward E + 15968 recurrent E) + 4000 recurrent I = 20000.
```

The model therefore contains exactly 20 billion active contacts. Self
recurrence is disabled. The smaller degree-matched control uses 4,001
inhibitory neurons because 4,000 distinct recurrent inhibitory sources require
at least 4,001 cells when self-connections are excluded.

## What fits on four H200s

The exact recurrent forward and normalized readout training now fit on one
node with four H200s. The recurrent core is frozen, but it is not replaced or
approximated: every sample is first propagated through the full morphology,
E/I pathways, nonlinearities, taus, shunting dynamics, and all 20 billion
contacts. Because that core is deterministic and frozen, its final state is
cached once per sample. Later epochs optimize the readout without repeating
an identical expensive recurrent calculation.

This is different from full-core Adam training. The current indexed topology
is replicated on every rank. Its 20 billion `int32` indices require about
80 GB (74.5 GiB) per GPU. A four-way shard of 20 billion FP32 synaptic
parameters is another 20 GB per GPU. Full Adam would additionally require
sharded gradients and two optimizer moments. Indices, parameters, gradients,
and moments alone exceed H200 capacity before activations, FSDP all-gathers,
or transformed-weight workspaces are included.

Consequently:

- Exact recurrent execution plus readout learning is supported on 4 H200s.
- Updating all 20 billion independent recurrent weights with Adam is not
  supported on 4 H200s.
- Full-core learning at this scale will require sharded or procedural
  connectivity indices, parameter-efficient recurrent adaptation, compressed
  optimizer state, more GPUs, or a combination of these.

## Validated results

| Run | GPUs | Learning result | Peak allocated memory |
| --- | ---: | --- | ---: |
| Paper-scale indexed/FSDP control, job `36223565` | 2 H100 | Validation loss `0.6936 -> 0.0225`; 100% validation accuracy | 0.105 GiB/GPU |
| 20k neurons / 10k contacts, frozen core, job `36250235` | 2 H100 | Best validation loss `0.2721`; best validation accuracy 90.6% | 6.94 GiB/GPU |
| 20,001 neurons / 20k contacts, frozen core, job `36260818` | 2 H100 | Validation loss `0.6934 -> 0.6572` over 30 epochs | 9.13 GiB/GPU |
| 1M neurons / 10k contacts, full-core step, job `36228210` | 8 H200 | Completed train, Adam, and validation; the unscaled first update was unstable | 135.46 GiB/GPU |
| 1M neurons / 20k contacts, exact uncached readout, job `36265339` | 4 H200 | Validation loss `0.6938 -> 0.6919` over 5 epochs; final validation accuracy 62.5% | 121.51 GiB/GPU |
| 1M neurons / 20k contacts, cached readout, job `36279943` | 4 H200 | Validation loss `0.6929 -> 0.6472` over 30 epochs; best loss `0.6470` | 121.51 GiB/GPU |

The uncached exact 20k-contact job took 36 minutes. Topology construction and
FSDP setup used about 602 GiB of host memory. Each uncached epoch took about
291 seconds for 32 training and 8 validation examples.

The final cached job completed in 31 minutes 43 seconds. Its one-time exact
core pass cached 128 training and 32 validation states, each with 800,000
features. The following 240 readout updates took about 15 seconds. Training
loss decreased from `0.6938` to `0.6227`; validation loss decreased smoothly
from `0.6929` to a best value of `0.6470` and ended at `0.6472`. The best
validation accuracy was 62.5%; accuracy is quantized in 3.125 percentage-point
increments because the validation set has 32 examples.

This exact run and the degree-matched control both used 128/32 samples, hidden
width 32, global batch 16, 30 epochs, and 240 optimizer updates. The exact
model achieved a slightly lower final validation loss than the 20,001-neuron
20k-contact control (`0.6472` versus `0.6572`), although the runs use different
random seeds and are intended as feasibility controls rather than a
statistical scale comparison. Together with the finite first-step feature and
gradient diagnostics, the monotonic loss curve establishes that the exact
20-billion-contact recurrent representation contains task-relevant signal
that a trained readout can learn.

FSDP setup peaked at 121.51 GiB allocated per GPU. Overall reserved memory
peaked at 137.72 GiB per GPU, close to the H200 limit, so four H200s are the
minimum validated GPU allocation for this configuration.

## Why the initial large runs did not learn

The failure was not one single model-size limit:

1. The FSDP launcher retained the global learning rate but discarded
   component-specific rates from the YAML configuration.
2. Indexed sparse weights were not assigned to the synapse optimizer group.
3. Adam's first update to a dense `800000 -> hidden` decoder accumulated
   coherently across width and produced an oversized logit change.
4. Large recurrent states had a substantial common offset and a much smaller
   task-dependent component.
5. Frozen transformed synaptic tensors were retained between large FSDP
   modules, and the exponential transform temporarily materialized two
   full-sized outputs.

The corrected path preserves component learning rates, gives only the first
decoder weight a width-scaled rate, applies parameter-free LayerNorm to the
readout input, supports a zero-initialized output head, excludes frozen core
parameters from Adam, transforms frozen indexed weights by bounded output
chunks, and optionally caches deterministic frozen-core states. First-step
feature, logit, gradient, and memory diagnostics are saved with the training
results.

## Run the exact four-H200 configuration

```bash
export PYTHON_BIN=/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python

sbatch \
  --job-name=dendrnn-1m-20k-cached \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=48 \
  --mem=768G \
  --time=00:50:00 \
  --output=logs/dendrnn_1m_20k_cached_%j.out \
  --error=logs/dendrnn_1m_20k_cached_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN",DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS=3600,DENDRITIC_FSDP_CPU_THREADS_PER_RANK=12,PYTORCH_ALLOC_CONF=expandable_segments:True,NCCL_MAX_NCHANNELS=8,NCCL_BUFFSIZE=2097152 \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_1m_10k/test_1m_20k_frozen_readout.yaml 4
```

Rank 0 constructs the exact topology before FSDP synchronizes it. The measured
peak host use is about 602 GiB, so the 768 GiB request leaves approximately
27% headroom. The validated job used a conservative 1 TiB request; 768 GiB is
the recommended resource-efficient request for subsequent identical runs.
The 32 MB indexed workspace and bounded frozen-weight transform are also
required; increasing the workspace is not beneficial near the H200 memory
limit.

Results are written to:

```text
results/dendritic_rnn/large_scale_1m_10k/test_1m_20k_frozen_readout/
```

The directory contains `training_results.json` and
`standard_loss_curves.png`. The preserved five-epoch uncached result is in the
neighboring `test_1m_20k_frozen_readout_uncached_5epoch/` directory.
A local comparison against the two matched smaller controls is written to
`results/dendritic_rnn/large_scale_1m_10k/frozen_readout_learning_comparison.png`.
