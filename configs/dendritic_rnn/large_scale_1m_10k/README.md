# Large dendritic RNN: 1M neurons / 10k contacts

The exact target is [`base.yaml`](base.yaml). It contains 800,000 excitatory
and 200,000 inhibitory neurons. Excitatory neurons use morphology `[3,3,3]`;
inhibitory neurons are point cells.

The branch multiplicities are `[27,9,3,1]`, so the target has 32 million
excitatory compartments and 0.2 million inhibitory compartments. Every neuron
receives exactly 8,000 excitatory-source and 2,000 inhibitory-source contacts,
giving 10 billion contacts overall.

For an excitatory target, excitatory-source contacts are

```text
27 * 64 + 9 * 192 + 3 * 576 + (32 + 2784) = 8000.
```

Inhibitory-source contacts are

```text
27 * 16 + 9 * 48 + 3 * 144 + 704 = 2000.
```

For an inhibitory target the allocation is
`(32 + 7968) + 2000 = 10000`.

## Staged test

[`gate_100k_10k.yaml`](gate_100k_10k.yaml) preserves the morphology and
contacts per neuron at one tenth of the neuron count. Its one billion contacts
are large enough to measure the dominant topology, parameter, recurrent, and
optimizer costs. It runs 30 epochs on 32 balanced training and 16 validation
examples and writes a real loss curve. It uses the analytical expectation
initialization that successfully trained the paper-scale Romo model.

[`diagnostic_20k_10k.yaml`](diagnostic_20k_10k.yaml) keeps the full 10,000
contacts per neuron at the smallest clean 80/20 population scale. Its 128
training examples and 30 epochs produce 960 distributed optimizer updates.
Use it to distinguish too few updates from degree- or scale-dependent loss of
trainable signal.

[`diagnostic_20k_1k.yaml`](diagnostic_20k_1k.yaml) is the matched degree
control. It changes only the E/I contact total from 8,000/2,000 to 800/200.

[`diagnostic_paper_scale_fsdp.yaml`](diagnostic_paper_scale_fsdp.yaml) runs the
64E/16I paper-scale morphology through the same indexed/FSDP execution path.
It separates an implementation-level training problem from a scaling effect.

## Measured results

The staged runs establish two separate points:

- Job `36195643` trained the 100k-neuron / one-billion-contact gate on four
  H200s. It used 18.22 GiB peak allocated memory per GPU and took about 37.2 s
  per epoch for 32 training and 16 validation examples. Its validation loss
  stayed at 0.6934 and validation accuracy stayed at 50%. An analytical
  initialization repeat (`36196837`) and the 20k-neuron 10k- and 1k-contact
  diagnostics (`36197511`, `36199137`) also remained at chance over comparable
  numbers of optimizer updates.
- Job `36200507` trained the paper-scale control through the same indexed
  connectivity, custom backward, and FSDP path on two H100s. Validation loss
  fell from 0.6931 to 0.0218 and validation accuracy reached 100% after 20
  epochs. Peak allocated memory was 0.105 GiB per GPU and an epoch took about
  34.4 s for 4,096 training and 1,024 validation examples.
- Job `36223565` repeats that control with the opt-in
  `auto_wrap_policy: indexed_synapses` used by the exact target. Its loss
  follows the same learning trajectory and reaches 0.0225 validation loss and
  100% validation accuracy. Peak allocated memory is again 0.105 GiB. The
  narrow wrapping policy therefore preserves learning, but its 87.4 s mean
  epoch is 2.54 times slower than whole-model FSDP for this small control. It
  is intended for contact-heavy models where the bounded all-gather is needed.
- The earlier 500k-neuron / 2.5-billion-contact systems gate (`36186369`) used
  49.87 GiB peak allocated memory per H200 and took 200.7 s for one epoch with
  64 training and 32 validation examples.
- An exact-target construction on 16 H100s (`36202200`) reached FSDP module
  synchronization but ran out of memory. Each rank had 75.21 GiB allocated and
  only 3.41 GiB free when a 5.96 GiB synchronization allocation was requested.
  More H100 ranks do not eliminate this replicated setup cost.
- The first exact H200 attempt (`36204181`) held about 83.7 GiB on each of
  seven waiting ranks. Rank 0 was still generating the 10-billion-contact
  topology when the default 10-minute process-group store timeout expired.
  This was a coordination timeout, not an H200 out-of-memory failure. The FSDP
  launcher now uses a 30-minute default process-group timeout, configurable
  with `DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS`.
- Job `36228210` completed the exact target on eight H200s across two nodes.
  It exercised topology construction, FSDP synchronization, forward,
  backward, Adam, and validation. FSDP setup peaked at 56.383 GiB allocated
  and 64.123 GiB reserved per GPU. Training and validation peaked at 135.457
  GiB allocated and 138.035 GiB reserved per GPU. The epoch took 65.36 s and
  the full Slurm job took 7 min 31 s. Each node used about 318 GB peak host
  memory.

The paper-scale result verifies that the optimized execution path learns. The
exact result establishes systems feasibility. It does not establish
large-scale task learning: one optimizer update changed validation loss from
the chance-scale training loss to 224 at 50% accuracy. Together with the flat
100k-neuron pilots, this identifies large-scale initialization, normalization,
and optimizer step size as the next questions; it is not evidence that FSDP
or the indexed operator is incorrect.

The combined learning, memory, and throughput diagnostic is written to
`results/dendritic_rnn/large_scale_1m_10k/learning_and_scaling_diagnostics.png`.

```bash
export PYTHON_BIN=/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python

sbatch \
  --job-name=dendrnn-100k-10k-learn \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=32 \
  --mem=256G \
  --time=02:00:00 \
  --output=logs/dendrnn_100k_10k_learn_%j.out \
  --error=logs/dendrnn_100k_10k_learn_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN" \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_1m_10k/gate_100k_10k.yaml 4
```

The exact systems test uses two nodes with four H200 GPUs each:

```bash
sbatch \
  --job-name=dendrnn-1m-10k \
  --partition=kempner_eng \
  --account=kempner_dev \
  --qos=normal \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gpus-per-node=nvidia_h200:4 \
  --cpus-per-task=48 \
  --mem=512G \
  --time=00:30:00 \
  --output=logs/dendrnn_1m_10k_%j.out \
  --error=logs/dendrnn_1m_10k_%j.err \
  --export=ALL,PYTHON_BIN="$PYTHON_BIN",DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS=3600,DENDRITIC_FSDP_CPU_THREADS_PER_RANK=12,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scripts/train_fsdp.sh \
  configs/dendritic_rnn/large_scale_1m_10k/base.yaml 4
```

The exact job uses eight total GPUs, one train step per rank, one validation
step per rank, and no checkpoint. It is a systems-feasibility test, not a
learning experiment. A full checkpoint would contain roughly 40 GB of weights
plus Adam state and should be expected to exceed 100 GB.

The exact H200 systems test requires the explicit process-group timeout, 12
CPU threads per rank, and expandable CUDA allocator shown above. Omitting the
expandable allocator caused a first-backward allocation failure despite
enough aggregate reserved memory. The verified job runs within the H200
capacity but has only a small memory margin; it should continue to use eight
H200s rather than a smaller allocation.
