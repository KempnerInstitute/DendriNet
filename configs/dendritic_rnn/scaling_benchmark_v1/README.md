# Dendritic-RNN scaling benchmark

This directory contains the standard source-code configurations for the
NeurIPS response scaling study. The matched frozen-core configurations vary
only network scale:

- 25,000, 100,000, 500,000, or 1,000,000 neurons;
- 80% excitatory and 20% inhibitory neurons;
- excitatory morphology `[3,3,3]`;
- exactly 20,000 active contacts per neuron;
- identical 128/32 train/validation examples;
- 150 normalized-readout epochs; and
- one node with four H200 GPUs.

The frozen-core experiments execute the complete dendritic E/I recurrent
forward model once for each example, cache its deterministic output, and
then train the readout. They test systems scaling and representation
learnability. They are not end-to-end recurrent-weight training claims.

`strong_100k_20k_1gpu.yaml` and `strong_100k_20k_2gpu.yaml` repeat the exact
100,000-neuron frozen-core experiment on one and two H200s. Together with the
four-H200 scale point, they measure strong scaling without changing the
model, data, or topology seed.

`e2e_100k_20k_canary.yaml` and `e2e_100k_20k_30ep.yaml` test full recurrent
optimization at the 100,000-neuron, 2-billion-contact scale on four
H100-80GB GPUs.
`e2e_100k_20k_stable_30ep.yaml` tests the same model with a smaller recurrent
learning rate and a larger wide-decoder learning rate. This separates
systems feasibility from learning-rate allocation. The canary must complete
with finite first-step diagnostics before either 30-epoch experiment is
submitted.

The paper-specific generator is:

```text
drafts/dendritic_RNN/scripts/generate_scaling_benchmark_configs.py
```

All model execution uses `src/dendritic_modeling` and `scripts/train_fsdp.sh`.
The generated resource and topology ledger is stored in `manifest.json`.
