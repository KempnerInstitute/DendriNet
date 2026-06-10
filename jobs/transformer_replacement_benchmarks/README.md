# Transformer Replacement Benchmark Jobs

These Slurm jobs run a two-stage benchmark for dendritic transformer MLP
replacement:

1. Train a `DendriticFFNReplacement` to distill a teacher MLP block.
2. Reload the saved replacement and compare the original teacher block with the
   replaced block on held-out hidden states.

The array job covers three synthetic transformer-style tasks:

- `synthetic_linear_projection.yaml`
- `synthetic_gelu_mlp.yaml`
- `synthetic_gated_silu_mlp.yaml`

Submit:

```bash
mkdir -p results/transformer_replacement_benchmarks/slurm
TRAIN_JOB=$(sbatch --parsable jobs/transformer_replacement_benchmarks/train_and_benchmark_array.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} jobs/transformer_replacement_benchmarks/aggregate.slurm
```

The first real Hugging Face language-model benchmark uses Pythia-70M on
WikiText and measures full causal-LM loss/perplexity before and after replacing
layer 0:

```bash
python -m pip install -e ".[transformer]"
sbatch jobs/transformer_replacement_benchmarks/pythia70m_train_and_benchmark.slurm
```

Two follow-up Pythia-70M variants compare the same layer-0 replacement with
standard dense TopK and with a larger/deeper indexed-dynamic DendriNet:

```bash
sbatch jobs/transformer_replacement_benchmarks/pythia70m_variant_benchmarks.slurm
```

The high-capacity multi-GPU variant uses a 1024-unit, three-level DendriNet with
DataParallel replacement modules:

```bash
sbatch jobs/transformer_replacement_benchmarks/pythia70m_xlarge_multigpu.slurm
```

DeepSeek-V3 is the real target for the LLM replacement experiments.  The
recommended workflow is:

1. Run DeepSeek-V3 with a model-parallel inference stack and cache MLP
   input/output tensors for layers 0, 1, and 2.
2. Train dendritic replacements from that hidden cache:

```bash
sbatch jobs/transformer_replacement_benchmarks/deepseek_v3_hidden_cache_replacement_array.slurm
```

For stock DeepSeek-V3 inference/serving, use the vLLM TP8+expert-parallel job:

```bash
sbatch jobs/transformer_replacement_benchmarks/deepseek_v3_vllm_tp8_ep_2node_h200.slurm
```

OLMo-2 replacement benchmarks use OLMo's open Dolmino training data and run on
``kempner_h100``:

```bash
sbatch jobs/transformer_replacement_benchmarks/olmo2_1b_dolmino_h100.slurm
sbatch jobs/transformer_replacement_benchmarks/olmo2_7b_dolmino_h100.slurm
```

To sweep every OLMo-2 1B MLP layer independently, then evaluate a model with
all trained dendritic replacements patched in together:

```bash
python -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_all_layer_configs --variant 1b
ARRAY_JOB=$(sbatch --parsable jobs/transformer_replacement_benchmarks/olmo2_1b_all_layers_dolmino_array.slurm)
sbatch --dependency=afterok:${ARRAY_JOB} jobs/transformer_replacement_benchmarks/olmo2_1b_all_layers_aggregate.slurm
```

The per-layer configs and manifest are written under
`jobs/transformer_replacement_benchmarks/configs/generated/olmo2_1b_dolmino_all_layers/`.
The per-layer and all-layer parameter/loss summaries are written under
`results/transformer_replacement_benchmarks/olmo2_1b_dolmino_all_layers/`.

To test whether all-layer degradation is caused by independently trained
replacements accumulating errors, run the joint all-layer OLMo-2 1B
fine-tuning sweep.  This patches all selected MLPs at once, freezes the dense
backbone, and trains only the dendritic replacements through LM/KL/hidden-state
losses.  The array currently includes:

- `[2, 2]` morphology warm-started from the layerwise all-layer checkpoints.
- `[2, 2]` morphology trained jointly from scratch.
- `[2, 2, 2]` morphology trained jointly from scratch.

```bash
python -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_joint_configs --variant 1b
sbatch jobs/transformer_replacement_benchmarks/olmo2_1b_joint_all_layers_array.slurm
```

The joint configs and manifest are written under
`jobs/transformer_replacement_benchmarks/configs/generated/olmo2_1b_joint_all_layers/`.
The joint run outputs are written under
`results/transformer_replacement_benchmarks/olmo2_1b_joint_all_layers/`.

To remove all OLMo-2 1B dense MLPs and replace the set with one shared
layerwise EI stack, use:

```bash
python -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_ei_stack_configs
sbatch jobs/transformer_replacement_benchmarks/olmo2_1b_ei_stack_all_mlp.slurm
```

The default EI-stack condition has 16 EI layers aligned to the 16 transformer
blocks, with 200 excitatory and 50 inhibitory cells per depth.  Outputs are
written under
`results/transformer_replacement_benchmarks/olmo2_1b_ei_stack_all_mlp/`.

Outputs:

- Per-task training metrics:
  `results/transformer_replacement_benchmarks/<task>/metrics.json`
- Per-task post-replacement benchmark:
  `results/transformer_replacement_benchmarks/<task>/benchmark_metrics.json`
- Aggregate comparison:
  `results/transformer_replacement_benchmarks/summary.csv`
  and `summary.md`
