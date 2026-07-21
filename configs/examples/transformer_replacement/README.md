# OLMo transformer-replacement examples

These configs cover the currently supported FFN replacement scope:

| Config | Replaced scope | Training path | Scale status |
| --- | --- | --- | --- |
| `olmo2_1b_single_ffn_layerwise.yaml` | layer 0 | layerwise teacher distillation | runnable on one suitable GPU |
| `olmo2_1b_multi_ffn_layerwise_ddp.yaml` | layers 0/4/8/12/15 | layerwise teacher distillation | runnable with `torchrun` DDP |
| `olmo2_1b_all_ffn_joint_ddp.yaml` | all 16 FFNs | joint LM/KL/hidden distillation | runnable with `torchrun` if teacher and student fit per GPU |
| `olmo2_1b_all_ffn_from_scratch.yaml` | all 16 FFNs | teacher-free full-model LM training | small-scale reference; replicated training only |

Every config uses the same 75%-density gated replacement: `K=1536` for the
2048-to-8192 gate/value projections and `K=6144` for the 8192-to-2048 output
projection. This retains 75% of FFN weights, or a 1.333x learned-parameter
reduction, before optimizer state and topology metadata. Distillation examples
copy teacher-TopK weights; the from-scratch example initializes the sparse
student directly. This is a fidelity-first baseline, not a runtime-speed claim.

The two DDP examples require `torchrun`; DDP replicates the full teacher/student
on every rank. Full-model 7B+ training remains gated on integrating the
repository's existing FSDP stack into this HF/OLMo path and adding sharded
checkpoint support. For independent layer jobs at 7B/13B scale, use the
package-owned OLMo config generator and assemble the resulting per-layer
checkpoints before joint training.
