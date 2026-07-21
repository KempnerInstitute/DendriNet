Dendritic LLM Scaling Workflow
==============================

The transformer-replacement stack supports three distinct workflows. Keep
them separate when interpreting results:

1. ``layerwise_distillation`` learns one or more FFN replacements from
   teacher activations.
2. ``joint_lm_distillation`` patches a causal LM and optimizes token-level LM,
   KL, and optional hidden-state losses.
3. The scaling profiler counts parameters without loading model weights. Its
   active count is a weight-use proxy, not a FLOP, latency, or energy result.

Parameter profiles
------------------

Profile a replacement config against a local Hugging Face ``config.json``:

.. code-block:: bash

   dendritic-profile-transformer-scaling \
     path/to/replacement.yaml \
     --hf-config-json path/to/model/config.json \
     --output results/scaling.json

The report distinguishes:

``stored_parameters``
   Learned scalar values stored in the model.

``active_weight_count_proxy``
   Dense non-FFN parameters plus the dendritic weights selected in a forward
   pass. This is useful for architectural comparisons but must not be reported
   as realized compute savings.

Measure tokens/second/GPU, model FLOPs utilization, peak device memory, and
wall-clock time separately on the target hardware. Indexed sparse storage can
reduce parameter count without producing a proportional speedup unless the
underlying kernels avoid dense gather and dispatch overheads.

The same profiler can compare non-FFN candidates without loading model weights:

.. code-block:: bash

   dendritic-profile-transformer-scaling \
     path/to/replacement.yaml \
     --hf-config-json path/to/model/config.json \
     --attention-layers all \
     --attention-query-topk 3072 \
     --attention-key-value-topk 3072 \
     --attention-output-topk 3072 \
     --embedding-rank 512 \
     --lm-head-rank 512

This keeps attention softmax, RoPE, and KV caching unchanged and counts only
indexed-sparse Q/K/V/O projections. Embedding reports distinguish storage from
weights touched per token: a normal embedding is a cheap row lookup, while a
rank-factorized embedding adds a rank-to-hidden projection. LM-head estimates
assume exact full-vocabulary logits; tied embedding/head models are explicitly
rejected as independent-head replacements.

Attention reports also expose runtime int64-index, compact uint16-index, and
row-bitmask topology bytes. At high retained density, unstructured runtime
indices can cost more memory than the weights they select. Prefer block-sparse
layouts for the first kernel experiment; a compact bitmask checkpoint is a
serialization result, not an efficient execution layout.

Replacement scope and example configs
-------------------------------------

The patcher accepts any non-empty unique list of decoder layers, so one,
multiple, or every FFN can be replaced. It preflights every target and
replacement constructor before mutating the model. The model and training
layer lists must match when both are set; this prevents training or checkpoint
metadata from silently referring to a different patched model.

Launch-oriented OLMo-2 1B examples live under
``configs/examples/transformer_replacement/``. They cover a single FFN,
multiple FFNs with layerwise DDP, all-FFN joint distillation, and all-FFN
teacher-free full-model training. The last example is a small-scale reference,
not authorization to run 7B+ full-model training before the existing FSDP
stack is integrated into this dedicated path and its scale gates pass.

Attention, embedding, and LM-head strategy
------------------------------------------

Use a staged contract-preserving path rather than replacing all three with a
nonlinear dendritic block at once:

1. **Attention projections:** preserve scaled dot-product attention, RoPE,
   causal masking, and the KV-cache layout. Distill indexed or block-sparse
   projections individually, starting with ``o_proj``, then ``q_proj``, and
   only then ``k_proj``/``v_proj``. K/V errors persist in the cache and are the
   highest-risk change. A measured sparse projection kernel is required for a
   compute claim.
2. **Token embedding:** do not feed a vocabulary-sized one-hot vector through
   the current dendritic FFN. Start with a factorized or codebook lookup and an
   optional small dendritic mixer, distilling embedding vectors and validating
   downstream perplexity. Factorization saves stored weights but may increase
   lookup compute.
3. **LM head:** for exact logits, begin with a low-rank head or tie it to the
   embedding when the architecture permits. Unstructured per-row gathers are
   unlikely to beat the dense vocabulary GEMM. Hierarchical vocabulary routing
   or a learned shortlist is a later approximate-output experiment and must be
   evaluated with exact NLL coverage and retrieval recall.

The immediate experimental order is therefore ``o_proj`` single-layer
distillation, factorized embedding/head storage-quality sweeps, Q/K/V
projection ablations, and only then combinations with all-FFN replacement.

OLMo all-layer preparation
--------------------------

The OLMo generator supports ``1b``, ``7b``, and ``13b`` variants. Generate
independent per-layer distillation runs and an all-layer comparison config:

.. code-block:: bash

   python -m \
     dendritic_modeling.scripts.transformer_replacement.generate_olmo_all_layer_configs \
     --variant 13b

The generated ``manifest.json`` records every expected replacement checkpoint.
After all independent jobs finish, assemble a flat warm-start directory:

.. code-block:: bash

   dendritic-assemble-transformer-replacements \
     generated/transformer_replacement_benchmarks/configs/olmo2_13b_all_layers/manifest.json

Symlinks are used by default so assembly does not duplicate large checkpoint
files. ``--mode hardlink`` and ``--mode copy`` are also available.

Joint training and scratch initialization
-----------------------------------------

The joint trainer accepts:

.. code-block:: yaml

   training:
     transformer_replacement:
       mode: joint_lm_distillation
       teacher_source: hf_text
       student_initialization: pretrained  # or from_config
       train_target: full_model
       save_student_model: false

``from_config`` constructs random causal-LM weights from the Hugging Face
architecture config before applying the dendritic replacements. It enables
small from-scratch controls; it does not by itself make the dedicated
replacement trainer distributed. ``save_student_model`` is opt-in because a
full 7B+ state dict is a large artifact. Replacement-only checkpoints remain
the default.

For layerwise distillation, ``loss: relative_mse`` (aliases
``normalized_mse`` and ``nmse``) divides batch MSE by detached teacher-target
power. This is useful when teacher-output magnitude changes substantially by
depth. ``cosine_weight`` can be combined with it as with ordinary MSE.

``indexed_rewire`` maintenance runs once after each optimizer step in all
dedicated transformer-replacement modes. Newly sampled slots have their
optimizer moments cleared. Older transformer results produced before this hook
was connected should be interpreted as fixed-index sparse runs even when their
configuration says ``indexed_rewire``.

Gated and teacher-TopK replacements
------------------------------------

``replacement_kwargs.kind: gated_dendritic_ffn`` builds separate dendritic
gate and value paths followed by a shared output projection. This mirrors the
multiplicative structure of SwiGLU-style OLMo/Llama FFNs while retaining the
existing indexed-synapse, calibration, and rewiring machinery.

A one-level sparse-GLU control can be initialized directly from a dense teacher:

.. code-block:: yaml

   model:
     transformer_replacement:
       input_transform: identity
       replacement_kwargs:
         kind: gated_dendritic_ffn
         gate_activation: silu
         teacher_topk_init: true
         output_topk: 384  # omit to keep the dense down projection
     core:
       # For the population-network form, use one population whose
       # branch_factors is [] and n_neurons equals the teacher FFN width.

Teacher-TopK initialization requires ``branch_factors: []``,
``reactivate: false``, ``weight_transform: identity``, and an indexed synapse.
It copies the largest-magnitude weights in every teacher gate/value row and,
when ``output_topk`` is set, every down-projection row. At full fan-in the
replacement is exactly equivalent to the dense teacher MLP. Retained weight
energy is saved in ``teacher_topk_initialization.json`` and benchmark metrics.

Layerwise training exports the validation-best replacement by default via
``restore_best_replacement: true``. Step 0 participates in model selection, so
fine-tuning cannot silently overwrite a better teacher-pruned initialization.

Transformer replacement checkpoints pack non-negative ``connection_indices``
as uint16 (or uint32 when required) and cast them back into the runtime buffer
on load. Convert an older checkpoint without overwriting it:

.. code-block:: bash

   dendritic-compact-transformer-checkpoint old.pt compact.pt

For moderately dense fixed topology, use row-wise masks to avoid paying one
integer index per retained weight:

.. code-block:: bash

   dendritic-compact-transformer-checkpoint \
     old.pt compact-bitmask.pt --encoding bitmask

Per-weight indices can still cost more bytes than a dense BF16 matrix at high
density. Bitmask checkpoints reorder each sparse row by input index and are
decoded transparently by the replacement loader. This reduces serialized
topology bytes, but it does not accelerate the current gather-based runtime;
kernel support is still required for an end-to-end compute claim.

Streaming text and DDP distillation
-----------------------------------

Set ``training.transformer_replacement.text.streaming: true`` to pack documents
incrementally instead of concatenating an in-memory corpus. ``data_files``
accepts datasets-compatible local paths and globs, including compressed JSON
shards. A bounded deterministic validation prefix is reserved with
``validation_documents`` and ``validation_max_tokens``; training skips that
prefix when no separate validation split/files are supplied. Every rank shards
the shuffled document stream, and its exact packing state is written beside the
run outputs. Layerwise validation uses fixed token offsets across experiment
cells and records per-layer MSE, relative MSE, cosine, R², and token count in
``metrics.json``. Training and benchmark artifacts also record wall time and
peak CUDA allocated/reserved bytes when running on GPU.

Set ``distributed_mode: ddp`` and launch the training entry point with
``torchrun`` for replicated-model data parallelism. DDP is appropriate for
layerwise or replacement-only distillation when the frozen teacher and student
fit on each GPU. The cluster launcher is
``scripts/train_transformer_replacement_ddp.sh``.

Data-driven reactivation policies such as ``occupancy_quantile`` are calibrated
on actual replacement inputs before optimization. The resolved voltage and
gate statistics are saved to ``reactivation_calibration.json``. Calibration can
be controlled with ``reactivation_calibration_batches`` and
``reactivation_calibration_iterations``; an empty
``reactivation_calibration_mode`` respects each layer's configured policy.

For preemptible layerwise jobs, ``checkpoint_every`` writes replacement,
optimizer, scaler, history, rank-local RNG, and rank-local packed-stream state.
Set ``resume_checkpoint`` to the saved ``training_checkpoint.pt`` to continue.
Joint-model sharded checkpointing and integration with the repository's
existing FSDP utilities remain separate milestones.

Current scale boundary
----------------------

The dedicated transformer-replacement loop now supports replicated ``torchrun``
DDP and packed streaming text. Full OLMo 7B/13B pretraining still requires the
following engineering milestones before claiming a fully sharded training run:

* integration of the existing FSDP stack plus decoder-block wrapping and
  distributed checkpoint/resume for the patched student;
* a teacher-offload or teacher-free pretraining path that avoids colocating two
  large models on every rank;
* measured sparse-kernel throughput and communication profiling.

Treat these as correctness gates, not optional performance refinements.
