.. _compact-checkpoints:

Compact checkpoints
===================

Top-K synapse layers train with a bank of candidate weights: a dense
``[out_features, in_features]`` matrix (``standard``, ``dense_to_sparse``) or
a candidate pool larger than ``K`` (``indexed_dynamic``). After training, each
output uses only its ``K`` active contacts. ``dendritic-compact-checkpoint``
replaces such layers with fixed indexed layers (``IndexedSparseLinear``) that
store only those contacts, ``K`` input indices and ``K`` weights per output,
and writes them to a self-describing compact checkpoint. The topology and the
stored weights do not change; outputs agree up to floating-point rounding,
because an indexed layer sums its contacts in a different order. The
conversion needs no optional dependency.

The compact checkpoint can also record a TorchAO weight-only quantization or
block-sparse transform for conventional dense ``nn.Linear`` layers, applied
when the checkpoint is loaded (:ref:`compact-dense-runtime`). Unlike topology
freezing, quantization changes the model's outputs.

Exporting a checkpoint
----------------------

``dendritic-compact-checkpoint`` takes the configuration that built the
model, a checkpoint holding its weights, and the path of the file to write:

.. code-block:: bash

   dendritic-compact-checkpoint \
     outputs/my_run_<timestamp>/config.json \
     outputs/my_run_<timestamp>/final_model.pt \
     outputs/my_run_<timestamp>/final_model.compact.pt \
     --compression-config configs/examples/deployment/compact_inference.yaml \
     --verify-full-model \
     --report outputs/my_run_<timestamp>/final_model.compact.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``experiment_config``
     - The training YAML, or the run's ``config.json``. The run's
       ``config.json`` also records the seeds and input shape that training
       resolved, so prefer it.
   * - ``source_checkpoint``
     - A file holding the model state dictionary, such as ``final_model.pt``
       or ``main_network/<prefix>best_model.pt`` of a run
       (:ref:`run-checkpoints`).
   * - ``output_checkpoint``
     - The compact checkpoint to write. It must not exist and must differ
       from the source.
   * - ``--compression-config``
     - A compression policy YAML (:ref:`compact-policy`). Without it the
       defaults apply.
   * - ``--verify-full-model``
     - Also compare the outputs of the whole model before export and after
       loading the written checkpoint (:ref:`compact-verification`).
   * - ``--device``
     - Device for the conversion and the checks; default ``cpu``.
   * - ``--report``
     - Also write the JSON report to this file. The file must not exist.

The command:

#. loads the configuration with the strict loader, builds the model with the
   same factory as training and loads the state dictionary strictly; with
   ``--verify-full-model`` it records the model's output for the check input
   and builds a fresh copy for the conversion;
#. converts the layers listed in :ref:`compact-conversion` and checks each
   converted layer against the original;
#. encodes the topology (:ref:`compact-encodings`) and writes the checkpoint
   to a temporary file that is then renamed, so an interrupted export never
   leaves a partial file;
#. with ``--verify-full-model``, rebuilds the model, loads the written
   checkpoint strictly and compares the outputs of the whole model;
#. prints its log records and then the JSON report to standard output, and
   with ``--report`` writes the report alone to the given file, in the same
   atomic way.

``--report`` refuses an existing file, or the path of the source or output
checkpoint, before anything is written.

The report lists:

- ``output``, ``output_sha256``, ``output_bytes`` and ``source``,
  ``source_bytes``, ``size_reduction_factor``: the two files and the ratio of
  their sizes;
- ``converted_modules`` and ``topology_encoding``;
- ``training_resources`` and ``deployment_resources``: parameter and synapse
  counts of the model before and after conversion, with candidate and active
  contacts per synapse layer, and ``core_parameter_reduction_factor``;
- ``training_runtime_storage``, ``deployment_runtime_storage`` and
  ``deployment_serialized_storage``: tensor bytes in memory before and after
  conversion and in the written file (parameters, buffers and topology
  separately). These are different quantities; do not add them up as one
  parameter count;
- ``full_model_verification``, ``experiment_config``, ``compression_config``,
  ``dense_runtime_backend`` and ``schema_version``.

The compact checkpoint holds the encoded state dictionary, one manifest
record per converted layer (path, source and target class, shape, ``K``,
weight transform, execution options and index dtype), the topology
encodings, the compression policy, and the path, SHA-256 hash and size of the
source checkpoint, with the paths and hashes of the experiment configuration
and policy files.

.. _compact-conversion:

Which layers are converted
--------------------------

Conversion matches the exact layer class. The synapse type is
``model.core.sparsity.type`` in structured E/I cores and ``topk_type`` in
population settings (:ref:`synapse-variants`).

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Synapse type
     - Layer class
     - Export
   * - ``standard``
     - ``TopKLinear``
     - Converted to its active contacts. Raises when ``noise_level`` is not
       zero.
   * - ``dense_to_sparse`` (``annealed_topk``)
     - ``DenseToSparseLinear``
     - Converted once its schedule has reached the target ``K``. Raises
       before that, unless ``require_final_dense_to_sparse_step: false``, and
       when ``noise_level`` is not zero.
   * - ``indexed_dynamic``
     - ``IndexedDynamicTopKLinear``
     - Converted to the ``K`` selected candidates when ``selection`` is
       ``standard`` and ``noise_level`` is zero; raises otherwise. Left
       unchanged with ``allow_indexed_dynamic: false``.
   * - ``indexed_rewire``
     - ``IndexedRewireLinear``
     - Converted to its current contacts; the rewiring state is dropped.
       Left unchanged with ``allow_indexed_rewire: false``.
   * - ``indexed``
     - ``IndexedSparseLinear``
     - Already compact; left unchanged.
   * - ``stochastic``, ``variance``
     - ``StochasticTopKLinear``, ``VarianceTopKLinear``
     - Not convertible losslessly: export raises (see below).
   * - ``deepst``, ``credit_deepst``
     - ``DeepstLinear``, ``CreditGatedDeepstLinear``
     - Not convertible losslessly: export raises.
   * - ``indexed_release``
     - ``IndexedReleaseLinear``
     - Already indexed; not converted and no error. Its contact statistics
       are kept with their contacts (:ref:`compact-encodings`).

Export fails closed: a sparse layer that topology freezing cannot convert
without changing the model raises, naming each such layer, because the
"compact" checkpoint would otherwise keep its dense candidate bank.
``sparse_topology.allow_unconverted: true`` exports such layers in their
training form instead, and ``converted_modules`` counts only the converted
ones. The affine MLP control converts in ``topk``
mode (``TopKLinear``) and stays unchanged in ``fixed`` mode. A layer object
that appears at two module paths raises.

A converted layer keeps the weight transform of the original layer, and its
indices take the policy's ``index_dtype``. Its execution options
(``projection_backend``, ``workspace_mb``, ``output_chunk_size``,
``cache_transformed_weights``) come from the original layer when that layer
has them (``indexed_dynamic``, ``indexed_rewire``); converted Top-K and
dense-to-sparse layers get the ``IndexedSparseLinear`` defaults, which
include ``projection_backend: eager``. The policy's
``sparse_topology.projection_backend`` and ``workspace_mb`` override both.

.. _compact-verification:

Numerical checks
----------------

Each converted layer is checked while it is replaced: the original and the
indexed layer receive the same fixed two-row standard-normal input, and their
outputs must agree within ``verification.atol`` and ``verification.rtol``
(both ``1e-5``) for float32 layers, or within a fixed ``5e-3`` for float16 and
bfloat16 layers. ``verification.enabled: false`` skips the check.

``--verify-full-model`` feeds one standard-normal sample, drawn with a fixed
seed, through the model before conversion and through a model rebuilt from
the written checkpoint, and compares the outputs with ``verification.atol``
and ``verification.rtol``. The report's ``full_model_verification`` gives the
input shape, the largest and mean absolute differences and whether the
predicted classes agree. Two requirements follow from the input it uses:

- Its shape comes from the configuration: a run's recorded ``config.json``
  holds the ``model.encoder.params.input_shape`` and ``input_dim`` training
  saw. A written ``input_shape`` other than its ``[3, 32, 32]`` default is
  used as is; otherwise the sample is flat with ``input_dim`` features (the
  image shape is kept only when it matches ``input_dim`` and
  ``data.processing.flatten`` is false). A hand-written configuration without
  either can state the shape, for example for flattened MNIST:

  .. code-block:: yaml

     model:
       encoder:
         type: identity
         params:
           input_shape: [784]

  Otherwise the check fails with a shape error such as ``PopulationNetwork
  expected last input dimension 784, got 32``.
- The sample has negative values. Structured shunting E/I cores reject signed
  input unless their transfer layer clamps it
  (``model.core.transfer.output_activation: relu``), so for them the check
  raises ``ValueError``; population networks log a warning and continue.
  Without the full-model check, the per-layer checks still run; evaluate the
  loaded compact model on held-out data instead.

.. _compact-encodings:

Topology encodings
------------------

``sparse_topology.topology_encoding`` selects how the checkpoint stores every
``connection_indices`` tensor of the model, converted or not:

``uint`` (default)
   Each index as ``uint16`` when the largest index is at most 65,535,
   otherwise as ``uint32``. Loading converts the indices back to the layer's
   index dtype. The order of the contacts is preserved.

``bitmask``
   One bit per possible input in each row. Smaller than ``uint`` when ``K`` is
   a large fraction of the input width; ``uint`` is smaller for very sparse
   rows. A bitmask records which inputs a row contacts but not the slot each
   contact is stored in, so the layers are stored in one of two layouts:

   - Fixed indexed layers (``IndexedSparseLinear``: the converted layers,
     ``indexed`` and degree-grouped layers) are stored with their contacts and
     weights in ascending input order. The weights are their only per-contact
     state and no update depends on the slot order. A layer whose contacts
     were not ascending sums them in another order after loading, so its
     outputs can differ in the last bits.
   - Every other indexed layer keeps its slot order exactly:
     ``indexed_release`` (its weights, ``impact`` and ``contact_age`` hold one
     entry per contact, and the release draws of training forwards and the
     cellwise impact rewiring follow the slots), ``indexed_rewire`` and
     ``indexed_dynamic`` layers left unconverted (``allow_indexed_rewire:
     false``, ``allow_indexed_dynamic: false``; ties of the pruning and the
     Top-K selection, the pairing of new inputs with pruned slots and the
     selection noise follow the slots), and layer types defined outside the
     library. For each row whose contacts are not ascending the file also
     stores the slot of every contact, one byte per contact up to 256
     contacts per row and two bytes up to 65,536, so ``bitmask`` saves less
     for these layers. Loading restores every tensor of these layers, and
     their next training update matches the source model's.

   The checkpoint's ``sparse_index_encoding`` names the layout of each layer:
   ``bitmask`` (ascending) or ``bitmask_slot_order``. Library versions before
   2026-10-13 cannot read a file with stored slot orders and raise ``Invalid
   bitmask metadata``.

Bitmask checkpoints written before 2026-10-13 stored every layer in the
ascending layout and reordered only the weights with the contacts. In such a
file, an ``indexed_release`` layer whose contacts were not ascending has its
``impact`` and ``contact_age`` attached to other contacts (its weights and
outputs are correct), and the next training update of an unconverted
``indexed_release``, ``indexed_rewire`` or ``indexed_dynamic`` layer can
differ from the source model's. The original order was not stored, so these
files cannot be repaired. ``load_compact_model_checkpoint`` reports each layer
of such a file whose contact order matters under ``contact_order_warnings``
and logs a warning; export the model again from its source checkpoint.

The export keeps the slot order of every layer that
``dendritic_modeling.deployment.bitmask_slot_order_keys(model)`` lists. A
layer may be reordered only when its class itself (not just a parent class)
sets ``SLOT_ORDER_MATTERS = False`` and names ``pre_w`` as its only
per-contact tensor in ``CONTACT_SLOT_STATE``, and its state holds no other
tensor with one entry per contact; a new indexed layer type keeps its slot
order until it declares otherwise. Code that encodes a state dictionary
directly with ``networks.checkpoints.compact_sparse_bitmask_state_dict`` gets
the ascending layout unless ``keep_slot_order`` names the
``connection_indices`` keys to keep (``True`` keeps all). The ascending layout
raises ``ValueError`` for a row it would reorder when the layer holds another
tensor with one entry per contact.

.. _compact-policy:

Compression policy
------------------

The policy is a separate YAML file with three sections; it does not change
training. ``configs/examples/deployment/compact_inference.yaml`` is an
example:

.. literalinclude:: ../configs/examples/deployment/compact_inference.yaml
   :language: yaml

Unknown sections raise ``ValueError``, and unknown keys inside a section
raise ``TypeError``. All keys and their defaults:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Default
     - Meaning
   * - ``sparse_topology.enabled``
     - ``true``
     - Convert layers. ``false`` converts no layer; the file still encodes
       the indices of indexed layers.
   * - ``sparse_topology.topology_encoding``
     - ``uint``
     - ``uint`` or ``bitmask`` (:ref:`compact-encodings`).
   * - ``sparse_topology.require_final_dense_to_sparse_step``
     - ``true``
     - Reject ``dense_to_sparse`` layers that have not reached their target
       ``K``.
   * - ``sparse_topology.allow_indexed_dynamic``
     - ``true``
     - Convert ``indexed_dynamic`` layers.
   * - ``sparse_topology.allow_indexed_rewire``
     - ``true``
     - Convert ``indexed_rewire`` layers.
   * - ``sparse_topology.index_dtype``
     - ``int64``
     - In-memory index dtype of converted layers: ``int32`` or ``int64``.
       There is no setting that keeps each layer's own dtype; set ``int32``
       for layers trained with ``index_dtype: auto`` or ``int32``.
   * - ``sparse_topology.projection_backend``
     - ``null``
     - Backend of converted layers (:ref:`indexed-backends`). ``null`` keeps
       the original layer's backend, or ``eager`` for layers without one. The
       per-layer check runs it on ``--device``, so a Triton backend needs a
       CUDA device.
   * - ``sparse_topology.workspace_mb``
     - ``null``
     - Gather workspace of converted layers. ``null`` keeps the original
       layer's value, or no budget for layers without one.
   * - ``dense_runtime.backend``
     - ``none``
     - ``none``, ``torchao_int8_weight_only``, ``torchao_int4_weight_only``
       or ``torchao_block_sparse`` (:ref:`compact-dense-runtime`).
   * - ``dense_runtime.include_patterns``
     - ``[]``
     - Regular expressions; when given, a layer's module path must match one.
   * - ``dense_runtime.exclude_patterns``
     - ``[]``
     - Regular expressions; a matching module path is skipped.
   * - ``dense_runtime.min_in_features``, ``min_out_features``
     - ``256``
     - Smallest input and output width of a transformed layer.
   * - ``dense_runtime.int4_group_size``
     - ``128``
     - Group size of int4 quantization: 32, 64, 128 or 256.
   * - ``dense_runtime.block_size``
     - ``64``
     - Square block size of ``torchao_block_sparse``.
   * - ``dense_runtime.minimum_block_sparsity``
     - ``0.8``
     - Fraction of all-zero blocks that each block-sparse layer must already
       have.
   * - ``verification.enabled``
     - ``true``
     - Run the per-layer check.
   * - ``verification.atol``, ``verification.rtol``
     - ``1e-5``
     - Tolerances of the float32 per-layer check and of the full-model check.

Loading a compact checkpoint
----------------------------

Build the model from the same configuration and load the compact checkpoint
into it. The loader rebuilds every converted layer as ``IndexedSparseLinear``
from the manifest, then loads the state dictionary strictly:

.. code-block:: python

   from dendritic_modeling.config import load_config
   from dendritic_modeling.deployment import load_compact_model_checkpoint
   from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

   config = load_config("outputs/my_run_<timestamp>/config.json")
   model, _ = initialize_model(config.model)
   report = load_compact_model_checkpoint(
       model, "outputs/my_run_<timestamp>/final_model.compact.pt"
   )
   model.eval()

The returned report gives the checkpoint path and SHA-256 hash, the missing
and unexpected keys (empty with the default ``strict=True``), the number of
converted layers, the ``contact_order_warnings`` of older bitmask files
(:ref:`compact-encodings`) and the dense-runtime result. A compact checkpoint
is not a plain state dictionary: load it with
``load_compact_model_checkpoint``, not with ``model.load_state_dict``.

A compact checkpoint does not keep the module versions of the state
dictionary (its ``_metadata``, :ref:`checkpoint-versions`), so loading treats
every module as written before its versioned entries existed. Every stored
value is loaded as stored. The visible effect is a ``UserWarning`` for each
``indexed_release`` layer, "contact ages in files written before 2026-10-04
count training forwards, not optimizer steps": the ages of a compact
checkpoint exported from a current model count optimizer steps, so the
warning can be ignored there. A versioned entry missing from the file would
likewise keep the module's constructed value instead of failing strict
loading; current exports write every entry.

Resume training from the original checkpoint, not the compact one: freezing
removes the inactive candidates and the rewiring state that training uses.

.. _compact-dense-runtime:

Dense runtime transforms with TorchAO
-------------------------------------

The TorchAO backends need the optional dependency (``torchao>=0.18,<0.19``):

.. code-block:: bash

   pip install -e '.[compression]'

Select the backend in the ``dense_runtime`` section of the policy given to
``dendritic-compact-checkpoint``; the policy is stored in the compact
checkpoint, so editing the YAML file afterwards has no effect. Loading applies
it on request:

.. code-block:: python

   report = load_compact_model_checkpoint(
       model, "outputs/my_run_<timestamp>/final_model.compact.pt",
       apply_dense_runtime=True,
   )

With ``apply_dense_runtime=True`` the loader puts the model in evaluation
mode and transforms every module whose class is exactly ``nn.Linear``, whose
widths reach ``min_in_features`` and ``min_out_features``, and whose path
passes the include and exclude patterns. Indexed dendritic layers are never
transformed. The call raises when no layer qualifies, and raises
``ImportError`` when TorchAO is not installed; with ``backend: none`` it
changes nothing. ``report["dense_runtime"]`` lists the transformed modules.

``torchao_int8_weight_only`` and ``torchao_int4_weight_only`` quantize the
weights, which changes predictions; measure the held-out metrics before and
after. ``torchao_block_sparse`` never zeroes weights: each selected layer must
have both weight dimensions divisible by ``block_size``, and a fraction of
all-zero blocks of at least ``minimum_block_sparsity`` already, as after
training with an explicit block mask; otherwise the call raises. A smaller
checkpoint does not by itself mean faster inference: measure latency on the
target device, batch size and input shape.

What to record
--------------

For each exported model, keep:

- the source and compact checkpoint SHA-256 hashes (in the report);
- the training configuration and the compression policy (their paths and
  hashes are stored in the compact checkpoint);
- held-out metrics before and after any transform that changes outputs;
- file size, peak device memory, latency and throughput, with the device,
  dtype, batch size and input shape used;
- the PyTorch, TorchAO and CUDA versions and the GPU model.

These separate storage savings from measured speed.

References:
`TorchAO sparsity documentation <https://docs.pytorch.org/ao/stable/contributing/sparsity.html>`_,
`PyTorch BSR tensor reference <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html>`_.
