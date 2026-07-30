Profiling with NVTX
===================

CUDA training runs emit NVTX ranges automatically.  The annotations are a
no-op on CPU-only runs and on PyTorch builds without NVTX support.

Capture a run with Nsight Systems, for example:

.. code-block:: bash

   nsys profile --trace=cuda,nvtx,osrt --output=dendrinet \
     python src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/config_exp.yaml

Training ranges
---------------

Each epoch has separate ``train``, ``validation``, and ``bookkeeping`` ranges.
Within a standard training batch, the trace separates:

* data transfer to the device
* model forward and loss computation
* regularization
* gradient clearing
* backward
* gradient checking or clipping
* optimizer step
* metric transfer to the host

Dendrite ranges
---------------

The dendritic forward pass is nested by network, branch layer, and synapse
type.  The key Top-K ranges are:

``dendrite/topk/select_mask``
   Selects the K active inputs for every output branch and constructs a dense
   binary mask.

``dendrite/topk/apply_dense_mask``
   Multiplies that mask by the full dense weight tensor.  This creates a
   logically sparse but physically dense weight tensor.

``dendrite/topk/dense_matmul``
   Calls ``torch.matmul`` with the masked dense tensor.  This is the dense GEMM
   that performs the logically sparse synaptic computation.

The non-efficient block-linear implementation is similarly split into
``materialize_dense_matrix`` and ``dense_matmul`` ranges.  The
``EfficientBlockLinear`` path instead appears as
``structured_multiply_reduce`` and does not construct the full block matrix.
