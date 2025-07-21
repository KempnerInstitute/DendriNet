GPU Memory Estimation Guide
==================================================

This guide explains how to estimate GPU memory requirements for dendritic models before training.

Overview
------------------------------

The GPU memory estimation module provides comprehensive memory analysis for dendritic neural networks, including:
* Parameter breakdown by layer type
* Memory requirements for single-GPU and multi-GPU (FSDP) training
* GPU recommendations based on model size
* Suggested training commands

Standalone Script Usage
------------------------------

Basic Usage
~~~~~~~~~~~~~~~~~~~~

`````bash
Estimate GPU requirements for a configuration
==================================================
python scripts/estimate_gpu_requirements.py configs/your_config.yaml

Override batch size
==================================================
python scripts/estimate_gpu_requirements.py configs/your_config.yaml --batch-size 256
`````

Example Output
~~~~~~~~~~~~~~~~~~~~

The script provides:
#. **Model Parameter Breakdown**: Shows parameters by layer type (TopK, BlockLinear, Reactivation, etc.)
#. **Single GPU Analysis**: Memory breakdown and feasibility
#. **Multi-GPU FSDP Analysis**: Memory per GPU for different GPU counts
#. **GPU Recommendations**: Specific GPU models and required counts
#. **Training Commands**: Ready-to-use commands for your setup

Integrated with FSDP Training
------------------------------

When you run ``train_large_model_fsdp.py``, GPU memory estimation is automatically performed:

`````bash
torchrun --nproc_per_node=4 src/dendritic_modeling/scripts/train_large_model_fsdp.py configs/large_model.yaml
`````

The estimation report will appear before training starts, allowing you to:
* Verify the model fits on your GPUs
* See warnings if memory is insufficient
* Get suggestions for optimization (CPU offloading, more GPUs, etc.)

Memory Components
------------------------------

The estimator considers all major memory components:

#. **Parameters**: Model weights
#. **Gradients**: Same size as parameters
#. **Optimizer States**: 2x parameters for Adam/AdamW
#. **Activations**: Scaled by batch size and dendritic complexity
#. **FSDP Overhead**: ~15% for communication buffers and metadata

Dendritic-Specific Considerations
------------------------------

For dendritic networks, activation memory is adjusted based on:
* Branch complexity (more branches = more intermediate activations)
* Shunting computations (additional variables)
* TopK operations (sparse but requires indices)

GPU Recommendations
------------------------------

The module knows specifications for common GPUs:
* **H200**: 141GB (best for very large models)
* **H100**: 80GB (recommended for large models)
* **A100**: 80GB/40GB variants
* **A40**: 48GB (good price/performance)
* **V100**: 32GB/16GB (older but capable)

Tips for Large Models
------------------------------

#. **Start with estimation**: Always run the estimator before training
#. **Use FSDP for large models**: Models >40GB should use multi-GPU
#. **Power-of-2 GPUs**: Use 2, 4, 8, or 16 GPUs for efficient communication
#. **CPU offloading**: Use ``--cpu_offload`` flag if close to memory limits
#. **Gradient checkpointing**: Reduces activation memory (future feature)

Configuration Example
------------------------------

Here's a large model configuration that requires FSDP:

`````yaml
model:
  task: classification
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [4096, 2048, 1024, 512]
      excitatory_branch_factors: [4, 4, 2]
      # ... more parameters
`````

This model would require ~150GB for single-GPU training but only ~38GB per GPU with 4-GPU FSDP.

Programmatic Usage
------------------------------

You can also use the estimation module in your own scripts:

`````python
from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator

Create estimator
==================================================
estimator = GPUMemoryEstimator(dtype='fp16', optimizer='adam')

Estimate memory
==================================================
breakdown = estimator.estimate_memory_breakdown(
    model, batch_size=32, input_shape=(3072,), 
    use_fsdp=True, num_gpus=4
)

Get recommendations
==================================================
recommendations = estimator.get_gpu_recommendations(breakdown)
`````

Troubleshooting
------------------------------

* **"Model does not fit"**: Try more GPUs, smaller batch size, or CPU offloading
* **High activation memory**: Reduce batch size or use gradient checkpointing
* **FSDP overhead too high**: Use fewer, larger GPUs instead of many small ones 