Frequently Asked Questions (FAQ)
=================================

This section addresses common questions and issues encountered when using the Dendritic Modeling Framework.

Installation and Setup
-----------------------

**Q: I'm getting import errors when trying to use the framework. What's wrong?**

A: This usually indicates an installation or environment issue:

1. Ensure you're in the correct conda environment: ``conda activate dendritic_env``
2. Verify the package is installed: ``pip list | grep dendritic``
3. Check if you installed in development mode: ``pip install -e .`` from the project root
4. If using custom modules, ensure the project root is in your Python path

**Q: The installation fails with CUDA errors. How do I fix this?**

A: CUDA compatibility issues are common:

1. Check your CUDA version: ``nvidia-smi``
2. Install the correct PyTorch version for your CUDA: Visit `pytorch.org <https://pytorch.org/>`_ for compatibility matrix
3. For CPU-only installation: ``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu``
4. If problems persist, try conda installation: ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``

**Q: How much disk space and memory do I need?**

A: Requirements depend on your use case:

- **Minimum**: 4GB RAM, 2GB disk space for basic experiments
- **Recommended**: 16GB RAM, 10GB disk space for typical research
- **Large-scale**: 32GB+ RAM, 50GB+ disk space for FSDP training
- **GPU memory**: 4GB for small models, 8GB+ for medium models, 24GB+ for large models

Training and Performance
------------------------

**Q: My training is very slow. How can I speed it up?**

A: Several optimization strategies can help:

1. **Enable mixed precision**: Set ``enable_amp: true`` in your config
2. **Increase batch size**: Use the largest batch size that fits in memory
3. **Use multiple workers**: Set ``dataloader_num_workers: 4-8`` in config
4. **Profile your code**: Use ``enable_profiling: true`` to identify bottlenecks
5. **Reduce model complexity**: Start with smaller networks and scale up

**Q: I'm running out of memory. What should I do?**

A: Memory issues have several solutions:

1. **Reduce batch size**: Lower ``batch_size`` in your config
2. **Enable gradient accumulation**: Use ``gradient_accumulation_steps: 4``
3. **Use gradient checkpointing**: Set ``gradient_checkpointing: true``
4. **Reduce model size**: Decrease layer sizes or number of branches
5. **For large models**: Enable FSDP with ``use_fsdp: true``

**Q: My model isn't converging. What's wrong?**

A: Convergence issues can have multiple causes:

1. **Check E/I balance**: Monitor excitation/inhibition ratios in logs
2. **Adjust learning rate**: Try lower rates (0.001 → 0.0001) or learning rate scheduling
3. **Verify data preprocessing**: Ensure proper normalization and data quality
4. **Use voltage stabilization**: For shunting inhibition, try ``voltage_stabilization: true``
5. **Check gradients**: Look for vanishing/exploding gradients in logs

**Q: How do I choose the right learning strategy?**

A: Strategy selection depends on your goals:

- **Standard**: Fast training, good for prototyping and baselines
- **Layer-wise**: Better for deep networks, hierarchical learning
- **Branch-wise**: Biological realism, interpretable representations
- **Two-step**: Complex tasks requiring unsupervised pretraining
- **Voltage stabilization**: Essential for shunting inhibition networks

Configuration and Architecture
------------------------------

**Q: How do I design a good dendritic architecture?**

A: Follow these guidelines:

1. **Start simple**: Begin with single-layer EINet before adding complexity
2. **E/I ratios**: Use 2:1 or 3:1 excitatory to inhibitory neuron ratios
3. **Branch factors**: Start with [4, 2] or [8, 4] for hierarchical branching
4. **Synapses per branch**: Begin with 32-64 synapses per branch
5. **Scale gradually**: Increase complexity based on task requirements

**Q: What's the difference between EINet and DendriNet?**

A: The architectures serve different purposes:

- **EINet**: Excitation-inhibition networks with separate E/I pathways, best for most tasks
- **DendriNet**: Pure dendritic networks with hierarchical processing, better for interpretability
- **MLP**: Standard networks for comparison and simple tasks
- **Hybrid**: Combinations for specialized applications

**Q: How do I configure sparse connectivity?**

A: Sparse connectivity options include:

- **standard**: Deterministic top-K selection (fastest)
- **stochastic**: Probabilistic selection (better exploration)
- **variance**: Variance-based selection (adaptive)
- **deepst**: Dynamic rewiring during training (most flexible)

Start with standard TopK and experiment with others for specific needs.

**Q: What input modes should I use?**

A: Input mode selection:

- **Mode 0**: Raw input to excitatory only (clear E/I separation)
- **Mode 1**: Same input to both pathways (simple but effective)
- **Mode 2**: Network-transformed inputs (most flexible, higher complexity)

Use Mode 0 for biological realism, Mode 1 for simple tasks, Mode 2 for complex preprocessing.

Analysis and Debugging
-----------------------

**Q: How do I interpret information theory results?**

A: Information theory metrics indicate:

- **High MI**: Strong information flow between components
- **Low MI**: Independent processing
- **I(E;I|C)**: How E/I interaction depends on class
- **Negative values**: Usually estimation errors (try different methods)

Always verify model performance before analyzing information flow.

**Q: My visualization isn't working. What's the issue?**

A: Common visualization problems:

1. **Missing dependencies**: Install visualization packages: ``pip install matplotlib seaborn plotly``
2. **Backend issues**: Try different matplotlib backends: ``matplotlib.use('Agg')``
3. **Memory issues**: Reduce plot complexity or data size
4. **File permissions**: Check write permissions for output directories

**Q: How do I run parameter sweeps?**

A: Parameter sweeps require:

1. **Create sweep config**: Define parameter ranges in YAML
2. **Use sweep manager**: ``python scripts/sweeps/unified/sweep_manager.py``
3. **SLURM setup**: Configure cluster parameters for HPC
4. **Analysis**: Use result analyzer for post-processing

See the sweep documentation for detailed examples.

**Q: What does "biological realism" mean in practice?**

A: Biological realism involves:

- **Dale's principle**: Separate E/I populations
- **Sparse connectivity**: Realistic connection patterns
- **Shunting inhibition**: Divisive normalization
- **Branch-wise learning**: Local credit assignment
- **Morphological constraints**: Realistic dendritic structures

Advanced Topics
---------------

**Q: How do I scale to very large models?**

A: Large-scale training requires:

1. **FSDP with intelligent delegation**: Use ``torchrun --nproc_per_node=4 train_experiments.py config.yaml --use_fsdp``
2. **Multiple GPUs**: Use appropriate cluster configurations
3. **Memory optimization**: Gradient checkpointing, CPU offload
4. **Monitoring**: Comprehensive logging and profiling

The framework provides automatic script selection and environment detection to simplify distributed training.

**Q: What's the difference between the training scripts?**

A: The framework provides intelligent script selection:

- **train_experiments.py**: Main script with intelligent FSDP delegation
  - Single GPU: ``python train_experiments.py config.yaml``
  - Multi-GPU: ``torchrun --nproc_per_node=4 train_experiments.py config.yaml --use_fsdp``
- **train_experiments_fsdp.py**: Direct FSDP script for advanced users
- **generate_slurm_job.py**: Automatic job generation for clusters

The main script automatically chooses the right approach based on your environment and flags.

**Q: I'm getting "FSDP requires distributed environment" error. What's wrong?**

A: This error occurs when using ``--use_fsdp`` without torchrun:

.. code-block:: bash

   # ❌ Wrong: Missing torchrun
   python train_experiments.py config.yaml --use_fsdp
   
   # ✅ Correct: With torchrun
   torchrun --nproc_per_node=4 train_experiments.py config.yaml --use_fsdp

The framework detects your environment and provides clear guidance on proper usage.

**Q: Can I use this framework with other deep learning libraries?**

A: Integration options:

- **PyTorch**: Full compatibility (framework is PyTorch-based)
- **Transformers**: Can integrate with Hugging Face models
- **Lightning**: Compatible with PyTorch Lightning
- **Custom**: Extensible architecture for custom integrations

**Q: How do I contribute new features?**

A: Contribution process:

1. **Open an issue**: Discuss the feature first
2. **Fork and branch**: Create a feature branch
3. **Follow standards**: Use existing code style and testing
4. **Documentation**: Update docs and add examples
5. **Pull request**: Submit for review

See the contributing guide for detailed instructions.

Troubleshooting Specific Errors
-------------------------------

**Q: "RuntimeError: CUDA out of memory"**

A: GPU memory issues:

.. code-block:: bash

   # Reduce batch size
   --train.batch_size 16
   
   # Enable gradient accumulation
   --train.gradient_accumulation_steps 4
   
   # Use CPU for large models
   --device cpu

**Q: "ImportError: No module named 'dendritic_modeling'"**

A: Installation issues:

.. code-block:: bash

   # Install in development mode
   pip install -e .
   
   # Or install from source
   pip install .
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/dendritic-modeling/src"

**Q: "ValueError: Inconsistent E/I dimensions"**

A: Architecture configuration issues:

.. code-block:: yaml

   # Ensure matching dimensions
   excitatory_layer_sizes: [128, 64]
   inhibitory_layer_sizes: [64, 32]  # Should be smaller or equal
   
   # Check branch factors
   excitatory_branch_factors: [4, 2]  # Must match layer count

**Q: "FileNotFoundError: Dataset not found"**

A: Data loading issues:

.. code-block:: yaml

   # Specify correct data path
   task:
     dataset: mnist
     data_path: "/path/to/data"
   
   # Or use auto-download
   task:
     dataset: mnist
     download: true

Getting Further Help
--------------------

If your question isn't answered here:

1. **Check the documentation**: Search the full documentation at :doc:`index`
2. **GitHub Issues**: Search existing issues or create a new one
3. **Discussion forum**: Join community discussions on GitHub
4. **Contact**: Reach out to the development team

**Useful Links**:

- :doc:`installation` - Installation guide
- :doc:`quickstart` - Quick start tutorial  
- :doc:`troubleshooting` - Detailed troubleshooting
- :doc:`contributing` - Contribution guidelines
- :doc:`glossary` - Key terminology 