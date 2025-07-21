Troubleshooting Guide
====================

This guide helps you diagnose and fix common issues with the Dendritic Modeling Framework. For quick answers to frequent questions, see the :doc:`faq`.

Quick Diagnosis
---------------

**Start Here**: Most issues fall into these categories:

1. :ref:`installation-issues` - Problems setting up the framework
2. :ref:`training-issues` - Convergence, performance, or memory problems
3. :ref:`configuration-issues` - YAML configuration errors
4. :ref:`gpu-issues` - CUDA and GPU-related problems
5. :ref:`analysis-issues` - Visualization and analysis tool problems

.. _installation-issues:

Installation Issues
-------------------

Problem: Import Errors
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ``ImportError: No module named 'dendritic_modeling'``

**Solutions**:

1. **Check conda environment**:
   
   .. code-block:: bash
   
      conda activate dendritic_env
      which python
      pip list | grep dendritic

2. **Reinstall in development mode**:
   
   .. code-block:: bash
   
      cd /path/to/dendritic-modeling
      pip install -e .

3. **Check Python path**:
   
   .. code-block:: python
   
      import sys
      print(sys.path)
      # Should include /path/to/dendritic-modeling/src

**Related**: See :doc:`faq` Q: "I'm getting import errors..."

Problem: CUDA Installation Conflicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Mixed CUDA versions, PyTorch conflicts

**Solutions**:

1. **Check CUDA version compatibility**:
   
   .. code-block:: bash
   
      nvidia-smi  # Check driver CUDA version
      nvcc --version  # Check toolkit version
      python -c "import torch; print(torch.version.cuda)"

2. **Clean reinstall PyTorch**:
   
   .. code-block:: bash
   
      pip uninstall torch torchvision torchaudio
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. **Use conda for CUDA packages**:
   
   .. code-block:: bash
   
      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

**Related**: :doc:`installation` for detailed setup instructions

.. _training-issues:

Training Issues
---------------

Problem: Poor Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: High loss, low accuracy, training plateau

**Diagnostic Steps**:

1. **Check data preprocessing**:
   
   .. code-block:: python
   
      # Verify data normalization
      print(f"Data range: {data.min():.3f} to {data.max():.3f}")
      print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")

2. **Monitor gradients**:
   
   .. code-block:: yaml
   
      train:
        enable_grad_monitoring: true
        log_grad_norms: true

3. **Check E/I balance**:
   
   .. code-block:: python
   
      # In training loop
      if hasattr(model.core_network, 'get_ei_ratio'):
           ei_ratio = model.core_network.get_ei_ratio()
           print(f"E/I ratio: {ei_ratio:.3f}")

**Solutions**:

- **Lower learning rate**: Try 0.0001 instead of 0.001
- **Use voltage stabilization**: For shunting inhibition networks
- **Gradient scaling**: Enable for deep dendritic networks
- **Architecture simplification**: Start with fewer layers/branches

**Related**: :doc:`training/index` for training strategies

Problem: Memory Issues
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ``CUDA out of memory``, system freezing

**Immediate Solutions**:

.. code-block:: yaml

   train:
     batch_size: 32              # Reduce from 128
     gradient_accumulation_steps: 4  # Maintain effective batch size
     enable_amp: true            # Mixed precision
     gradient_checkpointing: true # Trade compute for memory

**Long-term Solutions**:

- **Model architecture**: Reduce layer sizes or branches
- **FSDP**: For very large models, see :doc:`performance/index`
- **Hardware upgrade**: Consider more GPU memory

**Related**: :doc:`faq` Q: "I'm running out of memory..."

Problem: Slow Training
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Very slow epochs, low GPU utilization

**Performance Optimization**:

.. code-block:: yaml

   train:
     dataloader_num_workers: 8   # Parallel data loading
     pin_memory: true           # Faster GPU transfers
     enable_amp: true           # Mixed precision
   
   # In model config
   model:
     efficient_blocklinear: true # Memory-efficient implementation

**Profiling**:

.. code-block:: yaml

   train:
     enable_profiling: true
     profiling_output_dir: "./profiling"

View results: ``tensorboard --logdir=./profiling``

**Related**: :doc:`performance/index` for optimization techniques

.. _configuration-issues:

Configuration Issues
--------------------

Problem: YAML Syntax Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common Mistakes**:

.. code-block:: yaml

   # Wrong: Mixed tabs and spaces
   model:
   	core_network:  # Tab used here
       type: EINet    # Spaces used here
   
   # Correct: Consistent spacing
   model:
     core_network:
       type: EINet

   # Wrong: Missing quotes for special characters
   outputs:
     run_name: test-run@2024  # @ is special
   
   # Correct: Quoted strings
   outputs:
     run_name: "test-run@2024"

**Validation**:

.. code-block:: bash

   # Test configuration syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"

**Related**: :doc:`configuration/index` for complete configuration guide

Problem: Parameter Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common Issues**:

1. **Mismatched dimensions**:
   
   .. code-block:: yaml
   
      # Wrong: Incompatible sizes
      excitatory_layer_sizes: [128, 64]
      excitatory_branch_factors: [4, 2, 2]  # Too many elements
      
      # Correct: Matching dimensions
      excitatory_layer_sizes: [128, 64]
      excitatory_branch_factors: [4, 2]

2. **Invalid parameter ranges**:
   
   .. code-block:: yaml
   
      # Wrong: Invalid learning rate
      train:
        learning_rate: 10.0  # Too high
      
      # Correct: Reasonable range
      train:
        learning_rate: 0.001

**Related**: :doc:`glossary` for parameter definitions

.. _gpu-issues:

GPU Issues
----------

Problem: CUDA Device Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnostic Commands**:

.. code-block:: bash

   # Check GPU status
   nvidia-smi
   
   # Check PyTorch CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch; print(torch.cuda.device_count())"

**Solutions**:

1. **Force CPU usage** (temporary):
   
   .. code-block:: yaml
   
      device: cpu

2. **Reset GPU state**:
   
   .. code-block:: bash
   
      sudo nvidia-smi --gpu-reset

3. **Check driver compatibility**:
   
   .. code-block:: bash
   
      cat /proc/driver/nvidia/version

**Related**: :doc:`installation` for GPU setup

Problem: Mixed GPU/CPU Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ``RuntimeError: Expected all tensors to be on the same device``

**Solution**:

.. code-block:: python

   # Ensure consistent device placement
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   data = data.to(device)
   target = target.to(device)

.. _analysis-issues:

Analysis Issues
---------------

Problem: Visualization Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common Issues**:

1. **Missing backend**:
   
   .. code-block:: python
   
      import matplotlib
      matplotlib.use('Agg')  # For headless systems
      import matplotlib.pyplot as plt

2. **Memory issues with large plots**:
   
   .. code-block:: yaml
   
      visualization:
        max_samples: 1000  # Limit data size
        dpi: 100          # Reduce resolution

**Related**: :doc:`analysis/index` for visualization tools

Problem: Information Theory Analysis Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Negative MI values, estimation errors

**Solutions**:

1. **Check model performance first**:
   
   .. code-block:: python
   
      # Ensure accuracy > chance before analyzing
      assert accuracy > 0.5  # For binary classification

2. **Adjust estimation parameters**:
   
   .. code-block:: yaml
   
      analysis:
        information_theory:
          n_neighbors: 5      # Increase for better estimation
          estimation_method: sklearn  # Try different estimator

**Related**: :doc:`analysis/information_theory` for detailed guidance

Getting Help
------------

When All Else Fails
~~~~~~~~~~~~~~~~~~~

1. **Search existing issues**: `GitHub Issues <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_

2. **Create a minimal example**:
   
   .. code-block:: python
   
      # Minimal failing example
      import torch
      from dendritic_modeling.networks import ExcitationInhibitionNetwork
      
      # Your minimal failing code here
      model = ExcitationInhibitionNetwork(input_dim=10, ...)

3. **Gather system information**:
   
   .. code-block:: bash
   
      python --version
      pip list | grep torch
      nvidia-smi
      uname -a

4. **Check logs and error messages**:
   
   .. code-block:: bash
   
      # Look for detailed tracebacks
      python your_script.py 2>&1 | tee error_log.txt

**Include in your issue**:
- Minimal failing example
- Complete error traceback
- System information
- Configuration file (if relevant)

Community Resources
~~~~~~~~~~~~~~~~~~

- :doc:`faq` - Frequently asked questions
- :doc:`contributing` - How to contribute fixes
- :doc:`api/index` - Detailed API documentation
- `GitHub Discussions <https://github.com/KempnerInstitute/dendritic-modeling/discussions>`_ - Community support

**Emergency Contacts**: For critical research issues, contact the development team directly.

Advanced Debugging
------------------

Debug Mode
~~~~~~~~~~

Enable comprehensive debugging:

.. code-block:: yaml

   debug:
     enable: true
     log_level: DEBUG
     save_intermediates: true
     verbose_errors: true

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

Identify bottlenecks:

.. code-block:: python

   # Enable PyTorch profiler
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU, 
                  torch.profiler.ProfilerActivity.CUDA],
       record_shapes=True
   ) as prof:
       output = model(input_data)
   
   print(prof.key_averages().table(sort_by="cuda_time_total"))

Memory Debugging
~~~~~~~~~~~~~~~~

Track memory usage:

.. code-block:: python

   import torch
   
   # Monitor GPU memory
   if torch.cuda.is_available():
       print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
       print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

This comprehensive troubleshooting guide should help you resolve most issues. Remember to check the :doc:`faq` first for quick solutions! 