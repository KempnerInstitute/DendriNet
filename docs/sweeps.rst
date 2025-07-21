Sweeps Guide
============

The Dendritic Modeling Framework includes a unified sweep framework for systematic parameter exploration.

Unified Sweep Framework
-----------------------

Located in ``src/dendritic_modeling/sweeps/unified/``, this framework provides:

- **Types**: ``ei`` (E/I ratios), ``branch`` (branch factors), ``noise`` (noise levels), ``general`` (arbitrary parameters)
- **Implementation**: ``BaseSweepGenerator`` creates configs, ``SweepJobManager`` submits SLURM jobs, ``BaseSweepAnalyzer`` plots results
- **CLI Access**: Available via ``python -m dendritic_modeling.sweeps.unified.sweep_manager``

Sweep Types
-----------

1. **EI Sweep** (``ei``):
   - **Purpose**: Systematically explore excitatory/inhibitory synapse combinations
   - **Parameters**: ``ee_synapses_per_branch_per_layer``, ``ie_synapses_per_branch_per_layer``, ``use_shunting``
   - **Analysis**: Generates heatmaps of performance vs. E/I parameters
   - **Example**: ``python -m dendritic_modeling.sweeps.unified.sweep_manager ei --config configs/ei_sweep.yaml --run``

2. **Branch Sweep** (``branch``):
   - **Purpose**: Explore branch architecture variations
   - **Parameters**: ``num_branches_per_layer``, ``excitatory_branch_factors``
   - **Analysis**: Performance vs. branching complexity
   - **Example**: ``python -m dendritic_modeling.sweeps.unified.sweep_manager branch --config configs/branch_sweep.yaml --generate-only``

3. **Noise Sweep** (``noise``):
   - **Purpose**: Test network robustness to different noise types and levels
   - **Parameters**: ``noise_type`` (gaussian, poisson, uniform), ``noise_levels``
   - **Analysis**: Robustness curves and degradation analysis
   - **Example**: ``python -m dendritic_modeling.sweeps.unified.sweep_manager noise --config configs/noise_sweep.yaml --run``

4. **General Sweep** (``general``):
   - **Purpose**: Arbitrary parameter sweeps for custom experiments
   - **Parameters**: Any model/training parameter
   - **Analysis**: Flexible visualization based on swept parameters
   - **Example**: ``python -m dendritic_modeling.sweeps.unified.sweep_manager general --config configs/general_sweep.yaml --analyze --results-dir output/``

Command Line Options
-------------------

The sweep manager supports various execution modes:

- ``--generate-only``: Generate configurations without job submission
- ``--run``: Generate configs and submit SLURM jobs
- ``--analyze``: Analyze existing results (requires ``--results-dir``)
- ``--split-jobs N``: Split large sweeps into N smaller job batches
- ``--dry-run``: Show what would be submitted without actual execution

Configuration Examples
----------------------

**EI Sweep Configuration** (``configs/ei_sweep.yaml``):

.. code-block:: yaml

   sweep_config:
     model.network.parameters.ee_synapses_per_branch_per_layer: [[10], [20], [30], [40]]
     model.network.parameters.ie_synapses_per_branch_per_layer: [[10], [20], [30], [40]]
     model.network.parameters.use_shunting: [true, false]

   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42

   slurm_config:
     account: "kempner_dev"
     partition: "kempner"
     time: "02:00:00"
     gpus_per_node: 1
     max_concurrent_jobs: 50

**Branch Sweep Configuration** (``configs/branch_sweep.yaml``):

.. code-block:: yaml

   sweep_config:
     model.network.parameters.excitatory_branch_factors: [[2, 2, 1], [4, 2, 1], [8, 4, 2]]
     model.network.parameters.num_layers: [3, 4, 5]

   experiment_settings:
     seeds_per_condition: 5
     base_seed: 100

**General Sweep Configuration** (``configs/general_sweep.yaml``):

.. code-block:: yaml

   sweep_config:
     train.learning_rate: [0.0001, 0.001, 0.01]
     train.batch_size: [32, 64, 128]
     model.network.parameters.epsilon: [1e-6, 1e-5, 1e-4]

Workflow Examples
----------------

**Complete EI Sweep Workflow**:

.. code-block:: bash

   # 1. Generate and submit jobs
   python -m dendritic_modeling.sweeps.unified.sweep_manager ei \
       --config configs/ei_sweep.yaml --run

   # 2. Monitor job progress (check SLURM queue)
   squeue -u $USER

   # 3. Analyze results when complete
   python -m dendritic_modeling.sweeps.unified.sweep_manager ei \
       --analyze --results-dir output/ei_20240721_120000

**Development/Testing Workflow**:

.. code-block:: bash

   # 1. Generate configs only (no job submission)
   python -m dendritic_modeling.sweeps.unified.sweep_manager branch \
       --config configs/branch_sweep.yaml --generate-only

   # 2. Dry run to see what would be submitted
   python -m dendritic_modeling.sweeps.unified.sweep_manager branch \
       --config configs/branch_sweep.yaml --run --dry-run

   # 3. Split large sweep into smaller jobs
   python -m dendritic_modeling.sweeps.unified.sweep_manager general \
       --config configs/large_sweep.yaml --run --split-jobs 10

Result Analysis
--------------

The sweep framework automatically generates:

- **Heatmaps**: Performance metrics vs. swept parameters
- **Line Plots**: Trends across parameter ranges
- **Statistical Analysis**: Mean, std, significance tests across seeds
- **Missing Run Reports**: Identification of failed/incomplete jobs

Results are saved in timestamped directories: ``output/[sweep_type]_[timestamp]/``

Legacy Sweeps
------------

Individual sweep scripts are available in ``experiments/sweeps/`` for specialized use cases:

- ``ei_sweeps/``: Detailed EI parameter exploration
- ``experiment_sweeps/``: General experiment parameter sweeps
- ``general_sweeps/``: Utilities for custom parameter sweeps

For comprehensive documentation, see :doc:`CONFIGURATION_GUIDE` for parameter details and :doc:`examples` for usage examples. 