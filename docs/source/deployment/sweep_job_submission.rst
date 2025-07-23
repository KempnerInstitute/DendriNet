.. _sweep_job_submission:

Sweep Job Submission
====================

This document explains how to run large-scale parameter sweeps and submit jobs to a SLURM cluster using the dendritic modeling framework.

Overview
--------

Running large-scale experiments is essential for exploring the parameter space of dendritic networks. The dendritic modeling framework provides tools for:

1. **Generating parameter sweeps**: Creating multiple configurations by varying parameters
2. **Submitting array jobs**: Running many experiments in parallel on a cluster
3. **Managing experiment results**: Organizing and analyzing results from large sweeps

These capabilities allow researchers to efficiently explore the parameter space and identify optimal configurations for specific tasks.

Parameter Sweeps
----------------

The framework provides a dedicated script ``generate_sweep_configs.py`` for generating parameter sweeps:

.. code-block:: bash

    python scripts/generate_sweep_configs.py configs sweep_params.yaml

This script takes a sweep parameters file and generates multiple configuration files, each with a different combination of parameter values.

Sweep Parameters File
~~~~~~~~~~~~~~~~~~~~~

The sweep parameters file defines:

1. The base configuration file to use
2. Parameters to vary and their possible values
3. SLURM configuration for job submission

Here's an example sweep parameters file:

.. code-block:: yaml

    # Base training configuration
    train_config:
      model:
        task: classification
        probabilistic: false
        network:
          type: EINet
          parameters:
            excitatory_layer_sizes: [10]
            inhibitory_layer_sizes: [20]
            excitatory_branch_factors: [1,2]
            inhibitory_branch_factors: []
            reactivate: true
            somatic_synapses: true
            ee_synapses_per_branch_per_layer: [24]
            ei_synapses_per_branch_per_layer: [200]
            ie_synapses_per_branch_per_layer: [2]
            ii_synapses_per_branch_per_layer: [0]
            topk_init_method: xavier_normal
            use_shunting: true
            reactivation_type: param_tanh
            reactivation_strategy: inverse
            blocklinear_strategy: block_conductance_dynamic
            input_mode: 2
            input_net_dict:
              type: "IndexInputTransform"
              hidden_dims: []
              output_dim: 128

      train:
        learning_strategy: mle
        run_name: einet_sweep
        seed: 42
        train_valid_split: 0.8
        lr: 0.001
        epochs: 50
        batch_size: 64
        grad_clip_value: 1.0
        weight_decay_rate: 0.01

      task:
        dataset: mnist
        data_path:
        parameters: {}

      wandb:
        entity:
        project:
        group:
        tags:
        use_wandb: false

      visualization:
        enabled: false
        visualize_epochs: [10, 30]
        plots:
          - weights
          - activations
          - gradients

      outputs:
        dir: outputs

    # Parameters to sweep over
    sweep_config:
      model.network.parameters.excitatory_branch_factors:
        - [1,2]
        - [1,3]
        - [1,4]

      train.lr:
        - 0.001
        - 0.0005
        - 0.0001

      train.batch_size:
        - 32
        - 64
        - 128

    # SLURM configuration for job submission
    slurm_config:
      account: "lab-account"
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: 1
      cpus_per_task: 4
      time: "12:00:00"
      mem: "32G"
      partition: "gpu"
      modules_to_load:
        - "python/3.8"
        - "cuda/11.3"
      conda_env_path: "/path/to/conda/envs/dendritic-modeling"
      training_script: "scripts/train_experiments.py"
      output_dir: "outputs"
      config_filename_init: "config"
      max_concurrent_jobs: 10

The ``sweep_config`` section defines which parameters to vary and what values to use. The script will generate all possible combinations of these parameter values.

Generating Sweep Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you run the ``generate_sweep_configs.py`` script, it:

1. Reads the base configuration and sweep parameters
2. Generates all combinations of parameter values
3. Creates a separate configuration file for each combination
4. Creates a summary file with all the parameter combinations
5. Generates a SLURM batch script for submitting the jobs

For example, if you have 3 parameters with 3 values each, it will generate 27 configuration files.

Structure of Generated Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script generates three types of files:

1. **Configuration files**: One for each parameter combination (e.g., ``config_0.yaml``, ``config_1.yaml``, ...)
2. **Summary file**: A text file listing all parameter combinations (``config_sweeps.txt``)
3. **Batch script**: A SLURM batch script for submitting the jobs (``run.sh``)

The configuration files follow the structure of the base configuration but with parameter values replaced according to the sweep configuration.

SLURM Job Submission
--------------------

The framework can generate a SLURM batch script for running the parameter sweep as an array job:

.. code-block:: bash

    bash run.sh

This script contains all necessary SLURM directives and commands to run the experiments.

SLURM Array Jobs
~~~~~~~~~~~~~~~~

SLURM array jobs allow running multiple similar jobs with different parameters. The generated batch script uses the array job feature to run all configurations in parallel:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=einet_sweep
    #SBATCH --account=lab-account
    #SBATCH --output=outputs/einet_sweep_%A/%A_%a/output_%A_%a.out
    #SBATCH --error=outputs/einet_sweep_%A/%A_%a/error_%A_%a.out
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=1
    #SBATCH --cpus-per-task=4
    #SBATCH --time=12:00:00
    #SBATCH --mem=32G
    #SBATCH --partition=gpu
    #SBATCH --array=0-26%10  # Run configurations 0-26, max 10 at a time

This batch script includes:

1. **Job metadata**: Name, account, output/error files
2. **Resource requirements**: Nodes, tasks, GPUs, CPUs, memory, time limit
3. **Array job configuration**: Which configurations to run and concurrency limit

SLURM Array Task IDs
~~~~~~~~~~~~~~~~~~~~

Within the batch script, the array task ID (``SLURM_ARRAY_TASK_ID``) is used to select the appropriate configuration file:

.. code-block:: bash

    # Get the specific config file based on the array job ID
    CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml
    RUN_NAME=${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

This ensures each job uses the correct configuration file.

Output Organization
~~~~~~~~~~~~~~~~~~~

The batch script organizes output files into directories based on the job ID and array task ID:

.. code-block:: bash

    # Path to the folder to save outputs
    OUTPUT_FOLDER_PATH=${OUTPUT_DIR}/${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

This creates a hierarchical directory structure that keeps outputs from different runs organized.

.. note::
   Always check your generated plots carefully to ensure the experiments are producing expected trends.

Customizing SLURM Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the SLURM configuration by editing the ``slurm_config`` section in the sweep parameters file:

.. code-block:: yaml

    slurm_config:
      account: "your-account"          # SLURM account to use
      nodes: 1                         # Number of nodes per job
      ntasks_per_node: 1               # Number of tasks per node
      gpus_per_node: 1                 # Number of GPUs per node
      cpus_per_task: 4                 # Number of CPUs per task
      time: "12:00:00"                 # Time limit (HH:MM:SS)
      mem: "32G"                       # Memory per node
      partition: "gpu"                 # SLURM partition
      modules_to_load:                 # Modules to load
        - "python/3.8"
        - "cuda/11.3"
      conda_env_path: "/path/to/conda/envs/dendritic-modeling"  # Path to conda environment
      training_script: "scripts/train_experiments.py"          # Training script to run
      output_dir: "outputs"                                   # Output directory
      config_filename_init: "config"                          # Config filename prefix
      max_concurrent_jobs: 10                                # Maximum concurrent jobs

Each field in this configuration maps to a SLURM directive or environment variable in the generated batch script.

Analysis and Hyperparameter Search
----------------------------------

After running a parameter sweep, you need to analyze the results to identify the best parameter combination.

Collecting Results
~~~~~~~~~~~~~~~~~~

To collect results from all runs, you can use the following script:

.. code-block:: python

    import os
    import pandas as pd
    import yaml

    def collect_sweep_results(sweep_dir, metrics_filename="metrics.csv"):
        """
        Collect results from all runs in a sweep.

        Parameters
        ----------
        sweep_dir : str
            Directory containing sweep results
        metrics_filename : str
            Name of metrics file

        Returns
        -------
        pd.DataFrame
            DataFrame with results from all runs
        """
        results = []

        # Load sweep summary
        with open(os.path.join(sweep_dir, "config_sweeps.txt"), "r") as f:
            sweep_summary = f.readlines()

        # Parse sweep summary
        sweep_params = {}
        for line in sweep_summary:
            line = line.strip()
            if not line:
                continue
            config_file, params = line.split(":", 1)
            config_id = int(config_file.split("_")[1].split(".")[0])
            sweep_params[config_id] = eval(params)

        # Find all result directories
        for root, dirs, files in os.walk(sweep_dir):
            if metrics_filename in files:
                # Get config ID from directory path
                parts = root.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    config_id = int(parts[-1])

                    # Load metrics
                    metrics_path = os.path.join(root, metrics_filename)
                    metrics = pd.read_csv(metrics_path)

                    # Get best validation accuracy
                    best_valid_acc = metrics["valid_accuracy"].max()

                    # Add to results
                    result = {
                        "config_id": config_id,
                        "best_valid_acc": best_valid_acc,
                        **sweep_params.get(config_id, {})
                    }
                    results.append(result)

        return pd.DataFrame(results)

This script collects results from all runs in a sweep and returns a DataFrame with the best validation accuracy for each parameter combination.

Finding the Best Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can then find the best parameter combination:

.. code-block:: python

    # Collect results
    results = collect_sweep_results("outputs/sweep_20230501")

    # Find best configuration
    best_config = results.loc[results["best_valid_acc"].idxmax()]
    print(f"Best configuration: {best_config}")

This identifies the parameter combination that achieved the highest validation accuracy.

Visualizing Sweep Results
~~~~~~~~~~~~~~~~~~~~~~~~~

You can visualize how different parameters affect performance:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot effect of learning rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="train.lr", y="best_valid_acc", data=results)
    plt.title("Effect of Learning Rate on Validation Accuracy")
    plt.xlabel("Learning Rate")
    plt.ylabel("Best Validation Accuracy")
    plt.savefig("lr_effect.png")

This helps visualize the effect of different parameters on model performance.

Tips for Running Large-Scale Sweeps
-----------------------------------

1. **Start small**: Begin with a small subset of parameters and values to test the workflow
2. **Monitor resource usage**: Check memory and GPU usage to ensure efficient resource allocation
3. **Set appropriate time limits**: Estimate job duration and set time limits accordingly
4. **Use concurrency limits**: Limit the number of concurrent jobs to avoid overwhelming the cluster
5. **Add checkpointing**: Enable checkpointing to resume jobs if they are interrupted
6. **Use unique run names**: Include timestamp or unique identifier in run names

Advanced Sweep Strategies
-------------------------

Beyond the basic grid search, you can implement more advanced hyperparameter search strategies:

Random Search
~~~~~~~~~~~~~

Instead of trying all combinations, you can randomly sample from the parameter space. Modify ``generate_sweep_configs.py`` to sample random combinations:

.. code-block:: python

    import random

    def generate_random_configs(base_config, sweep_params, num_configs=10):
        """Generate random parameter combinations."""
        configs = []
        for i in range(num_configs):
            new_config = deepcopy(base_config)
            param_details = {}
            for key, values in sweep_params.items():
                value = random.choice(values)
                keys = key.split(".")
                sub_config = new_config
                for sub_key in keys[:-1]:
                    sub_config = sub_config[sub_key]
                sub_config[keys[-1]] = value
                param_details[key] = value
            configs.append((new_config, param_details))
        return configs

This approach can be more efficient when exploring large parameter spaces.

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

For even more efficient hyperparameter search, you can use Bayesian optimization. This requires additional libraries like ``scikit-optimize`` or ``ax-platform``:

.. code-block:: python

    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical

    # Define parameter space
    space = [
        Real(0.0001, 0.01, name="lr", prior="log-uniform"),
        Integer(16, 128, name="batch_size"),
        Categorical([[1,2], [1,3], [1,4]], name="branch_factors")
    ]

    # Define objective function (requires running an experiment)
    def objective(params):
        lr, batch_size, branch_factors = params
        # Generate config with these parameters
        # Run experiment
        # Return -validation_accuracy (minimize negative accuracy)
        return -validation_accuracy

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    # Best parameters
    best_params = result.x
    print(f"Best parameters: lr={best_params[0]}, batch_size={best_params[1]}, branch_factors={best_params[2]}")

This approach requires more integration effort but can find good hyperparameters with fewer trials.
