Quickstart
==========

DendriNet models replace the point neurons of a neural network with dendritic
neurons: each neuron is a tree of branch compartments that receive sparse
excitatory (E) and inhibitory (I) synapses. A compartment combines its
excitation and inhibition by shunting (divisive) or additive integration,
passes the result through a branch nonlinearity (a reactivation gate), and
feeds its parent compartment, up to the soma. A population network arranges
these neurons in named E and I populations connected by explicit, directed
connections, in feedforward or recurrent layers.

This page trains a small population network on MNIST, reads the files the run
writes, analyzes the trained checkpoint and generates a parameter sweep. Run
the commands from the root of the repository checkout, after installing the
package as described in :doc:`installation`.

Train a population network on MNIST
-----------------------------------

.. code-block:: bash

   dendritic-train configs/examples/population_network_feedforward.yaml --epochs 1

``configs/examples/population_network_feedforward.yaml`` defines one
feedforward layer with an inhibitory population ``i`` (32 cells) and an
excitatory population ``e`` (128 cells with ``[4, 4]`` branch factors). The
input drives both populations, ``i`` inhibits ``e``, and ``e`` feeds a small
MLP decoder. The model part of the file, without its synapse and gate
options, is:

.. code-block:: yaml

   model:
     task: classification
     encoder:
       type: identity
     core:
       type: population_network
       population_network:
         input_transform: nonnegative
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 30
               ff_inhibitory_synapses: 15
             populations:
               - {name: i, polarity: inhibitory, n_neurons: 32, branch_factors: [1]}
               - {name: e, polarity: excitatory, n_neurons: 128, branch_factors: [4, 4]}
             connections:
               - {source: input, target: i}
               - {source: input, target: e}
               - {source: i, target: e}
     decoder:
       type: MLP
       params:
         hidden_dims: [32]
         output_dim: 10

:doc:`population_networks` explains every field. The file allows up to 50
epochs, with early stopping; ``--epochs 1`` shortens the run for an initial
check. Without a GPU the run trains on the CPU.

The first run downloads MNIST into ``<data root>/datasets/mnist/``. The data
root is ``data.base_dir`` when it is set, else the ``DENDRITIC_DATA_DIR``
environment variable, else ``$XDG_CACHE_HOME/dendritic_modeling`` (by
default ``~/.cache/dendritic_modeling``); ``--data_dir <dir>`` sets
``data.base_dir`` for one run. See :doc:`datasets`.

Other command-line options change the run without editing the file:
``--seed``, ``--epochs``, ``--output_dir``, ``--run_name``, ``--data_dir``,
the learning rates (``--lr``, ``--topk_lr``, ``--blocklinear_lr``,
``--reactivation_lr``, ``--decoder_lr``), ``--learning_strategy`` and
``--error_broadcast_mode``. Every other option is set in the YAML file or
varied with a sweep. :ref:`cli-reference` lists the options of every
command.

To check a configuration without training, add ``--validate-only``: the
command loads and validates the file, builds the datasets, the model, the
calibrated gates, the optimizer and the trainer, writes the construction
records and stops.

Read the run directory
----------------------

The run writes into ``outputs/population_network_feedforward_<YYYYMMDD_HHMMSS>/``
(``outputs.results_dir`` and ``outputs.run_name`` of the example). The
one-epoch CPU run produced:

.. code-block:: text

   outputs/population_network_feedforward_<YYYYMMDD_HHMMSS>/
   |-- config.json                       # resolved configuration
   |-- dendritic_modeling.log            # complete log of the run
   |-- final_model.pt                    # final weights
   |-- init_gate_stats.json              # gate parameters after building
   |-- main_network/
   |   `-- standard_best_model.pt        # best-validation weights
   |-- model_resources.json              # parameter and synapse counts
   |-- nparams                           # total parameter count (JSON)
   |-- operator_card.json                # integration rule, gates, morphology
   |-- performance/
   |   `-- final.json                    # accuracy, AUC, log-likelihood
   |-- post_calibration_gate_stats.json  # gate parameters after calibration
   |-- reactivation_calibration.json     # data-driven gate calibration
   |-- resolved_seeds.json               # every random seed and numerics flag
   |-- runtime_environment.json          # library commit, dependency versions
   |-- train.log                         # run-setup log
   `-- training_summary.json             # losses per epoch, best epoch

``performance/final.json`` holds the metrics that ``analysis.performance``
enables, on the training, validation and test splits:

.. code-block:: text

   {
       "accuracy": {"train": 0.830, "valid": 0.831, "test": 0.838},
       "auc": {"train": 0.979, "valid": 0.978, "test": 0.979},
       "categorical_loglikelihood": {"train": -0.643, "valid": -0.654, "test": -0.636}
   }

These values come from the one-epoch CPU run (rounded); yours will differ.
:ref:`run-outputs` describes every file, the checkpoint names and how to
choose the run directory.

Analyze the checkpoint
----------------------

``dendritic-analyze-checkpoint`` rebuilds the model and the dataset splits
from a run's ``config.json``, loads a checkpoint and runs the analyses that
an analysis profile enables, without training. Save this profile as
``weight_analysis.yaml``:

.. code-block:: yaml

   analysis:
     weight:
       enabled: true

and run it on the trained weights:

.. code-block:: bash

   # Replace the timestamp below with the directory created by your run.
   RUN="outputs/population_network_feedforward_<YYYYMMDD_HHMMSS>"
   dendritic-analyze-checkpoint \
     --config "$RUN/config.json" \
     --checkpoint "$RUN/final_model.pt" \
     --analysis-config weight_analysis.yaml \
     --output-dir outputs/weight_analysis

The output directory receives ``weight_analysis/`` (synaptic weight
statistics and figures), ``checkpoint_analysis_provenance.json`` (digests of
the config, checkpoint and profile, and the seeds and dataset sizes used), the
seed and environment records and the logs. :ref:`analyzer-catalogue` lists
every analyzer, and :ref:`checkpoint-analysis` describes the command.

Generate a sweep
----------------

``configs/sweep_config.yaml`` varies the synapse counts and the excitatory
branch factors of a population network over five seeds per combination.
Generate its configurations and job scripts without submitting them:

.. code-block:: bash

   dendritic-sweep --config configs/sweep_config.yaml --generate-only

This writes 40 configurations to ``outputs/sweep_run_<YYYYMMDDHHMMSS>/configs/``,
the Slurm job scripts to ``jobs/``, a ``submit_all_jobs.sh`` script and a
``frozen_sweep_manifest.json`` that records the source checkout. Without a
Slurm account and partition the job scripts hold placeholders; generate them
again with ``--slurm-account <account> --slurm-partition <partition>``, or run
``dendritic-sweep --config configs/sweep_config.yaml`` with those options to
generate and submit in one step. :doc:`sweeps` describes the sweep file and
the result collection.

Train on several GPUs
---------------------

``torchrun`` launches distributed training. Distributed data parallel (DDP)
runs the training module in each process; fully sharded data parallel (FSDP)
uses its own entry point and a config with ``distributed.mode: fsdp``. Write
``distributed.mode: ddp`` in a config meant for DDP: a config with
``distributed.mode: none``, such as ``configs/config_exp.yaml``, also trains
with DDP under ``torchrun``, and the run warns and records ``ddp`` in its
``config.json``.

.. code-block:: bash

   # DDP on 4 GPUs
   torchrun --nproc_per_node=4 \
     -m dendritic_modeling.scripts.training.train_experiments \
     configs/config_exp.yaml

   # FSDP on 4 GPUs
   torchrun --nproc_per_node=4 \
     -m dendritic_modeling.scripts.training.train_experiments_fsdp \
     configs/config_exp_fsdp.yaml

See :ref:`distributed-training`.

Use DendriNet from Python
-------------------------

The model builder that ``dendritic-train`` uses is available directly:

.. code-block:: python

   import torch

   from dendritic_modeling.config import load_config
   from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

   config = load_config("configs/examples/population_network_feedforward.yaml")
   config.model.encoder.params.input_dim = 784  # dendritic-train sets it from the data
   model, encoder = initialize_model(config.model)
   logits = model(torch.rand(8, 784))  # shape (8, 10)

``initialize_model`` builds the model only. ``dendritic-train`` also
calibrates data-driven gates on training data, trains, and writes the run
directory.

Next steps
----------

* :doc:`examples`: every shipped configuration and what it shows.
* :doc:`network_architectures`: the encoder, core and decoder of a model and
  the core types.
* :doc:`training_strategies`: backpropagation, feedback alignment and local
  credit-assignment rules.
* :doc:`analysis`: the analyzers and their outputs.
* :doc:`configuration_guide`: how configuration files are loaded and
  checked.
