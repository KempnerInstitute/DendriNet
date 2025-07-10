Training Pipeline
=================

High-level helper :pyfunc:`dendritic_modeling.training.utils.train_model` wraps the full training loop: dataloaders → forward pass → loss → back-prop → logging.

Basic Usage
-----------

.. code-block:: python

   from dendritic_modeling.training.utils import train_model
   train_model(model, train_ds, val_ds,
               epochs=100,
               optimizer_cfg={
                   "type": "AdamW",
                   "lr": 2e-4,
                   "weight_decay": 1e-2,
               },
               scheduler_cfg={
                   "type": "CosineAnnealingLR",
                   "T_max": 100,
               })

Branch-wise vs. Joint Training
------------------------------
Pass ``branch_wise=True`` to train dendrite branches individually for the first *N* epochs (``branch_wise_epochs``).  This can stabilise early learning.

Gradient Scaling
----------------
Enable by setting ``gradient_scaling: true`` in network layers or YAML.  The trainer automatically computes per-layer scaling factors and applies them every backward pass via :pyfunc:`dendritic_modeling.training.grad_utils.apply_gradient_scaling`.

Experiment Tracking
-------------------
If ``wandb`` is installed and the environment variable ``WANDB_API_KEY`` is set, the trainer logs metrics, gradients, and parameter histograms automatically.

Checkpoints
~~~~~~~~~~~
Weights are saved every ``save_every`` epochs to ``results/{run_name}/checkpoints``.  Use :pyfunc:`dendritic_modeling.training.utils.load_checkpoint` to resume.

Mixed Precision
~~~~~~~~~~~~~~~
Set ``amp=True`` to enable PyTorch Automatic Mixed Precision for faster training on GPUs.

Hyper-parameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~
The CLI command ``scripts/sweeps/submit_sweep.py`` submits SLURM jobs reading a sweep YAML describing ranges for *K*, E/I ratio, etc. 