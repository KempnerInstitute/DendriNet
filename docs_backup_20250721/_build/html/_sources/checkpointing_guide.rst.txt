Checkpoint Saving Guide
==================================================

This guide explains how to configure checkpoint saving in the dendritic modeling framework for both ``train_experiments.py`` and ``train_large_model_fsdp.py``.

Overview
------------------------------

Both training scripts support periodic checkpoint saving with a consistent directory structure:

`````
results/
└── your_run_name/
    ├── _main_network/          # Standard trainer saves best model here
    ├── checkpoints/            # Periodic checkpoints saved here
    │   ├── checkpoint_epoch_10.pt
    │   ├── checkpoint_epoch_20.pt
    │   └── ...
    └── other_outputs/
`````

Configuration
------------------------------

For train_experiments.py (Standard Trainer)
~~~~~~~~~~~~~~~~~~~~

Add checkpoint configuration to your YAML config under ``train.main.trainer``:

`````yaml
train:
  main:
    trainer:
      # ... other trainer parameters ...
      save_every_n_epochs: 10        # Save checkpoint every N epochs (required to enable checkpointing)
`````

For train_large_model_fsdp.py (FSDP Trainer)
~~~~~~~~~~~~~~~~~~~~

Add checkpoint configuration to your YAML config under ``train.main.trainer``:

`````yaml
train:
  main:
    trainer:
      # ... other trainer parameters ...
      save_every_n_epochs: 10        # Save checkpoint every N epochs (required to enable checkpointing)
`````

Checkpoint Contents
------------------------------

Standard Trainer Checkpoints
~~~~~~~~~~~~~~~~~~~~

Each checkpoint file contains:

`````python
{
    'epoch': int,                    # Current epoch number
    'model_state_dict': dict,        # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state
    'best_loss': float,              # Best validation loss so far
    'best_epoch': int,               # Epoch with best validation loss
    'loss_function': str,            # Name of loss function used
}
`````

FSDP Trainer Checkpoints
~~~~~~~~~~~~~~~~~~~~

Each checkpoint file contains:

`````python
{
    'epoch': int,                    # Current epoch number
    'model_state_dict': dict,        # Model parameters (full state dict)
    'optimizer_state_dict': dict,    # Optimizer state
}
`````

Examples
------------------------------

Example 1: Save checkpoints every 5 epochs
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  main:
    trainer:
      epochs: 100
      batch_size: 128
      loss_function: cat_nll
      save_every_n_epochs: 5         # Save at epochs 5, 10, 15, 20, ...
`````

Example 2: Disable checkpoint saving (save only best model)
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  main:
    trainer:
      epochs: 100
      batch_size: 128
      loss_function: cat_nll
      # Don't specify save_every_n_epochs to disable periodic checkpointing
      # Best model is still saved automatically
`````

Example 3: Save checkpoints every epoch for debugging
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  main:
    trainer:
      epochs: 20
      batch_size: 64
      loss_function: cat_nll
      save_every_n_epochs: 1         # Save every epoch
`````

Loading Checkpoints
------------------------------

To load a checkpoint for resuming training or evaluation:

`````python
import torch

Load checkpoint
==================================================
checkpoint = torch.load('results/my_run/checkpoints/checkpoint_epoch_50.pt')

Extract information
==================================================
epoch = checkpoint['epoch']
model_state = checkpoint['model_state_dict']
optimizer_state = checkpoint['optimizer_state_dict']

Load into model and optimizer
==================================================
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)

print(f"Loaded checkpoint from epoch {epoch}")
`````

Best Practices
------------------------------

#. **Storage Space**: Checkpoints can be large. Consider saving every 10-20 epochs for long training runs.

#. **Critical Epochs**: Save more frequently during critical phases (e.g., every epoch for the first 20 epochs, then every 10 epochs).

#. **Cleanup**: Remove old checkpoints to save disk space, keeping only recent ones and key milestones.

#. **Validation**: Always save checkpoints after validation to ensure they represent stable training states.

Behavior Differences
------------------------------

When save_every_n_epochs is NOT specified:
~~~~~~~~~~~~~~~~~~~~
* **Standard Trainer**: Only saves the best model based on validation loss
* **FSDP Trainer**: Only saves the best model and final model

When save_every_n_epochs IS specified:
~~~~~~~~~~~~~~~~~~~~
* **Both Trainers**: Save periodic checkpoints + best model + final model

File Naming Convention
------------------------------

* **Periodic checkpoints**: ``checkpoint_epoch_{epoch_number}.pt``
* **Best model**: ``best_model.pt`` (standard trainer) or ``best_model.pt`` (FSDP trainer)
* **Final model**: ``final_model.pt`` (FSDP trainer only)

Directory Structure
------------------------------

Both trainers use the same checkpoint directory structure:

`````
{output_dir}/
├── _main_network/              # Standard trainer working directory
│   └── best_model.pt          # Best model (standard trainer)
├── checkpoints/               # Periodic checkpoints
│   ├── checkpoint_epoch_10.pt
│   ├── checkpoint_epoch_20.pt
│   ├── best_model.pt         # Best model (FSDP trainer)
│   └── final_model.pt        # Final model (FSDP trainer)
└── other_experiment_outputs/
````` 