#!/usr/bin/env python

import os
import time
import argparse
from pathlib import Path

import wandb
import torch
import torchinfo
from omegaconf import OmegaConf

# Reuse your logging & training code
from dendritic_modeling import logger, logger_manager
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import EINetClassifierNLL
from dendritic_modeling.utils import accuracy_score, save_dict

# Our new unified dataset loader
from dendritic_modeling.synthetic_datasets import get_unified_dataloaders

# A simple MLP baseline (for comparison)
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    A simple feedforward MLP with one hidden layer, for baseline comparison.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits


def main(config_path: str, output_dir: str, experiment_name: str):

    # 1. Setup output dir & logging
    os.makedirs(output_dir, exist_ok=True)
    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))

    # 2. Load config
    config = OmegaConf.load(config_path)
    model_cfg = config.model
    train_cfg = config.train
    wandb_cfg = config.get("wandb", None)

    logger.debug("="*50)
    logger.debug(f" Experiment Name: {experiment_name}")
    logger.debug(" Model Hyperparameters:\n" + str(model_cfg))
    logger.debug(" Training Hyperparameters:\n" + str(train_cfg))
    logger.debug(f" Output Directory: {output_dir}")
    logger.debug("="*50)

    # 3. Possibly init W&B
    if wandb_cfg:
        try:
            wandb_dir = Path(output_dir) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_run_name = experiment_name
            wandb.init(
                dir=wandb_dir,
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                group=wandb_cfg.group,
                name=wandb_run_name,
                tags=wandb_cfg.tags,
                config=OmegaConf.to_container(config, resolve=True),
            )
            logger.debug("W&B initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            wandb_cfg = None

    # 4. Load dataset
    dataset_name = train_cfg.get("dataset_name", "mnist")
    train_loader, valid_loader, test_loader = get_unified_dataloaders(
        dataset_name=dataset_name,
        train_size=train_cfg.get("train_size", 8000),
        valid_size=train_cfg.get("valid_size", 1000),
        test_size=train_cfg.get("test_size", 1000),
        image_size=train_cfg.get("image_size", 10),
        n_orient_classes=train_cfg.get("n_orient_classes", 8),
        n_bits=train_cfg.get("n_bits", 8),
        batch_size=train_cfg.get("batch_size", 64),
        mnist_train_valid_split=train_cfg.get("train_valid_split", 0.9),
    )
    logger.info(f"Loaded dataset: {dataset_name}")

    # Peek at one batch to find input_dim & num_classes
    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[1]  # e.g. 784 for MNIST, or 100 for line
    num_classes = len(torch.unique(sample_y))

    # 5. Build model: either MLP or EINet
    model_type = model_cfg.get("type", "EINet")  # "MLP" or "EINet"
    if model_type == "MLP":
        hidden_dim = model_cfg.get("hidden_dim", 50)
        clf = MLPBaseline(input_dim, hidden_dim, num_classes)
    else:
        # build EINet
        clf = EINetClassifierNLL(
            input_dim=input_dim,
            excitatory_layer_sizes=model_cfg.get("excitatory_layer_sizes", [10]),
            inhibitory_layer_sizes=model_cfg.get("inhibitory_layer_sizes", [20]),
            excitatory_branch_factors=model_cfg.get("excitatory_branch_factors", [1,2]),
            inhibitory_branch_factors=model_cfg.get("inhibitory_branch_factors", []),
            ee_synapses_per_branch_per_layer=model_cfg.get("ee_synapses_per_branch_per_layer", [24]),
            ei_synapses_per_branch_per_layer=model_cfg.get("ei_synapses_per_branch_per_layer", [200]),
            ie_synapses_per_branch_per_layer=model_cfg.get("ie_synapses_per_branch_per_layer", [1]),
            ii_synapses_per_branch_per_layer=model_cfg.get("ii_synapses_per_branch_per_layer", []),
            reactivate=model_cfg.get("reactivate", True),
            somatic_synapses=model_cfg.get("somatic_synapses", True),
            topk_init_method=model_cfg.get("topk_init_method", "xavier_normal"),
        )

    # 6. Optional: Model summary with torchinfo
    try:
        import torchinfo
        summary = torchinfo.summary(
            clf,
            input_size=(train_cfg.get("batch_size", 64), input_dim),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7
        )
        logger.debug("\n" + str(summary))
        if wandb_cfg:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except:
        pass

    # 7. Trainer
    optimizer = torch.optim.Adam(clf.parameters(), lr=train_cfg.get("lr",1e-3))
    trainer = TrainerMLE(
        optimizer=optimizer,
        suppress_prints=False,
        print_every=10,
    )

    # 8. Train
    epochs = train_cfg.get("epochs", 10)
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train(
        model=clf,
        train_data=train_loader,
        valid_data=valid_loader,
        epochs=epochs,
        batch_size=train_cfg.get("batch_size", 64),
        grad_clip_value=train_cfg.get("grad_clip_value", None),
        shuffle=True,
        plot_losses=True,
        save_path=output_dir,
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} min.")

    # 9. Evaluate
    logger.info("Evaluating performance...")
    t0 = time.time()
    train_acc = accuracy_score(clf, *next(iter(train_loader))).item()
    valid_acc = accuracy_score(clf, *next(iter(valid_loader))).item()
    test_acc  = accuracy_score(clf, *next(iter(test_loader))).item()
    inference_time = time.time() - t0

    performance = {
        "train accuracy": train_acc,
        "valid accuracy": valid_acc,
        "test accuracy":  test_acc,
        "training time": f"{training_time:.3f} sec",
        "inference time": f"{inference_time:.3f} sec"
    }
    logger.info(f"Performance: {performance}")

    # 10. Save info
    save_dict(model_cfg, output_dir, "model_hparams.json")
    save_dict(train_cfg, output_dir, "training_hparams.json")
    save_dict(performance, output_dir, "performance.json")

    if wandb_cfg:
        wandb.log(performance)
        wandb.finish()

    logger.info("----------- Done! -----------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--experiment_name", default="example", help="Experiment name")
    args = parser.parse_args()

    main(args.config, args.output_dir, args.experiment_name)