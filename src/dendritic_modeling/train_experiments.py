"""
train_experiments.py
--------------------
Main script that reads config_exp.yaml, builds the model,
chooses a training strategy, trains, and evaluates using
a unified data-loading function from synthetic_datasets.py.
"""

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict

import wandb
import torch
import torchinfo

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.learning_strategies import get_trainer
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, FeedforwardNN)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager

from synthetic_datasets import get_unified_datasets


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters.__dict__
    enable_branch_outputs = net_params.pop('enable_branch_outputs', False)

    if task == 'classification':
        if net_type == 'feedforward':
            logger.info(f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = FeedforwardNN(**net_params)
            output_dim = net_params.get('output_dim', 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)

        elif net_type == 'EINet':
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(
                **net_params,
                enable_branch_outputs=enable_branch_outputs
            )
            output_dim = net_params['excitatory_layer_sizes'][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)

        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Invalid task: {task}")


def main(config_path: str, output_dir: str, experiment_name: str):
    os.makedirs(output_dir, exist_ok=True)
    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))

    # Load config
    config = load_config(config_path)
    model_cfg = config.model
    train_cfg = config.train
    wandb_cfg = config.wandb

    # Initialize W&B if desired
    if getattr(wandb_cfg, "use_wandb", False):
        try:
            wandb_dir = Path(output_dir) / "wandb"
            wandb_run_name = experiment_name
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_dict = asdict(config)  # Convert config to dict for logging
            wandb.init(
                dir=str(wandb_dir),
                entity=wandb_cfg.entity,
                project=wandb_cfg.project,
                group=wandb_cfg.group,
                name=wandb_run_name,
                tags=wandb_cfg.tags,
                config=wandb_dict,
            )
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            raise e

    logger.info("Loading dataset...")

    train_ds, valid_ds, test_ds = get_unified_datasets(config.task, train_cfg)

    # Build the model
    clf = initialize_model(model_cfg)

    # Optional: log a summary. We'll guess input dim from the first training sample
    try:
        sample_input, _ = train_ds[0]  # a single (features, label)
        sample_input_dim = sample_input.numel()  # flatten if needed
        summary = torchinfo.summary(
            clf,
            input_size=(train_cfg.batch_size, sample_input_dim),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7
        )
        logger.debug("\n" + str(summary))
        if wandb.run:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except:
        pass

    # ================== CHOOSE & RUN TRAINING STRATEGY =================
    learning_strategy = getattr(train_cfg, "learning_strategy", "mle")
    logger.info(f"Using learning strategy: {learning_strategy}")

    # Create optimizer
    optimizer = torch.optim.Adam(clf.parameters(), lr=train_cfg.lr)

    # Acquire the trainer from learning_strategies.py
    trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)

    # If "freeze_layers"
    if learning_strategy == "freeze_layers":
        # Use epochs_per_layer from config
        epochs_per_layer = getattr(train_cfg, "epochs_per_layer", 5)
        trainer.train_freeze_layers(
            model=clf,
            train_data=train_ds,
            valid_data=valid_ds,
            epochs_per_layer=epochs_per_layer,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            grad_clip_value=train_cfg.grad_clip_value,
            plot_losses=True,
            save_path=output_dir
        )

    elif learning_strategy == "two_step":
        # Use pretrain & maintrain from config
        pretrain_epochs = getattr(train_cfg, "pretrain_epochs", 5)
        maintrain_epochs = getattr(train_cfg, "maintrain_epochs", 45)
        trainer.train(
            model=clf,
            train_data=train_ds,
            valid_data=valid_ds,
            pretrain_epochs=pretrain_epochs,
            maintrain_epochs=maintrain_epochs,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            load_best_state_dict=True,
            plot_losses=True,
            save_path=output_dir
        )

    else:
        # "mle", "local_credit_assignment", "train_only_b_react"
        trainer.train(
            model=clf,
            train_data=train_ds,
            valid_data=valid_ds,
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            load_best_state_dict=True,
            plot_losses=True,
            save_path=output_dir
        )

    # ================== Evaluate final model ==================
    logger.info("Evaluating performance...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)

    def evaluate_accuracy(model, dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=False)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in loader:
                preds = model.predict(xb.to(device))
                all_preds.append(preds.cpu())
                all_labels.append(yb)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return (all_preds == all_labels).float().mean().item()

    train_acc = evaluate_accuracy(clf, train_ds)
    valid_acc = evaluate_accuracy(clf, valid_ds) if valid_ds else None
    test_acc  = evaluate_accuracy(clf, test_ds)
    logger.info(f"train acc={train_acc:.3f}, valid acc={valid_acc:.3f}, test acc={test_acc:.3f}")

    # Log final accuracies to W&B
    if wandb.run is not None:
        wandb.log({"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc})
        wandb.finish()

    # ================== Visualization ==================
    if config.visualization.enabled:
        plot_manager = PlotManager(clf, {"train": train_ds, "valid": valid_ds, "test": test_ds}, config, device=device)
        for plot_name in config.visualization.plots:
            plot_manager.run_plot(plot_name)

    logger.info("All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    parser.add_argument("--experiment_name", default="example", help="Experiment name (used for W&B run name).")
    args = parser.parse_args()

    main(config_path=args.config, output_dir=args.output_dir, experiment_name=args.experiment_name)