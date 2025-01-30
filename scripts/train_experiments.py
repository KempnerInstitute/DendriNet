"""
train_experiments.py
====================
Your main experiment entry script that:

1) Parses arguments for config path, output_dir, experiment_name
2) Loads that config via load_config
3) Builds model & trainer
4) Runs the training, logs results
"""

import os
import json
import time
import argparse
from pathlib import Path

import wandb
import torch
import torchinfo
from omegaconf import OmegaConf

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.learning_strategies import TrainerMLE
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, FeedforwardNN)
from dendritic_modeling.config import load_config
from dendritic_modeling.utils import load_MNIST, accuracy_score
from dendritic_modeling.visualization_manager import PlotManager

def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.

    model_cfg is a ModelConfig object with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters.__dict__  # or vars(...)

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

    # 1) Load config
    config = load_config(config_path)

    model_cfg = config.model
    train_cfg = config.train
    wandb_cfg = config.wandb

    # 2) Possibly init W&B
    if wandb_cfg and wandb_cfg.entity:
        try:
            wandb_dir = Path(output_dir) / "wandb"
            wandb_run_name = experiment_name
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(
                dir=wandb_dir,
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                group=wandb_cfg.group,
                name=wandb_run_name,
                tags=wandb_cfg.tags,
                config=OmegaConf.to_container(config, resolve=True),
            )
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B ... ")
            # no wandb

    # 3) Load dataset
    logger.info("Loading dataset...")
    data = load_MNIST(train_valid_split=train_cfg.train_valid_split)

    # 4) Build model
    clf = initialize_model(model_cfg)

    # 5) Optionally show summary
    try:
        sample_input_dim = data['train'].x.shape[1]
        summary = torchinfo.summary(
            clf,
            input_size=(train_cfg.batch_size, sample_input_dim),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7
        )
        logger.debug("\n" + str(summary))
        if wandb.run is not None:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except:
        pass

    # 6) Build optimizer
    optimizer = torch.optim.Adam(clf.parameters(), lr=train_cfg.lr)
    trainer = TrainerMLE(optimizer=optimizer, suppress_prints=False, print_every=10)

    # 7) Check if we want freeze_layer or normal MLE
    grad_strat = getattr(model_cfg.network.parameters, "gradient_strategy", "none")
    if grad_strat == "freeze_layer":
        trainer.train_freeze_layers(
            model=clf,
            train_data=data['train'],
            valid_data=data.get('valid', None),
            epochs_per_layer=train_cfg.epochs_per_layer,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            grad_clip_value=train_cfg.grad_clip_value,
            plot_losses=True,
            save_path=output_dir
        )
    else:
        trainer.train(
            model=clf,
            train_data=data['train'],
            valid_data=data.get('valid', None),
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            plot_losses=True,
            save_path=output_dir
        )

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

    train_acc = evaluate_accuracy(clf, data['train'])
    valid_acc = evaluate_accuracy(clf, data['valid']) if 'valid' in data else None
    test_acc  = evaluate_accuracy(clf, data['test'])
    logger.info(f"train acc={train_acc}, valid acc={valid_acc}, test acc={test_acc}")

    if wandb.run is not None:
        wandb.log({"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc})
        wandb.finish()

    # 8) Possibly do visualization
    if config.visualization.enabled:
        plot_manager = PlotManager(clf, data, config, device=device)
        for plot_name in config.visualization.plots:
            plot_manager.run_plot(plot_name)

    logger.info("All done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Optional output directory.")
    parser.add_argument("--experiment_name", default="example", help="Optional experiment name.")
    args = parser.parse_args()

    main(config_path=args.config, output_dir=args.output_dir, experiment_name=args.experiment_name)