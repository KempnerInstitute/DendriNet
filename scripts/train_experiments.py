######## train_experiments.py ########


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
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, FeedforwardNN)
from dendritic_modeling.config import load_config
from dendritic_modeling.utils import load_MNIST, accuracy_score
from dendritic_modeling.visualization_manager import PlotManager

def initialize_model(model_hp):
    """
    Builds the specified model: 
    'classification' uses ProbabilisticClassifier or Classifier.
    """
    task = model_hp['task']
    probabilistic = model_hp['probabilistic']
    net_type = model_hp['network']['type']
    net_params = model_hp['network']['parameters']

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

    config = load_config(config_path)

    model_hp = config.model.asdict()
    train_hp = config.train.asdict()

    vis_cfg = config.visualization
    wandb_cfg = config.wandb

    if wandb_cfg is not None and wandb_cfg.entity:
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
                config=config.asdict(exclude=["wandb"]),
            )
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B ... ")
            wandb_cfg = None

    logger.info("Loading dataset...")
    data = load_MNIST(train_valid_split=train_hp.get("train_valid_split", 1.0), cache_dir=None)
    
    clf = initialize_model(model_hp)

    try:
        sample_input_dim = data['train'].x.shape[1]
        summary = torchinfo.summary(
            clf,
            input_size=(train_hp.get("batch_size", 64), sample_input_dim),
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

    optimizer = torch.optim.Adam(clf.parameters(), lr=train_hp.get("lr", 1e-3))
    trainer = TrainerMLE(optimizer=optimizer, suppress_prints=False, print_every=10)

    grad_strat = model_hp["network"]["parameters"].get("gradient_strategy", "none")
    if grad_strat == "freeze_layer":
        trainer.train_freeze_layers(
            model=clf,
            train_data=data['train'],
            valid_data=data.get('valid', None),
            epochs_per_layer=train_hp.get("epochs_per_layer", 5),
            batch_size=train_hp.get("batch_size", 64),
            shuffle=True,
            grad_clip_value=train_hp.get("grad_clip_value", 5),
            plot_losses=True,
            save_path=output_dir
        )
    else:
        trainer.train(
            model=clf,
            train_data=data['train'],
            valid_data=data.get('valid', None),
            epochs=train_hp.get("epochs", 50),
            batch_size=train_hp.get("batch_size", 64),
            grad_clip_value=train_hp.get("grad_clip_value", 5),
            shuffle=True,
            plot_losses=True,
            save_path=output_dir
        )

    logger.info("Evaluating performance...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)

    def evaluate_accuracy(model, dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=train_hp.get("batch_size", 64), shuffle=False)
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

    if vis_cfg.enabled:
        plot_manager = PlotManager(clf, data, config, device=device)
        for plot_name in vis_cfg.plots:
            print(plot_name)
            plot_manager.run_plot(plot_name)

    logger.info("All done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Optional output directory.")
    parser.add_argument("--experiment_name", default="example", help="Optional experiment name.")
    args = parser.parse_args()
    main(config_path=args.config, output_dir=args.output_dir, experiment_name=args.experiment_name)