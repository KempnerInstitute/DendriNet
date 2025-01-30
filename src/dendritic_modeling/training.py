"""
training.py
===========
A sample script that:
1) Loads a config from config.py (which references a .yaml file),
2) Builds a model,
3) Creates a trainer from learning_strategies.py,
4) Runs training according to the chosen method.
"""

import os
import wandb
import torch
from pathlib import Path
import argparse

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.utils import load_MNIST
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, FeedforwardNN
from dendritic_modeling.learning_strategies import get_trainer


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is a dataclass from config.py (ModelConfig).
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = vars(model_cfg.network.parameters)

    enable_branch_outputs = net_params.pop('enable_branch_outputs', False)

    if task == 'classification':
        if net_type.lower() == 'einet':
            logger.info(f"Building EINet with {'probabilistic' if probabilistic else 'basic'} classifier.")
            net = ExcitationInhibitionNetwork(**net_params, enable_branch_outputs=enable_branch_outputs)
            output_dim = net_params['excitatory_layer_sizes'][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        elif net_type.lower() == 'feedforward':
            logger.info(f"Building Feedforward with {'probabilistic' if probabilistic else 'basic'} classifier.")
            net = FeedforwardNN(**net_params)
            output_dim = net_params.get('output_dim', 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Unsupported task: {task}")

def main(config_path):
    # 1) Setup logging, load config
    output_dir = "results"
    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))
    config = load_config(config_path)

    # 2) Possibly init wandb
    if config.wandb.use_wandb:
        wandb_dir = Path(output_dir) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            tags=config.wandb.tags,
            name=config.train.run_name,
            config=config
        )

    # 3) Load data
    logger.info("Loading dataset...")
    data = load_MNIST(train_valid_split=config.train.train_valid_split)
    train_data = data['train']
    valid_data = data.get('valid', None)
    test_data = data['test']

    # 4) Build model
    model = initialize_model(config.model)

    # 5) Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # 6) Retrieve trainer
    trainer = get_trainer(
        strategy=config.train.learning_strategy, 
        optimizer=optimizer, 
        suppress_prints=False, 
        print_every=10
    )

    # 7) Decide how to run training
    if config.train.learning_strategy == "freeze_layers":
        results = trainer.train_freeze_layers(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            epochs_per_layer=config.train.epochs_per_layer,
            batch_size=config.train.batch_size,
            shuffle=True,
            grad_clip_value=config.train.grad_clip_value,
            plot_losses=True,
            save_path=config.train.save_path
        )
    elif config.train.learning_strategy == "two_step":
        # If we used TwoStepTrainer, we do:
        # (We must cast the trainer to TwoStepTrainer, or rely on the same interface)
        results = trainer.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            pretrain_epochs=5,  # from config if needed
            maintrain_epochs=45,
            batch_size=config.train.batch_size,
            grad_clip_value=config.train.grad_clip_value
        )
    else:
        # default MLE
        results = trainer.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            grad_clip_value=config.train.grad_clip_value,
            shuffle=True,
            plot_losses=True,
            save_path=config.train.save_path
        )

    logger.info("Training finished.")
    logger.info(f"Best epoch: {results.get('best epoch', None)}")

    # Evaluate test performance if desired
    def evaluate_accuracy(model, dataset):
        from torch.utils.data import DataLoader
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False)
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

    test_acc = evaluate_accuracy(model, test_data)
    logger.info(f"Test accuracy: {test_acc:.3f}")

    if wandb.run is not None:
        wandb.log({"test_acc": test_acc})
        wandb.finish()

    # 8) Possibly do visualization
    if config.visualization.enabled:
        pm = PlotManager(model, data, config, device='cuda' if torch.cuda.is_available() else 'cpu')
        for plot_name in config.visualization.plots:
            pm.run_plot(plot_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("config", type=str, help="Path to config YAML.")
    args = parser.parse_args()
    main(args.config)