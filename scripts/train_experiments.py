#!/usr/bin/env python

import os
import json
import time
import argparse
from pathlib import Path

import wandb
import torch
import torchinfo
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.plot_utils import (
    plot_einet_params, 
    plot_einet_activations
)
from dendritic_modeling.utils import (
    save_dict,
    accuracy_score, 
    Shaper
)
from dendritic_modeling.config import load_config
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.analysis import (
    analyze_branch_ablation,
    analyze_branch_information,
    record_branch_activations
)
from dendritic_modeling.networks import (
    ExcitationInhibitionNetwork, 
    FeedforwardNN,
    MLPBaseline
)

def initialize_model(model_hp):
    """
    Initialize a model based on the config's 'task' and 'network' type.
    model_hp is the dictionary from config.model.asdict().
    """
    task = model_hp['task']
    probabilistic = model_hp['probabilistic']
    network_type = model_hp['network']['type'].lower()
    network_parameters = model_hp['network']['parameters']

    # Decide which net to build
    if task == 'classification':
        if network_type == 'feedforward':
            logger.info(
                f"Running feedforward network with "
                f"{'probabilistic' if probabilistic else 'simple'} classifier")
            net = FeedforwardNN(**network_parameters)
            output_dim = network_parameters.get('output_dim', 10)
        elif network_type == 'einet':
            logger.info(
                f"Running EINet with "
                f"{'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(**network_parameters)
            # by default, excitatory_layer_sizes[-1] is output dimension
            output_dim = network_parameters['excitatory_layer_sizes'][-1]
        else:
            raise ValueError(f"Invalid network type: {network_type}")

        # wrap in classifier
        if probabilistic:
            clf = ProbabilisticClassifier(net, output_dim)
        else:
            clf = Classifier(net)
    
    else:
        raise ValueError(f"Invalid 'task' in model config: {task}")

    return clf


def main(config_path: str, output_dir: str, experiment_name: str):
    # 1. Make output dir & logging
    save_path = output_dir
    os.makedirs(save_path, exist_ok=True)

    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))
    config = load_config(config_path)

    model_hp = config.model.asdict()   # dictionary from ModelConfig
    train_hp = config.train.asdict()   # dictionary from TrainConfig
    task_hp  = config.task.asdict()    # dictionary from TaskConfig
    wandb_cfg = config.wandb           # raw object, or None
    
    logger.debug("="*50)
    logger.debug(" "*10 + "Config and run hyperparameters" + " "*10)
    logger.debug("="*50)

    logger.debug(f" Experiment Name: {experiment_name}")
    logger.debug(" Model Hyperparameters:\n" + str(model_hp))
    logger.debug(" Training Hyperparameters:\n" + str(train_hp))
    logger.debug(" Task Hyperparameters:\n" + str(task_hp))
    logger.debug(f" Output Directory: {save_path}")
    logger.debug("="*50)

    # 2. Possibly init W&B
    if wandb_cfg.entity != "":  # or if config.wandb is not None
        try:
            wandb_dir = Path(save_path) / "wandb"
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
            logger.debug("W&B initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B ... ")
            # set wandb_cfg = None if needed

    # 3. Load dataset
    logger.info('Loading dataset ...')
    train_dataset, valid_dataset, test_dataset = get_unified_datasets(
        task_cfg=task_hp, 
        train_cfg=train_hp
    )
    logger.info(f"Loaded dataset: {task_hp['dataset']}")

    # Peek at one batch to find input_dim & num_classes
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=train_hp["batch_size"], 
                                   shuffle=False)
    sample_x, sample_y = next(iter(train_data_loader))
    input_dim = sample_x.shape[1]
    num_classes = len(torch.unique(sample_y))

    # 4. Initialize model
    clf = initialize_model(model_hp)

    # 4b. Optional summary
    try:
        summary = torchinfo.summary(
            clf,
            input_size=(train_hp.get("batch_size", 64), input_dim),
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

    # 5. Build optimizer & trainer
    optimizer = torch.optim.Adam(
        clf.parameters(), 
        lr=train_hp.get("lr", 1e-3),
        weight_decay=train_hp.get("weight_decay", 0.0)
    )
    trainer = TrainerMLE(
        optimizer=optimizer,
        suppress_prints=False,
        print_every=10,
    )

    # 6. Training
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train(
        model=clf,
        train_data=train_dataset,
        valid_data=valid_dataset,
        epochs=train_hp["epochs"],
        batch_size=train_hp["batch_size"],
        grad_clip_value=train_hp["grad_clip_value"],
        shuffle=True,
        plot_losses=True,
        save_path=output_dir,
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} min.")

    # 7. Evaluate
    from torch.utils.data import DataLoader
    eval_train_loader = DataLoader(train_dataset, batch_size=train_hp["batch_size"], shuffle=False)
    eval_valid_loader = DataLoader(valid_dataset, batch_size=train_hp["batch_size"], shuffle=False)
    eval_test_loader  = DataLoader(test_dataset,  batch_size=train_hp["batch_size"], shuffle=False)

    t0 = time.time()
    train_acc = accuracy_score(clf, *next(iter(eval_train_loader))).item()
    valid_acc = accuracy_score(clf, *next(iter(eval_valid_loader))).item()
    test_acc  = accuracy_score(clf, *next(iter(eval_test_loader))).item()
    inference_time = time.time() - t0

    performance = {
        "train accuracy": train_acc,
        "valid accuracy": valid_acc,
        "test accuracy":  test_acc,
        "training time": f"{training_time:.3f} sec",
        "inference time": f"{inference_time:.3f} sec"
    }
    logger.info(f"Performance: {performance}")

    if wandb.run is not None:
        wandb.log(performance)
        wandb.finish()

    logger.info("----------- Training & Basic Eval Done! -----------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--experiment_name", default="example", help="Experiment name")
    args = parser.parse_args()

    main(args.config, args.output_dir, args.experiment_name)