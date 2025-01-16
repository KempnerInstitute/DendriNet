#!/usr/bin/env python

import os
import argparse
from pathlib import Path
import time

import wandb
import torch
import torchinfo
from torch.utils.data import DataLoader

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.utils import accuracy_score, save_dict
from dendritic_modeling.config import load_config
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.networks import ExcitationInhibitionNetwork, FeedforwardNN

def initialize_model(model_hp):
    task = model_hp['task']
    probabilistic = model_hp['probabilistic']
    net_type = model_hp['network']['type'].lower()
    net_params = model_hp['network']['parameters']

    if task=='classification':
        if net_type=='feedforward':
            net = FeedforwardNN(**net_params)
            output_dim = net_params.get('output_dim', 10)
        elif net_type=='einet':
            net = ExcitationInhibitionNetwork(**net_params)
            output_dim = net_params['excitatory_layer_sizes'][-1]
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        if probabilistic:
            return ProbabilisticClassifier(net, output_dim)
        else:
            return Classifier(net)
    else:
        raise ValueError(f"Unsupported task: {task}")

def main(config_path, output_dir, experiment_name):
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger_manager.set_log_file(os.path.join(output_dir, "log.txt"))

    config = load_config(config_path)
    model_hp = config.model.asdict()
    train_hp = config.train.asdict()
    task_hp  = config.task.asdict()

    wandb_cfg = config.wandb

    logger.debug("Loaded config:")
    logger.debug(model_hp)
    logger.debug(train_hp)
    logger.debug(task_hp)

    # Possibly init W&B
    if wandb_cfg.entity != "":
        try:
            wandb_dir = Path(output_dir)/"wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_run_name = experiment_name
            wandb.init(
                dir=str(wandb_dir),
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                group=wandb_cfg.group,
                name=wandb_run_name,
                tags=wandb_cfg.tags,
                config=config.asdict(exclude=["wandb"])
            )
        except Exception as e:
            logger.error(f"Error initializing wandb: {e}")

    # Load dataset
    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=task_hp,
        train_cfg=train_hp
    )
    logger.info(f"Loaded dataset: {task_hp['dataset']}")

    # Peek
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=train_hp['batch_size'], shuffle=False)
    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[1]
    num_classes = len(torch.unique(sample_y))

    # Init model
    clf = initialize_model(model_hp)

    # Summaries
    try:
        summary = torchinfo.summary(
            clf,
            input_size=(train_hp['batch_size'], input_dim),
            col_names=["input_size","output_size","num_params"],
            depth=7
        )
        logger.debug(summary)
        if wandb.run is not None:
            wandb.log({"total_params": summary.total_params,
                       "trainable_params": summary.trainable_params})
    except:
        pass

    # Trainer
    optimizer = torch.optim.Adam(
        clf.parameters(),
        lr=train_hp.get('lr',1e-3),
        weight_decay=train_hp.get('weight_decay',0.0)
    )
    trainer = TrainerMLE(optimizer, suppress_prints=False, print_every=5)

    # Train
    start_time = time.time()
    trainer.train(
        model=clf,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_hp['epochs'],
        batch_size=train_hp['batch_size'],
        grad_clip_value=train_hp.get('grad_clip_value',5),
        shuffle=True,
        plot_losses=True,
        save_path=output_dir
    )
    elapsed = time.time()-start_time
    logger.info(f"Training took {elapsed/60:.2f} minutes.")

    # Evaluate
    eval_train_loader = DataLoader(train_ds, batch_size=train_hp['batch_size'], shuffle=False)
    eval_valid_loader = DataLoader(valid_ds, batch_size=train_hp['batch_size'], shuffle=False)
    eval_test_loader  = DataLoader(test_ds,  batch_size=train_hp['batch_size'], shuffle=False)

    train_acc = accuracy_score(clf, *next(iter(eval_train_loader))).item()
    valid_acc = accuracy_score(clf, *next(iter(eval_valid_loader))).item()
    test_acc  = accuracy_score(clf, *next(iter(eval_test_loader))).item()

    performance = {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc" : test_acc,
        "training_time_s": f"{elapsed:.3f}"
    }
    logger.info(f"Performance: {performance}")

    if wandb.run is not None:
        wandb.log(performance)
        wandb.finish()

    # Done
    logger.info("Done with train & eval. Exiting.")


if __name__=="__main__":
    import sys
    parser=argparse.ArgumentParser()
    parser.add_argument("config",help="Path to config.yaml")
    parser.add_argument("--output_dir",default="results",help="Output dir")
    parser.add_argument("--experiment_name",default="example",help="Name for W&B or logs")
    args=parser.parse_args()
    main(args.config,args.output_dir,args.experiment_name)