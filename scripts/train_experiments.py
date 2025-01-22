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
from dendritic_modeling.plot_utils import (plot_einet_params, 
                                           plot_einet_activations)
from dendritic_modeling.utils import (load_MNIST, 
                                      split_MNIST_inputs, 
                                      accuracy_score, 
                                      save_dict, 
                                      Shaper)

from dendritic_modeling.config import load_config
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import (analyze_branch_ablation,
                                      analyze_branch_information,
                                      record_branch_activations)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, 
                                         FeedforwardNN,
                                         MLPBaseline)


def initialize_model(model_hp):
    task = model_hp['task']
    probabilistic = model_hp['probabilistic']
    network_type = model_hp['network']['type']
    network_parameters = model_hp['network']['parameters']
    
    if task == 'classification':
        if network_type == 'feedforward':
            logger.info(
                f"Running simple feedforward neural network with "
                f"{'probabilistic' if probabilistic else 'simple'} classifier"
            )
            net = FeedforwardNN(**network_parameters)
            output_dim = network_parameters.get('output_dim', 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim=output_dim)
            else:
                return Classifier(net)

        elif network_type == 'EINet':
            logger.info(
                f"Running EINet with "
                f"{'probabilistic' if probabilistic else 'simple'} classifier"
            )
            net = ExcitationInhibitionNetwork(**network_parameters)
            # final excitatory dimension is classification size
            output_dim = network_parameters['excitatory_layer_sizes'][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        else:
            raise ValueError(f"Invalid network type: {network_type}")

    elif task == 'regression':
        if probabilistic:
            raise NotImplementedError(
                "Probabilistic regression not implemented yet.")
        else:
            raise NotImplementedError(
                "Non-probabilistic regression not yet implemented.")
    else:
        raise ValueError(f"Invalid task: {task}")


def main(config_path: str, output_dir: str, experiment_name: str):

    save_path = output_dir
    os.makedirs(save_path, exist_ok=True)

    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))

    config = load_config(config_path)
    model_hp = config.model.asdict()
    train_hp = config.train.asdict()
    task_hp = config.task.asdict()

    logger.debug("="*50)
    logger.debug(" "*10 + "Config and run hyperparameters" + " "*10)
    logger.debug("="*50)

    logger.debug(f" Experiment Name: {experiment_name}")
    logger.debug(" Model Hyperparameters:\n" + str(model_hp))
    logger.debug(" Training Hyperparameters:\n" + str(train_hp))
    logger.debug(f" Output Directory: {save_path}")

    logger.debug("="*50)

    if config.wandb is not None:
        try:
            wandb_dir = Path(save_path) / "wandb"
            wandb_run_name = experiment_name
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(
                dir=wandb_dir,
                project=config.wandb.project,
                entity=config.wandb.entity,
                group=config.wandb.group,
                name=wandb_run_name,
                tags=config.wandb.tags,
                config=config.asdict(exclude=["wandb"]),
            )
            logger.debug("W&B initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B ... ")
            config.wandb = None

    logger.info('Loading mnist ...')

    dataset_name = task_hp['dataset']
    train_dataset, valid_dataset, test_dataset = get_unified_datasets(
        task_cfg=task_hp,
        train_cfg=train_hp
    )
    logger.info(f"Loaded dataset: {dataset_name}")

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=train_hp["batch_size"], 
                                   shuffle=False)
    sample_x, sample_y = next(iter(train_data_loader))
    input_dim = sample_x.shape[1]  # e.g. 784 for MNIST
    num_classes = len(torch.unique(sample_y))

    clf = initialize_model(model_hp)

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
        if config.wandb:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except:
        pass

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

    einet_params = model_hp["network"].get("parameters", {})
    grad_strat = einet_params.get("gradient_strategy", "none")
    logger.info(f"Gradient strategy in config: {grad_strat}")

    start_time = time.time()

    # If freeze-layers, call trainer.train_freeze_layers(...), else standard trainer.train(...)
    if grad_strat == "freeze_layers":
        trainer.train_freeze_layers(
            model=clf,
            train_data=train_dataset,
            valid_data=valid_dataset,
            epochs_per_layer=5,         # or set from config
            batch_size=train_hp.get("batch_size", 64),
            shuffle=True,
            grad_clip_value=train_hp.get("grad_clip_value", 5),
            plot_losses=True,
            save_path=output_dir
        )
    else:
        trainer.train(
            model=clf,
            train_data=train_dataset,
            valid_data=valid_dataset,
            epochs=train_hp["epochs"],
            batch_size=train_hp.get("batch_size", 64),
            grad_clip_value=train_hp.get("grad_clip_value", 5),
            shuffle=True,
            plot_losses=True,
            save_path=output_dir
        )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} min.")

    logger.info("Evaluating performance...")
    eval_train_loader = DataLoader(
        train_dataset, batch_size=train_hp.get("batch_size", 64), shuffle=False
    )
    eval_valid_loader = DataLoader(
        valid_dataset, batch_size=train_hp.get("batch_size", 64), shuffle=False
    )
    eval_test_loader  = DataLoader(
        test_dataset,  batch_size=train_hp.get("batch_size", 64), shuffle=False
    )

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

    if config.wandb:
        wandb.log(performance)
        wandb.finish()

    logger.info("----------- Training & Basic Eval Done! -----------")

    # Post-Training Dendritic Analysis (If model has a dendritic net)
    if hasattr(clf, 'net') and hasattr(clf.net, 'forward_with_branch_outputs'):
        logger.info("Performing Dendritic Branch Analysis ...")
        branch_info = analyze_branch_information(
            clf.net,
            train_dataset,
            batch_size=train_hp.get("batch_size", 64),
            device='cpu', 
            n_bins=10
        )
        logger.info(f"Branch Info (MI per layer): {branch_info}")

        # Demo ablation
        if getattr(clf.net, 'branch_layers', None) is not None:
            if len(clf.net.branch_layers) > 0:
                layer_idx = 0
                branch_ids = [0,1]  # ablate first 2 branches, as a demo
                baseline_acc, ablation_res = analyze_branch_ablation(
                    clf.net, test_dataset, 
                    layer_idx=layer_idx, branch_ids=branch_ids, 
                    mode='direct', device='cpu'
                )
                logger.info(f"Ablation baseline acc on test = {baseline_acc}")
                for b_id, (acc_after, delta) in ablation_res.items():
                    logger.info(f"Layer {layer_idx}, branch {b_id}: acc={acc_after:.3f}, drop={delta:.3f}")

        from dendritic_modeling.plot_utils import (
            build_dendritic_tree_graph,
            visualize_dendritic_tree,
            plot_depth_info
        )
        ablation_info = {}
        for lyr, val in branch_info.items():
            ablation_info[lyr] = 0  # placeholder

        G = build_dendritic_tree_graph(
            model=clf.net, 
            branch_info=branch_info, 
            ablation_info=ablation_info
        )
        visualize_dendritic_tree(G, mode='info')
        plot_depth_info(branch_info)
        logger.info("Dendritic Analysis Completed.")
    else:
        logger.info("No dendritic analysis performed (model lacks forward_with_branch_outputs).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run dendritic modeling training and evaluation.'
    )
    parser.add_argument(
        "config", 
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--output_dir", 
        default="results", 
        help="Optional output directory for saving results."
    )
    parser.add_argument(
        "--experiment_name", 
        default="example", 
        help="Optional experiment name for saving results."
    )

    args = parser.parse_args()

    main(
        config_path=args.config, 
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Example usage:
    # python scripts/train_experiments.py configs/config_exp.yaml --experiment_name example --output_dir results