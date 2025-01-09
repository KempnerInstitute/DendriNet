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
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.plot_utils import (plot_einet_params, 
                                           plot_einet_activations)
from dendritic_modeling.utils import (load_MNIST, 
                                      split_MNIST_inputs, 
                                      accuracy_score, 
                                      save_dict, 
                                      Shaper)

from dendritic_modeling.config import load_config
from dendritic_modeling.excitation_inhibition import (ExcitationInhibitionNetwork, 
                                                      FeedforwardNN)



def initialize_model(model_hp):
    task = model_hp['task']
    probabilistic = model_hp['probabilistic']
    network_type = model_hp['network']['type']
    network_parameters = model_hp['network']['parameters']
    
    if task == 'classification':
        if network_type == 'feedforward':
            logger.info(
                f"Running simple feedforward neural network with "
                f" {'probabilistic' if probabilistic else 'simple'} classifier")
            net = FeedforwardNN(**network_parameters)
            output_dim = network_parameters.get('output_dim')
        elif network_type == 'einet':
            logger.info(
                f"Running EINet with "
                f"{'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(**network_parameters)
            output_dim = network_parameters['excitatory_layer_sizes'][-1]
        else:
            raise ValueError(f"Invalid network type: {network_type}")

        clf = (
            ProbabilisticClassifier(net, output_dim) if probabilistic 
            else Classifier(net)
        )
    
    elif task == 'regression':
        if probabilistic:
            raise NotImplementedError(
                "Probabilistic regression not implemented yet.")
        else:
            raise NotImplementedError(
                "Non-probabilistic regression not yet implemented.")
    else:
        raise ValueError(f"Invalid task: {task}")

    return clf
    



def main(config_path: str, output_dir: str,  experiment_name: str):

    
    # create output directory
    save_path = output_dir
    os.makedirs(save_path, exist_ok = True)

    
    # Set up the logging environment
    logger_manager.set_log_file(os.path.join(output_dir, 
                                             "dendritic_modeling.log"))

    # Load configuration
    config = load_config(config_path)
    model_hp = config.model.asdict()
    train_hp = config.train.asdict()

    logger.debug("="*50)
    logger.debug(" "*10 + "Config and run hyperparameters" + " "*10)
    logger.debug("="*50)

    logger.debug(f" Experiment Name: {experiment_name}")
    logger.debug(" Model Hyperparameters:\n" + str(model_hp))
    logger.debug(" Training Hyperparameters:\n" + str(train_hp))
    logger.debug(f" Output Directory: {save_path}")

    logger.debug("="*50)


    # intialize W&B
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
    data = load_MNIST(train_valid_split = train_hp['train_valid_split'], 
                      cache_dir=os.getcwd())
    
    
    # Initialize model
    clf = initialize_model(model_hp)


    model_summary = torchinfo.summary(
                    clf,
                    input_size=(train_hp['batch_size'], 
                                model_hp['network']['parameters']['input_dim']),
                    col_names=["input_size", 
                                "output_size", 
                                "num_params"],
                    col_width=25,
                    row_settings=["depth"],
                    depth=7
                    )

    logger.debug("".join(["\n", str(model_summary), "\n"]))

    # send model summary to wandb
    if wandb.run is not None:
        model_summary_params = {
            "total_params": model_summary.total_params,
            "trainable_params": model_summary.trainable_params
        }
        wandb.log(model_summary_params)



    # initialize trainer class
    trainer = TrainerMLE(
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr = train_hp['lr']),
        suppress_prints = False, 
        print_every = 10,
    )



    logger.info('Started training the model...')
    training_start_time = time.time()
    trainer.train(
        model=clf,
        train_data=data['train'], 
        valid_data=data['valid'],
        epochs=train_hp['epochs'],
        batch_size=train_hp['batch_size'], 
        grad_clip_value=train_hp['grad_clip_value'],
        shuffle=True,
        plot_losses=True, 
        save_path=save_path,
    )
    training_time = time.time() - training_start_time
    formatted_training_time = f'{training_time:.6f} sec -> {training_time / 60:.6f} min'
    logger.info(f'Training completed in {formatted_training_time}.')

    # Evaluate performance
    logger.info('Started evaluating performance...')
    inference_start = time.time()
    performance = {
        'train accuracy': accuracy_score(clf, *data['train'][:]).item(),
        'valid accuracy': accuracy_score(clf, *data['valid'][:]).item(),
        'test accuracy': accuracy_score(clf, *data['test'][:]).item(),
    }
    inference_time = time.time() - inference_start
    performance['training time'] = formatted_training_time
    formatted_inference_time = f'{inference_time:.6f} sec -> {inference_time / 60:.6f} min'
    performance['inference time'] = formatted_inference_time

    # Saving
    save_dict(model_hp, save_path, 'model_hparams.json')
    save_dict(train_hp, save_path, 'training_hparams.json')
    save_dict(performance, save_path, 'performance.json')

    logger.info(f"Completed evaluation in {formatted_inference_time}.")

    # Visualize and save model parameters and activations
    if config.settings.visualizations:
        shaper = Shaper(shape=(28, 28))

        logger.info('Creating parameter visuals ...')
        visual_start_time = time.time()

        plot_einet_params(
            einet=clf.net, save_root=os.path.join(save_path, 'param_visuals'),
            reshape_fn=shaper.reshape, reshape_exc_syn=True,
        )

        input_list = split_MNIST_inputs(*data['train'][:])
        print('Creating activation visuals...')
        plot_einet_activations(
            einet=clf.net, input_list=input_list,
            save_root=os.path.join(save_path, 'activation_visuals'),
        )
        visual_time = time.time() - visual_start_time
        formatted_visual_time = f'{visual_time:.6f} sec -> {visual_time / 60:.6f} min'
        logger.info(f'Completed visualization in {formatted_visual_time}.')
    else:
        visual_time = 0


    # send wall clock time to wandb
    if wandb.run is not None:
        wc_time = {
            "training_time": training_time,
            "inference_time": inference_time,
            "visualization_time": visual_time,
            "total_time": training_time + inference_time + visual_time,
            "normalized_training_time": (training_time) / config.train.epochs

        }
        wandb.log(wc_time)


    logger.info('----------- Done! -----------')

    wandb.finish()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", 
                        help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", 
                        default="results", 
                        help="Optional output directory for saving results.")
    parser.add_argument("--experiment_name", 
                        default="example", 
                        help="Optional experiment name for saving results.")

 
    args = parser.parse_args()

    # Run the main function
    main(config_path=args.config, 
         output_dir=args.output_dir,
         experiment_name=args.experiment_name)

    # Example usage:
    # python scripts/example_with_config.py configs/example_config.yaml --experiment_name example --output_dir results