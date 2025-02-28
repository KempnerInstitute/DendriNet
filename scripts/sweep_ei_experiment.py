"""
sweep_ei_experiment.py
---------------------
Script that performs parameter sweeps over excitatory-excitatory and 
inhibitory-excitatory synaptic connections while keeping other parameters constant.
"""
import os
import sys
import itertools
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict

import torch

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.utils import evaluate_accuracy, save_dict
from dendritic_modeling.plot_utils import plot_ei_sweep_results

def initialize_model(model_cfg):
    """Initialize EINet model with given configuration."""
    net_params = deepcopy(model_cfg.network.parameters)
    enable_branch_outputs = net_params.pop('enable_branch_outputs', False)
    learning_strategy = net_params.pop('learning_strategy', 'mle')
    
    net = ExcitationInhibitionNetwork(
        enable_branch_outputs=enable_branch_outputs,
        learning_strategy=learning_strategy,
        **net_params
    )
    
    output_dim = net_params['excitatory_layer_sizes'][-1]
    return ProbabilisticClassifier(net, output_dim) if model_cfg.probabilistic else Classifier(net)


def train_and_evaluate(model, train_ds, valid_ds, test_ds, train_cfg, run_save_path):
    """Train model and evaluate performance."""
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate
    )
    
    trainer = get_trainer(train_cfg.learning_strategy, optimizer, suppress_prints=True)
    
    # Train the model
    trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=True,
        save_path=run_save_path
    )
    
    # Evaluate performance
    train_acc, valid_acc, test_acc = evaluate_accuracy(
        model, train_ds, valid_ds, test_ds,
        save_path=run_save_path,
        filename="performance"
    )
    
    return {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python sweep_ei_experiment.py <path_to_config>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    # Create base output directory
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = os.path.join(config.outputs.dir, f"sweep_ei_{now}")
    os.makedirs(base_save_path, exist_ok=True)
    
    # Setup logging
    logger_manager.set_log_file(os.path.join(base_save_path, "sweep_ei.log"))
    logger.info(f"Starting parameter sweep experiment")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Results will be saved to: {base_save_path}")
    
    # Load datasets
    train_ds, valid_ds, test_ds = get_unified_datasets(config.task, config.train)

    config.model.network.parameters.input_dim = train_ds[0][0].shape[-1]
    config.model.network.parameters.output_dim = torch.unique(train_ds[:][1]).shape[-1]
    config.model.network.parameters.learning_strategy = getattr(config.train, "learning_strategy", "mle")
    
    # Get parameter combinations
    ee_configs = config.model.network.parameters.ee_synapses_per_branch_per_layer
    ie_configs = config.model.network.parameters.ie_synapses_per_branch_per_layer
    
    # Create results file
    results_file = os.path.join(base_save_path, "ei_sweep_results.csv")
    with open(results_file, 'w') as f:
        f.write("exc,inh,train,valid,test\n")
    
    # Run experiments for each combination
    total_combinations = len(ee_configs) * len(ie_configs)
    current_combination = 0
    
    for ee_config, ie_config in itertools.product(ee_configs, ie_configs):
        current_combination += 1
        logger.info(f"\nRunning combination {current_combination}/{total_combinations}")
        logger.info(f"EE config: {ee_config}, IE config: {ie_config}")
        
        # Create experiment directory
        exp_name = f"exc{'_'.join(map(str, ee_config))}_inh{'_'.join(map(str, ie_config))}"
        exp_save_path = os.path.join(base_save_path, 'sweeps', exp_name)
        os.makedirs(exp_save_path, exist_ok=True)
        
        # Update config with current parameter combination
        exp_config = deepcopy(config)
        exp_config.model.network.parameters.ee_synapses_per_branch_per_layer = ee_config
        exp_config.model.network.parameters.ie_synapses_per_branch_per_layer = ie_config
        
        # Initialize and train model
        model = initialize_model(exp_config.model)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        try:
            results = train_and_evaluate(
                model, train_ds, valid_ds, test_ds, 
                exp_config.train, exp_save_path
            )
            
            # Log results
            logger.info(f"Results for exc={ee_config}, inh={ie_config}:")
            logger.info(f"Train acc: {results['train_acc']:.3f}, Valid acc: {results['valid_acc']:.3f}, Test acc: {results['test_acc']:.3f}")
            
            # Save results to CSV
            with open(results_file, 'a') as f:
                f.write(f"{ee_config[0]},{ie_config[0]},"
                       f"{results['train_acc']:.4f},{results['valid_acc']:.4f},"
                       f"{results['test_acc']:.4f}\n")
                
        except Exception as e:
            logger.error(f"Error in combination {exp_name}: {str(e)}")
            continue
    
    logger.info("\nParameter sweep completed. Results saved to: " + results_file)
    
    logger.info("Plotting results...")
    logger.info(results_file)
    plot_ei_sweep_results(
        results_csv_path = results_file, metric = 'acc', split = 'test',
    )
    logger.info("  done...")

    logger.info("Saving config...")
    save_dict(asdict(config), base_save_path, "config.yaml")
    logger.info(f"Saved config to {base_save_path}/config.yaml")
    logger.info("  done...")

if __name__ == "__main__":
    main()
