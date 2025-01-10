import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

# Reuse your existing code:
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import EINetClassifierNLL
from dendritic_modeling.utils import accuracy_score
from dendritic_modeling.synthetic_datasets import get_synthetic_line_dataloaders
from dendritic_modeling.information_metrics import estimate_hidden_output_MI

def run_experiment(
    e_cells, i_cells, 
    remove_inhibition=False,
    dataset="synthetic_line",
    output_dir="experiment_results",
    config=None
):
    """
    e_cells: number of excitatory neurons in final classification layer
    i_cells: number of inhibitory neurons in final classification layer
    remove_inhibition: bool, whether to ablate inhibition
    dataset: 'synthetic_line' or 'mnist'
    config: dictionary or object with standard hyperparams
    """

    # 1. Load data
    if dataset == "synthetic_line":
        train_loader, valid_loader, test_loader = get_synthetic_line_dataloaders()
        input_dim = 10*10
        n_classes = 3
    else:
        # Reuse your MNIST loader
        from dendritic_modeling.utils import load_MNIST
        data = load_MNIST(train_valid_split=config["train_valid_split"])
        train_loader = DataLoader(data['train'], batch_size=config["batch_size"], shuffle=True)
        valid_loader = DataLoader(data['valid'], batch_size=config["batch_size"], shuffle=False)
        test_loader  = DataLoader(data['test'],  batch_size=config["batch_size"], shuffle=False)
        input_dim = 28*28
        n_classes = 10

    # 2. Build model hyperparams
    excitatory_layer_sizes = [e_cells]
    inhibitory_layer_sizes = [i_cells] if not remove_inhibition else [0]  # ablation means i_cells=0 or None

    # minimal example: single-layer net with some synapses
    model_hp = {
        "input_dim": input_dim,
        "excitatory_layer_sizes": excitatory_layer_sizes,
        "inhibitory_layer_sizes": inhibitory_layer_sizes,
        "excitatory_branch_factors": [4],   # or some other factor
        "inhibitory_branch_factors": [4],
        "ee_synapses_per_branch_per_layer": [20],
        "ei_synapses_per_branch_per_layer": [20],
        "ie_synapses_per_branch_per_layer": [1],
        "ii_synapses_per_branch_per_layer": [1] if not remove_inhibition else [],
        "reactivate": True,
        "somatic_synapses": True,
        "topk_init_method": "xavier_normal",
    }

    clf = EINetClassifierNLL(**model_hp)

    # 3. Set up trainer
    train_hp = {
        "epochs": config.get("epochs", 10),
        "batch_size": config.get("batch_size", 64),
        "lr": config.get("lr", 1e-3),
        "grad_clip_value": config.get("grad_clip_value", None)
    }

    optimizer = torch.optim.Adam(clf.parameters(), lr=train_hp["lr"])
    trainer = TrainerMLE(
        optimizer = optimizer, 
        suppress_prints = False, 
        print_every = 1,
    )

    # 4. Training
    trainer.train(
        model=clf,
        train_data=train_loader, 
        valid_data=valid_loader,
        epochs=train_hp['epochs'],
        batch_size=train_hp['batch_size'], 
        grad_clip_value=train_hp['grad_clip_value'],
        shuffle=True,
        plot_losses=True, 
        save_path=output_dir
    )

    # 5. Evaluation
    train_acc = accuracy_score(clf, *next(iter(train_loader)))
    valid_acc = accuracy_score(clf, *next(iter(valid_loader)))
    test_acc  = accuracy_score(clf, *next(iter(test_loader)))

    # 6. (Optional) Info analysis
    hidden_MI = estimate_hidden_output_MI(clf, test_loader)

    return {
        "train_acc": train_acc.item(),
        "valid_acc": valid_acc.item(),
        "test_acc": test_acc.item(),
        "hidden_MI": hidden_MI
    }


if __name__ == "__main__":

    # Example usage of the experiment function
    # We'll systematically vary excitatory/inhibitory ratio
    results = []

    # Suppose we fix total = 10 neurons, vary E/I from (9,1) to (5,5)
    E_list = [9,8,7,6,5]
    I_list = [1,2,3,4,5]

    for e, i in zip(E_list, I_list):
        res = run_experiment(
            e_cells=e, 
            i_cells=i,
            remove_inhibition=False, 
            dataset="synthetic_line",  # or "mnist"
            output_dir=f"results/E_{e}_I_{i}",
            config={
                "epochs": 10,
                "batch_size": 64,
                "lr": 1e-3,
            }
        )
        print(f"E={e}, I={i} => {res}")
        results.append((e, i, res))

    # Ablation: remove inhibition altogether
    no_inh_res = run_experiment(
        e_cells=10, 
        i_cells=0, 
        remove_inhibition=True, 
        dataset="synthetic_line",
        output_dir=f"results/no_inhibition",
        config={
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-3
        }
    )
    print(f"No Inhibition => {no_inh_res}")