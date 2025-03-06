"""
two_phase_train.py
==================
Example script demonstrating a two-phase approach:

Phase 1: Train a simple MLP classifier end-to-end on MNIST (or any classification task).
Phase 2: Remove final classification layer, wrap leftover as mlp_transform that outputs 
         (exc, inh). Build an ExcitationInhibitionNetwork using input_mode=2 (or 1).
         Freeze the MLP's parameters so only the EiNet's dendritic weights are trained.

We fix the "Unexpected file type" error by ensuring we only load the config once.
We do so in __main__, then pass the loaded config to main(config).
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import wandb
import torch
import torch.nn as nn
import torchinfo

### CHANGED ###
# We import load_config here so we can call it in __main__ only
from dendritic_modeling.config import load_config
from dendritic_modeling import logger, logger_manager
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.models import ProbabilisticClassifier
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict


###############################################################################
# PHASE 1: Train a simple MLP classifier
###############################################################################
class SimpleMLPClassifier(nn.Module):
    """
    A basic MLP: (Linear->ReLU)* + final Linear->10 classes (MNIST)
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10):
        super().__init__()
        layers = []
        in_feat = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_feat, hdim))
            layers.append(nn.ReLU())
            in_feat = hdim
        # final classification layer
        layers.append(nn.Linear(in_feat, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_classifier(model, train_ds, valid_ds, epochs=10, lr=1e-3, device='cpu'):
    """
    Train the MLP classifier in a standard classification loop.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=128, shuffle=False)

    best_val_loss = float('inf')
    best_state = model.state_dict()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for vx, vy in valid_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                l = loss_fn(v_logits, vy)
                val_loss += l.item() * vx.size(0)
                preds = v_logits.argmax(dim=1)
                correct += (preds == vy).sum().item()
        val_loss /= len(valid_loader.dataset)
        val_acc = correct / len(valid_loader.dataset)

        logger.info(f"[Pretrain MLP] Epoch={epoch}/{epochs} | "
                    f"train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")

        # track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model

###############################################################################
# Phase 2: use the pretrained MLP's hidden layers -> (exc, inh) transform
###############################################################################
class TwoHeadTransform(nn.Module):
    """
    Takes a pretrained MLP *without* its final classification layer
    as the 'backbone'. Then we add two heads that produce excitatory_input
    and inhibitory_input for input_mode=2.
    """
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone  # all hidden layers
        # We produce excit + inhib from feature_dim
        self.head_exc = nn.Linear(feature_dim, feature_dim)
        self.head_inh = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        feats = self.backbone(x)
        excit = self.head_exc(feats)
        inhib = self.head_inh(feats)
        return excit, inhib

def freeze_parameters(module):
    """
    Utility: set requires_grad=False for all parameters in the given module
    so they won't be updated by the optimizer.
    """
    for param in module.parameters():
        param.requires_grad = False


### CHANGED: main now accepts 'config' not 'config_path' ###
def main(config):
    """
    We do NOT call load_config again inside 'main'. 'config' is already
    a loaded OmegaConf object or a config structure from __main__.
    """
    # Unpack config
    model_cfg = config.model
    train_cfg = config.train
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    now = now.replace("_", "").replace("-", "")
    run_save_path = os.path.join(outputs_cfg.dir, f"{train_cfg.run_name}_{now}")

    os.makedirs(run_save_path, exist_ok=True)
    logger_manager.set_log_file(os.path.join(run_save_path, "dendritic_modeling.log"))
    logger.info(f"Saving results to: {run_save_path}")

    if getattr(wandb_cfg, "use_wandb", False):
        from dataclasses import asdict
        wandb_dir = Path(run_save_path) / "wandb"
        wandb_run_name = train_cfg.run_name
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_dict = asdict(config)
        wandb.init(
            dir=str(wandb_dir),
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
            group=wandb_cfg.group,
            name=wandb_run_name,
            tags=wandb_cfg.tags,
            config=wandb_dict,
        )

    logger.info("Loading dataset...")
    train_ds, valid_ds, test_ds = get_unified_datasets(config.task, train_cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ###########################################################################
    # 1) Train an MLP classifier (Phase 1)
    ###########################################################################
    logger.info("=== Phase 1: Pretrain MLP classifier ===")
    from torch.utils.data import DataLoader
    input_dim = train_ds[0][0].numel()
    mlp_hidden_dims = [512, 256]
    pretrain_mlp = SimpleMLPClassifier(input_dim=input_dim, hidden_dims=mlp_hidden_dims, num_classes=10)

    pretrain_epochs = getattr(train_cfg, "pretrain_epochs", 10)
    pretrain_mlp = train_mlp_classifier(pretrain_mlp, train_ds, valid_ds, epochs=pretrain_epochs, lr=1e-3, device=device)

    # Evaluate MLP on test set
    pretrain_mlp.eval()
    test_loader = DataLoader(test_ds, batch_size=128)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = pretrain_mlp(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    logger.info(f"[Pretrain MLP] test accuracy = {test_acc:.3f}")

    ###########################################################################
    # 2) Remove final classification layer, wrap leftover => (exc, inh)
    #    Freeze these parameters so we do NOT train them in Phase 2
    ###########################################################################
    logger.info("=== Phase 2: Build EiNet with the pretrained MLP as input transform ===")

    # final classification layer is the last in pretrain_mlp.net
    # remove it:
    backbone_layers = list(pretrain_mlp.net[:-1])  # everything except final
    feature_dim = mlp_hidden_dims[-1]
    backbone = nn.Sequential(*backbone_layers).to(device)

    mlp_transform = TwoHeadTransform(backbone, feature_dim).to(device)
    freeze_parameters(mlp_transform)

    ###########################################################################
    # 3) Convert config node => dict, insert mlp_transform => build EiNet
    ###########################################################################
    from dendritic_modeling.networks import ExcitationInhibitionNetwork
    import omegaconf
    from omegaconf import OmegaConf

    einet_params = model_cfg.network.parameters
    # Force input_mode=2
    einet_params["input_mode"] = 2
    einet_params["input_dim"] = input_dim  # ensure correct input dim

    # Convert OmegaConf to dict
    einet_params_dict = OmegaConf.to_container(einet_params, resolve=True)
    # remove mlp_transform_dict if present
    einet_params_dict.pop("mlp_transform_dict", None)
    # Insert actual PyTorch transform
    einet_params_dict["mlp_transform"] = mlp_transform

    # Now build
    dendri_net = ExcitationInhibitionNetwork(**einet_params_dict).to(device)

    # Possibly build a classifier on top
    output_dim = einet_params_dict.get("output_dim", 10)
    clf = ProbabilisticClassifier(dendri_net, output_dim=output_dim).to(device)

    # Summarize
    try:
        sample_input, _ = train_ds[0]
        summary = torchinfo.summary(clf, input_size=(128, sample_input.numel()))
        logger.info(str(summary))
    except:
        pass

    ###########################################################################
    # 4) Train only EiNet (mlp_transform is frozen)
    ###########################################################################
    optimizer = CustomWeightDecayOptimizer(
        model=clf,
        optimizer=torch.optim.Adam(clf.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )
    trainer = get_trainer(train_cfg.learning_strategy, optimizer, suppress_prints=False, print_every=10)

    results = trainer.train(
        model=clf,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=True,
    )

    # Evaluate final
    from dendritic_modeling.utils import evaluate_accuracy
    train_acc, val_acc, test_acc = evaluate_accuracy(
        clf, train_ds, valid_ds, test_ds,
        save_path=run_save_path,
        filename="final"
    )
    logger.info(f"Phase2 EiNet final: train_acc={train_acc:.3f}, valid_acc={val_acc:.3f}, test_acc={test_acc:.3f}")

    # Done
    logger.info("All done with two-phase approach.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Two-phase training example.")
    parser.add_argument("config", help="Path to the config YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    parser.add_argument("--run_name", default="example", help="Run name (used for W&B run name).")
    args = parser.parse_args()

    ### CHANGED: we load config here, then pass the loaded config object to main() ###
    cfg = load_config(args.config)  # This is the only place we call load_config
    if args.output_dir:
        cfg.outputs.dir = args.output_dir
    if args.run_name:
        cfg.train.run_name = args.run_name

    # pass the loaded config to main
    main(cfg)