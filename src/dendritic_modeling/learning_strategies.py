"""
learning_strategies.py
======================
Houses multiple learning/training strategies:

  - TrainerMLE            (classic NLL backprop)
      * includes train_freeze_layers for freeze/unfreeze approach with a final full-network tuning stage.
  - LocalCreditAssignment (advanced local learning using feedback alignment)
  - TwoStepTrainer        (pretrain each layer with local supervision using cross-entropy loss, then full end-to-end training)
  - TwoStepTrainerWithKL  (pretrain each layer with local supervision using a combined loss (cross-entropy + KL divergence), then full end-to-end training)
  - VoltageStabilizationTrainer (add a warm-up phase that stabilizes voltages to a target)

A factory function get_trainer(...) returns the chosen strategy based on a string.
"""

import os
import time
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies import BaseTrainer, compute_metrics_classification


# compute_metrics_classification has been moved to training/strategies/base.py


# CustomWeightDecayOptimizer has been moved to training/optimizers/custom.py


# BaseTrainer has been moved to training/strategies/base.py


class TrainerMLE(BaseTrainer):
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        load_best_state_dict=True,
        best_loss=float("inf"),
        early_stopping=False,
        patience=10,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        start_time = time.time()
        # Early stopping setup
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())

            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            valid_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                patience_counter = 0  # Reset counter on improvement
                if save_path and os.path.exists(save_path):
                    torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            else:
                patience_counter += 1
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)

            # Early stopping check
            if early_stopping and patience_counter >= patience:
                if not self.suppress:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )

            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                        "best_epoch": best_epoch,
                    }
                )

        logger.info(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")
        if load_best_state_dict:
            model.load_state_dict(best_state)
            logger.info("Final model loaded from best checkpoint")

        results = {
            "best epoch": best_epoch,
            "best loss": best_loss,
            "best state dict": best_state,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }

        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epoch, save_path)
            else:
                results["loss curves"] = plot_NLL_loss_curves(
                    train_losses, valid_losses, epoch
                )

        return results

    def train_freeze_branches(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
        **kwargs,
    ):
        """Train individual dendritic branches sequentially from soma to input (distal).
        This method trains each branch in the dendritic tree separately, starting from the branches
        closest to the soma and moving outward."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Get the branch layers from the model
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"train_freeze_branches: found {n_branches} branch layers. Training them sequentially from soma to input."
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_branches * epochs_per_branch
        epoch_counter = 0
        start_time = time.time()

        # Train branches from soma (near) to distal (far) - typically index 0 is closest to soma
        for branch_idx in range(n_branches):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        logger.info(
            "Done freeze-branches. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        return final_results

    def train_freeze_branches_reversed(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
        **kwargs,
    ):
        """Train individual dendritic branches sequentially from input (distal) to soma.
        This method trains each branch in the dendritic tree separately, starting from the branches
        farthest from the soma and moving inward."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Get the branch layers from the model
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"train_freeze_branches_reversed: found {n_branches} branch layers. Training them sequentially from input to soma."
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_branches * epochs_per_branch
        epoch_counter = 0
        start_time = time.time()

        # Train branches from distal (far) to soma (near) - reverse order
        for branch_idx in range(n_branches - 1, -1, -1):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-reversed] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_reversed": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        logger.info(
            "Done freeze-branches-reversed. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        return final_results

    def train_freeze_branches_with_pretraining(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=10,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
        **kwargs,
    ):
        """Train with a three-phase approach:
        1. Initial full network training (all parameters unfrozen) for pretrain_epochs
        2. Branch-by-branch training from soma to input (freezing all branches except the current one)
        3. Final full network fine-tuning
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Phase 1: Initial full network training
        logger.info(
            f"Starting initial full network training for {pretrain_epochs} epochs"
        )
        initial_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=False,
        )

        # Save the pretrained model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            pretrain_path = os.path.join(save_path, "pretrained_model.pt")
            torch.save(model.state_dict(), pretrain_path)
            logger.info(f"Saved pretrained model to {pretrain_path}")

        # Phase 2: Branch-by-branch training
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"Starting branch-by-branch training (soma to input): found {n_branches} branches."
        )
        train_losses = initial_results.get("train_losses", [])
        valid_losses = initial_results.get("valid_losses", [])
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = (
            pretrain_epochs + (n_branches * epochs_per_branch) + final_tune_epochs
        )
        epoch_counter = pretrain_epochs
        start_time = time.time()

        # Train branches from soma (near) to distal (far)
        for branch_idx in range(n_branches):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-with-pretraining] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_with_pretraining": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        # Phase 3: Final full-network fine-tuning
        logger.info(
            "Done branch-by-branch training. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        # Combine all results
        final_results["train_losses"] = train_losses + final_results.get(
            "train_losses", []
        )
        final_results["valid_losses"] = valid_losses + final_results.get(
            "valid_losses", []
        )

        return final_results

    def train_freeze_branches_reversed_with_pretraining(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=10,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
        **kwargs,
    ):
        """Train with a three-phase approach:
        1. Initial full network training (all parameters unfrozen) for pretrain_epochs
        2. Branch-by-branch training from input to soma (freezing all branches except the current one)
        3. Final full network fine-tuning
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Phase 1: Initial full network training
        logger.info(
            f"Starting initial full network training for {pretrain_epochs} epochs"
        )
        initial_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=False,
        )

        # Save the pretrained model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            pretrain_path = os.path.join(save_path, "pretrained_model.pt")
            torch.save(model.state_dict(), pretrain_path)
            logger.info(f"Saved pretrained model to {pretrain_path}")

        # Phase 2: Branch-by-branch training from input to soma
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"Starting branch-by-branch training (input to soma): found {n_branches} branches."
        )
        train_losses = initial_results.get("train_losses", [])
        valid_losses = initial_results.get("valid_losses", [])
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = (
            pretrain_epochs + (n_branches * epochs_per_branch) + final_tune_epochs
        )
        epoch_counter = pretrain_epochs
        start_time = time.time()

        # Train branches from distal (far) to soma (near) - reverse order
        for branch_idx in range(n_branches - 1, -1, -1):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-reversed-with-pretraining] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_reversed_with_pretraining": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        # Phase 3: Final full-network fine-tuning
        logger.info(
            "Done branch-by-branch training. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        # Combine all results
        final_results["train_losses"] = train_losses + final_results.get(
            "train_losses", []
        )
        final_results["valid_losses"] = valid_losses + final_results.get(
            "valid_losses", []
        )

        return final_results


class TrainerRegression(BaseTrainer):
    def __init__(self, optimizer, suppress_prints=False, print_every=10, use_scheduler=True, min_lr=1e-6):
        super().__init__(optimizer, suppress_prints, print_every)
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr

    def _eval_dataset(self, model, dataset, device, batch_size=256):
        if dataset is None or len(dataset) == 0:
            return 0.0, 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_mse = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                total_loss += nll.item()
                pred = model.predict(xb)
                mse = F.mse_loss(pred, yb)
                total_mse += mse.item()
        return total_loss / len(loader), total_mse / len(loader)

    def train(
        self, 
        model, 
        train_data, 
        valid_data=None, 
        grad_clip_value=5, 
        epochs=100,
        early_stopping=False, 
        patience=10,
        batch_size=256, 
        shuffle=True, 
        load_best_state_dict=True, 
        best_loss=float('inf'),
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False, 
        save_path=None, 
        seed=None, 
        **kwargs
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed is not None:
            model.set_seed(seed)
        model = model.to(device)
        
        # Initialize the scheduler using the wrapped optimizer if needed
        if self.use_scheduler:
            # Get the actual optimizer from the wrapper
            actual_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            scheduler = CosineAnnealingLR(actual_optimizer, T_max=epochs, eta_min=self.min_lr)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        start_time = time.time()

        patience_counter = 0
        epoch_counter = 0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_counter += 1
            epoch_train_loss = 0.0
            epoch_train_mse = 0.0
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                
                # Calculate MSE for monitoring
                with torch.no_grad():
                    pred = model.predict(xb)
                    mse = F.mse_loss(pred, yb)
                    epoch_train_mse += mse.item()
            
            epoch_train_loss /= len(train_loader)
            epoch_train_mse /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            val_loss, val_mse = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            
            # Update scheduler (new)
            if self.use_scheduler:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                lr_info = f", lr={current_lr:.2e}" if self.use_scheduler else ""
                logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                          f"train_mse={epoch_train_mse:.4f}, valid_loss={val_loss:.4f}, "
                          f"valid_mse={val_mse:.4f}, elapsed={elapsed:.1f}s{lr_info}")
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch_counter,
                    "train_loss": epoch_train_loss,
                    "train_mse": epoch_train_mse,
                    "valid_loss": val_loss,
                    "valid_mse": val_mse,
                    "best_valid_mse": best_loss
                })
                if self.use_scheduler:
                    wandb.log({
                        "learning_rate": current_lr
                    })
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch_counter
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
                if save_path and os.path.exists(save_path):
                    torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            else:
                patience_counter += 1
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
                
            if early_stopping and patience_counter >= patience:
                if not self.suppress:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
            
        if load_best_state_dict:
            model.load_state_dict(best_state)
        
        results = {
            "best epoch": best_epoch,
            "best loss": best_loss,
            "best state dict": best_state,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }
        
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                results['loss curves'] = plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        
        return results


class LocalCreditAssignment(BaseTrainer):
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        best_loss = float("inf")
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        train_losses = []
        valid_losses = []
        start_time = time.time()
        logger.info(
            "LocalCreditAssignment: Using feedback alignment for local updates."
        )
        if hasattr(model, "net") and hasattr(model.net, "layers"):
            if (
                not hasattr(self, "feedback_matrices_initialized")
                or not self.feedback_matrices_initialized
            ):
                self.feedback_matrices = []
                for layer in model.net.layers:
                    if hasattr(layer, "weight"):
                        B = torch.randn(layer.weight.shape).to(layer.weight.device)
                        self.feedback_matrices.append(B)
                    else:
                        self.feedback_matrices.append(None)
                self.feedback_matrices_initialized = True
        else:
            logger.info(
                "No 'model.net.layers' found => feedback alignment may not apply."
            )
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            for p in model.parameters():
                p.requires_grad_(False)
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.no_grad():
                    logits = model(xb)
                    loss_val = model.compute_loss(xb, yb)
                epoch_train_loss += loss_val.item()
                preds = model.predict(xb)
                preds_list.append(preds.cpu())
                labels_list.append(yb.cpu())
                alpha = 1e-3
                error_signal = (yb - preds.to(device)).float()
                batch_error = error_signal.sum().item()
                sign_of_error = (
                    1.0 if batch_error > 0 else -1.0 if batch_error < 0 else 0.0
                )
                for p in model.parameters():
                    p.data -= alpha * sign_of_error
                if grad_clip_value is not None:
                    for p in model.parameters():
                        p.data = torch.clamp(p.data, -grad_clip_value, grad_clip_value)
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[LocalCredit] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "best epoch": best_epoch,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results


class TrainOnlyMReactivation(BaseTrainer):
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(
            "TrainOnlyMReactivation: Freezing log_b and training all other parameters."
        )
        for name, param in model.named_parameters():
            if "log_b" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss_val = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TrainOnlyMReactivation] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "best epoch": best_epoch,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results


class TwoStepTrainer(BaseTrainer):
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
        if hasattr(net_obj, "layers"):
            if not hasattr(self, "local_heads"):
                self.local_heads = []
                for i, layer in enumerate(net_obj.layers):
                    out_dim = getattr(layer, "out_features", None)
                    if out_dim is None:
                        excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                        if isinstance(excit_sizes, list) and i < len(excit_sizes):
                            out_dim = excit_sizes[i]
                        else:
                            try:
                                dummy = torch.randn(
                                    1, getattr(net_obj, "input_dim", 784)
                                ).to(device)
                            except Exception:
                                dummy = torch.randn(1, 784).to(device)
                            dummy_out = layer(dummy)
                            if isinstance(dummy_out, tuple):
                                dummy_out = dummy_out[0]
                            out_dim = dummy_out.size(1)
                    if hasattr(layer, "weight"):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    head = torch.nn.Linear(
                        out_dim, getattr(model, "output_dim", 10)
                    ).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses,
                save_path=save_path,
            )
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        logger.info(
            f"=== TwoStepTrainer: Stage 1 (local pretraining) for {pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            for layer_idx in range(n_layers):
                for name, param in model.named_parameters():
                    param.requires_grad = False
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h = net_obj.layers[j].activation(h)
                    loss_val = F.cross_entropy(local_head(h), yb)
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    if grad_clip_value is not None:
                        params_to_clip = list(
                            filter(lambda p: p.requires_grad, model.parameters())
                        )
                        if params_to_clip:
                            clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            clip_grad_value_(params_local, grad_clip_value)
                    self.optimizer.step()
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
            epoch_train_loss = epoch_loss_accum / (ct_batches if ct_batches > 0 else 1)
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step1_epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "layerwise-pretrain",
                        "train_loss": epoch_train_loss,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        logger.info(
            f"=== TwoStepTrainer: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ==="
        )
        for param in model.parameters():
            param.requires_grad = True
        for ep in range(1, maintrain_epochs + 1):
            ep_counter += 1
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                if grad_clip_value is not None:
                    clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step2_epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "main-train",
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {"train losses": all_train_losses, "valid losses": all_valid_losses}
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(
                    all_train_losses, all_valid_losses, ep_counter, save_path
                )
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        return results


class TwoStepTrainerWithKL(BaseTrainer):
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
        if hasattr(net_obj, "layers"):
            if not hasattr(self, "local_heads"):
                self.local_heads = []
                n_layers = len(net_obj.layers)
                for i, layer in enumerate(net_obj.layers):
                    out_dim = getattr(layer, "out_features", None)
                    if out_dim is None:
                        excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                        if isinstance(excit_sizes, list) and i < len(excit_sizes):
                            out_dim = excit_sizes[i]
                        else:
                            try:
                                dummy = torch.randn(
                                    1, getattr(net_obj, "input_dim", 784)
                                ).to(device)
                            except Exception:
                                dummy = torch.randn(1, 784).to(device)
                            dummy_out = layer(dummy)
                            if isinstance(dummy_out, tuple):
                                dummy_out = dummy_out[0]
                            out_dim = dummy_out.size(1)
                    if hasattr(layer, "weight"):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    head = torch.nn.Linear(
                        out_dim, getattr(model, "output_dim", 10)
                    ).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses,
                save_path=save_path,
            )
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        logger.info(
            f"=== TwoStepTrainerWithKL: Stage 1 (local pretraining with KL) for {pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            for layer_idx in range(n_layers):
                for name, param in model.named_parameters():
                    param.requires_grad = False
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h = net_obj.layers[j].activation(h)
                    local_logits = local_head(h)
                    ce_loss = F.cross_entropy(local_logits, yb)
                    with torch.no_grad():
                        final_logits = model(xb).detach()
                    kl_loss = F.kl_div(
                        torch.log_softmax(local_logits, dim=-1),
                        torch.softmax(final_logits, dim=-1),
                        reduction="batchmean",
                    )
                    weight_alpha = (layer_idx + 1) / n_layers
                    weight_beta = 1.0 - weight_alpha
                    loss_val = weight_alpha * ce_loss + weight_beta * kl_loss
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    if grad_clip_value is not None:
                        params_to_clip = list(
                            filter(lambda p: p.requires_grad, model.parameters())
                        )
                        if params_to_clip:
                            clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            clip_grad_value_(params_local, grad_clip_value)
                    self.optimizer.step()
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
            epoch_train_loss = epoch_loss_accum / (ct_batches if ct_batches > 0 else 1)
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepWithKL: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step1_epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "layerwise-pretrain-kl",
                        "train_loss": epoch_train_loss,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        logger.info(
            f"=== TwoStepTrainerWithKL: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ==="
        )
        for param in model.parameters():
            param.requires_grad = True
        for ep in range(1, maintrain_epochs + 1):
            ep_counter += 1
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                if grad_clip_value is not None:
                    clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepWithKL: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step2_epoch{epoch}', training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "main-train-kl",
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {"train losses": all_train_losses, "valid losses": all_valid_losses}
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(
                    all_train_losses, all_valid_losses, ep_counter, save_path
                )
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        return results


def voltage_warmup_loss(model, x, target_val=0.5):
    outputs = model(x)
    if hasattr(model, "net"):
        if hasattr(model.net, "branch_layers"):
            vs = []
            for b in model.net.branch_layers:
                if hasattr(b, "branch_excitation") and b.branch_excitation is not None:
                    pass
                if hasattr(b, "branch_inhibition") and b.branch_inhibition is not None:
                    pass
            # for demonstration, we just do a single approach:
    # the simplest approach is to treat 'outputs' as "voltage" if possible
    return torch.mean((outputs - target_val) ** 2)


def generate_warmup_report(
    model, data_loader, pre_state, post_state, device, save_path
):
    """Generate a report on the differences before and after voltage warm-up.

    Args:
        model: The model being trained
        data_loader: DataLoader with a batch of data to test
        pre_state: Model state dict before warm-up
        post_state: Model state dict after warm-up
        device: Device to run model on
        save_path: Directory to save report files

    Returns:
        report_data: Dictionary with report metrics
    """

    report_data = {}

    # Create report directory
    warmup_report_dir = os.path.join(save_path, "warmup_report")
    os.makedirs(warmup_report_dir, exist_ok=True)

    # Get a batch of data for visualization
    x_batch, _ = next(iter(data_loader))
    x_batch = x_batch.to(device)

    # Save current model state
    current_state = deepcopy(model.state_dict())

    # Function to collect outputs and activations
    def collect_model_stats(state_dict):
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            outputs = model(x_batch)
            output_stats = {
                "mean": outputs.mean().item(),
                "std": outputs.std().item(),
                "min": outputs.min().item(),
                "max": outputs.max().item(),
                "histogram": outputs.detach().cpu().numpy().flatten(),
            }

            # Collect activations from each layer if they're available
            layer_activations = {}
            if hasattr(model, "net"):
                if hasattr(model.net, "branch_layers"):
                    # Store the original state of activation tracking
                    activation_tracking_state = {}

                    # Try to enable activation tracking if it exists
                    for i, layer in enumerate(model.net.branch_layers):
                        if hasattr(layer, "enable_branch_outputs"):
                            activation_tracking_state[i] = layer.enable_branch_outputs
                            layer.enable_branch_outputs = True

                    # Forward pass to collect activations
                    _ = model(x_batch)

                    # Collect branch activations if they exist
                    for i, layer in enumerate(model.net.branch_layers):
                        if hasattr(layer, "branch_activations"):
                            layer_activations[f"layer_{i}"] = {
                                "mean": layer.branch_activations.mean().item(),
                                "std": layer.branch_activations.std().item(),
                                "min": layer.branch_activations.min().item(),
                                "max": layer.branch_activations.max().item(),
                                "histogram": layer.branch_activations.detach()
                                .cpu()
                                .numpy()
                                .flatten(),
                            }

                    # Restore original state
                    for i, state in activation_tracking_state.items():
                        model.net.branch_layers[i].enable_branch_outputs = state

        return output_stats, layer_activations

    # Collect statistics before and after warm-up
    pre_output_stats, pre_activations = collect_model_stats(pre_state)
    post_output_stats, post_activations = collect_model_stats(post_state)

    # Calculate differences
    output_diff = {
        "mean_diff": post_output_stats["mean"] - pre_output_stats["mean"],
        "std_diff": post_output_stats["std"] - pre_output_stats["std"],
        "min_diff": post_output_stats["min"] - pre_output_stats["min"],
        "max_diff": post_output_stats["max"] - pre_output_stats["max"],
    }

    # Store report data
    report_data["pre_warmup"] = {
        "outputs": pre_output_stats,
        "activations": pre_activations,
    }
    report_data["post_warmup"] = {
        "outputs": post_output_stats,
        "activations": post_activations,
    }
    report_data["diff"] = {"outputs": output_diff}

    # Generate output distribution plots
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.hist(pre_output_stats["histogram"], bins=50, alpha=0.5, label="Pre-warmup")
    plt.hist(post_output_stats["histogram"], bins=50, alpha=0.5, label="Post-warmup")
    plt.axvline(x=0.5, color="r", linestyle="--", label="Target voltage (0.5)")
    plt.title("Output Distribution Before and After Warm-up")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 1, 2)
    for key in ["mean", "std", "min", "max"]:
        x = [0, 1]
        y = [pre_output_stats[key], post_output_stats[key]]
        plt.plot(x, y, marker="o", label=key)
    plt.axhline(y=0.5, color="r", linestyle="--", label="Target voltage (0.5)")
    plt.xticks([0, 1], ["Pre-Warmup", "Post-Warmup"])
    plt.title("Output Statistics Before and After Warm-up")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(warmup_report_dir, "output_distribution.png"))
    plt.close()

    # Generate activation comparison plots for each layer
    common_layers = set(pre_activations.keys()) & set(post_activations.keys())
    for layer in common_layers:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.hist(
            pre_activations[layer]["histogram"], bins=50, alpha=0.5, label="Pre-warmup"
        )
        plt.hist(
            post_activations[layer]["histogram"],
            bins=50,
            alpha=0.5,
            label="Post-warmup",
        )
        plt.title(f"{layer} Activation Distribution Before and After Warm-up")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2, 1, 2)
        for key in ["mean", "std", "min", "max"]:
            x = [0, 1]
            y = [pre_activations[layer][key], post_activations[layer][key]]
            plt.plot(x, y, marker="o", label=key)
        plt.xticks([0, 1], ["Pre-Warmup", "Post-Warmup"])
        plt.title(f"{layer} Activation Statistics Before and After Warm-up")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(warmup_report_dir, f"{layer}_activation.png"))
        plt.close()

    # Save report summary as text file
    with open(os.path.join(warmup_report_dir, "warmup_report_summary.txt"), "w") as f:
        f.write("Voltage Warm-up Report Summary\n")
        f.write("==============================\n\n")
        f.write("Output Statistics:\n")
        f.write(
            f"  Before warm-up: mean={pre_output_stats['mean']:.4f}, std={pre_output_stats['std']:.4f}, min={pre_output_stats['min']:.4f}, max={pre_output_stats['max']:.4f}\n"
        )
        f.write(
            f"  After warm-up: mean={post_output_stats['mean']:.4f}, std={post_output_stats['std']:.4f}, min={post_output_stats['min']:.4f}, max={post_output_stats['max']:.4f}\n"
        )
        f.write(
            f"  Change: mean_diff={output_diff['mean_diff']:.4f}, std_diff={output_diff['std_diff']:.4f}, min_diff={output_diff['min_diff']:.4f}, max_diff={output_diff['max_diff']:.4f}\n\n"
        )

        if common_layers:
            f.write("Layer Activation Statistics:\n")
            for layer in common_layers:
                pre = pre_activations[layer]
                post = post_activations[layer]
                f.write(f"  {layer}:\n")
                f.write(
                    f"    Before warm-up: mean={pre['mean']:.4f}, std={pre['std']:.4f}, min={pre['min']:.4f}, max={pre['max']:.4f}\n"
                )
                f.write(
                    f"    After warm-up: mean={post['mean']:.4f}, std={post['std']:.4f}, min={post['min']:.4f}, max={post['max']:.4f}\n"
                )
                f.write(
                    f"    Change: mean_diff={post['mean']-pre['mean']:.4f}, std_diff={post['std']-pre['std']:.4f}, min_diff={post['min']-pre['min']:.4f}, max_diff={post['max']-pre['max']:.4f}\n\n"
                )

    # Save numeric data as JSON for later analysis
    json_data = {
        "pre_warmup": {
            "outputs": {k: v for k, v in pre_output_stats.items() if k != "histogram"},
            "activations": {
                layer: {k: v for k, v in stats.items() if k != "histogram"}
                for layer, stats in pre_activations.items()
            },
        },
        "post_warmup": {
            "outputs": {k: v for k, v in post_output_stats.items() if k != "histogram"},
            "activations": {
                layer: {k: v for k, v in stats.items() if k != "histogram"}
                for layer, stats in post_activations.items()
            },
        },
        "diff": {"outputs": output_diff},
    }

    with open(os.path.join(warmup_report_dir, "warmup_data.json"), "w") as f:
        import json

        json.dump(json_data, f, indent=2)

    # Restore original model state
    model.load_state_dict(current_state)

    return report_data


class VoltageStabilizationTrainer(BaseTrainer):
    def __init__(
        self,
        optimizer,
        suppress_prints=False,
        print_every=10,
        warmup_epochs=1,
        stability_target_voltage=0.5,
    ):
        super().__init__(optimizer, suppress_prints, print_every)
        self.warmup_epochs = warmup_epochs
        self.stability_target_voltage = stability_target_voltage

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

        train_losses = []
        valid_losses = []
        best_loss = float("inf")

        # Store the model state before warm-up
        pre_warmup_state = deepcopy(model.state_dict())

        start_time = time.time()

        logger.info("=== Starting warm-up phase ===")
        for epoch in range(1, self.warmup_epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                self.optimizer.zero_grad()
                loss_val = voltage_warmup_loss(
                    model, xb, target_val=self.stability_target_voltage
                )
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            val_loss, _ = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[VoltageWarmup] epoch={epoch}/{self.warmup_epochs}, warmup_loss={epoch_train_loss:.4f}, valid_loss={val_loss:.4f}, elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f"warmup_epoch_{epoch}", training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "warmup_loss": epoch_train_loss,
                        "valid_loss": val_loss,
                    }
                )

        logger.info("=== Warm-up phase ended ===")

        # Store the model state after warm-up
        post_warmup_state = deepcopy(model.state_dict())

        # Print brief warmup statistics instead of generating a full report
        with torch.no_grad():
            model.eval()
            # Get a small batch of data for quick statistics
            sample_loader = DataLoader(
                train_data, batch_size=min(32, len(train_data)), shuffle=True
            )
            x_sample, _ = next(iter(sample_loader))
            x_sample = x_sample.to(device)

            # Get pre-warmup stats
            model.load_state_dict(pre_warmup_state)
            pre_outputs = model(x_sample)
            pre_mean = pre_outputs.mean().item()
            pre_std = pre_outputs.std().item()

            # Get post-warmup stats
            model.load_state_dict(post_warmup_state)
            post_outputs = model(x_sample)
            post_mean = post_outputs.mean().item()
            post_std = post_outputs.std().item()

            # Calculate changes
            mean_change = post_mean - pre_mean
            mean_pct_change = (mean_change / (abs(pre_mean) + 1e-8)) * 100
            std_change = post_std - pre_std
            std_pct_change = (std_change / (abs(pre_std) + 1e-8)) * 100

            # Print summary
            logger.info("=== Warm-up Effect Summary ===")
            logger.info(f"Target voltage: {self.stability_target_voltage:.4f}")
            logger.info(
                f"Output mean: {pre_mean:.4f} → {post_mean:.4f} (Δ: {mean_change:.4f}, {mean_pct_change:+.1f}%)"
            )
            logger.info(
                f"Output std: {pre_std:.4f} → {post_std:.4f} (Δ: {std_change:.4f}, {std_pct_change:+.1f}%)"
            )

            # Calculate distance to target
            pre_dist = abs(pre_mean - self.stability_target_voltage)
            post_dist = abs(post_mean - self.stability_target_voltage)
            dist_change = pre_dist - post_dist
            logger.info(
                f"Distance to target: {pre_dist:.4f} → {post_dist:.4f} (Improved by: {dist_change:.4f})"
            )
            logger.info("=============================")

            # Restore the state after warmup
            model.load_state_dict(post_warmup_state)

        remaining_epochs = epochs - self.warmup_epochs
        start_main_phase = time.time()
        for epoch in range(1, remaining_epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss_val = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                "accuracy"
            ]
            val_loss, val_acc = self._eval_dataset(
                model, valid_data, device, batch_size
            )
            train_losses.append(epoch_train_loss)
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_main_phase
                logger.info(
                    f"[VoltageStab Main] epoch={epoch}/{remaining_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f"main_epoch_{epoch}", training=True)
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": self.warmup_epochs + epoch,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )

        model.load_state_dict(best_state)
        results = {"train_losses": train_losses, "valid_losses": valid_losses}
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results


def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10):
    strategy = strategy.lower()
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_reversed":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_with_pretraining":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_reversed_with_pretraining":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(optimizer, suppress_prints, print_every)
    elif strategy == "voltage_stabilization":
        return VoltageStabilizationTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "mse":
        return TrainerRegression(optimizer, suppress_prints, print_every)
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)


# Add a function to train the simple MLP
def train_simple_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    train_ds,
    valid_ds,
    test_ds,
    epochs=20,
    batch_size=128,
    lr=0.001,
    run_save_path=None,
    device=None,
):
    """
    Train a simple MLP model with better performance.
    Returns the trained model and its accuracy.

    Args:
        input_dim (int): Input dimension
        hidden_dims (list): List of hidden layer dimensions
        output_dim (int): Output dimension (number of classes)
        train_ds, valid_ds, test_ds (Dataset): PyTorch datasets
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        run_save_path (str): Path to save model checkpoints
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        Tuple of (model, (train_acc, valid_acc, test_acc))
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import here to avoid circular imports
    from dendritic_modeling.models import SimpleClassifier

    # Create the model
    model = SimpleClassifier(input_dim, hidden_dims, output_dim)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    # Training loop
    best_valid_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                outputs = model(xb)
                loss = model.criterion(outputs, yb)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())

            if run_save_path:
                os.makedirs(run_save_path, exist_ok=True)
                save_path = os.path.join(run_save_path, "best_simple_mlp.pt")
                torch.save(best_state, save_path)

        # Log progress
        logger.info(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()

    test_acc = test_correct / test_total
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Return the model and its accuracy
    return model, (train_acc, best_valid_acc, test_acc)


# Function to train an MLP with a classification head for feature learning
def train_mlp_with_classification_head(
    feature_extractor,
    num_features,
    num_classes,
    train_ds,
    valid_ds,
    test_ds,
    epochs=20,
    patience=10,
    batch_size=128,
    lr=0.001,
    run_save_path=None,
    device=None,
):
    """
    Train an MLP with a classification head, then return the trained feature extractor part.

    Args:
        feature_extractor: The feature extraction part of the MLP
        num_features: The output dimension of the feature extractor
        num_classes: The number of classes for classification
        train_ds, valid_ds, test_ds: The datasets
        epochs, batch_size, lr: Training parameters
        run_save_path: Where to save the model and logs
        device: Device to use ('cuda' or 'cpu')

    Returns:
        The trained feature extractor and performance metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import here to avoid circular imports
    from dendritic_modeling.models import MLPWithClassificationHead

    # Create the model with classification head
    model = MLPWithClassificationHead(feature_extractor, num_features, num_classes)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    # Training loop
    best_valid_acc = 0.0
    best_state = None
    best_epoch = 0

    patience_counter = 0

    # Create the save directory if it doesn't exist
    if run_save_path:
        # Ensure the main directory exists
        os.makedirs(run_save_path, exist_ok=True)
        logger.info(f"Will save best model to {run_save_path} at the end of training")

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                outputs = model(xb)
                loss = model.criterion(outputs, yb)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total

        # Track best model but don't save yet
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            logger.info(
                f"New best model found at epoch {epoch} with validation accuracy: {valid_acc:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy")
                break

        # Log progress
        logger.info(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    "mlp_train_loss": train_loss,
                    "mlp_train_acc": train_acc,
                    "mlp_valid_loss": valid_loss,
                    "mlp_valid_acc": valid_acc,
                }
            )

    # Save only the best model at the end of training
    if best_state and run_save_path:
        save_path = os.path.join(run_save_path, "best_mlp_with_classification.pt")
        torch.save(best_state, save_path)
        logger.info(f"Saved best model (from epoch {best_epoch}) to {save_path}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        logger.info(
            f"Loaded best model from epoch {best_epoch} with validation accuracy: {best_valid_acc:.4f}"
        )

    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()

    test_acc = test_correct / test_total
    logger.info(f"Test accuracy with classification head: {test_acc:.4f}")

    # Return the feature extractor part (without the classification head)
    return model.feature_extractor, (train_acc, best_valid_acc, test_acc)


class InhibitoryHeadTrainer:
    """
    Custom trainer for the inhibitory head with different objective functions.
    """

    def __init__(
        self,
        model,
        optimizer,
        objective="classification",
        excit_head_model=None,
        alpha=0.5,
        suppress_prints=False,
        print_every=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.excit_head_model = excit_head_model
        self.alpha = alpha  # Weight for the diversity loss
        self.suppress = suppress_prints
        self.print_every = print_every

    def compute_classification_loss(self, x, y):
        """Standard classification loss"""
        return self.model.compute_loss(x, y)

    def compute_diversity_loss(self, x):
        """
        Diversity loss: encourages inhibitory head to learn features different from excitatory head
        by maximizing the cosine distance between their representations.
        """
        if self.excit_head_model is None:
            return torch.tensor(0.0, device=x.device)

        # Get representations from both heads
        with torch.no_grad():
            if hasattr(self.excit_head_model, "net"):
                excit_repr = self.excit_head_model.net(x)
            elif hasattr(self.excit_head_model, "network"):
                excit_repr = self.excit_head_model.network(x)
            else:
                excit_repr = self.excit_head_model(x)

        if hasattr(self.model, "net"):
            inhib_repr = self.model.net(x)
        elif hasattr(self.model, "network"):
            inhib_repr = self.model.network(x)
        else:
            inhib_repr = self.model(x)

        # Normalize the representations
        excit_repr = F.normalize(excit_repr, p=2, dim=1)
        inhib_repr = F.normalize(inhib_repr, p=2, dim=1)

        # Compute cosine similarity (we want to minimize this, or maximize negative)
        similarity = torch.sum(excit_repr * inhib_repr, dim=1).mean()

        # Return negative similarity to maximize distance
        return similarity

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        epochs=100,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=False,
        save_path=None,
    ):
        """
        Train the inhibitory head with the specified objective.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if self.excit_head_model is not None:
            self.excit_head_model = self.excit_head_model.to(device)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=shuffle
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                # Compute the appropriate loss based on the objective
                if self.objective == "classification":
                    loss = self.compute_classification_loss(xb, yb)
                elif self.objective == "diversity":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
                elif self.objective == "combined":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
                else:
                    raise ValueError(f"Unknown objective: {self.objective}")

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Evaluate on validation set
            model.eval()
            val_loss = 0.0

            if valid_data is not None:
                valid_loader = torch.utils.data.DataLoader(
                    valid_data, batch_size=batch_size
                )
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)

                        # Use classification loss for validation
                        loss = self.compute_classification_loss(xb, yb)
                        val_loss += loss.item()

                val_loss /= len(valid_loader)
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    if save_path and os.path.exists(save_path):
                        torch.save(best_state, os.path.join(save_path, "best_model.pt"))

            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                if valid_data is not None:
                    logger.info(
                        f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                        f"valid_loss={val_loss:.4f}, elapsed={elapsed:.1f}s"
                    )
                else:
                    logger.info(
                        f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                        f"elapsed={elapsed:.1f}s"
                    )

            if wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                }
                if valid_data is not None:
                    log_dict["valid_loss"] = val_loss
                wandb.log(log_dict)

        if load_best_state_dict and valid_data is not None:
            model.load_state_dict(best_state)

        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses if valid_data is not None else [],
            "best_loss": best_loss if valid_data is not None else None,
        }

        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                from dendritic_modeling.plot_utils import plot_NLL_loss_curves

                plot_NLL_loss_curves(
                    train_losses,
                    valid_losses if valid_data is not None else [],
                    epochs,
                    save_path,
                )

        return results


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.

    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(
        f"Transferring weights from MLP to {head_type} head of InputNetTransform"
    )

    # Extract the MLP layers
    if hasattr(mlp_net, "layers"):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, "net") and hasattr(mlp_net.net, "layers"):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(
            f"Could not extract layers from MLP model. Using random initialization for {head_type} head."
        )
        return

    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)

    target_head = (
        input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    )
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)

    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if (
                i < len(mlp_layers) - 1
            ):  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, "excit_pre_w"):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(
                        f"Unexpected MLP layer structure. Using random initialization for shared layer {i}."
                    )

    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, "excit_pre_w"):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(
                f"Unexpected MLP layer structure. Using random initialization for {head_type} head."
            )

    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.

    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")

    # Get network modules
    mlp_modules = list(simple_mlp.net.network.children())
    mlp_linear_modules = [m for m in mlp_modules if isinstance(m, torch.nn.Linear)]

    # Get shared backbone modules of InputNetTransform
    shared_modules = []
    for module in input_net_transform.shared:
        if isinstance(module, torch.nn.Linear):
            shared_modules.append(module)

    # Get head modules
    exc_head_modules = []
    for module in input_net_transform.head_exc:
        if isinstance(module, torch.nn.Linear):
            exc_head_modules.append(module)

    inh_head_modules = []
    for module in input_net_transform.head_inh:
        if isinstance(module, torch.nn.Linear):
            inh_head_modules.append(module)

    # Transfer weights for the shared backbone (from early layers of MLP)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:  # All but the last MLP layer
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")

    # Transfer weights for heads (from last layer of MLP)
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        # Use the second-to-last layer for the heads (since the last layer is the classifier)
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]

        # Transfer to excitatory headß
        exc_head_modules[0].weight.data = source_layer.weight.data[
            : exc_head_modules[0].weight.shape[0], :
        ].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[
            : exc_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to excitatory head")

        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[
            : inh_head_modules[0].weight.shape[0], :
        ].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[
            : inh_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to inhibitory head")

    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")

    def train_freeze_layers_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train layers sequentially from soma to input (previously named train_freeze_layers)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )
        n_layers = len(model.net.layers)
        logger.info(
            f"train_freeze_layers_by_layer: found {n_layers} layers. Training them sequentially."
        )
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_layers * epochs_per_layer
        epoch_counter = 0
        start_time = time.time()
        for layer_idx in range(n_layers):
            for name, param in model.named_parameters():
                param.requires_grad = False
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True
            logger.info(
                f"Training only layer {layer_idx} for {epochs_per_layer} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            for e in range(1, epochs_per_layer + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())
                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-layers-by-layer] layer={layer_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_layer": layer_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
        logger.info(
            "Done freeze-layers-by-layer. Unfreezing all layers and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)
        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )
        return final_results

    # Alias for backward compatibility
    train_freeze_layers = train_freeze_layers_by_layer

    def train_freeze_layers_reversed_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train layers sequentially from input to soma (previously named train_freeze_layers_reversed)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )
        n_layers = len(model.net.layers)
        logger.info(
            f"train_freeze_layers_reversed_by_layer: found {n_layers} layers. Training them sequentially from input to soma."
        )
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_layers * epochs_per_layer
        epoch_counter = 0
        start_time = time.time()
        # Iterate in reverse order (n_layers-1 to 0), training from input towards soma
        for layer_idx in range(n_layers - 1, -1, -1):
            for name, param in model.named_parameters():
                param.requires_grad = False
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True
            logger.info(
                f"Training only layer {layer_idx} (farther from soma) for {epochs_per_layer} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            for e in range(1, epochs_per_layer + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())
                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-layers-reversed-by-layer] layer={layer_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_layer_reversed": layer_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
        logger.info(
            "Done freeze-layers-reversed-by-layer. Unfreezing all layers and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)
        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )
        return final_results

    # Alias for backward compatibility
    train_freeze_layers_reversed = train_freeze_layers_reversed_by_layer

    def train_freeze_branches(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train individual dendritic branches sequentially from soma to input (distal).
        This method trains each branch in the dendritic tree separately, starting from the branches
        closest to the soma and moving outward."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Get the branch layers from the model
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"train_freeze_branches: found {n_branches} branch layers. Training them sequentially from soma to input."
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_branches * epochs_per_branch
        epoch_counter = 0
        start_time = time.time()

        # Train branches from soma (near) to distal (far) - typically index 0 is closest to soma
        for branch_idx in range(n_branches):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        logger.info(
            "Done freeze-branches. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        return final_results

    def train_freeze_branches_reversed(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train individual dendritic branches sequentially from input (distal) to soma.
        This method trains each branch in the dendritic tree separately, starting from the branches
        farthest from the soma and moving inward."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Get the branch layers from the model
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"train_freeze_branches_reversed: found {n_branches} branch layers. Training them sequentially from input to soma."
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_branches * epochs_per_branch
        epoch_counter = 0
        start_time = time.time()

        # Train branches from distal (far) to soma (near) - reverse order
        for branch_idx in range(n_branches - 1, -1, -1):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-reversed] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_reversed": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        logger.info(
            "Done freeze-branches-reversed. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        return final_results

    def train_freeze_branches_with_pretraining(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=10,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train with a three-phase approach:
        1. Initial full network training (all parameters unfrozen) for pretrain_epochs
        2. Branch-by-branch training from soma to input (freezing all branches except the current one)
        3. Final full network fine-tuning
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Phase 1: Initial full network training
        logger.info(
            f"Starting initial full network training for {pretrain_epochs} epochs"
        )
        initial_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=False,
        )

        # Save the pretrained model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            pretrain_path = os.path.join(save_path, "pretrained_model.pt")
            torch.save(model.state_dict(), pretrain_path)
            logger.info(f"Saved pretrained model to {pretrain_path}")

        # Phase 2: Branch-by-branch training
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"Starting branch-by-branch training (soma to input): found {n_branches} branches."
        )
        train_losses = initial_results.get("train_losses", [])
        valid_losses = initial_results.get("valid_losses", [])
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = (
            pretrain_epochs + (n_branches * epochs_per_branch) + final_tune_epochs
        )
        epoch_counter = pretrain_epochs
        start_time = time.time()

        # Train branches from soma (near) to distal (far)
        for branch_idx in range(n_branches):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-with-pretraining] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_with_pretraining": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        # Phase 3: Final full-network fine-tuning
        logger.info(
            "Done branch-by-branch training. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        # Combine all results
        final_results["train_losses"] = train_losses + final_results.get(
            "train_losses", []
        )
        final_results["valid_losses"] = valid_losses + final_results.get(
            "valid_losses", []
        )

        return final_results

    def train_freeze_branches_reversed_with_pretraining(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=10,
        epochs_per_branch=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """Train with a three-phase approach:
        1. Initial full network training (all parameters unfrozen) for pretrain_epochs
        2. Branch-by-branch training from input to soma (freezing all branches except the current one)
        3. Final full network fine-tuning
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Check if model has the right structure
        if (
            not hasattr(model, "net")
            or not hasattr(model.net, "layers")
            or not hasattr(model.net.layers[0], "excitatory_cells")
            or not hasattr(model.net.layers[0].excitatory_cells, "branch_layers")
        ):
            logger.info(
                "Model structure not compatible with branch-wise training. Falling back to normal train."
            )
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + epochs_per_branch,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )

        # Phase 1: Initial full network training
        logger.info(
            f"Starting initial full network training for {pretrain_epochs} epochs"
        )
        initial_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=False,
        )

        # Save the pretrained model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            pretrain_path = os.path.join(save_path, "pretrained_model.pt")
            torch.save(model.state_dict(), pretrain_path)
            logger.info(f"Saved pretrained model to {pretrain_path}")

        # Phase 2: Branch-by-branch training from input to soma
        branch_layers = model.net.layers[0].excitatory_cells.branch_layers
        n_branches = len(branch_layers)

        logger.info(
            f"Starting branch-by-branch training (input to soma): found {n_branches} branches."
        )
        train_losses = initial_results.get("train_losses", [])
        valid_losses = initial_results.get("valid_losses", [])
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = (
            pretrain_epochs + (n_branches * epochs_per_branch) + final_tune_epochs
        )
        epoch_counter = pretrain_epochs
        start_time = time.time()

        # Train branches from distal (far) to soma (near) - reverse order
        for branch_idx in range(n_branches - 1, -1, -1):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only parameters for the current branch
            target_str = f"excitatory_cells.branch_layers.{branch_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(
                f"Training only branch {branch_idx} for {epochs_per_branch} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )

            for e in range(1, epochs_per_branch + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())

                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())

                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-branches-reversed-with-pretraining] branch={branch_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_branch_reversed_with_pretraining": branch_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )

        # Phase 3: Final full-network fine-tuning
        logger.info(
            "Done branch-by-branch training. Unfreezing all parameters and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )

        # Combine all results
        final_results["train_losses"] = train_losses + final_results.get(
            "train_losses", []
        )
        final_results["valid_losses"] = valid_losses + final_results.get(
            "valid_losses", []
        )

        return final_results
