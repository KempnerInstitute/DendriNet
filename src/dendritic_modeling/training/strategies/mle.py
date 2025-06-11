"""
Maximum Likelihood Estimation (MLE) training strategy.

This module contains the TrainerMLE class which implements various MLE-based 
training strategies for dendritic networks, including standard training and 
sophisticated branch-wise training approaches.
"""

import os
import time
from copy import deepcopy
from typing import Optional
import torch
import wandb
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from .base import BaseTrainer, compute_metrics_classification


class TrainerMLE(BaseTrainer):
    """
    Maximum Likelihood Estimation trainer with advanced branch-wise training capabilities.
    
    This trainer implements:
    - Standard MLE training with negative log-likelihood loss
    - Branch-wise training from soma to input (distal)
    - Branch-wise training from input (distal) to soma
    - Multi-phase training with pretraining and branch-wise fine-tuning
    
    All methods support early stopping, checkpointing, and analysis integration.
    """
    
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
        """
        Standard MLE training with negative log-likelihood loss.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            grad_clip_value: Gradient clipping value
            epochs: Number of training epochs
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            load_best_state_dict: Load best model state at end
            best_loss: Initial best loss for comparison
            early_stopping: Enable early stopping
            patience: Early stopping patience
            analysis_manager: Analysis manager for periodic analysis
            plot_losses: Whether to plot loss curves
            save_path: Path to save model checkpoints
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and metrics
        """
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
        """
        Train individual dendritic branches sequentially from soma to input (distal).
        
        This method trains each branch in the dendritic tree separately, starting from the branches
        closest to the soma and moving outward. After training all branches individually, performs
        final full-network fine-tuning.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            epochs_per_branch: Number of epochs to train each branch
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot loss curves
            save_path: Path to save model checkpoints
            final_tune_epochs: Number of final full-network tuning epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results
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
        """
        Train individual dendritic branches sequentially from input (distal) to soma.
        
        This method trains each branch in the dendritic tree separately, starting from the branches
        farthest from the soma and moving inward. After training all branches individually, performs
        final full-network fine-tuning.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            epochs_per_branch: Number of epochs to train each branch
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot loss curves
            save_path: Path to save model checkpoints
            final_tune_epochs: Number of final full-network tuning epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results
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
        """
        Train with a three-phase approach.
        
        Phase 1: Initial full network training (all parameters unfrozen) for pretrain_epochs
        Phase 2: Branch-by-branch training from soma to input (freezing all branches except the current one)
        Phase 3: Final full network fine-tuning
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            pretrain_epochs: Number of pretraining epochs
            epochs_per_branch: Number of epochs to train each branch
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot loss curves
            save_path: Path to save model checkpoints
            final_tune_epochs: Number of final full-network tuning epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing combined training results
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
        """
        Train with a three-phase approach using reversed branch order.
        
        Phase 1: Initial full network training (all parameters unfrozen) for pretrain_epochs
        Phase 2: Branch-by-branch training from input to soma (freezing all branches except the current one)
        Phase 3: Final full network fine-tuning
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            pretrain_epochs: Number of pretraining epochs
            epochs_per_branch: Number of epochs to train each branch
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot loss curves
            save_path: Path to save model checkpoints
            final_tune_epochs: Number of final full-network tuning epochs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing combined training results
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


__all__ = ["TrainerMLE"] 