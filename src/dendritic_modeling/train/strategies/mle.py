"""
Maximum Likelihood Estimation (MLE) training strategies.
"""

import os
import time
from copy import deepcopy
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from .base import BaseTrainer, compute_metrics_classification


class TrainerMLE(BaseTrainer):
    """Classic maximum likelihood estimation trainer with backpropagation."""
    
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
        """Train individual dendritic branches sequentially from soma to input (distal)."""
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

        logger.info("Done freeze-branches. Unfreezing all parameters and loading best state.")
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        logger.info(f"Starting final full-network tuning for {final_tune_epochs} epochs.")
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


class TrainOnlyMReactivation(BaseTrainer):
    """Trainer that only trains on M (motor) reactivation patterns."""
    
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
                
                # Only train on M reactivation patterns
                motor_mask = (yb == 2)  # Assuming M class is labeled as 2
                if motor_mask.sum() == 0:
                    continue
                    
                xb_motor = xb[motor_mask]
                yb_motor = yb[motor_mask]
                
                nll = model.compute_loss(xb_motor, yb_motor)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                
                p = model.predict(xb_motor)
                preds_list.append(p.cpu())
                labels_list.append(yb_motor.cpu())

            if len(preds_list) == 0:
                continue
                
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            train_losses.append(epoch_train_loss)
            
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())

            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)

            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[M-only Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                    "best_epoch": best_epoch,
                })

        logger.info(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")
        if load_best_state_dict:
            model.load_state_dict(best_state)
            logger.info("Final model loaded from best checkpoint")

        return {
            "best epoch": best_epoch,
            "best loss": best_loss,
            "best state dict": best_state,
            "train losses": train_losses,
            "valid losses": valid_losses,
        } 