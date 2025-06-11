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