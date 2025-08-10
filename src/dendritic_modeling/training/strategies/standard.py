"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

import logging
import os
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Tuple, List

import torch
import torch.profiler as torch_profiler
import torch.utils.data
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from dendritic_modeling.training.loss.functions import get_loss_function
from dendritic_modeling.config.training import (
    OptimizerConfig,
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting import plot_loss_curves
from dendritic_modeling.training.loss.functions import LossFunction
from dendritic_modeling import logger


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: str,
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value=5,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.loss_function: LossFunction = get_loss_function(loss_function)
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grad_clip_value = grad_clip_value
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager

        # TEST 1B: Add expensive config instantiation from iss294
        regularization_config = kwargs.get('regularization_config')
        pruning_config = kwargs.get('pruning_config') 
        reporting_config = kwargs.get('reporting_config')
        
        # Initialize enhanced feature configs with safe defaults (EXPENSIVE?)
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        self.reporting_config = reporting_config or ReportingConfig()
        
        self.filename_prefix = ""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _dataloaders(
        self, 
        train_data: torch.utils.data.Dataset, 
        valid_data: torch.utils.data.Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        valid_loader = DataLoader(
            valid_data, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        return train_loader, valid_loader

    def _initialize_attributes(
        self, model: BaseModel, total_epochs: int
    ) -> None:
        if not hasattr(self, "device"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self.device)
        if not hasattr(self, "best_state_dict"):
            self.best_state_dict = deepcopy(model.state_dict())
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "epoch_counter"):
            self.epoch_counter = 0
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "total_epochs"):
            self.total_epochs = total_epochs
        self.patience_counter = 0
    
    def train(
        self, 
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
    ) -> dict:
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        
        train_losses = []
        valid_losses = []
        
        self._initialize_attributes(model, self.epochs)

        train_losses, valid_losses = self._run_training_loop(
            n_epochs=self.epochs,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_losses=train_losses,
            valid_losses=valid_losses,
            logger_info="[Standard] ",
            wandb_info={}
        )
        
        logger.info(
            f"Best epoch: {self.best_epoch}, "
            f"Best loss: {self.best_loss:.4f}"
        )
        if self.load_best_state_dict:
            model.load_state_dict(self.best_state_dict)
            logger.info("Final model loaded from best checkpoint")
        
        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }
        
        self._loss_plotting(train_losses, valid_losses)
        
        return results
    
    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: List[float],
        valid_losses: List[float],
        logger_info: str = "",
        wandb_info: dict = {}
    ) -> Tuple[List[float], List[float]]:
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses = self._run_epoch(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                logger_info=logger_info,
                wandb_info=wandb_info
            )
            
            if self._check_early_stopping():
                break
        
        return train_losses, valid_losses
    
    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: List[float],
        valid_losses: List[float],
        logger_info: str = "",
        wandb_info: dict = {}
    ) -> Tuple[List[float], List[float]]:
        self.epoch_counter += 1

        train_loss = self._epoch_train(model, train_loader)
        train_losses.append(train_loss)
        valid_loss = self._epoch_valid(model, valid_loader)
        valid_losses.append(valid_loss)
        
        self._update_best(model, valid_loss)
        self._analyze()
        self._tracking(train_loss, valid_loss, logger_info, wandb_info)
        return train_losses, valid_losses
    
    def _epoch_train(
        self,
        model: BaseModel,
        train_loader: DataLoader
    ) -> float:
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch: torch.Tensor
            y_batch: torch.Tensor
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_value_(model.parameters(), self.grad_clip_value)
            self.optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        return train_loss

    def _epoch_valid(
        self,
        model: BaseModel,
        valid_loader: DataLoader
    ) -> float:
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch: torch.Tensor
                y_batch: torch.Tensor
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                valid_loss += loss.item()
            
        valid_loss /= len(valid_loader)
        return valid_loss
    
    def _update_best(
        self,
        model: BaseModel,
        valid_loss: float
    ) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_dict = deepcopy(model.state_dict())
            self.patience_counter = 0
            if self.save_path and os.path.exists(self.save_path):
                torch.save(
                    self.best_state_dict, 
                    os.path.join(
                        self.save_path, 
                        f"{self.filename_prefix}best_model.pt"
                    )
                )
        else:
            self.patience_counter += 1
    
    def _analyze(self):
        if self.analysis_manager is not None:
            # Conditional import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager
            if isinstance(self.analysis_manager, AnalysisManager):
                self.analysis_manager.run_analysis(
                    filename=f"epoch{self.epoch_counter}", 
                    training=True
                )
    
    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        wandb_info: dict = {}
    ) -> None:
        elapsed = time.time() - self.start_time
        if (
            not self.suppress_prints 
            and (self.epoch_counter % self.print_every == 0)
        ):
            logger_info = logger_info + \
                f"epoch={self.epoch_counter}/{self.total_epochs}, " + \
                f"train_loss={train_loss:.4f}, " + \
                f"valid_loss={valid_loss:.4f}, " + \
                f"elapsed={elapsed:.1f}s"
            logger.info(logger_info)
        
        if wandb.run is not None:
            wandb_info["epoch"] = self.epoch_counter
            wandb_info["train_loss"] = train_loss
            wandb_info["valid_loss"] = valid_loss
            wandb_info["elapsed"] = elapsed
            wandb.log(wandb_info)
    
    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered "
                    f"after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self,
        train_losses: List[float],
        valid_losses: List[float]
    ) -> None:
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                plot_loss_curves(
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    loss_name=self.loss_function._loss_name,
                    save_dir=self.save_path,
                    filename_prefix=self.filename_prefix
                )


__all__ = ["Trainer"] 
