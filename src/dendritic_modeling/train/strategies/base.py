"""
Base training strategies and utilities.
"""

import time
from copy import deepcopy
from typing import Optional
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dendritic_modeling import logger


def compute_metrics_classification(preds, labels):
    """Compute classification metrics from predictions and labels."""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(
            labels_np, preds_np, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels_np, preds_np, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(labels_np, preds_np, average="weighted", zero_division=0),
    }


class CustomWeightDecayOptimizer:
    """Custom optimizer wrapper that applies weight decay before optimization step."""
    
    def __init__(self, model, optimizer, weight_decay=0.1):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        lr = self.optimizer.param_groups[0]["lr"]
        self.model.decay_weights(weight_decay=lr * self.weight_decay)
        self.optimizer.step()


class BaseTrainer:
    """Base class for all training strategies."""
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def _eval_dataset(self, model, dataset, device, batch_size=256):
        """Evaluate model on a dataset and return loss and accuracy."""
        if dataset is None or len(dataset) == 0:
            return 0.0, 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        preds_list = []
        labels_list = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                total_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
        total_loss /= len(loader)
        preds_cat = torch.cat(preds_list, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)
        metrics = compute_metrics_classification(preds_cat, labels_cat)
        return total_loss, metrics["accuracy"] 