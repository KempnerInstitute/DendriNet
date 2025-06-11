"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics_classification(preds, labels):
    """
    Compute classification metrics including accuracy, precision, recall, and F1-score.
    
    Args:
        preds: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1_score
    """
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


class BaseTrainer:
    """
    Base class for all training strategies.
    
    This class provides common functionality shared across all training strategies,
    including dataset evaluation and basic initialization.
    
    Args:
        optimizer: PyTorch optimizer instance
        suppress_prints: Whether to suppress print statements during training
        print_every: Print training progress every N epochs
    """
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def _eval_dataset(self, model, dataset, device, batch_size=256) -> Tuple[float, float]:
        """
        Evaluate model performance on a dataset.
        
        Args:
            model: The model to evaluate
            dataset: Dataset to evaluate on
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
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


__all__ = ["BaseTrainer", "compute_metrics_classification"] 