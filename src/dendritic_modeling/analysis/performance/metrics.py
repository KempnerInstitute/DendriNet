"""
Performance metrics analysis module.

This module contains classes for evaluating model performance including
accuracy, categorical log-likelihood, and mean squared error.
"""

from typing import Optional
import torch
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import (
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    evaluate_mse,
    save_dict,
)
from dendritic_modeling.config.analysis import PerformanceAnalysisParams


class PerformanceAnalyzer:
    """
    Analyzer for evaluating model performance on various metrics.
    
    Supports accuracy, categorical log-likelihood, and MSE evaluation
    on train, validation, and test datasets.
    """
    
    def __init__(self, params: PerformanceAnalysisParams):
        self.accuracy = params.accuracy
        self.categorical_loglikelihood = params.categorical_loglikelihood
        self.mse = params.mse

    def analyze(
        self,
        model: BaseModel,
        train_ds: Optional[torch.utils.data.Dataset] = None,
        valid_ds: Optional[torch.utils.data.Dataset] = None,
        test_ds: Optional[torch.utils.data.Dataset] = None,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze model performance on specified datasets.
        
        Args:
            model: The model to evaluate
            train_ds: Training dataset (optional)
            valid_ds: Validation dataset (optional)
            test_ds: Test dataset (optional)
            device: Device to run evaluation on
            save_path: Path to save results (optional)
            filename: Filename for saved results
            
        Returns:
            Dictionary containing results if save_path is None, otherwise None
        """
        results = {}
        
        if self.accuracy:
            results["accuracy"] = {}
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["accuracy"]["train"] = train_acc
            if valid_ds is not None:
                results["accuracy"]["valid"] = valid_acc
            if test_ds is not None:
                results["accuracy"]["test"] = test_acc

        if self.categorical_loglikelihood:
            results["categorical_loglikelihood"] = {}
            train_catll, valid_catll, test_catll = evaluate_categorical_loglikelihood(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["categorical_loglikelihood"]["train"] = train_catll
            if valid_ds is not None:
                results["categorical_loglikelihood"]["valid"] = valid_catll
            if test_ds is not None:
                results["categorical_loglikelihood"]["test"] = test_catll

        if self.mse:
            results["mse"] = {}
            train_mse, valid_mse, test_mse = evaluate_mse(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["mse"]["train"] = train_mse
            if valid_ds is not None:
                results["mse"]["valid"] = valid_mse
            if test_ds is not None:
                results["mse"]["test"] = test_mse

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results 