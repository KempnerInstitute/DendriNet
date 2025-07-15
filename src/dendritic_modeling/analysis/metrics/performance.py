"""
Performance metrics analysis module.

This module contains classes for evaluating model performance including
accuracy, categorical log-likelihood, and mean squared error.
"""

from typing import Callable, Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import PerformanceAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict

import os
import logging


class PerformanceAnalyzer:
    """
    Analyzer for evaluating model performance on various metrics.

    Supports accuracy, categorical log-likelihood, and MSE evaluation
    on train, validation, and test datasets.
    """

    def __init__(self, params: PerformanceAnalysisParams):
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_loglikelihood = getattr(
            params, "categorical_loglikelihood", False
        )
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

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

        def _dataset_to_device(
            dataset: Optional[torch.utils.data.Dataset], device: str = "cpu"
        ):
            if dataset is not None:
                inputs, labels = dataset[:]
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)
                return TensorDataset(inputs, labels)
            return None

        results = {}

        def _evaluate_metric(metric_func: Callable, metric_key: str):
            results[metric_key] = {}
            train_metric, valid_metric, test_metric = metric_func(
                model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                move_device=False,
            )
            if train_ds is not None:
                results[metric_key]["train"] = train_metric
            if valid_ds is not None:
                results[metric_key]["valid"] = valid_metric
            if test_ds is not None:
                results[metric_key]["test"] = test_metric

        train_ds = _dataset_to_device(train_ds, device)
        valid_ds = _dataset_to_device(valid_ds, device)
        test_ds = _dataset_to_device(test_ds, device)

        if self.accuracy:
            _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")

        if self.auc:
            _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")

        if self.categorical_loglikelihood:
            _evaluate_metric(
                metric_func=evaluate_categorical_loglikelihood,
                metric_key="categorical_loglikelihood",
            )

        if self.mse:
            _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")

        if self.cosine_similarity:
            _evaluate_metric(
                metric_func=evaluate_cosine_similarity, metric_key="cosine_similarity"
            )

        if save_path is not None:
            save_dict(results, save_path, filename)

            # Generate performance plots
            try:
                import matplotlib.pyplot as plt
                from dendritic_modeling.plotting.base import BarPlotter, PlotConfig

                plot_save_path = os.path.join(save_path, "plots")
                os.makedirs(plot_save_path, exist_ok=True)

                # Create bar plot of metrics across datasets
                config = PlotConfig()
                plotter = BarPlotter(config)

                for metric_name, metric_values in results.items():
                    if isinstance(metric_values, dict) and len(metric_values) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))

                        datasets = list(metric_values.keys())
                        values = list(metric_values.values())

                        bars = ax.bar(datasets, values)
                        ax.set_xlabel("Dataset")
                        ax.set_ylabel(metric_name.replace("_", " ").title())
                        ax.set_title(f"Performance: {metric_name.replace('_', ' ').title()}")

                        # Add value labels on bars
                        for bar, val in zip(bars, values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.3f}', ha='center', va='bottom')

                        plt.tight_layout()
                        plot_path = os.path.join(plot_save_path, f"{filename}_{metric_name}.png")
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()

            except Exception as e:
                logging.warning(f"Could not generate performance plots: {e}")
        else:
            return results
