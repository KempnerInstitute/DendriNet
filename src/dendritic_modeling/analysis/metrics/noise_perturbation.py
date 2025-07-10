"""
Noise perturbation analysis module.

This module contains classes for evaluating model robustness under various noise conditions
including uniform and Gaussian noise perturbations.
"""

from typing import Callable, Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import NoisePerturbationAnalysisParams
from dendritic_modeling.datasets.augmentation.noise import (
    add_gaussian_noise,
    add_uniform_noise,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict


class NoisePerturbationAnalyzer:
    """
    Analyzer for evaluating model robustness under noise perturbations.

    Supports uniform and Gaussian noise at various magnitudes, evaluating
    accuracy and categorical log-likelihood on perturbed test data.
    """

    def __init__(self, params: NoisePerturbationAnalysisParams):
        """
        Initialize the noise perturbation analyzer.

        Args:
            uniform_noise: Whether to evaluate uniform noise robustness
            uniform_magnitudes: list of uniform noise magnitudes to test
            gaussian_noise: Whether to evaluate Gaussian noise robustness
            gaussian_sdevs: list of Gaussian noise standard deviations to test
            n_samples: Number of noise samples to generate per test point
            accuracy: Whether to compute accuracy on noisy data
            categorical_loglikelihood: Whether to compute categorical log-likelihood on noisy data
        """
        self.uniform_noise = getattr(params, "uniform_noise", False)
        self.uniform_magnitudes = getattr(params, "uniform_magnitudes", [0.1, 0.2])
        self.gaussian_noise = getattr(params, "gaussian_noise", False)
        self.gaussian_sdevs = getattr(params, "gaussian_sdevs", [0.1, 0.2])
        self.n_samples = getattr(params, "n_samples", 10)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze model robustness under noise perturbations.

        Args:
            model: The model to evaluate
            test_dataset: Test dataset to perturb with noise
            device: Device to run evaluation on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing noise robustness results if save_path is None, otherwise None
        """
        model = model.to(device)
        x, y = test_dataset[:]
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)
        y = y[None, ...].expand(self.n_samples, *y.shape)

        noise_results = {}

        def _evaluate_metric(
            noise_type: str,
            noise_level: float,
            metric_func: Callable,
            metric_key: str,
            noisy_test_ds: torch.utils.data.Dataset
        ):
            metric_tensor: torch.Tensor = metric_func(
                model,
                test_ds=noisy_test_ds,
                reduce_dim=1,
                move_device=False
            )[-1]
            metric_list = metric_tensor.cpu().tolist()
            noise_results[noise_type][f"{noise_level}"][metric_key] = metric_list

        if self.uniform_noise:
            noise_results["uniform"] = {}
            for mag in self.uniform_magnitudes:
                noise_results["uniform"][f"{mag}"] = {}
                noisy_x = add_uniform_noise(x, spread=mag, n_samples=self.n_samples)
                noisy_test_ds = TensorDataset(noisy_x, y)
                if self.accuracy:
                    _evaluate_metric(
                        noise_type="uniform",
                        noise_level=mag,
                        metric_func=evaluate_accuracy,
                        metric_key="accuracy",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.auc:
                    _evaluate_metric(
                        noise_type="uniform",
                        noise_level=mag,
                        metric_func=evaluate_auc,
                        metric_key="auc",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.categorical_ll:
                    _evaluate_metric(
                        noise_type="uniform",
                        noise_level=mag,
                        metric_func=evaluate_categorical_loglikelihood,
                        metric_key="categorical_ll",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.mse:
                    _evaluate_metric(
                        noise_type="uniform",
                        noise_level=mag,
                        metric_func=evaluate_mse,
                        metric_key="mse",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.cosine_similarity:
                    _evaluate_metric(
                        noise_type="uniform",
                        noise_level=mag,
                        metric_func=evaluate_cosine_similarity,
                        metric_key="cosine_similarity",
                        noisy_test_ds=noisy_test_ds
                    )

        if self.gaussian_noise:
            noise_results["gaussian"] = {}
            for sdev in self.gaussian_sdevs:
                noise_results["gaussian"][f"{sdev}"] = {}
                noisy_x = add_gaussian_noise(x, std=sdev, n_samples=self.n_samples)
                noisy_test_ds = TensorDataset(noisy_x, y)
                if self.accuracy:
                    _evaluate_metric(
                        noise_type="gaussian",
                        noise_level=sdev,
                        metric_func=evaluate_accuracy,
                        metric_key="accuracy",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.auc:
                    _evaluate_metric(
                        noise_type="gaussian",
                        noise_level=sdev,
                        metric_func=evaluate_auc,
                        metric_key="auc",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.categorical_ll:
                    _evaluate_metric(
                        noise_type="gaussian",
                        noise_level=sdev,
                        metric_func=evaluate_categorical_loglikelihood,
                        metric_key="categorical_ll",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.mse:
                    _evaluate_metric(
                        noise_type="gaussian",
                        noise_level=sdev,
                        metric_func=evaluate_mse,
                        metric_key="mse",
                        noisy_test_ds=noisy_test_ds
                    )
                if self.cosine_similarity:
                    _evaluate_metric(
                        noise_type="gaussian",
                        noise_level=sdev,
                        metric_func=evaluate_cosine_similarity,
                        metric_key="cosine_similarity",
                        noisy_test_ds=noisy_test_ds
                    )

        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results
