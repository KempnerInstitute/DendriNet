"""
Noise perturbation analysis module.

This module contains classes for evaluating model robustness under various noise conditions
including uniform and Gaussian noise perturbations.
"""

from typing import Optional
import torch
from torch.utils.data import TensorDataset
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import (
    add_gaussian_noise,
    add_uniform_noise,
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    save_dict,
)
from dendritic_modeling.config import NoisePerturbationAnalysisParams


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
            uniform_magnitudes: List of uniform noise magnitudes to test
            gaussian_noise: Whether to evaluate Gaussian noise robustness
            gaussian_sdevs: List of Gaussian noise standard deviations to test
            n_samples: Number of noise samples to generate per test point
            accuracy: Whether to compute accuracy on noisy data
            categorical_loglikelihood: Whether to compute categorical log-likelihood on noisy data
        """
        self.uniform_noise = getattr(params, "uniform_noise", True)
        self.uniform_magnitudes = getattr(params, "uniform_magnitudes", [0.1, 0.2])
        self.gaussian_noise = getattr(params, "gaussian_noise", True)
        self.gaussian_sdevs = getattr(params, "gaussian_sdevs", [0.1, 0.2])
        self.n_samples = getattr(params, "n_samples", 10)
        self.accuracy = getattr(params, "accuracy", True)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", True)

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

        if self.uniform_noise:
            noise_results["uniform"] = {}
            for mag in self.uniform_magnitudes:
                noise_results["uniform"][f"{mag}"] = {}
                noisy_x = add_uniform_noise(x, spread=mag, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["categorical_ll"] = cat_lls

        if self.gaussian_noise:
            noise_results["gaussian"] = {}
            for sdev in self.gaussian_sdevs:
                noise_results["gaussian"][f"{sdev}"] = {}
                noisy_x = add_gaussian_noise(x, std=sdev, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["categorical_ll"] = cat_lls

        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results 