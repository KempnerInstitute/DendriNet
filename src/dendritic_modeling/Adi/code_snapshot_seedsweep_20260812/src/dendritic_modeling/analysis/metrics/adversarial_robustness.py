"""
Adversarial robustness analysis via FGSM.

Evaluates model accuracy under Fast Gradient Sign Method (FGSM) attacks
at various epsilon values to assess adversarial robustness.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    subset_dataset_for_runtime,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class AdversarialRobustnessAnalyzer:
    """Evaluate model robustness under FGSM adversarial attacks."""

    def __init__(self, params):
        self.epsilons = getattr(params, "epsilons", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
        self.batch_size = getattr(params, "batch_size", 256)

    def fgsm_attack(
        self,
        model: BaseModel,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """Single-step FGSM attack.

        The model stays in eval mode; gradients flow through ``x_adv`` only.
        Uses ``nll_loss`` (appropriate for models with LogSoftmax output) with
        an auto-detection fallback to ``cross_entropy`` for raw-logit models.

        Args:
            model: Target model (eval mode is preserved).
            x: Input tensor [B, ...].
            y: Target labels [B].
            epsilon: Perturbation magnitude.

        Returns:
            Adversarial examples [B, ...].
        """
        x_adv = x.clone().detach().requires_grad_(True)
        y_hat = model(x_adv)

        # Auto-detect output type: log-probs (all <= 0) vs raw logits
        if y_hat.max().item() <= 0:
            loss = nn.functional.nll_loss(y_hat, y)
        else:
            loss = nn.functional.cross_entropy(y_hat, y)

        loss.backward()
        sign_grad = x_adv.grad.sign()
        x_adv = x_adv + epsilon * sign_grad
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv.detach()

    def evaluate_accuracy_under_attack(
        self,
        model: BaseModel,
        test_dataset: TensorDataset,
        epsilon: float,
        device: str = "cpu",
    ) -> float:
        """Evaluate accuracy under FGSM attack at given epsilon."""
        model.eval()
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        correct = 0
        total = 0

        for batch in loader:
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            if epsilon > 0:
                # Model stays in eval mode. FGSM only needs input gradients,
                # not train-mode behaviour (BatchNorm/Dropout should remain off).
                x_adv = self.fgsm_attack(model, x_batch, y_batch, epsilon)
            else:
                x_adv = x_batch

            with torch.no_grad():
                y_hat = model(x_adv)
                preds = y_hat.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return correct / total if total > 0 else 0.0

    def analyze(
        self,
        model: BaseModel,
        test_dataset: TensorDataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ) -> dict:
        """Run FGSM at each epsilon, compute accuracy on adversarial examples.

        Returns:
            Dict mapping epsilon -> accuracy.
        """
        if runtime is not None:
            self.batch_size = runtime.batch_size
        test_dataset = subset_dataset_for_runtime(test_dataset, runtime)
        results = {}
        with analysis_device_context(model, device) as analysis_device:
            for eps in self.epsilons:
                acc = self.evaluate_accuracy_under_attack(
                    model,
                    test_dataset,
                    eps,
                    str(analysis_device),
                )
                results[f"eps_{eps:.3f}"] = {"accuracy": acc}
                logger.info(f"FGSM eps={eps:.3f}: accuracy={acc:.4f}")

        if save_path is not None:
            save_dict(
                results,
                save_path,
                f"adversarial_robustness_{filename}.json",
            )

        return results
