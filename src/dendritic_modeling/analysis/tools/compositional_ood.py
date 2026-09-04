"""Compositional OOD performance dump for ContextualGainDataset."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.tools.epoch_aggregation import load_epoch_aggregation
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    evaluation_kwargs_from_runtime,
    materialize_dataset,
)
from dendritic_modeling.config.analysis import (
    CompositionalOODAnalysisParams,
    EvaluationRuntimeConfig,
)
from dendritic_modeling.datasets.contextual_gain import ContextualGainDataset
from dendritic_modeling.models import BaseModel, Classifier, Regressor
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
    evaluate_pred_label_mi,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)

METRIC_MAPPING = {
    "accuracy": "Compositional OOD Accuracy",
    "auc": "Compositional OOD AUC",
    "categorical_loglikelihood": "Compositional OOD Log-Likelihood",
    "mse": "Compositional OOD MSE",
    "cosine_similarity": "Compositional OOD Cosine Similarity",
    "pred_label_mi_bits": "Compositional OOD Pred-Label MI",
}

_CATEGORICAL_LL_ALIASES = frozenset(
    {
        "categorical_loglikelihood",
        "categorical_ll",
        "loglikelihood",
        "log_likelihood",
    }
)
_COSINE_ALIASES = frozenset({"cosine_similarity", "cosine"})
_PRED_LABEL_MI_ALIASES = frozenset(
    {"pred_label_mi_bits", "pred_label_mi", "pred_mi"}
)


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


class CompositionalOODAnalyzer(AbstractAnalyzer):
    """Score configured metrics on one frozen compositional-OOD draw."""

    def __init__(self, params: CompositionalOODAnalysisParams):
        super().__init__("CompositionalOODAnalyzer")
        metrics_raw = list(getattr(params, "metrics", []) or [])
        metrics = {_norm(x) for x in metrics_raw if str(x).strip()}

        self.accuracy = "accuracy" in metrics
        self.auc = "auc" in metrics
        self.categorical_ll = bool(_CATEGORICAL_LL_ALIASES.intersection(metrics))
        self.mse = "mse" in metrics
        self.cosine_similarity = bool(_COSINE_ALIASES.intersection(metrics))
        self.compute_pred_label_mi = bool(_PRED_LABEL_MI_ALIASES.intersection(metrics))

    def _metric_functions(self, model: BaseModel) -> list[tuple[str, Callable]]:
        is_classifier = isinstance(model, Classifier)
        is_regressor = isinstance(model, Regressor)
        candidates: list[tuple[bool, bool, str, Callable]] = [
            (self.accuracy, is_classifier, "accuracy", evaluate_accuracy),
            (self.auc, is_classifier, "auc", evaluate_auc),
            (
                self.categorical_ll,
                is_classifier,
                "categorical_loglikelihood",
                evaluate_categorical_loglikelihood,
            ),
            (self.mse, is_regressor, "mse", evaluate_mse),
            (
                self.cosine_similarity,
                is_regressor,
                "cosine_similarity",
                evaluate_cosine_similarity,
            ),
            (
                self.compute_pred_label_mi,
                is_classifier,
                "pred_label_mi_bits",
                evaluate_pred_label_mi,
            ),
        ]
        return [
            (key, func)
            for enabled, compatible, key, func in candidates
            if enabled and compatible
        ]

    def load_epochs_data(self, save_path: str):
        """Load aggregated metric lists from ``save_path/epochs``."""
        return load_epoch_aggregation(save_path, preserved_leaf_keys=())

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        if not (
            isinstance(test_dataset, ContextualGainDataset)
            and test_dataset._ood_enabled
        ):
            return

        metric_funcs = self._metric_functions(model)
        eval_kwargs = evaluation_kwargs_from_runtime(runtime)
        with analysis_device_context(model, device) as analysis_device:
            previous_ood_sampling = bool(test_dataset._ood_sampling)
            try:
                test_dataset._set_OOD_sampling(True)
                items = materialize_dataset(
                    test_dataset,
                    runtime,
                    device=analysis_device,
                )
            finally:
                test_dataset._set_OOD_sampling(previous_ood_sampling)

            frozen = TensorDataset(items[0], items[1])
            results: dict[str, float] = {}
            for metric_key, metric_func in metric_funcs:
                results[metric_key] = float(
                    metric_func(
                        model,
                        test_ds=frozen,
                        device=analysis_device,
                        **eval_kwargs,
                    )[-1]
                )

        if save_path is not None:
            if training:
                save_dict(
                    results,
                    os.path.join(save_path, "epochs"),
                    f"{filename}.json",
                )
            else:
                epochs_data, epoch_numbers = self.load_epochs_data(save_path)
                if epochs_data and epoch_numbers:
                    for metric in results:
                        plot_performance_vs_epoch(
                            epochs_data,
                            epoch_numbers,
                            metric,
                            save_path,
                        )
                save_dict(results, save_path, f"{filename}.json")
            return
        return results


def plot_performance_vs_epoch(
    epochs_data: dict,
    epoch_numbers: list,
    metric: str,
    save_path: str,
) -> None:
    """Plot one compositional-OOD metric across training epochs."""
    metric_values = epochs_data.get(metric)
    if metric_values is None or len(metric_values) != len(epoch_numbers):
        return

    ylabel = METRIC_MAPPING.get(metric, metric)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, metric_values, marker="o", linewidth=2, markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs Epoch", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, f"{metric}_vs_epoch.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Plot saved to %s", plot_path)
    plt.close()


__all__ = ["CompositionalOODAnalyzer", "plot_performance_vs_epoch"]
