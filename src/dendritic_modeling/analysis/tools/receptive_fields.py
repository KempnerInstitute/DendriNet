import os
from math import prod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config import ReceptiveFieldAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import remove_hook_handles


class ReceptiveFieldAnalyzer(AbstractAnalyzer):
    """Analyzer for dendritic receptive fields and optional branch tuning."""

    _RAW_VERSION = 1

    def __init__(self, params: ReceptiveFieldAnalysisParams):
        super().__init__("ReceptiveFieldAnalyzer")

        self.n_rows: int = getattr(params, "n_rows", 2)
        self.n_cols: int = getattr(params, "n_cols", 2)

        rf_shape = self._as_shape(getattr(params, "rf_shape", None))
        self.exc_rf_shape = self._as_shape(getattr(params, "exc_rf_shape", rf_shape))
        self.inh_rf_shape = self._as_shape(getattr(params, "inh_rf_shape", rf_shape))

        self.fig_save_format = str(getattr(params, "fig_save_format", "png")).lstrip(
            "."
        )
        if not self.fig_save_format:
            self.fig_save_format = "png"

        self.compute_activation_tuning = bool(
            getattr(params, "compute_activation_tuning", False)
        )
        self.save_raw_data = bool(getattr(params, "save_raw_data", True))
        self.load_raw_data = bool(getattr(params, "load_raw_data", False))
        self.raw_filename = getattr(params, "raw_filename", None)
        self.epsilon = float(getattr(params, "epsilon", 1e-8))

    @staticmethod
    def _as_shape(value) -> Optional[tuple[int, int]]:
        if value is None:
            return None
        shape = tuple(int(v) for v in value)
        if len(shape) != 2:
            raise ValueError(f"receptive-field shape must be length 2, got {shape}")
        return shape

    @staticmethod
    def _reshape_receptive_field(
        weights: torch.Tensor, shape: Optional[tuple[int, int]]
    ) -> Optional[torch.Tensor]:
        """Reshape input-level receptive fields and skip incompatible weights."""
        weights = weights.detach().cpu()
        if shape is None:
            return weights if weights.ndim > 1 else weights.unsqueeze(0)
        if weights.numel() != prod(shape):
            return None
        return weights.reshape(shape)

    @staticmethod
    def _effective_weight(module) -> Optional[torch.Tensor]:
        if module is None:
            return None
        if hasattr(module, "pruned_weight"):
            return module.pruned_weight().detach().cpu()
        if hasattr(module, "weight"):
            weight = module.weight
            if callable(weight):
                weight = weight()
            return weight.detach().cpu()
        return None

    def _raw_checkpoint_path(self, save_path: str, filename: str) -> str:
        raw_filename = self.raw_filename or f"{filename}_raw_receptive_fields.pt"
        return os.path.join(save_path, raw_filename)

    def _save_raw_checkpoint(self, save_path: str, filename: str) -> None:
        if not self.save_raw_data:
            return
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "version": self._RAW_VERSION,
                "raw_dict": self.raw_dict,
                "exc_rf_shape": self.exc_rf_shape,
                "inh_rf_shape": self.inh_rf_shape,
            },
            self._raw_checkpoint_path(save_path, filename),
        )

    def _load_raw_checkpoint(self, save_path: str, filename: str) -> bool:
        if not self.load_raw_data:
            return False
        raw_path = self._raw_checkpoint_path(save_path, filename)
        if not os.path.exists(raw_path):
            return False
        try:
            payload = torch.load(raw_path, map_location="cpu")
        except TypeError:
            payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid receptive-field checkpoint at {raw_path}")
        raw_dict = payload.get("raw_dict", payload)
        if not isinstance(raw_dict, dict):
            raise ValueError(f"Invalid receptive-field raw_dict at {raw_path}")
        self.raw_dict = raw_dict
        return True

    def collect_raw_data(self, model: BaseModel):
        """Collect input-level receptive-field weights from all branch layers."""
        self.raw_dict = {}

        branch_counter = 0
        for module_name, module in model.named_modules():
            if not isinstance(module, DendriticBranchLayer):
                continue

            exc_weights = self._effective_weight(module.branch_excitation)
            inh_weights = self._effective_weight(module.branch_inhibition)
            if exc_weights is None and inh_weights is None:
                continue

            for branch_idx in range(module.n_branches):
                item = {
                    "depth": module.layer_idx,
                    "branch_idx": branch_idx,
                    "module": module_name,
                    "exc": None,
                    "inh": None,
                }
                if exc_weights is not None:
                    item["exc"] = self._reshape_receptive_field(
                        exc_weights[branch_idx], self.exc_rf_shape
                    )
                if inh_weights is not None:
                    item["inh"] = self._reshape_receptive_field(
                        inh_weights[branch_idx], self.inh_rf_shape
                    )
                if item["exc"] is not None or item["inh"] is not None:
                    self.raw_dict[branch_counter] = item
                    branch_counter += 1

    def _activation_hook(
        self,
        output: torch.Tensor,
        layer_key: str,
        activation_type: str,
    ) -> None:
        if output is None:
            return
        output = output.detach().cpu()
        layer_data = self.activation_raw[layer_key]
        for branch_idx in range(output.shape[-1]):
            layer_data[branch_idx][activation_type] = output[..., branch_idx]

    def _collect_activation_raw(self, model: BaseModel, x: torch.Tensor) -> None:
        self.activation_raw = {}
        handles = []

        for module_name, module in model.named_modules():
            if not isinstance(module, DendriticBranchLayer):
                continue

            self._initialize_activation_records(module_name, module)
            handles.extend(self._attach_activation_hooks(module_name, module))

        try:
            with torch.no_grad():
                _ = model(x)
        finally:
            remove_hook_handles(handles)

    def _initialize_activation_records(
        self, module_name: str, module: DendriticBranchLayer
    ) -> None:
        self.activation_raw[module_name] = {
            branch_idx: {"depth": module.layer_idx}
            for branch_idx in range(module.n_branches)
        }

    def _register_activation_hook(
        self,
        module: torch.nn.Module,
        layer_key: str,
        activation_type: str,
    ) -> torch.utils.hooks.RemovableHandle:
        return module.register_forward_hook(
            lambda module, inputs, output, key=layer_key: (
                self._activation_hook(output, key, activation_type)
            )
        )

    def _attach_activation_hooks(
        self, module_name: str, module: DendriticBranchLayer
    ) -> list[torch.utils.hooks.RemovableHandle]:
        handles = []
        if module.branch_excitation is not None:
            handles.append(
                self._register_activation_hook(
                    module.branch_excitation, module_name, "exc"
                )
            )
        if module.branch_inhibition is not None:
            handles.append(
                self._register_activation_hook(
                    module.branch_inhibition, module_name, "inh"
                )
            )
        handles.append(
            self._register_activation_hook(module.reactivation, module_name, "vout")
        )
        return handles

    def _summarize_signal(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, object]:
        values = values.reshape(values.shape[0], -1).mean(dim=-1)
        classes = torch.unique(labels)
        class_means = {}
        for label in classes:
            mask = labels == label
            if mask.any():
                class_means[str(int(label.item()))] = float(values[mask].mean().item())

        if not class_means:
            return {}

        preferred_class, preferred_mean = max(
            class_means.items(), key=lambda item: item[1]
        )
        unpreferred = [
            value for label, value in class_means.items() if label != preferred_class
        ]
        unpreferred_mean = (
            float(torch.tensor(unpreferred).mean().item()) if unpreferred else 0.0
        )

        return {
            "class_means": class_means,
            "preferred_class": int(preferred_class),
            "preferred_mean": float(preferred_mean),
            "unpreferred_mean": unpreferred_mean,
            "selectivity": float(preferred_mean - unpreferred_mean),
        }

    def _summarize_activation_tuning(self, labels: torch.Tensor) -> dict[str, object]:
        labels = labels.detach().cpu()
        summary = {}

        for layer_key, layer_data in self.activation_raw.items():
            layer_summary = {}
            for branch_idx, branch_data in layer_data.items():
                branch_summary = {"depth": branch_data["depth"], "signals": {}}
                for signal in ("exc", "inh", "vout"):
                    if signal in branch_data:
                        branch_summary["signals"][signal] = self._summarize_signal(
                            branch_data[signal], labels
                        )

                if "exc" in branch_data and "inh" in branch_data:
                    ratio = branch_data["exc"] / (
                        branch_data["inh"].abs() + self.epsilon
                    )
                    branch_summary["signals"]["ei_ratio"] = self._summarize_signal(
                        ratio, labels
                    )

                layer_summary[str(branch_idx)] = branch_summary
            summary[layer_key] = layer_summary

        return summary

    def _compute_activation_tuning(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
    ) -> dict[str, object]:
        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            labels: torch.Tensor = items[1].cpu()
            self._collect_activation_raw(model, x)
        return self._summarize_activation_tuning(labels)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ) -> Optional[dict[str, object]]:
        """Analyze receptive fields in a dendritic E/I model."""
        if not has_einet_core(model):
            self.logger.error(
                "Model must expose an ExcitationInhibitionNetwork "
                "for receptive field analysis"
            )
            return None

        loaded = False
        if save_path is not None:
            loaded = self._load_raw_checkpoint(save_path, filename)
        if not loaded:
            self.collect_raw_data(model)
            if save_path is not None:
                self._save_raw_checkpoint(save_path, filename)

        if not self.raw_dict:
            self.logger.warning("No dendritic branch receptive fields were found")
            return None

        results: dict[str, object] = {
            "n_receptive_fields": len(self.raw_dict),
            "raw_checkpoint_loaded": loaded,
        }

        if self.compute_activation_tuning:
            if test_dataset is None:
                self.logger.warning(
                    "Activation tuning requested but no test_dataset was provided"
                )
            else:
                results["activation_tuning"] = self._compute_activation_tuning(
                    model=model,
                    test_dataset=test_dataset,
                    device=device,
                    runtime=runtime,
                )

        if save_path is not None:
            out_path = os.path.join(save_path, "epochs") if training else save_path
            os.makedirs(out_path, exist_ok=True)
            save_dict(results, out_path, f"{filename}_summary.json")
            if "activation_tuning" in results:
                save_dict(
                    results["activation_tuning"],
                    out_path,
                    f"{filename}_activation_tuning.json",
                )
            if not training:
                for synapse_type in ("exc", "inh"):
                    available = [
                        ix
                        for ix, item in self.raw_dict.items()
                        if item.get(synapse_type) is not None
                    ]
                    if not available:
                        self.logger.warning(
                            "No %s receptive fields matched the requested shape",
                            synapse_type,
                        )
                        continue
                    n_branches = min(self.n_rows * self.n_cols, len(available))
                    perm = (
                        torch.randperm(len(available))[:n_branches].sort()[0].tolist()
                    )
                    ixs = [available[ix] for ix in perm]
                    self._plot_receptive_fields(ixs, synapse_type, save_path)

        return results

    def _plot_receptive_fields(self, ixs: list[int], synapse_type: str, save_path: str):
        """Plot receptive fields."""
        mins = []
        maxs = []
        for ix in ixs:
            data = self.raw_dict[ix][synapse_type]
            mins.append(float(torch.min(data)))
            maxs.append(float(torch.max(data)))
        vmin = min(mins)
        vmax = max(maxs)

        fig, axes = plt.subplots(
            self.n_rows, self.n_cols, figsize=(5 * self.n_cols, 5 * self.n_rows)
        )

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        axes_flat = np.atleast_1d(axes).flatten()

        for ax in axes_flat:
            ax.set_axis_off()

        for ix, ax in zip(ixs, axes_flat):
            ax.set_axis_on()
            data = self.raw_dict[ix][synapse_type]
            sns.heatmap(
                data,
                ax=ax,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                cbar_ax=cax,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            depth = self.raw_dict[ix]["depth"]
            branch_idx = self.raw_dict[ix]["branch_idx"]
            ax.set_title(f"Depth {depth}, Branch {branch_idx}")

        fig.suptitle(f"{synapse_type.title()} Receptive Fields")
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(
                save_path,
                f"{synapse_type}_receptive_fields.{self.fig_save_format}",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
