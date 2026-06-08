import os
from math import prod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.config import ReceptiveFieldAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer


class ReceptiveFieldAnalyzer(AbstractAnalyzer):
    """
    Analyzer for visualizing receptive fields of the model.
    """

    def __init__(self, params: ReceptiveFieldAnalysisParams):
        super().__init__("ReceptiveFieldAnalyzer")

        self.n_rows: int = getattr(params, "n_rows", 2)
        self.n_cols: int = getattr(params, "n_cols", 2)

        exc_rf_shape: Optional[tuple[int, int]] = getattr(params, "exc_rf_shape", None)
        if exc_rf_shape is not None:
            exc_rf_shape = tuple(exc_rf_shape)
        self.exc_rf_shape = exc_rf_shape

        inh_rf_shape: Optional[tuple[int, int]] = getattr(params, "inh_rf_shape", None)
        if inh_rf_shape is not None:
            inh_rf_shape = tuple(inh_rf_shape)
        self.inh_rf_shape = inh_rf_shape

    @staticmethod
    def _reshape_receptive_field(
        weights: torch.Tensor, shape: Optional[tuple[int, int]]
    ):
        """Reshape input-level receptive fields and skip branch-to-branch weights."""
        if shape is None:
            return weights if weights.ndim > 1 else weights.unsqueeze(0)
        if weights.numel() != prod(shape):
            return None
        return weights.reshape(shape)

    def collect_raw_data(self, model: BaseModel):
        """Collect raw data from the model."""
        self.raw_dict = {}

        branch_counter = 0
        for module in model.modules():
            if (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                exc_weights = module.branch_excitation.pruned_weight().detach().cpu()
                inh_weights = module.branch_inhibition.pruned_weight().detach().cpu()

                for branch_idx in range(module.n_branches):
                    exc_w = exc_weights[branch_idx]
                    inh_w = inh_weights[branch_idx]

                    exc_w = self._reshape_receptive_field(exc_w, self.exc_rf_shape)
                    inh_w = self._reshape_receptive_field(inh_w, self.inh_rf_shape)

                    self.raw_dict[branch_counter] = {
                        "depth": module.layer_idx,
                        "branch_idx": branch_idx,
                        "exc": exc_w,
                        "inh": inh_w,
                    }

                    branch_counter += 1

    def analyze(
        self,
        model: BaseModel,
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze receptive fields in the model.
        """
        if not has_einet_core(model):
            self.logger.error(
                "Model must expose an ExcitationInhibitionNetwork "
                + "for receptive field analysis"
            )
            return

        self.collect_raw_data(model)

        n_branches = self.n_rows * self.n_cols
        if not self.raw_dict:
            self.logger.warning("No dendritic branch receptive fields were found")
            return

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
            perm = torch.randperm(len(available))[:n_branches].sort()[0].tolist()
            ixs = [available[ix] for ix in perm]
            self._plot_receptive_fields(ixs, synapse_type, save_path)

    def _plot_receptive_fields(self, ixs: list[int], synapse_type: str, save_path: str):
        """Plot receptive fields."""
        mins = []
        maxs = []
        for ix in ixs:
            mins.append(float(torch.min(self.raw_dict[ix][synapse_type])))
            maxs.append(float(torch.max(self.raw_dict[ix][synapse_type])))
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
        # plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{synapse_type}_receptive_fields.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
