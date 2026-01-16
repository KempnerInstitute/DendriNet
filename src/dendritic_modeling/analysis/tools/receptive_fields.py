from typing import Optional

import os

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import ReceptiveFieldAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    TopKLinear,
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict




class ReceptiveFieldAnalyzer(AbstractAnalyzer):
    """
    Analyzer for visualizing receptive fields of the model.
    """

    def __init__(self, params: ReceptiveFieldAnalysisParams):
        super().__init__("ReceptiveFieldAnalyzer")
        
        self.n_rows: int = getattr(params, "n_rows", 2)
        self.n_cols: int = getattr(params, "n_cols", 2)
        
        exc_rf_shape: Optional[tuple[int, int]] = getattr(
            params, "exc_rf_shape", None)
        if exc_rf_shape is not None:
            exc_rf_shape = tuple(exc_rf_shape)
        self.exc_rf_shape = exc_rf_shape

        inh_rf_shape: Optional[tuple[int, int]] = getattr(
            params, "inh_rf_shape", None)
        if inh_rf_shape is not None:
            inh_rf_shape = tuple(inh_rf_shape)
        self.inh_rf_shape = inh_rf_shape
                
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

                    if self.exc_rf_shape is not None:
                        exc_w = exc_w.reshape(self.exc_rf_shape)

                    if self.inh_rf_shape is not None:
                        inh_w = inh_w.reshape(self.inh_rf_shape)

                    self.raw_dict[branch_counter] = {
                        "depth": module.layer_idx, 
                        "branch_idx": branch_idx,
                        "exc": exc_w,
                        "inh": inh_w
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
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        self.collect_raw_data(model)

        n_branches = self.n_rows * self.n_cols
        ixs = torch.randperm(len(self.raw_dict))[:n_branches].sort()[0].tolist()

        self._plot_receptive_fields(ixs, "exc", save_path)
        self._plot_receptive_fields(ixs, "inh", save_path)

    def _plot_receptive_fields(
        self, 
        ixs: list[int], 
        synapse_type: str,
        save_path: str
    ):
        """Plot receptive fields."""
        mins = []
        maxs = []
        for ix in ixs:
            mins.append(torch.min(self.raw_dict[ix][synapse_type]))
            maxs.append(torch.max(self.raw_dict[ix][synapse_type]))
        vmin = min(mins)
        vmax = max(maxs)

        fig, axes = plt.subplots(
            self.n_rows, self.n_cols, 
            figsize=(5 * self.n_cols, 5 * self.n_rows)
        )

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        for ix, ax in zip(ixs, axes.flatten()):
            data = self.raw_dict[ix][synapse_type]
            sns.heatmap(
                data, ax=ax, cmap="viridis",
                vmin=vmin, vmax=vmax,
                cbar_ax=cax,
                square=True,
                xticklabels=False, yticklabels=False
            )
            depth = self.raw_dict[ix]["depth"]
            branch_idx = self.raw_dict[ix]["branch_idx"]
            ax.set_title(f"Depth {depth}, Branch {branch_idx}")
        
        fig.suptitle(f"{synapse_type.title()} Receptive Fields")
        # plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{synapse_type}_receptive_fields.png"), 
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)




