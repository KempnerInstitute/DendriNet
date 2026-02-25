"""
RoutingAnalyzer: branch utilization (alpha) analysis for recurrent dendritic models.

Temporarily enables store_routing on the model, runs forward passes to collect
per-timestep alpha tensors, then resets store_routing. Computes:
- Population-averaged alpha per branch per condition
- Temporal dynamics of alpha within trials
"""

import logging
from typing import Any, Optional

import torch
import torch.utils.data

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.architectures.recurrent import DendriticRNNNetwork
from dendritic_modeling.utils.general import save_dict

logger = logging.getLogger(__name__)


class RoutingAnalyzer(AbstractAnalyzer):
    """Analyzes branch routing coefficients (alpha) in recurrent dendritic models.

    Sets model.core_network.store_routing = True before evaluation passes,
    collects _routing_info from forward passes, then resets store_routing.
    """

    def __init__(self, params: Optional[Any] = None):
        super().__init__(logger_name="RoutingAnalyzer")
        self.params = params

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "routing",
        batch_size: int = 128,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform routing analysis on a recurrent dendritic model.

        Args:
            model: Model with DendriticRNNNetwork as core.
            data: Dataset returning (input_seq, target) tuples.
            device: Device for computation.
            save_path: Optional path to save results.
            filename: Filename for saved results.
            batch_size: Batch size for forward passes.

        Returns:
            Dictionary with routing statistics, or empty dict if model
            is not a recurrent dendritic model.
        """
        self.log_analysis_start("routing_analysis")

        # Check if model has a recurrent core
        if not isinstance(model.core_network, DendriticRNNNetwork):
            self.logger.info(
                "Skipping routing analysis: core_network is not DendriticRNNNetwork"
            )
            return {}

        core: DendriticRNNNetwork = model.core_network

        # Enable routing storage temporarily
        original_store_routing = core.store_routing
        was_training = model.training
        core.store_routing = True

        model.eval()
        model.to(device)

        # Collect alpha values per layer across batches
        # alpha shape from cell: [batch, n_exc, n_branches]
        per_layer_alphas: dict[str, list[torch.Tensor]] = {}

        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        try:
            with torch.no_grad():
                for x_batch, _ in loader:
                    x_batch = x_batch.to(device)
                    _ = model(x_batch)

                    # Extract alpha from _routing_info
                    # _routing_info is a list of dicts (one per timestep)
                    # Each dict has "layer_0": {"alpha": [batch, n_exc, n_branches]}
                    for timestep_info in core._routing_info:
                        for layer_key, layer_info in timestep_info.items():
                            if "alpha" in layer_info:
                                if layer_key not in per_layer_alphas:
                                    per_layer_alphas[layer_key] = []
                                per_layer_alphas[layer_key].append(
                                    layer_info["alpha"].cpu()
                                )
        finally:
            # Always restore flags even if analysis fails mid-run.
            core.store_routing = original_store_routing
            model.train(was_training)

        if not per_layer_alphas:
            self.logger.warning("No routing info collected (no alpha tensors found)")
            return {}

        # Compute per-layer statistics
        # For each layer, concatenate along batch dim → [N, n_exc, n_branches]
        # then average across batch and neurons → [n_branches]
        results: dict[str, Any] = {}
        for layer_key, alpha_list in sorted(per_layer_alphas.items()):
            # [N, n_exc, n_branches] where N = total_timesteps * batch_size
            alpha_cat = torch.cat(alpha_list, dim=0)
            # Population average: mean across samples and neurons → [n_branches]
            pop_mean = alpha_cat.mean(dim=(0, 1))
            pop_std = alpha_cat.std(dim=(0, 1))

            results[layer_key] = {
                "mean_alpha_per_branch": pop_mean.tolist(),
                "std_alpha_per_branch": pop_std.tolist(),
                "n_branches": alpha_cat.shape[-1],
                "n_neurons": alpha_cat.shape[1],
                "n_samples": alpha_cat.shape[0],
            }

        self.log_analysis_end("routing_analysis", num_results=len(results))

        if save_path is not None:
            save_dict(results, save_path, filename)

        return results
