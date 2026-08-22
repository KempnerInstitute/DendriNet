"""
RoutingAnalyzer: per-level routing analysis for recurrent dendritic models.

Temporarily enables store_routing on the model, runs forward passes to collect
per-timestep routing info, then resets store_routing. Computes:
- Per-level contribution (beta_l): how much each dendritic level contributes
- FF/REC decomposition (gamma_l): fraction of signal from feedforward compartments
- Temporal dynamics of these readouts within trials
"""

import logging
from typing import Any, Optional

import torch
import torch.utils.data

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils.general import save_dict

logger = logging.getLogger(__name__)


def _extract_seq_lengths(batch) -> Optional[torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[2]
    return None


class RoutingAnalyzer(AbstractAnalyzer):
    """Analyzes per-level routing in recurrent dendritic E-I networks.

    Sets model.core_network.store_routing = True before evaluation passes,
    collects _routing_info from forward passes, then resets store_routing.

    Routing info format per timestep per layer:
        {
            "layer_0": {
                "excitatory": {
                    "level_contributions": Tensor[batch, n_levels],
                    "ff_rec_ratios": Tensor[batch, n_levels],
                },
                "inhibitory": {  # if present
                    "level_contributions": Tensor[batch, n_levels_i],
                    "ff_rec_ratios": Tensor[batch, n_levels_i],
                },
            }
        }
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
            model: Model with a recurrent core network (e.g. EINetwork).
            data: Dataset returning (input_seq, target) tuples.
            device: Device for computation.
            save_path: Optional path to save results.
            filename: Filename for saved results.
            batch_size: Batch size for forward passes.

        Returns:
            Dictionary with per-level routing statistics, or empty dict if
            model is not a recurrent dendritic model.
        """
        self.log_analysis_start("routing_analysis")

        core = model.core_network
        if not getattr(core, "is_recurrent", False):
            self.logger.info("Skipping routing analysis: core_network is not recurrent")
            return {}
        if not hasattr(core, "store_routing") or not hasattr(core, "_routing_info"):
            self.logger.info(
                "Skipping routing analysis: core_network has no routing support"
            )
            return {}

        original_store_routing = core.store_routing
        core.store_routing = True

        # Accumulators: per (layer, population) -> list of per-timestep tensors
        # level_contributions[key] = list of [n_valid, n_levels] tensors
        # ff_rec_ratios[key] = list of [n_valid, n_levels] tensors
        level_contribs: dict[str, list[torch.Tensor]] = {}
        ff_rec_ratios: dict[str, list[torch.Tensor]] = {}

        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        try:
            with analysis_device_context(model, device) as analysis_device:
                with torch.no_grad():
                    for batch in loader:
                        x_batch = batch[0].to(analysis_device)
                        seq_lengths = _extract_seq_lengths(batch)
                        if seq_lengths is not None:
                            seq_lengths = seq_lengths.to(analysis_device)
                        _ = model(x_batch, seq_lengths=seq_lengths)

                        for timestep_idx, timestep_info in enumerate(
                            core._routing_info
                        ):
                            valid_mask = None
                            if seq_lengths is not None:
                                valid_mask = (seq_lengths > timestep_idx).detach().cpu()
                                if not bool(valid_mask.any()):
                                    continue
                            for layer_key, layer_info in timestep_info.items():
                                for pop_key in ("excitatory", "inhibitory"):
                                    if pop_key not in layer_info:
                                        continue
                                    pop_info = layer_info[pop_key]
                                    full_key = f"{layer_key}/{pop_key}"

                                    if "level_contributions" in pop_info:
                                        values = (
                                            pop_info["level_contributions"]
                                            .detach()
                                            .cpu()
                                        )
                                        if valid_mask is not None:
                                            values = values[valid_mask]
                                        if values.numel() == 0:
                                            continue
                                        if full_key not in level_contribs:
                                            level_contribs[full_key] = []
                                        level_contribs[full_key].append(values)

                                    if "ff_rec_ratios" in pop_info:
                                        ratios = (
                                            pop_info["ff_rec_ratios"].detach().cpu()
                                        )
                                        if valid_mask is not None:
                                            ratios = ratios[valid_mask]
                                        if ratios.numel() == 0:
                                            continue
                                        if full_key not in ff_rec_ratios:
                                            ff_rec_ratios[full_key] = []
                                        ff_rec_ratios[full_key].append(ratios)
        finally:
            core.store_routing = original_store_routing

        if not level_contribs:
            self.logger.warning("No routing info collected")
            return {}

        results: dict[str, Any] = {}
        for key in sorted(level_contribs.keys()):
            # Concatenate: [N, n_levels] where N = total_timesteps * batch_size
            contribs = torch.cat(level_contribs[key], dim=0)
            # Normalize to get fractional contribution beta_l
            contribs_sum = contribs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            beta = contribs / contribs_sum  # [N, n_levels]

            beta_mean = beta.mean(dim=0)  # [n_levels]
            beta_std = beta.std(dim=0)

            entry: dict[str, Any] = {
                "n_levels": contribs.shape[-1],
                "n_samples": contribs.shape[0],
                "mean_level_contribution": beta_mean.tolist(),
                "std_level_contribution": beta_std.tolist(),
                "mean_raw_voltage_norm": contribs.mean(dim=0).tolist(),
            }

            if key in ff_rec_ratios:
                ratios = torch.cat(ff_rec_ratios[key], dim=0)  # [N, n_levels]
                entry["mean_ff_rec_ratio"] = ratios.mean(dim=0).tolist()
                entry["std_ff_rec_ratio"] = ratios.std(dim=0).tolist()

            results[key] = entry

        self.log_analysis_end("routing_analysis", num_results=len(results))

        if save_path is not None:
            save_dict(results, save_path, filename)

        return results
