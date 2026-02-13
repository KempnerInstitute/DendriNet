"""
Parameter matching for dendritic layer swaps.
Rule: baseline layer params < 50k -> 5% tolerance; >= 50k -> 2%.
Procedure: grid over knobs, pick argmin absolute percent error, write to adi_metadata.json.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def tolerance_for_params(baseline_params: int) -> float:
    """Return allowed relative tolerance: 0.05 for <50k, 0.02 for >=50k."""
    return 0.05 if baseline_params < 50_000 else 0.02


def match_conv_layer(
    baseline_params: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Dict[str, Any]:
    """
    DendriteConv2d has 2x weights (E/I) plus biases and apical; param count differs from Conv2d.
    For conv we only control out_channels (and in_channels fixed by prev layer).
    Grid: try dendritic with same out_channels; report actual params and percent error.
    """
    from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

    best_err = float("inf")
    best_out_ch = out_channels
    best_params = 0
    for out_ch in [out_channels]:  # Single candidate; extend to [out_ch//2, out_ch, out_ch*2] if needed
        layer = DendriteConv2d(
            in_channels, out_ch, kernel_size, stride=stride, padding=padding
        )
        n = count_params(layer)
        err = abs(n - baseline_params) / max(baseline_params, 1)
        if err < best_err:
            best_err = err
            best_out_ch = out_ch
            best_params = n
    tol = tolerance_for_params(baseline_params)
    return {
        "baseline_params": baseline_params,
        "matched_params": best_params,
        "percent_error": best_err,
        "tolerance": tol,
        "within_tolerance": best_err <= tol,
        "chosen_out_channels": best_out_ch,
    }


def match_fc_layer(
    baseline_params: int,
    in_features: int,
    out_features: int,
    n_branches_grid: Optional[List[int]] = None,
    branch_width_factors: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Grid over n_branches and branch width; pick argmin percent error.
    DendriticLinearWrapper uses DendriticBranchLayer with TopK; param count depends on K and dimensions.
    """
    from dendritic_modeling.Adi.dendritic_linear import DendriticLinearWrapper

    n_branches_grid = n_branches_grid or [1, 2, 4, 8]
    branch_width_factors = branch_width_factors or [0.5, 1.0, 2.0]
    best_err = float("inf")
    best_knobs: Dict[str, Any] = {}
    best_params = 0
    for n_b in n_branches_grid:
        for bw in branch_width_factors:
            out_dim = max(1, int(out_features * bw))
            K = min(64, in_features, out_dim)
            try:
                layer = DendriticLinearWrapper(in_features, out_dim, K=K)
            except Exception:
                continue
            n = count_params(layer)
            err = abs(n - baseline_params) / max(baseline_params, 1)
            if err < best_err:
                best_err = err
                best_knobs = {"n_branches": n_b, "branch_width_factor": bw, "out_dim": out_dim, "K": K}
                best_params = n
    tol = tolerance_for_params(baseline_params)
    return {
        "baseline_params": baseline_params,
        "matched_params": best_params,
        "percent_error": best_err,
        "tolerance": tol,
        "within_tolerance": best_err <= tol,
        "chosen_knobs": best_knobs,
    }


def write_adi_metadata(
    run_dir: str,
    baseline_total_params: int,
    hybrid_total_params: int,
    per_layer_param_counts: Optional[Dict[str, int]] = None,
    swapped_layers_match: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> None:
    """Write adi_metadata.json next to config.json in the run directory."""
    data = {
        "baseline_total_params": baseline_total_params,
        "hybrid_total_params": hybrid_total_params,
        "per_layer_param_counts": per_layer_param_counts or {},
        "swapped_layers_match": swapped_layers_match or {},
        **kwargs,
    }
    path = os.path.join(run_dir, "adi_metadata.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
