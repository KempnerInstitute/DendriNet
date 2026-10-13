"""Explicit fan-in-scaled initialization for positive conductance trees."""

import math

import torch

from dendritic_modeling.networks.utils.weight_transforms import inverse_weight_transform

FAN_IN_LOGNORMAL_LOG_SD = 0.5
FAN_IN_LOGNORMAL_TRANSFORMS = ("exp", "softplus")


@torch.no_grad()
def fan_in_lognormal_init(branch):
    """Set mean contact conductance to 1 / total local synaptic fan-in.

    Log standard deviation is 0.5; E and I use the same distribution. Child
    couplings use initial_child_conductance, optionally log-normal with mean
    initial_child_conductance and log standard deviation
    initial_child_conductance_log_sd. These are opt-in initialization
    semantics, independent of the legacy analytical E/I balancing policy.
    Softplus weights draw the same conductances as exp weights from the same
    random stream and store their inverse-softplus raw values.
    """
    if (
        branch.weight_transform not in FAN_IN_LOGNORMAL_TRANSFORMS
        or branch.reactivation_init_policy != "fixed"
    ):
        raise ValueError(
            "fan_in_lognormal requires exp or softplus weights and fixed "
            "activation initialization"
        )
    layers = [
        getattr(branch, key, None)
        for key in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        )
    ]
    total = sum(layer.K for layer in layers if layer is not None)
    scale = branch.initial_conductance_scale
    for i, layer in enumerate(layers):
        if layer is None:
            continue
        log_mean = -math.log(total) - 0.5 * FAN_IN_LOGNORMAL_LOG_SD**2
        with branch.initialization_seed_scope(f"fan_in_lognormal.{i}"):
            layer.pre_w.normal_(log_mean, FAN_IN_LOGNORMAL_LOG_SD)
        if scale != 1:
            layer.pre_w.add_(math.log(scale))
        if layer.weight_transform != "exp":
            layer.pre_w.copy_(
                inverse_weight_transform(layer.pre_w.exp(), layer.weight_transform)
            )
        # Recorded law for initialization-matched contact regrowth.
        layer.initial_log_mean = log_mean + math.log(scale)
        layer.initial_log_sd = FAN_IN_LOGNORMAL_LOG_SD
        if (
            hasattr(layer, "release_weight_scale")
            and branch.synapse_config.indexed.release_weight_scale is None
        ):
            layer.release_weight_scale = scale / total
    if branch.input_branches:
        _initialize_child_couplings(branch, scale)
    if branch.reactivate:
        initialize = getattr(branch.reactivation, "initialize", None)
        # Fixed activations have no slope/midpoint to initialize. Keep the
        # same contract as the other conductance initialization policies.
        if callable(initialize) and getattr(
            branch.reactivation, "supports_slope_midpoint_init", True
        ):
            initialize(branch.reactivation_init_m, branch.reactivation_init_b)


def _initialize_child_couplings(branch, scale):
    coupling = branch.branches_to_output
    log_sd = branch.initial_child_conductance_log_sd
    coupling.initialize(g_branch=branch.initial_child_conductance)
    if log_sd == 0 and coupling.weight_transform == "exp":
        if scale != 1:
            coupling.log_weight.add_(math.log(scale))
        return
    conductance = torch.full_like(
        coupling.log_weight, branch.initial_child_conductance * scale
    )
    if log_sd > 0:
        with branch.initialization_seed_scope("fan_in_lognormal.child_coupling"):
            noise = torch.randn_like(coupling.log_weight)
        # Mean-preserving log-normal: E[g] = initial_child_conductance * scale.
        conductance = conductance * torch.exp(log_sd * noise - 0.5 * log_sd**2)
    coupling.log_weight.copy_(
        inverse_weight_transform(conductance, coupling.weight_transform)
    )
