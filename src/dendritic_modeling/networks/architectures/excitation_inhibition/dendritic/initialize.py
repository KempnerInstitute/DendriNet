"""
Initialization methods for dendritic branch layers.
"""

from math import exp as mexp, log as mlog, sqrt
from typing import Optional

import torch
from scipy.stats import norm


def compute_expectation_truncated_log_normal(
    mean,
    std,
    upper_quantile,
):
    z = mean + std * norm.ppf(1 - upper_quantile).item()
    # E[e^X | X < z]
    expectation = (
        torch.exp(mean + 0.5 * std**2)
        * norm.cdf((z - mean - std**2) / std)
        / norm.cdf((z - mean) / std)
    )
    return expectation


def compute_expectation_truncated_inverse_softplus_normal(
    mean,
    std,
    upper_quantile,
):
    z = mean + std * norm.ppf(1 - upper_quantile).item()

    normal = torch.distributions.Normal(mean, std)

    truncated_samples_list = []
    max_iter = 100
    iter_counter = 0
    done = False
    while not done:
        iter_counter += 1
        samples = normal.sample((10000,))
        truncated_samples = samples[samples > z]
        truncated_samples_list.append(truncated_samples)
        all_truncated_samples = torch.cat(truncated_samples_list)
        if (all_truncated_samples.shape[0] >= 10000) or (iter_counter >= max_iter):
            done = True

    transformed_samples = torch.log(1 + torch.exp(all_truncated_samples))
    return torch.mean(transformed_samples).item()


# -----------------------------
# Transform-aware expectation helpers
# -----------------------------


def _truncated_normal_moments(
    mean: float, std: float, upper_quantile: float
) -> tuple[float, float, float]:
    """
    Compute moments of X | X > alpha where X ~ Normal(mean, std^2),
    and alpha is the threshold corresponding to the given upper_quantile (top proportion kept).

    Returns:
        (alpha, mean_c, var_c) where mean_c and var_c are conditional mean and variance
    """
    # alpha is the threshold: P(X > alpha) = upper_quantile
    alpha = norm.ppf(1 - upper_quantile, loc=mean, scale=std)

    # Standard normal equivalent
    a = (alpha - mean) / std

    # Mill's ratio
    phi_a = norm.pdf(a)
    Phi_a_c = upper_quantile  # 1 - Phi(a)

    # Conditional mean and variance
    lambda_a = phi_a / Phi_a_c
    delta_a = lambda_a * (lambda_a - a)

    mean_c = mean + std * lambda_a
    var_c = std**2 * (1 - delta_a)

    return alpha, mean_c, var_c


def _compute_expectation_truncated_softplus(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[softplus(X) | X > alpha] via delta method using truncated normal moments.
    softplus(x) = ln(1 + e^x),
    softplus'(x) = sigmoid(x) = 1/(1 + e^{-x})
    """
    _, mu_c, var_c = _truncated_normal_moments(mean, std, upper_quantile)
    # softplus at conditional mean
    sp = mlog(1 + mexp(mu_c))
    # second-order correction
    sig = 1.0 / (1.0 + mexp(-mu_c))
    sp_corr = sp + 0.5 * var_c * sig * (1 - sig)
    return sp_corr


def _compute_expectation_truncated_relu(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[ReLU(X) | X > alpha]. Since conditioning ensures X > alpha,
    and for typical TopK alpha is positive, we use E[X | X > alpha] as a good approximation.
    """
    _, mu_c, _ = _truncated_normal_moments(mean, std, upper_quantile)
    # For ReLU: max(0, mu_c)
    return max(0.0, mu_c)


def _expected_weight_for_transform(
    transform: str, mean: float, std: float, upper_quantile: float
) -> float:
    """
    Compute/approximate E[g(X) | X > alpha] where g is the transform and X ~ N(mean, std^2).

    For exp: exact formula
    For softplus/relu: approximations via delta method
    """
    t = (transform or "exp").lower()
    if t == "exp":
        return compute_expectation_truncated_log_normal(
            mean=mean, std=std, upper_quantile=upper_quantile
        )
    if t == "softplus":
        return _compute_expectation_truncated_softplus(mean, std, upper_quantile)
    if t == "relu":
        return _compute_expectation_truncated_relu(mean, std, upper_quantile)
    # Fallback to exp-style expectation if unknown
    return compute_expectation_truncated_log_normal(
        mean=mean, std=std, upper_quantile=upper_quantile
    )


# -----------------------------
# Layer initialization utilities
# -----------------------------


def _no_grad_normal_(tensor, mean, std):
    """Initialize tensor with normal distribution without gradient tracking."""
    with torch.no_grad():
        tensor.normal_(mean, std)


def _sample_sign(p_pos, n_samples, device, dtype):
    """Sample signs with probability p_pos of being positive."""
    return (
        2 * torch.bernoulli(torch.full((n_samples,), p_pos, device=device, dtype=dtype))
        - 1
    )


# -----------------------------
# Main initialization functions
# -----------------------------


def analytical_expectation_dbl_init(
    dbl, adaptive: bool = True, branch_factors: Optional[list] = None
):
    """
    Initialize a dendritic branch layer using analytical expectations
    with optional adaptive conductance balancing.

    Args:
        dbl: The DendriticBranchLayer to initialize
        adaptive: Whether to use adaptive conductance balancing
        branch_factors: List of branch factors for hierarchical networks
    """
    from .branch_layer import DendriticBranchLayer
    from .dendriarchblock import DendriarchBlockLinear

    assert isinstance(dbl, DendriticBranchLayer) or isinstance(
        dbl, DendriarchBlockLinear
    )

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0

        # Initialize excitatory synapses
        if dbl.input_excitatory:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_mean = 0
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            # Determine transform for excitation
            exc_transform = getattr(
                getattr(dbl, "branch_excitation", None), "weight_transform", "exp"
            )

            if not dbl.input_inhibitory and not dbl.input_branches:
                step = 0.01
                done = False
                while not done:
                    E_w_exc = _expected_weight_for_transform(
                        transform=exc_transform,
                        mean=exc_mean,
                        std=exc_std,
                        upper_quantile=exc_upper_quantile,
                    )
                    E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc
                    if E_g_exc >= 1:
                        done = True
                    else:
                        exc_std += step

            else:
                E_w_exc = _expected_weight_for_transform(
                    transform=exc_transform,
                    mean=exc_mean,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )
                E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

            # Initialize the excitatory weights
            if hasattr(dbl.branch_excitation, "initialize"):
                dbl.branch_excitation.initialize()
            else:
                _no_grad_normal_(dbl.branch_excitation.pre_w, exc_mean, exc_std)

        # Initialize inhibitory synapses
        if dbl.input_inhibitory:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_mean = 0
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            inh_transform = getattr(
                getattr(dbl, "branch_inhibition", None), "weight_transform", "exp"
            )

            if dbl.input_excitatory:
                ie_ratio = dbl.branch_inhibition.K / dbl.branch_excitation.K
                inh_std = inh_std * ie_ratio if ie_ratio < 1 else inh_std
                E_w_inh = _expected_weight_for_transform(
                    transform=inh_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )
                E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

                # only tune the IE ratio if it's not on adaptive initialization
                if not adaptive:
                    step = 0.01
                    done = False
                    if E_w_exc >= ie_ratio * E_w_inh:
                        while not done:
                            E_w_inh = _expected_weight_for_transform(
                                transform=inh_transform,
                                mean=inh_mean,
                                std=inh_std,
                                upper_quantile=inh_upper_quantile,
                            )
                            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh
                            if E_w_exc < ie_ratio * E_w_inh:
                                done = True
                            else:
                                inh_std += step

                    else:
                        while not done:
                            E_w_inh = _expected_weight_for_transform(
                                transform=inh_transform,
                                mean=inh_mean,
                                std=inh_std,
                                upper_quantile=inh_upper_quantile,
                            )
                            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh
                            if E_w_exc >= ie_ratio * E_w_inh:
                                done = True
                            else:
                                inh_std -= step

            else:
                E_w_inh = _expected_weight_for_transform(
                    transform=inh_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )
                E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

            # Initialize the inhibitory weights
            if hasattr(dbl.branch_inhibition, "initialize"):
                dbl.branch_inhibition.initialize()
            else:
                _no_grad_normal_(dbl.branch_inhibition.pre_w, inh_mean, inh_std)

        # Initialize branch-to-output connections
        if dbl.input_branches:
            if dbl.use_shunting:
                if adaptive and branch_factors and len(branch_factors) > 0:
                    # Use hierarchical conductance balancing
                    hierarchy_scale = 1.0
                    for bf in branch_factors:
                        if bf is not None:
                            hierarchy_scale *= sqrt(float(bf))
                    g_branch = hierarchy_scale * (E_g_exc + E_g_inh)
                else:
                    # Standard shunting initialization
                    g_branch = E_g_exc + E_g_inh + 1
            else:
                # Standard initialization for non-shunting
                g_branch = 1

            dbl.branches_to_output.initialize(g_branch=g_branch)

        # Initialize reactivation if present
        reactivation = getattr(dbl, "reactivation", None)
        if reactivation is None:
            reactivation = getattr(dbl, "activation", None)

        if reactivation and not isinstance(reactivation, torch.nn.Identity):
            # Adaptive initialization for reactivation based on expected voltage
            if adaptive and dbl.use_shunting:
                # For shunting: E[V] ≈ E[g_exc] / (E[g_exc] + E[g_inh] + 1)
                E_Vinf = (
                    E_g_exc / (E_g_exc + E_g_inh + 1)
                    if (E_g_exc + E_g_inh) > 0
                    else 0.5
                )
                # Target middle of activation range
                m = 1.5
                b = E_Vinf
            else:
                # For non-shunting: use standard initialization
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1
                b = 0.0

            if hasattr(reactivation, "initialize"):
                reactivation.initialize(m, b)


def ei_equivalence_dbl_init(dbl):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        step = 0.01
        done = False
        if (E_g_exc < E_g_inh) and dbl.branch_excitation is not None:
            while not done:
                exc_std += step
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=0,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )
                E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc
                if E_g_exc > E_g_inh:
                    done = True

        if dbl.branch_excitation is not None:
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

            E_w_exc = _expected_weight_for_transform(
                transform=getattr(dbl.branch_excitation, "weight_transform", "exp"),
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_inhibition is not None:
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=getattr(dbl.branch_inhibition, "weight_transform", "exp"),
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + 1
            else:
                g_branch = 1
            dbl.branches_to_output.initialize(g_branch=g_branch)

        # Initialize reactivation if present
        reactivation = getattr(dbl, "reactivation", None)
        if reactivation is None:
            reactivation = getattr(dbl, "activation", None)

        if reactivation and not isinstance(reactivation, torch.nn.Identity):
            E_Vinf = (
                E_g_exc / (E_g_exc + E_g_inh + 1)
                if dbl.use_shunting
                else E_g_exc - E_g_inh
            )
            m = 1.5 if dbl.use_shunting else 0.1
            if hasattr(reactivation, "initialize"):
                reactivation.initialize(m, E_Vinf)


def naive_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        exc_upper_quantile = 0
        inh_upper_quantile = 0
        E_g_exc = 0
        E_g_inh = 0

        if dbl.input_excitatory:
            # we use the default topk initialization
            if hasattr(dbl.branch_excitation, "initialize"):
                dbl.branch_excitation.initialize()
            else:
                _no_grad_normal_(
                    dbl.branch_excitation.pre_w,
                    0,
                    sqrt(2 / (dbl.excitatory_input_dim + 1)),
                )

            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))

            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

            E_w_exc = _expected_weight_for_transform(
                transform=getattr(dbl.branch_excitation, "weight_transform", "exp"),
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.input_inhibitory:
            # we use the default topk initialization
            if hasattr(dbl.branch_inhibition, "initialize"):
                dbl.branch_inhibition.initialize()
            else:
                _no_grad_normal_(
                    dbl.branch_inhibition.pre_w,
                    0,
                    sqrt(2 / (dbl.inhibitory_input_dim + 1)),
                )

            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))

            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=getattr(dbl.branch_inhibition, "weight_transform", "exp"),
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + 1
            else:
                g_branch = 1

            dbl.branches_to_output.initialize(g_branch=g_branch)

        # Initialize reactivation if present
        reactivation = getattr(dbl, "reactivation", None)
        if reactivation is None:
            reactivation = getattr(dbl, "activation", None)

        if reactivation and not isinstance(reactivation, torch.nn.Identity):
            if hasattr(reactivation, "initialize"):
                reactivation.initialize(
                    1.5 if dbl.use_shunting else 0.1, 0.5 if dbl.use_shunting else 0.0
                )


def default_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        if dbl.input_excitatory:
            # we use the default topk initialization
            if hasattr(dbl.branch_excitation, "initialize"):
                dbl.branch_excitation.initialize()
            else:
                _no_grad_normal_(
                    dbl.branch_excitation.pre_w,
                    0,
                    sqrt(2 / (dbl.excitatory_input_dim + 1)),
                )

        if dbl.input_inhibitory:
            # we use the default topk initialization
            if hasattr(dbl.branch_inhibition, "initialize"):
                dbl.branch_inhibition.initialize()
            else:
                _no_grad_normal_(
                    dbl.branch_inhibition.pre_w,
                    0,
                    sqrt(2 / (dbl.inhibitory_input_dim + 1)),
                )

        if dbl.input_branches:
            dbl.branches_to_output.initialize(g_branch=1)

        # Initialize reactivation if present
        reactivation = getattr(dbl, "reactivation", None)
        if reactivation is None:
            reactivation = getattr(dbl, "activation", None)

        if reactivation and not isinstance(reactivation, torch.nn.Identity):
            if hasattr(reactivation, "initialize"):
                reactivation.initialize(1, 0)
