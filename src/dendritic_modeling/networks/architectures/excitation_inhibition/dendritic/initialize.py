from math import erf, exp, log, sqrt

import torch
from scipy.stats import norm
from torch.nn.init import _no_grad_normal_


def compute_expectation_truncated_log_normal(
    mean,
    std,
    upper_quantile,
):
    log_alpha = mean + std * norm.ppf(1 - upper_quantile).item()

    numerator = exp(mean + 0.5 * std**2) * (
        1 - erf((log_alpha - mean - std**2) / (std * sqrt(2)))
    )
    denominator = 1 - erf((log_alpha - mean) / (std * sqrt(2)))

    expectation = numerator / denominator
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
        alpha: threshold
        mean_c: E[X | X > alpha]
        var_c: Var[X | X > alpha]
    """
    # Determine threshold alpha such that P(X > alpha) = upper_quantile
    # gamma = (alpha - mean) / std
    gamma = norm.ppf(1 - upper_quantile)
    alpha = mean + std * gamma

    # Avoid numerical issues for extreme quantiles
    Q = max(1e-12, 1 - norm.cdf(gamma))
    phi = norm.pdf(gamma)
    lam = phi / Q

    mean_c = mean + std * lam
    var_c = (std**2) * (1 + gamma * lam - lam**2)
    return alpha, mean_c, var_c


def _compute_expectation_truncated_softplus(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[softplus(X) | X > alpha] via delta method using truncated normal moments.
    softplus(x) = ln(1 + e^x),
    g'(x) = sigmoid(x), g''(x) = sigmoid(x) * (1 - sigmoid(x)).
    """
    _, mu_c, var_c = _truncated_normal_moments(mean, std, upper_quantile)
    # softplus at conditional mean
    from math import exp as mexp, log as mlog

    sp = mlog(1 + mexp(mu_c))
    # second-order correction
    sig = 1.0 / (1.0 + mexp(-mu_c))
    sp_corr = sp + 0.5 * sig * (1 - sig) * var_c
    return sp_corr


def _compute_expectation_truncated_relu(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[ReLU(X) | X > alpha]. Since conditioning ensures X > alpha,
    and for typical TopK alpha is positive, we use E[X | X > alpha] as a good approximation.
    """
    alpha, mu_c, _ = _truncated_normal_moments(mean, std, upper_quantile)
    # If alpha <= 0 but mu_c still small, ensure non-negativity
    return max(0.0, mu_c)


def _expected_weight_for_transform(
    transform: str, mean: float, std: float, upper_quantile: float
) -> float:
    """
    Compute/approximate E[g(X) | X > alpha] where g is the transform and X ~ N(mean, std^2).

    For 'exp' uses the existing closed-form truncated log-normal expectation.
    For 'softplus' and 'relu' uses truncated normal moments with a delta-method approximation.
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


def compute_network_scale_factor(dbl, branch_factors=None):
    """
    Compute a scale factor based on the total network architecture.
    This prevents overflow when many branches accumulate in shunting computation.

    Args:
        dbl: DendriticBranchLayer to initialize
        branch_factors: List of branch factors from the parent network

    Returns a scale factor to reduce initial weights in log space.
    """
    # Calculate total branches
    total_branches = 1

    if branch_factors:
        for bf in branch_factors:
            total_branches *= bf
    else:
        # Conservative estimate if branch factors not provided
        total_branches = 100

    # Calculate approximate total conductance
    k_exc = dbl.branch_excitation.K if dbl.input_excitatory else 0
    k_inh = dbl.branch_inhibition.K if dbl.input_inhibitory else 0

    # Total approximate conductance with all weights at 1
    # Scale by sqrt of branches for more conservative scaling
    total_conductance = (k_exc + k_inh) * sqrt(total_branches)

    # Target a reasonable initial total conductance
    target_conductance = 5.0

    if total_conductance > target_conductance:
        scale_factor = log(total_conductance / target_conductance)
    else:
        scale_factor = 0.0

    return scale_factor


def analytical_expectation_dbl_init(dbl, adaptive=True, branch_factors=None):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
        adaptive: If True, use network-aware scaling to prevent overflow.
                 If False, use the original initialization method.
        branch_factors: List of branch factors from parent network (for adaptive mode)
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        # Compute network-aware scale factor if adaptive mode is enabled
        network_scale_factor = (
            compute_network_scale_factor(dbl, branch_factors) if adaptive else 0.0
        )

        if dbl.input_excitatory:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_mean = 0
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))

            if not dbl.input_inhibitory and not dbl.input_branches:
                step = 0.01
                done = False
                while not done:
                    E_w_exc = _expected_weight_for_transform(
                        transform=weight_transform,
                        mean=exc_mean,
                        std=exc_std,
                        upper_quantile=exc_upper_quantile,
                    )
                    if 0.5 * dbl.branch_excitation.K * E_w_exc <= 1:
                        done = True
                    else:
                        exc_mean -= step

            else:
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=exc_mean,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )

            # Apply the network scale factor to reduce initial weights
            exc_mean_adjusted = exc_mean - network_scale_factor

            _no_grad_normal_(dbl.branch_excitation.pre_w, exc_mean_adjusted, exc_std)
            # Adjust expected conductance by the scale factor
            E_g_exc = (
                0.5 * dbl.branch_excitation.K * E_w_exc * exp(-network_scale_factor)
            )

        if dbl.input_inhibitory:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_mean = 0
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))

            if dbl.input_excitatory:
                ie_ratio = dbl.branch_inhibition.K / dbl.branch_excitation.K
                inh_std = inh_std * ie_ratio if ie_ratio < 1 else inh_std
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

                step = 0.01
                done = False
                if E_w_exc >= ie_ratio * E_w_inh:
                    while not done:
                        E_w_inh = _expected_weight_for_transform(
                            transform=weight_transform,
                            mean=inh_mean,
                            std=inh_std,
                            upper_quantile=inh_upper_quantile,
                        )
                        if E_w_exc <= ie_ratio * E_w_inh:
                            done = True
                        else:
                            inh_mean += step
                else:
                    while not done:
                        E_w_inh = _expected_weight_for_transform(
                            transform=weight_transform,
                            mean=inh_mean,
                            std=inh_std,
                            upper_quantile=inh_upper_quantile,
                        )
                        if E_w_exc >= ie_ratio * E_w_inh:
                            done = True
                        else:
                            inh_mean -= step

            else:
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

            # Apply the network scale factor to reduce initial weights
            inh_mean_adjusted = inh_mean - network_scale_factor

            _no_grad_normal_(dbl.branch_inhibition.pre_w, inh_mean_adjusted, inh_std)
            # Adjust expected conductance by the scale factor
            E_g_inh = (
                0.5 * dbl.branch_inhibition.K * E_w_inh * exp(-network_scale_factor)
            )

        if dbl.input_branches:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)

                if E_Vinf < 0.5:
                    E_Vinf = max(0.49, 0.5 * E_Vinf + 0.25)
                else:
                    E_Vinf = min(0.51, 0.5 * E_Vinf + 0.25)

                sum_g_branch = ((E_g_exc + E_g_inh + 1) * E_Vinf - E_g_exc) / (
                    0.5 - E_Vinf
                )
                g_branch = sum_g_branch / dbl.branches_to_output.block_size
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                E_Vinf = E_g_exc - E_g_inh + 0.5 * g_branch
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        if not isinstance(dbl.reactivation, torch.nn.Identity):
            dbl.reactivation.initialize(m, E_Vinf)


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

        elif (E_g_exc > E_g_inh) and dbl.branch_inhibition is not None:
            while not done:
                inh_std += step
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=0,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )
                E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh
                if E_g_inh >= E_g_exc:
                    done = True

        if dbl.branch_excitation is not None:
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

        if dbl.branch_inhibition is not None:
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + 1
                sum_g_branch = g_branch * dbl.branches_to_output.block_size
                E_Vinf = (E_g_exc + 0.5 * sum_g_branch) / (
                    E_g_exc + E_g_inh + sum_g_branch + 1
                )
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                E_Vinf = E_g_exc - E_g_inh + 0.5 * g_branch
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        if not isinstance(dbl.reactivation, torch.nn.Identity):
            dbl.reactivation.initialize(m, E_Vinf)


def naive_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

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
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            g_branch = E_g_exc + E_g_inh + 1
            dbl.branches_to_output.initialize(g_branch=g_branch)

            if dbl.use_shunting:
                E_Vinf = (E_g_exc + 0.5 * g_branch) / (E_g_exc + E_g_inh + g_branch + 1)
                m = 1.5
            else:
                E_Vinf = E_g_exc - E_g_inh + 0.5 * g_branch
                m = 0.1
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

        if not isinstance(dbl.reactivation, torch.nn.Identity):
            dbl.reactivation.initialize(m, E_Vinf)


def default_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

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
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
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

        if not isinstance(dbl.reactivation, torch.nn.Identity):
            dbl.reactivation.initialize(
                1.5 if dbl.use_shunting else 0.1, 0.5 if dbl.use_shunting else 0.0
            )
