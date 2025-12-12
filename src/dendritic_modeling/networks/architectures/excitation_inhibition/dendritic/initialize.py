from math import erf, exp, log, sqrt
from typing import Optional

import numpy as np
import torch
from scipy.stats import norm
from torch.nn.init import _no_grad_normal_

from dendritic_modeling.utils.lda import fit_nonnegative_lda


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

    # Ensure variance is non-negative (can be slightly negative due to numerical errors)
    var_c = max(0.0, var_c)

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

    # Clip mu_c to prevent overflow
    mu_c = max(-50.0, min(50.0, mu_c))

    # Safe computation of softplus to avoid overflow
    if mu_c > 20:
        sp = mu_c  # For large x, softplus(x) ≈ x
    else:
        sp = mlog(1 + mexp(mu_c))

    # Safe computation of sigmoid to avoid overflow
    if mu_c > 0:
        sig = 1.0 / (1.0 + mexp(-mu_c))
    else:
        exp_mu = mexp(mu_c)
        sig = exp_mu / (1.0 + exp_mu)

    sp_corr = sp + 0.5 * sig * (1 - sig) * var_c
    return sp_corr


def _compute_expectation_truncated_relu(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[ReLU(X) | X > alpha]. Since conditioning ensures X > alpha,
    and for typical TopK alpha is positive, we use E[X | X > alpha] as a good approximation.
    """
    _, mu_c, _ = _truncated_normal_moments(mean, std, upper_quantile)
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


# =============================================================================
# LDA-Based Initialization
# =============================================================================


def _fit_nonnegative_lda_weights(
    X: np.ndarray,
    y: np.ndarray,
    weight_norm_order: Optional[int] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Fit non-negative LDA weights for initialization.

    Delegates to the utility function in dendritic_modeling.utils.lda.

    Returns:
        Non-negative weights, shape [n_features,]
    """
    weights, _ = fit_nonnegative_lda(X, y, weight_norm_order, gamma)
    return weights


def _weights_to_pre_w(weights: np.ndarray, weight_transform: str) -> torch.Tensor:
    """Convert positive weights to pre_w parameter space.

    Args:
        weights: Positive weights, shape [n_features,]
        weight_transform: The transform used ("exp", "softplus", "relu")

    Returns:
        pre_w values in the appropriate parameter space
    """
    weights = np.clip(weights, 1e-8, None)  # Ensure positive

    if weight_transform == "exp":
        # w = exp(pre_w), so pre_w = log(w)
        pre_w = np.log(weights)
    elif weight_transform == "softplus":
        # w = log(1 + exp(pre_w)), so pre_w = log(exp(w) - 1)
        # Use inverse softplus with numerical stability
        pre_w = np.where(
            weights > 20,
            weights,  # For large w, softplus(x) ≈ x
            np.log(np.exp(weights) - 1 + 1e-8),
        )
    elif weight_transform == "relu":
        # w = max(0, pre_w), so pre_w = w (for positive w)
        pre_w = weights
    else:
        # Default to exp
        pre_w = np.log(weights)

    return torch.tensor(pre_w, dtype=torch.float32)


def lda_analytical_dbl_init(
    dbl,
    X: torch.Tensor,
    y: torch.Tensor,
    inhibitory_input: Optional[torch.Tensor] = None,
    adaptive: bool = True,
    branch_factors: Optional[list] = None,
    use_lda_sparsity: bool = False,
    use_lda_weights: bool = True,
):
    """Initialize a dendritic branch layer using LDA weights + analytical scaling.

    This combines LDA-optimal weight directions (maximizing class separability)
    with analytical expectation scaling (proper conductance balance for shunting).

    Args:
        dbl: The DendriticBranchLayer to initialize
        X: Excitatory input data, shape [n_samples, excitatory_input_dim]
        y: Class labels, shape [n_samples,]
        inhibitory_input: Optional inhibitory input data, shape [n_samples, inhibitory_input_dim]
        adaptive: If True, use network-aware scaling to prevent overflow
        branch_factors: List of branch factors from parent network
        use_lda_sparsity: If True, use LDA importance scores to select which inputs are active
        use_lda_weights: If True, use LDA to initialize synaptic weight values
    """
    from dendritic_modeling.utils.lda import compute_lda_importance_scores

    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    # Convert inputs to numpy
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.array(X)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().flatten()
    else:
        y_np = np.array(y).flatten()

    if inhibitory_input is not None:
        if isinstance(inhibitory_input, torch.Tensor):
            inh_np = inhibitory_input.detach().cpu().numpy()
        else:
            inh_np = np.array(inhibitory_input)
    else:
        inh_np = X_np  # Use same input for inhibitory if not provided

    weight_transform = dbl.weight_transform

    # Compute network-aware scale factor
    network_scale_factor = (
        compute_network_scale_factor(dbl, branch_factors) if adaptive else 0.0
    )

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0

        # Initialize excitatory weights with LDA
        if dbl.input_excitatory and dbl.branch_excitation is not None:
            n_branches = dbl.branch_excitation.out_features
            K = dbl.branch_excitation.K
            weight_norm_order = getattr(
                dbl.branch_excitation, "weight_norm_order", None
            )
            gamma = getattr(dbl.branch_excitation, "gamma", 1.0)

            # Compute LDA importance scores if using LDA for sparsity selection
            if use_lda_sparsity:
                exc_importance = compute_lda_importance_scores(X_np, y_np)
                # Set pre_w based on importance so TopK naturally selects best features
                # Higher importance -> higher pre_w -> more likely to be in TopK
                # Scale to reasonable range for the weight transform
                scaled_importance = (exc_importance - exc_importance.min()) / (
                    exc_importance.max() - exc_importance.min() + 1e-10
                )
                # Initialize all branches with the same importance-based pre_w
                for branch_idx in range(n_branches):
                    dbl.branch_excitation.pre_w.data[branch_idx, :] = torch.tensor(
                        scaled_importance, dtype=torch.float32
                    ).to(dbl.branch_excitation.pre_w.device)

            # Get the TopK mask to know which inputs are active per branch
            # (This now reflects LDA-based selection if use_lda_sparsity=True)
            exc_mask = dbl.branch_excitation.weight_mask().detach().cpu().numpy()

            # Initialize weights with LDA if enabled
            if use_lda_weights:
                for branch_idx in range(n_branches):
                    branch_mask = exc_mask[branch_idx, :]
                    active_indices = np.where(branch_mask > 0)[0]

                    if len(active_indices) > 0:
                        # Fit LDA on the active inputs for this branch
                        branch_inputs = X_np[:, active_indices]
                        lda_weights = _fit_nonnegative_lda_weights(
                            branch_inputs, y_np, weight_norm_order, gamma
                        )

                        # Apply network scale factor
                        lda_weights = lda_weights * exp(-network_scale_factor)

                        # Convert to pre_w space and set
                        pre_w_values = _weights_to_pre_w(lda_weights, weight_transform)
                        dbl.branch_excitation.pre_w.data[branch_idx, active_indices] = (
                            pre_w_values.to(dbl.branch_excitation.pre_w.device)
                        )

            # Estimate expected conductance
            exc_upper_quantile = K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * K * E_w_exc * exp(-network_scale_factor)

        # Initialize inhibitory weights with LDA
        if dbl.input_inhibitory and dbl.branch_inhibition is not None:
            n_branches = dbl.branch_inhibition.out_features
            K = dbl.branch_inhibition.K
            weight_norm_order = getattr(
                dbl.branch_inhibition, "weight_norm_order", None
            )
            gamma = getattr(dbl.branch_inhibition, "gamma", 1.0)

            # Compute LDA importance scores if using LDA for sparsity selection
            if use_lda_sparsity:
                inh_importance = compute_lda_importance_scores(inh_np, y_np)
                # Set pre_w based on importance so TopK naturally selects best features
                scaled_importance = (inh_importance - inh_importance.min()) / (
                    inh_importance.max() - inh_importance.min() + 1e-10
                )
                for branch_idx in range(n_branches):
                    dbl.branch_inhibition.pre_w.data[branch_idx, :] = torch.tensor(
                        scaled_importance, dtype=torch.float32
                    ).to(dbl.branch_inhibition.pre_w.device)

            # Get the TopK mask
            inh_mask = dbl.branch_inhibition.weight_mask().detach().cpu().numpy()

            # Initialize weights with LDA if enabled
            if use_lda_weights:
                for branch_idx in range(n_branches):
                    branch_mask = inh_mask[branch_idx, :]
                    active_indices = np.where(branch_mask > 0)[0]

                    if (
                        len(active_indices) > 0
                        and active_indices.max() < inh_np.shape[1]
                    ):
                        # Fit LDA on the active inputs for this branch
                        branch_inputs = inh_np[:, active_indices]
                        lda_weights = _fit_nonnegative_lda_weights(
                            branch_inputs, y_np, weight_norm_order, gamma
                        )

                        # Apply network scale factor
                        lda_weights = lda_weights * exp(-network_scale_factor)

                        # Convert to pre_w space and set
                        pre_w_values = _weights_to_pre_w(lda_weights, weight_transform)
                        dbl.branch_inhibition.pre_w.data[branch_idx, active_indices] = (
                            pre_w_values.to(dbl.branch_inhibition.pre_w.device)
                        )

            # Estimate expected conductance
            inh_upper_quantile = K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * K * E_w_inh * exp(-network_scale_factor)

        # Initialize branch weights (use analytical method for BlockLinear)
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

        # Initialize reactivation
        if not isinstance(dbl.reactivation, torch.nn.Identity):
            dbl.reactivation.initialize(m, E_Vinf)


def initialize_model_with_lda(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    n_samples: int = 1000,
    adaptive: bool = True,
    use_lda_sparsity: bool = False,
    use_lda_weights: bool = True,
):
    """Initialize all dendritic branch layers in a model using LDA weights.

    This is a convenience function to apply LDA initialization to an entire model
    after it has been constructed.

    Args:
        model: The model containing DendriticBranchLayers
        X: Input data, shape [n_samples, input_dim]
        y: Class labels, shape [n_samples,]
        device: Device to run the model on
        n_samples: Number of samples to use for LDA fitting
        adaptive: Whether to use network-aware scaling
        use_lda_sparsity: If True, use LDA to select which inputs are active (TopK selection)
        use_lda_weights: If True, use LDA to initialize synaptic weight values

    Returns:
        The model with LDA-initialized weights
    """
    from .branch_layer import DendriticBranchLayer

    # Subsample if needed
    if X.shape[0] > n_samples:
        indices = torch.randperm(X.shape[0])[:n_samples]
        X = X[indices]
        y = y[indices]

    # Move data to device
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)

    # Forward pass to get intermediate activations and masks
    model.eval()
    with torch.no_grad():
        # Run a forward pass to establish TopK masks
        _ = model(X)

    # Find all DendriticBranchLayers and initialize them
    branch_factors = None
    for _name, module in model.named_modules():
        if isinstance(module, DendriticBranchLayer):
            # Get branch_factors from the layer if available
            if hasattr(module, "branch_factors"):
                branch_factors = module.branch_factors

            # For now, use X as both excitatory and inhibitory input
            # In practice, you might want to extract the actual layer inputs
            lda_analytical_dbl_init(
                dbl=module,
                X=X,
                y=y,
                inhibitory_input=X,
                adaptive=adaptive,
                branch_factors=branch_factors,
                use_lda_sparsity=use_lda_sparsity,
                use_lda_weights=use_lda_weights,
            )

    return model
