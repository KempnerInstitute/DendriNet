from __future__ import annotations

from math import exp, sqrt

import torch
from torch.nn.init import _no_grad_normal_

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_expectations import (
    _expected_weight_for_transform,
    _mean_for_expected_weight,
    _solve_first_crossing_expected_weight_mean,
    normalize_adaptive_initialization_policy,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_reactivation import (
    _apply_reactivation_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_scaling import (
    _center_preserving_shunting_conductances,
    compute_network_scale_factor,
)


def _seeded_parameter_init(dbl, component: str, initializer, *args, **kwargs):
    """Apply a random initializer in the branch layer's keyed RNG scope."""
    with dbl.initialization_seed_scope(f"final.{component}"):
        return initializer(*args, **kwargs)


def identity_weighttransform_dbl_init(dbl):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        if dbl.branch_excitation is not None:
            _seeded_parameter_init(
                dbl,
                "ff_excitatory",
                torch.nn.init.xavier_normal_,
                dbl.branch_excitation.pre_w,
            )

        if dbl.branch_recurrent is not None:
            _seeded_parameter_init(
                dbl,
                "rec_excitatory",
                torch.nn.init.xavier_normal_,
                dbl.branch_recurrent.pre_w,
            )

        if dbl.branch_inhibition is not None:
            _seeded_parameter_init(
                dbl,
                "ff_inhibitory",
                torch.nn.init.xavier_normal_,
                dbl.branch_inhibition.pre_w,
            )

        if dbl.branch_rec_inhibition is not None:
            _seeded_parameter_init(
                dbl,
                "rec_inhibitory",
                torch.nn.init.xavier_normal_,
                dbl.branch_rec_inhibition.pre_w,
            )

        if hasattr(dbl, "branches_to_output"):
            _seeded_parameter_init(
                dbl,
                "branch_aggregation",
                torch.nn.init.xavier_normal_,
                dbl.branches_to_output.log_weight,
            )

        if dbl.reactivate:
            _apply_reactivation_init(dbl, m_auto=1.0, b_auto=0.0)


def mechanism_neutral_dbl_init(dbl):
    """Initialize paired shunting/additive controls identically.

    This initializer is intentionally mechanism agnostic: it never branches on
    ``use_shunting``.  With the same model seed and architecture, synaptic
    parameters, child-branch couplings, and reactivation parameters therefore
    start bit-identically in a shunting/additive pair.  It is intended for
    causal rule-only controls; mechanism-specific initializers remain the right
    choice for best-tuned system comparisons.
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        synapse_specs = (
            ("branch_excitation", "excitatory_input_dim"),
            ("branch_inhibition", "inhibitory_input_dim"),
            ("branch_recurrent", "recurrent_input_dim"),
            ("branch_rec_inhibition", "rec_inhibitory_input_dim"),
        )
        for module_attr, input_dim_attr in synapse_specs:
            synapse = getattr(dbl, module_attr, None)
            if synapse is None:
                continue
            input_dim = int(getattr(dbl, input_dim_attr))
            component = {
                "branch_excitation": "ff_excitatory",
                "branch_inhibition": "ff_inhibitory",
                "branch_recurrent": "rec_excitatory",
                "branch_rec_inhibition": "rec_inhibitory",
            }[module_attr]
            _seeded_parameter_init(
                dbl,
                component,
                _no_grad_normal_,
                synapse.pre_w,
                0.0,
                sqrt(2 / (input_dim + 1)),
            )

        if dbl.input_branches:
            # A fixed positive child coupling is shared by both mechanisms.
            # The default remains exactly the historical value of 1.0.
            dbl.branches_to_output.initialize(
                g_branch=float(getattr(dbl, "initial_child_conductance", 1.0))
            )

        if dbl.reactivate:
            # ``fixed`` policy uses the explicit configured values; the
            # fallback is also shared if a caller selects another policy.
            _apply_reactivation_init(dbl, m_auto=1.5, b_auto=0.5)


def _normal_for_expected_weight(
    transform: str, mean: float, std: float, upper_quantile: float, target: float
) -> tuple[float, float]:
    """Return a raw-weight (mean, std) whose top-K expected weight is ``target``.

    ``exp`` and ``softplus`` move the mean. ``relu`` scales mean and std
    instead: ``relu(a x) = a relu(x)``, so every contact keeps its activity,
    where moving the mean down would switch contacts off.
    """
    if (transform or "exp").lower() == "relu":
        current = _expected_weight_for_transform(
            transform="relu", mean=mean, std=std, upper_quantile=upper_quantile
        )
        scale = max(float(target), 1e-12) / max(current, 1e-12)
        return mean * scale, std * scale
    return (
        _mean_for_expected_weight(
            transform, std, upper_quantile, target, reference_mean=mean
        ),
        std,
    )


def _scaled_normal(
    transform: str, mean: float, std: float, upper_quantile: float, scale_factor: float
) -> tuple[float, float]:
    """Return a raw-weight (mean, std) that scales the expected weight by ``exp(-f)``.

    For ``exp`` weights this is the mean shift ``mean - f``.
    """
    if scale_factor <= 0.0:
        return mean, std
    if (transform or "exp").lower() == "exp":
        return mean - scale_factor, std
    target = _expected_weight_for_transform(
        transform=transform, mean=mean, std=std, upper_quantile=upper_quantile
    ) * exp(-scale_factor)
    return _normal_for_expected_weight(transform, mean, std, upper_quantile, target)


def shunting_child_coupling(
    e_g_exc: float, e_g_inh: float, block_size: int, leak: float = 1.0
) -> tuple[float, float]:
    """Return the per-child coupling and gate centre of a shunting compartment.

    With its children at their output 0.5, a total child coupling ``G`` holds
    a compartment with expected conductances ``E`` and ``I`` and leak ``L`` at
    ``V = (E + 0.5 G) / (E + I + L + G)``, between ``E / (E + I + L)`` at
    ``G = 0`` and 0.5 for large ``G``. ``G`` is chosen so that ``V`` is the
    returned gate centre: 0.49 when ``E / (E + I + L) < 0.48`` (a strong
    coupling), otherwise the midpoint between ``E / (E + I + L)`` and 0.5,
    which ``G = E + I + L`` realizes. The midpoint rule also covers
    ``E >= I + L`` (for example a compartment without inhibition), where no
    coupling brings ``V`` below 0.5.
    """
    e_vinf = e_g_exc / (e_g_exc + e_g_inh + leak)
    if e_vinf < 0.5:
        e_vinf = max(0.49, 0.5 * e_vinf + 0.25)
        sum_g_branch = ((e_g_exc + e_g_inh + leak) * e_vinf - e_g_exc) / (0.5 - e_vinf)
    else:
        e_vinf = 0.5 * e_vinf + 0.25
        sum_g_branch = e_g_exc + e_g_inh + leak
    return max(sum_g_branch / block_size, 1e-6), e_vinf


def _initial_leak(dbl) -> float:
    """The compartment leak at initialization (the learnable leak starts there)."""
    return float(getattr(dbl, "leak_conductance", 1.0))


def _require_branch_contacts(dbl, method: str) -> None:
    """Raise when a pathway the initializer divides by has no contacts per branch.

    The analytical initializers scale by the expected contacts per branch,
    ``K``. DeepST sets ``K = int(target_density * in_features)``, which is
    zero when the density gives less than one contact per branch.
    """
    for name, synapses in (
        ("excitatory", dbl.branch_excitation),
        ("inhibitory", dbl.branch_inhibition),
    ):
        if synapses is None or getattr(synapses, "K", 1):
            continue
        raise ValueError(
            f"dbl_init_method {method!r} needs at least one {name} contact per "
            f"branch, but this layer's {name} synapses expect "
            f"K={getattr(synapses, 'K', None)} over "
            f"{getattr(synapses, 'in_features', '?')} inputs (DeepST rounds "
            "target_density x in_features down). Raise the target density, use "
            "rewiring_mode: constant_branch (which keeps the pathway's synapse "
            "count per branch), or choose another dbl_init_method."
        )


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
    _require_branch_contacts(dbl, "analytical_expectation")

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        E_w_exc = 0
        E_w_inh = 0
        exc_mean = 0
        inh_mean = 0
        exc_std = None
        inh_std = None
        exc_upper_quantile = None
        inh_upper_quantile = None
        weight_transform = dbl.weight_transform
        adaptive_policy = normalize_adaptive_initialization_policy(
            getattr(dbl, "adaptive_initialization_policy", "preserve_shunting_center")
        )
        adaptive_target_conductance = float(
            getattr(dbl, "adaptive_target_conductance", 5.0)
        )

        # Compute network-aware scale factor if adaptive mode is enabled
        network_scale_factor = (
            compute_network_scale_factor(
                dbl,
                branch_factors,
                target_conductance=adaptive_target_conductance,
            )
            if adaptive
            else 0.0
        )

        if dbl.input_excitatory:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_mean = 0
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))

            if not dbl.input_inhibitory and not dbl.input_branches:
                exc_mean = _solve_first_crossing_expected_weight_mean(
                    weight_transform,
                    exc_std,
                    exc_upper_quantile,
                    2.0 / dbl.branch_excitation.K,
                    direction=-1,
                    reference_mean=exc_mean,
                )
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=exc_mean,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )

            else:
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=exc_mean,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )

            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.input_inhibitory:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_mean = 0
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))

            if dbl.input_excitatory:
                ie_ratio = dbl.branch_inhibition.K / dbl.branch_excitation.K
                if ie_ratio <= 0.0:
                    raise ValueError(
                        "analytical E/I initialization requires positive "
                        "excitatory and inhibitory contact counts"
                    )
                inh_std = inh_std * ie_ratio if ie_ratio < 1 else inh_std
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

                if E_w_exc >= ie_ratio * E_w_inh:
                    direction = 1
                else:
                    direction = -1
                inh_mean = _solve_first_crossing_expected_weight_mean(
                    weight_transform,
                    inh_std,
                    inh_upper_quantile,
                    E_w_exc / ie_ratio,
                    direction=direction,
                    reference_mean=inh_mean,
                )
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

            else:
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        E_g_exc_target = E_g_exc
        E_g_inh_target = E_g_inh

        if adaptive and network_scale_factor > 0.0:
            if adaptive_policy == "preserve_shunting_center" and dbl.use_shunting:
                E_g_exc_target, E_g_inh_target = (
                    _center_preserving_shunting_conductances(
                        E_g_exc,
                        E_g_inh,
                        adaptive_target_conductance,
                        leak=_initial_leak(dbl),
                    )
                )
            else:
                scale = exp(-network_scale_factor)
                E_g_exc_target = E_g_exc * scale
                E_g_inh_target = E_g_inh * scale

        if dbl.input_excitatory:
            if (
                adaptive
                and adaptive_policy == "legacy_scale"
                and network_scale_factor > 0.0
            ):
                exc_mean_adjusted, exc_std = _scaled_normal(
                    weight_transform,
                    exc_mean,
                    exc_std,
                    exc_upper_quantile,
                    network_scale_factor,
                )
            elif E_g_exc > 0.0 and E_g_exc_target != E_g_exc:
                target_w_exc = 2.0 * E_g_exc_target / dbl.branch_excitation.K
                exc_mean_adjusted, exc_std = _normal_for_expected_weight(
                    weight_transform,
                    exc_mean,
                    exc_std,
                    exc_upper_quantile,
                    target_w_exc,
                )
            else:
                exc_mean_adjusted = exc_mean
            _seeded_parameter_init(
                dbl,
                "ff_excitatory",
                _no_grad_normal_,
                dbl.branch_excitation.pre_w,
                exc_mean_adjusted,
                exc_std,
            )
            E_g_exc = E_g_exc_target

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            rec_mean_adjusted, rec_std = _scaled_normal(
                weight_transform,
                0.0,
                rec_std,
                dbl.branch_recurrent.K / dbl.recurrent_input_dim,
                network_scale_factor,
            )
            _seeded_parameter_init(
                dbl,
                "rec_excitatory",
                _no_grad_normal_,
                dbl.branch_recurrent.pre_w,
                rec_mean_adjusted,
                rec_std,
            )

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            rec_inh_mean_adjusted, rec_inh_std = _scaled_normal(
                weight_transform,
                0.0,
                rec_inh_std,
                dbl.branch_rec_inhibition.K / dbl.rec_inhibitory_input_dim,
                network_scale_factor,
            )
            _seeded_parameter_init(
                dbl,
                "rec_inhibitory",
                _no_grad_normal_,
                dbl.branch_rec_inhibition.pre_w,
                rec_inh_mean_adjusted,
                rec_inh_std,
            )

        if dbl.input_inhibitory:
            if (
                adaptive
                and adaptive_policy == "legacy_scale"
                and network_scale_factor > 0.0
            ):
                inh_mean_adjusted, inh_std = _scaled_normal(
                    weight_transform,
                    inh_mean,
                    inh_std,
                    inh_upper_quantile,
                    network_scale_factor,
                )
            elif E_g_inh > 0.0 and E_g_inh_target != E_g_inh:
                target_w_inh = 2.0 * E_g_inh_target / dbl.branch_inhibition.K
                inh_mean_adjusted, inh_std = _normal_for_expected_weight(
                    weight_transform,
                    inh_mean,
                    inh_std,
                    inh_upper_quantile,
                    target_w_inh,
                )
            else:
                inh_mean_adjusted = inh_mean
            _seeded_parameter_init(
                dbl,
                "ff_inhibitory",
                _no_grad_normal_,
                dbl.branch_inhibition.pre_w,
                inh_mean_adjusted,
                inh_std,
            )
            E_g_inh = E_g_inh_target

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch, E_Vinf = shunting_child_coupling(
                    E_g_exc,
                    E_g_inh,
                    dbl.branches_to_output.block_size,
                    leak=_initial_leak(dbl),
                )
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + _initial_leak(dbl))
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


def ei_equivalence_dbl_init(dbl):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)
    _require_branch_contacts(dbl, "ei_equivalence")

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
            _seeded_parameter_init(
                dbl,
                "ff_excitatory",
                _no_grad_normal_,
                dbl.branch_excitation.pre_w,
                0,
                exc_std,
            )

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_excitatory",
                _no_grad_normal_,
                dbl.branch_recurrent.pre_w,
                0,
                rec_std,
            )

        if dbl.branch_inhibition is not None:
            _seeded_parameter_init(
                dbl,
                "ff_inhibitory",
                _no_grad_normal_,
                dbl.branch_inhibition.pre_w,
                0,
                inh_std,
            )

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_inhibitory",
                _no_grad_normal_,
                dbl.branch_rec_inhibition.pre_w,
                0,
                rec_inh_std,
            )

        if dbl.input_branches:
            if dbl.use_shunting:
                leak = _initial_leak(dbl)
                g_branch = E_g_exc + E_g_inh + leak
                sum_g_branch = g_branch * dbl.branches_to_output.block_size
                E_Vinf = (E_g_exc + 0.5 * sum_g_branch) / (
                    E_g_exc + E_g_inh + sum_g_branch + leak
                )
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + _initial_leak(dbl))
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


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
            _seeded_parameter_init(
                dbl,
                "ff_excitatory",
                _no_grad_normal_,
                dbl.branch_excitation.pre_w,
                0,
                exc_std,
            )

            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_excitatory",
                _no_grad_normal_,
                dbl.branch_recurrent.pre_w,
                0,
                rec_std,
            )

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_inhibitory",
                _no_grad_normal_,
                dbl.branch_rec_inhibition.pre_w,
                0,
                rec_inh_std,
            )

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "ff_inhibitory",
                _no_grad_normal_,
                dbl.branch_inhibition.pre_w,
                0,
                inh_std,
            )

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            leak = _initial_leak(dbl)
            g_branch = E_g_exc + E_g_inh + leak
            dbl.branches_to_output.initialize(g_branch=g_branch)

            if dbl.use_shunting:
                E_Vinf = (E_g_exc + 0.5 * g_branch) / (
                    E_g_exc + E_g_inh + g_branch + leak
                )
                m = 1.5
            else:
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + _initial_leak(dbl))
                m = 1.5
            else:
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


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
            _seeded_parameter_init(
                dbl,
                "ff_excitatory",
                _no_grad_normal_,
                dbl.branch_excitation.pre_w,
                0,
                exc_std,
            )

            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_excitatory",
                _no_grad_normal_,
                dbl.branch_recurrent.pre_w,
                0,
                rec_std,
            )

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "rec_inhibitory",
                _no_grad_normal_,
                dbl.branch_rec_inhibition.pre_w,
                0,
                rec_inh_std,
            )

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            _seeded_parameter_init(
                dbl,
                "ff_inhibitory",
                _no_grad_normal_,
                dbl.branch_inhibition.pre_w,
                0,
                inh_std,
            )

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + _initial_leak(dbl)
            else:
                g_branch = 1

            dbl.branches_to_output.initialize(g_branch=g_branch)

        if dbl.use_shunting:
            m_auto = 1.5
            b_auto = 0.5
        else:
            m_auto = 0.1
            b_auto = 0.0
        _apply_reactivation_init(dbl, m_auto=m_auto, b_auto=b_auto)


__all__ = [
    "analytical_expectation_dbl_init",
    "default_dbl_init",
    "ei_equivalence_dbl_init",
    "identity_weighttransform_dbl_init",
    "mechanism_neutral_dbl_init",
    "naive_dbl_init",
]
