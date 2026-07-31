"""
Parametric activation functions.

This module contains various parametric activation functions with learnable parameters.
"""

import logging
import math
import os

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import safe_exp

# --------------------------------------------------------------------------- #
# Positive-tanh fast path                                                      #
# --------------------------------------------------------------------------- #
# Every "positive tanh" reactivation in this file computes (tanh(z) + 1) / 2,
# which is exactly sigmoid(2z). The naive spelling runs four full-size
# elementwise kernels (tanh, add, div, plus the scale) and allocates a full-size
# temporary for each; the identity runs two and allocates one. An allocation
# profile of the 12-layer stress config attributed 39.9% of live memory to this
# single line, so the temporaries are not incidental.
#
# ON by default since 2026-07-30. It shipped off pending evidence; the evidence
# arrived from two directions:
#   * an allocation profile of the 12-layer stress config (cuda, batch 512) puts
#     THIS LINE at 14,943 MB = 57.6% of live allocation -- the single largest
#     thing in the model, ahead of branch_dynamics at 36.8% combined;
#   * equivalence is settled, not assumed: bit-identical in fp64 for every
#     touched activation (so the fp32 gap of 1.2e-07 value / 2.4e-07 through the
#     whole network is rounding, not a different function), and 200 Adam steps
#     down each path end at a final-loss relative difference of exactly 0.00e+00.
# Measured effect: 1.19x peak memory (4.8 GB at batch 512), ~1.03x step time.
#
# Set DENDRINET_POSITIVE_TANH_SIGMOID=0 or call set_positive_tanh_sigmoid(False)
# to get the old spelling back -- the switch is kept precisely so a numerics
# question can be answered by flipping it rather than by reverting a commit.
_USE_SIGMOID_IDENTITY = os.environ.get("DENDRINET_POSITIVE_TANH_SIGMOID", "1") != "0"


def set_positive_tanh_sigmoid(enabled: bool) -> None:
    """Enable/disable the sigmoid(2z) fast path for positive-tanh activations."""
    global _USE_SIGMOID_IDENTITY
    _USE_SIGMOID_IDENTITY = bool(enabled)


def positive_tanh_sigmoid_enabled() -> bool:
    """Whether the sigmoid(2z) fast path is currently active."""
    return _USE_SIGMOID_IDENTITY


# Fuse the whole positive-tanh into one autograd Function that retains only its
# OUTPUT. Default on; DENDRINET_POSITIVE_TANH_FUSED=0 disables.
_USE_FUSED_POSITIVE_TANH = os.environ.get("DENDRINET_POSITIVE_TANH_FUSED", "1") != "0"


def set_fused_positive_tanh(enabled: bool) -> None:
    """Enable/disable the single-Function positive-tanh that saves only y."""
    global _USE_FUSED_POSITIVE_TANH
    _USE_FUSED_POSITIVE_TANH = bool(enabled)


def fused_positive_tanh_enabled() -> bool:
    return _USE_FUSED_POSITIVE_TANH


class _PositiveTanhCentered(torch.autograd.Function):
    """``y = sigmoid(2m(x - c))``, retaining **only** ``y``.

    Written as one Function because of an asymmetry in what each gradient needs.
    With ``z = 2m(x-c)`` and ``s = dy/dz = y(1-y)``:

        grad_x = g*s*2m               <- needs only y
        grad_c = -2m * sum(g*s)       <- needs only y
        grad_m = (1/m) * sum(g*s*z)   <- needs z

    So only ``grad_m`` wants anything besides ``y`` -- and ``z`` is recoverable
    from ``y`` as ``logit(y)``. The eager spelling instead retains TWO full-size
    tensors per call: ``x - c`` (MulBackward needs it for grad_m) and ``y``
    (SigmoidBackward needs it). An allocation profile put this line at 57.6% of
    live allocation, so halving its retained set is the largest single memory
    lever in the model.

    The reconstruction is safe *because of* the saturation that makes it look
    unsafe. Once |z| exceeds ~17 the fp32 sigmoid returns exactly 1.0 (or 0.0),
    ``logit`` returns +-inf, and inf*0 would be nan -- but at exactly those
    points ``y*(1-y)`` is exactly 0, and PyTorch's own sigmoid backward computes
    the same ``g*y*(1-y)``, so the reference contribution is 0 too. Bounding the
    logit argument (never the ``y`` used for ``s``) replaces inf with a finite
    stand-in that the 0 annihilates, reproducing the reference rather than
    approximating it. Measured against autograd at saturation fractions from 0%
    to 58.8%: grad_log_m agrees to <= 3.8e-07 with no nan
    (probe_positive_tanh_saved.py).
    """

    @staticmethod
    def forward(ctx, x, scale, center):
        two_m = scale + scale
        y = torch.sigmoid(two_m * (x - center))
        center_t = center if torch.is_tensor(center) else None
        ctx.save_for_backward(y, scale, center_t)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        y, scale, center_t = ctx.saved_tensors
        needs_x, needs_scale, needs_center = ctx.needs_input_grad[:3]
        two_m = scale + scale
        gs = grad_out * y * (1.0 - y)          # g * dy/dz
        reduce_dims = tuple(range(gs.dim() - 1))

        grad_x = gs * two_m if needs_x else None
        grad_scale = None
        if needs_scale:
            # z = logit(y); eps bounds ONLY the logit argument. Where y has
            # saturated, gs is exactly 0, so the finite stand-in contributes 0.
            eps = torch.finfo(y.dtype).eps
            z = torch.logit(y, eps)
            grad_scale = (gs * z).sum(dim=reduce_dims) / scale
        grad_center = None
        if needs_center and center_t is not None:
            grad_center = -(gs * two_m).sum(dim=reduce_dims)
        return grad_x, grad_scale, grad_center


def _positive_tanh_centered(scale, x, center):
    """(tanh(scale * (x - center)) + 1) / 2.

    The 2 is folded into ``scale`` (shape [output_dim]) rather than applied to
    ``x``, so the fast path costs the same number of full-size kernels as the
    slow one minus the add and the divide. ``x - center`` is kept intact rather
    than expanded to ``scale*x - scale*center``: the expanded form cancels
    catastrophically exactly where ``x ~ center``, which is the steep part of
    the curve.
    """
    if _USE_SIGMOID_IDENTITY and _USE_FUSED_POSITIVE_TANH:
        return _PositiveTanhCentered.apply(x, scale, center)
    if _USE_SIGMOID_IDENTITY:
        return torch.sigmoid((2.0 * scale) * (x - center))
    return (torch.tanh(scale * (x - center)) + 1) / 2


class ParametricActivation(nn.Module):
    """
    Base class for parametric activation functions.

    Provides common initialization and parameter management functionality for parametric
    activation functions with learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, *args, **kwargs):
        """
        Initialize the activation function parameters.
        Should be implemented by child classes.
        """
        raise NotImplementedError(
            "Initialize method must be implemented by child class"
        )

    def forward(self, x):
        """
        Forward pass of the activation function.
        Should be implemented by child classes.
        """
        raise NotImplementedError("Forward method must be implemented by child class")


class ParametricTanh(ParametricActivation):
    """
    A parametric tanh reactivation function with learnable parameters m and b.
    Both parameters are trainable.
    """

    def __init__(self, output_dim, init_m=1.5, init_b=1.0):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        # self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)

        nn.init.constant_(self.log_m, math.log(init_m))
        nn.init.constant_(self.b, init_b)

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.b  # midpoint
        return _positive_tanh_centered(m, V, b)

    def initialize(self, slope_val, mid_val):
        self.log_m.data.fill_(math.log(slope_val))
        self.b.data.fill_(mid_val)


class ParametricTanhOnlyM(ParametricActivation):
    """
    A parametric tanh activation function where b is fixed and only m is
    trainable. The fixed value of b is provided via the fixed_b argument.
    """

    def __init__(self, output_dim, init_m=1.5, fixed_b=0.5):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.fixed_b = fixed_b  # fixed midpoint value
        with torch.no_grad():
            self.log_m.data.fill_(math.log(init_m))

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.fixed_b  # fixed midpoint
        return _positive_tanh_centered(m, V, b)

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearSigmoid(ParametricActivation):
    """
    A parametric activation function that interpolates between linear and sigmoid functions.
    The function is: h(x)*g(x) + f(x)*(1-g(x)) where:
    - h(x) = x (linear function)
    - g(x) = exp(x/2) (exponential weighting)
    - f(x) = 4/(1 + exp(-(mx + 2))) (sigmoid function)

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    init_m : float
        Initial value for the slope parameter
    init_b : float or None
        Initial value for the bias parameter
    trainable : bool
        Whether to train the slope parameter (default: True)
    """

    def __init__(self, output_dim, init_m=4, init_b=None, trainable=True):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=trainable)

        # Initialize trainable parameter
        if init_m == 0:
            init_m = 1e-8
        nn.init.constant_(self.log_m, math.log(init_m))

        # Fixed parameters for sigmoid
        if init_b is None:
            self.fixed_b = -2.0
        else:
            self.fixed_b = init_b

    def forward(self, x):
        m = self.log_m.exp()  # slope

        # Calculate the three component functions
        h_x = x  # linear function
        g_x = safe_exp(-3 * x, max_exponent=20.0)  # exponential weighting
        f_x = torch.sigmoid(m * x + self.fixed_b)  # sigmoid

        # Combine using the interpolation formula
        return h_x * g_x + f_x * (1 - g_x)

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearTanh(ParametricActivation):
    """
    A parametric activation function that interpolates between linear and tanh functions.
    The function is: h(x)*g(x) + f(x)*(1-g(x)) where:
    - h(x) = x (linear function)
    - g(x) = exp(x/2) (exponential weighting)
    - f(x) = (tanh(mx + b)+1)/2 (positivetanh function)

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    init_m : float
        Initial value for the slope parameter
    init_b : float or None
        Initial value for the bias parameter
    trainable : bool
        Whether to train the slope parameter (default: True)
    """

    def __init__(self, output_dim, init_m=4, init_b=None, trainable=True):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=trainable)

        # Initialize trainable parameter
        if init_m == 0:
            init_m = 1e-8
        nn.init.constant_(self.log_m, math.log(init_m))

        # Fixed parameters for sigmoid
        if init_b is None:
            self.fixed_b = -2.0
        else:
            self.fixed_b = init_b

    def forward(self, x):
        m = self.log_m.exp()  # slope

        # Calculate the three component functions
        h_x = x  # linear function
        g_x = safe_exp(-3 * x, max_exponent=20.0)  # exponential weighting
        # Deliberately NOT routed through the sigmoid fast path. The (1 - g_x)
        # factor below reaches ~1e5, which amplifies the identity's 1e-7
        # rounding into a 4e-4 relative shift on the parameter gradients (in
        # fp64 the two forms are bit-identical, so it is rounding, not a
        # different function -- see check_positive_tanh_equiv.py). No config
        # selects this activation, so there is nothing to win by taking that on.
        f_x = (torch.tanh(m * x + self.fixed_b) + 1) / 2  # tanh
        # Combine using the interpolation formula
        return h_x * g_x + f_x * (1 - g_x)

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearTanhTransition(ParametricActivation):
    """
    A differentiable activation function operating element-wise for a given output dimension.
    Each element starts linear (scalar slope linear_slope) for x < threshold[i],
    and transitions into a scaled and shifted tanh function for x >= threshold[i],
    saturating at 1.
    The transition threshold and transition slope are potentially learnable parameters for each dimension.

    Requires threshold[i] < 1 / linear_slope for proper behavior (saturation at 1).
    Requires transition_slope[i] > 0.
    """

    def __init__(
        self,
        output_dim: int,
        init_threshold=0.0,
        linear_slope=1.0,
        init_transition_slope=None,
        trainable_threshold=True,
        trainable_transition_slope=True,
    ):
        super().__init__()

        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        self.output_dim = output_dim

        if linear_slope <= 0:
            raise ValueError("linear_slope must be positive.")
        # linear_slope remains a scalar shared across dimensions
        self.linear_slope = linear_slope

        # --- Initialize Threshold ---
        max_init_threshold = (1.0 / self.linear_slope) - 1e-6
        threshold_vals = self._initialize_param_val(
            init_threshold, output_dim, "init_threshold"
        )
        # Clamp initial values
        clamped_threshold_vals = torch.clamp(threshold_vals, max=max_init_threshold)
        if torch.any(threshold_vals > max_init_threshold):
            logging.warning(
                f"Some initial thresholds violated constraint threshold < {1.0/self.linear_slope:.4f}. They were clamped."
            )

        if trainable_threshold:
            self.threshold = nn.Parameter(clamped_threshold_vals)
        else:
            self.register_buffer("threshold", clamped_threshold_vals)

        # --- Initialize Transition Slope ---
        if init_transition_slope is None:
            # Default to linear_slope if not provided
            init_transition_slope = self.linear_slope
        transition_slope_vals = self._initialize_param_val(
            init_transition_slope, output_dim, "init_transition_slope"
        )
        # Clamp initial values to be positive
        clamped_transition_slope_vals = torch.clamp(transition_slope_vals, min=1e-9)
        if torch.any(transition_slope_vals <= 0):
            logging.warning(
                "Some initial transition slopes were <= 0. They were clamped to min=1e-9."
            )

        if trainable_transition_slope:
            self.transition_slope = nn.Parameter(clamped_transition_slope_vals)
        else:
            self.register_buffer("transition_slope", clamped_transition_slope_vals)

    def _initialize_param_val(self, init_val, dim, name):
        """Helper to initialize parameter values from scalar or tensor/list."""
        if isinstance(init_val, (int, float)):
            return torch.full((dim,), float(init_val))
        elif isinstance(init_val, torch.Tensor):
            if init_val.shape == (dim,):
                return init_val.clone().float()
            else:
                raise ValueError(
                    f"{name} tensor has wrong shape ({init_val.shape}), expected ({dim},)"
                )
        elif isinstance(init_val, (list, tuple)):
            if len(init_val) == dim:
                return torch.tensor(init_val, dtype=torch.float32)
            else:
                raise ValueError(
                    f"{name} list/tuple has wrong length ({len(init_val)}), expected {dim}"
                )
        else:
            raise TypeError(f"Unsupported type for {name}: {type(init_val)}")

    def forward(self, x):
        # x shape: (batch_size, ..., output_dim) or (output_dim,)
        # parameters shape: (output_dim,) - broadcasting will handle element-wise ops

        # Clamp threshold to ensure 1 - m * threshold > 0
        max_threshold = (1.0 / self.linear_slope) - 1e-6  # scalar
        threshold = torch.clamp(
            self.threshold, max=max_threshold
        )  # shape (output_dim,)

        # Clamp transition_slope to ensure > 0
        transition_slope = torch.clamp(
            self.transition_slope, min=1e-8
        )  # shape (output_dim,)

        # Parameters A, B, D now depend on threshold and transition_slope per dimension
        # Broadcasting applies: (output_dim,) op (output_dim,) -> (output_dim,)
        A = 1.0 - self.linear_slope * threshold
        D = self.linear_slope * threshold
        # Add epsilon for stability when A is close to zero
        B = transition_slope / (A + 1e-8)

        # Apply element-wise logic using broadcasting
        # x: (..., output_dim), threshold: (output_dim,) -> comparison: (..., output_dim)
        linear_part = self.linear_slope * x
        # Tanh part involves ops between x and params.
        # (x - threshold) -> shape (..., output_dim)
        # B * (x - threshold) -> shape (..., output_dim)
        # A * tanh(...) + D -> shape (..., output_dim)
        tanh_part = A * torch.tanh(B * (x - threshold)) + D

        output = torch.where(x < threshold, linear_part, tanh_part)

        return output

    def extra_repr(self):
        is_thresh_trainable = isinstance(self.threshold, nn.Parameter)
        is_slope_trainable = isinstance(self.transition_slope, nn.Parameter)
        return (
            f"output_dim={self.output_dim}, linear_slope={self.linear_slope}, "
            f"threshold=Parameter(shape={tuple(self.threshold.shape)}, trainable={is_thresh_trainable}), "
            f"transition_slope=Parameter(shape={tuple(self.transition_slope.shape)}, trainable={is_slope_trainable})"
        )

    def initialize(self, slope_val, mid_val):
        max_threshold = (1.0 / self.linear_slope) - 1e-6
        threshold_val = min(float(mid_val), max_threshold)
        safe_slope = max(float(slope_val), 1e-8)
        self.threshold.data.fill_(threshold_val)
        self.transition_slope.data.fill_(safe_slope)


__all__ = [
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "ParametricTanh",
    "ParametricTanhOnlyM",
    "positive_tanh_sigmoid_enabled",
    "set_positive_tanh_sigmoid",
]
