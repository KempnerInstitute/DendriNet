from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ParametricActivation,
    TopKLinear,
)
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseVariationalAutoencoder,
)
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_modules_of_type,
    register_forward_hook_groups,
)

LossBuilder = Callable[[], "LossFunction"]
LOSS_FUNCTION_REGISTRY: dict[str, LossBuilder] = {}


def _identity_reduction(loss: torch.Tensor) -> torch.Tensor:
    """Return unreduced loss tensors for non-mean/sum reductions."""

    return loss


def _loss_reduction_fn(reduction: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the callable form of the project loss reduction convention."""

    if reduction == "mean":
        return torch.mean
    if reduction == "sum":
        return torch.sum
    return _identity_reduction


def _reduce_tensor_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply the project loss reduction convention to a tensor."""

    return _loss_reduction_fn(reduction)(loss)


def _difference_first_mean_error(
    reduction_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the historical voltage mean-error term."""
    return reduction_fn(x - y)


def _separate_reduction_mean_error(
    reduction_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the historical E/I balance mean-error term."""
    return reduction_fn(x) - reduction_fn(y)


def _build_pair_metric_fn(
    loss_metric: str,
    *,
    reduction: str,
    eps: float,
    error_context: str,
    reduce_mean_error_difference_first: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a two-tensor metric while preserving domain-specific formulas."""
    reduction_fn = _loss_reduction_fn(reduction)
    mean_error_fn = (
        _difference_first_mean_error
        if reduce_mean_error_difference_first
        else _separate_reduction_mean_error
    )

    if loss_metric == "mse":
        return lambda x, y: reduction_fn((x - y) ** 2)
    if loss_metric == "log_mse":
        return lambda x, y: torch.log(reduction_fn((x - y) ** 2) + eps)
    if loss_metric == "mae":
        return lambda x, y: reduction_fn(torch.abs(x - y))
    if loss_metric == "log_mae":
        return lambda x, y: torch.log(reduction_fn(torch.abs(x - y)) + eps)
    if loss_metric == "squared_mean_error":
        return lambda x, y: mean_error_fn(reduction_fn, x, y) ** 2
    if loss_metric == "log_squared_mean_error":
        return lambda x, y: torch.log(mean_error_fn(reduction_fn, x, y) ** 2 + eps)
    if loss_metric == "abs_mean_error":
        return lambda x, y: torch.abs(mean_error_fn(reduction_fn, x, y))
    if loss_metric == "log_abs_mean_error":
        return lambda x, y: torch.log(
            torch.abs(mean_error_fn(reduction_fn, x, y)) + eps
        )

    raise ValueError(f"Invalid metric for {error_context}: {loss_metric}")


def register_loss_function(
    name: str,
    builder: LossBuilder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    allow_override: bool = False,
) -> None:
    """Register a loss-function builder.

    Loss names remain case-sensitive for backward compatibility.
    Builders are zero-argument callables that return a ``LossFunction``.
    """
    names = [name, *aliases]
    if not names or any(not str(item).strip() for item in names):
        raise ValueError("loss function name and aliases must be non-empty")
    for key in names:
        if key in LOSS_FUNCTION_REGISTRY and not allow_override:
            raise ValueError(f"Loss function '{key}' is already registered")
        LOSS_FUNCTION_REGISTRY[str(key)] = builder


def unregister_loss_function(name: str) -> None:
    """Remove a registered loss function if present."""
    LOSS_FUNCTION_REGISTRY.pop(str(name), None)


def get_available_loss_functions() -> list[str]:
    """Return registered loss function names."""
    return sorted(LOSS_FUNCTION_REGISTRY)


class LossFunction(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self._loss_name = "Base Loss"

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class MSELoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Mean Squared Error"
        self.mse = nn.MSELoss(reduction=self.reduction)

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(model(x), y)


class LogMSELoss(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Log Mean Squared Error"

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        mse = super().forward(model, x, y)
        return torch.log(mse + 1e-12)


class CrossEntropyLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Cross Entropy"
        self.ce = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.ce(model(x), y)


class BCELoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Binary Cross Entropy"
        self.bce = nn.BCELoss(reduction=self.reduction)

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(model(x), y)


class CategoricalNLLLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Negative Log Likelihood"

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits = model(x)
        nll = -1 * Categorical(logits=logits).log_prob(y)
        return _reduce_tensor_loss(nll, self.reduction)


class VaeElboLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "VAE ELBO"
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self, model: BaseVariationalAutoencoder, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(model, BaseVariationalAutoencoder), (
            "Model must be an instance of BaseVariationalAutoencoder"
            + "but got "
            + str(type(model))
        )

        z_mu, z_sigma = model.encoder.latent_distribution(x)
        z = torch.sigmoid(Normal(z_mu, z_sigma).rsample())
        x_reconstructed: torch.Tensor = model.decoder(z)

        mse = torch.mean(
            self.mse(x_reconstructed, y), dim=tuple(range(1, x_reconstructed.dim()))
        )

        z_var = z_sigma**2
        kl_div = 0.5 * torch.sum(
            z_mu**2 + z_var - torch.log(z_var) - 1, dim=tuple(range(1, z_mu.dim()))
        )

        loss = mse + kl_div

        return _reduce_tensor_loss(loss, self.reduction)


class NegativeCosineSimilarityLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Negative Cosine Similarity"
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        y_hat = model(x)
        cosine_similarity = self.cosine_similarity(y_hat, y)
        return _reduce_tensor_loss(-1 * cosine_similarity, self.reduction)


class LogAdjustedNegativeCosineSimilarityLoss(NegativeCosineSimilarityLoss):
    """
    Enable finetuning near perfect cosine similarity.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Log Adjusted Negative Cosine Similarity"

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        neg_cos_sim = super().forward(model, x, y)
        return torch.log(neg_cos_sim + 1 + 1e-12)


class VoltageStabilizationLoss(ForwardHookRemovalMixin, LossFunction):
    """Voltage / gate-occupancy stabilization loss.

    Modes:

    ``vinf``, ``vout``, ``both``
        Legacy raw-voltage modes.  Drive V (input to the reactivation) or r
        (output) toward a fixed ``target_voltage``.  Works well for shunting
        morphology where V ∈ [0, 1] and 0.5 is a natural center, but poorly
        for additive where the raw-voltage scale is architecture-dependent.

    ``gate_occupancy`` **(recommended general mode)**
        Stabilize in gate-relative coordinates instead of raw voltages.
        Operates on the *post-gate output* ``r = (tanh(m*(V-b))+1)/2`` and
        drives it toward a healthy occupancy regime:

        * **Centering:** ``median(r) → 0.5`` (gate is centered).
        * **Spread:** ``sat_lo + sat_hi → target_saturation`` where
          ``sat_lo = mean(r < 0.1)`` and ``sat_hi = mean(r > 0.9)``
          (default ``target_saturation = 0.2``, i.e. ~10% at each rail).

        This is morphology-invariant by construction: it asks each branch to
        reach the same *gate-output* operating regime regardless of whether
        the underlying voltage is bounded (shunting) or unbounded (additive).
        The mode is defined only for real parametric gates, so it hooks only
        layers whose reactivation is a ``ParametricActivation``.
    """

    def __init__(
        self,
        stabilize_mode: str = "vinf",
        target_voltage: float = 0.5,
        target_saturation: float = 0.2,
        loss_metric: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self._loss_name = "Voltage Stabilization"
        self.stabilize_mode = stabilize_mode
        self.target_voltage = target_voltage
        self.target_saturation = target_saturation
        self.eps = torch.finfo(torch.float32).eps

        valid_modes = {"vinf", "vout", "both", "gate_occupancy"}
        if self.stabilize_mode not in valid_modes:
            raise ValueError(
                f"Invalid stabilize mode: {stabilize_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )

        self.save_inputs = False
        self.save_outputs = False
        if self.stabilize_mode in ("vinf", "both"):
            self.save_inputs = True
        if self.stabilize_mode in ("vout", "both"):
            self.save_outputs = True
        if self.stabilize_mode == "gate_occupancy":
            # gate_occupancy hooks on the reactivation output (r)
            self.save_outputs = True

        self.metric_fn = _build_pair_metric_fn(
            loss_metric,
            reduction=self.reduction,
            eps=self.eps,
            error_context="voltage stabilization",
            reduce_mean_error_difference_first=True,
        )

    def attach_forward_hooks(self, model: BaseModel):
        return register_forward_hook_groups(
            model,
            DendriticBranchLayer,
            self._attach_reactivation_hooks,
            predicate=self._should_hook_reactivation,
        )

    def _attach_reactivation_hooks(
        self,
        module: DendriticBranchLayer,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return [self._register_reactivation_hook(module)]

    def _should_hook_reactivation(self, module: DendriticBranchLayer) -> bool:
        if self.stabilize_mode != "gate_occupancy":
            return True
        return isinstance(module.reactivation, ParametricActivation)

    def _register_reactivation_hook(
        self,
        module: DendriticBranchLayer,
    ) -> torch.utils.hooks.RemovableHandle:
        return module.reactivation.register_forward_hook(self.forward_hook)

    def forward_hook(
        self,
        module: Union[nn.Identity, ParametricActivation],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        if self.save_inputs:
            self.voltages.append(inputs[0])
        if self.save_outputs:
            self.voltages.append(outputs)

    def reset_activation_lists(self):
        self.voltages = []

    def _gate_occupancy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss in gate-output coordinates.

        Drives: median(r) → 0.5 and total saturation → target_saturation.
        """
        if not self.voltages:
            return torch.tensor(
                0.0, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad
            )

        total_loss = torch.tensor(
            0.0, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad
        )

        for r in self.voltages:
            # r is the gate output, shape [B, D] or [B, ...].
            r_flat = r.reshape(-1)

            # Centering: drive median(r) toward 0.5.
            # Use a differentiable proxy: mean(r) → 0.5.
            # (median is not differentiable; mean tracks it for unimodal
            # distributions, which r approximately is.)
            center_loss = (r_flat.mean() - 0.5) ** 2

            # Spread: drive saturation toward target.
            # sat_lo = fraction with r < 0.1, sat_hi = fraction with r > 0.9.
            # Use soft thresholds for differentiability:
            #   soft_sat_lo = mean(sigmoid(k * (0.1 - r)))
            #   soft_sat_hi = mean(sigmoid(k * (r - 0.9)))
            # with k = 20 (sharp enough to approximate the indicator).
            k = 20.0
            sat_lo = torch.sigmoid(k * (0.1 - r_flat)).mean()
            sat_hi = torch.sigmoid(k * (r_flat - 0.9)).mean()
            total_sat = sat_lo + sat_hi
            sat_target = self.target_saturation
            spread_loss = (total_sat - sat_target) ** 2

            total_loss = total_loss + center_loss + spread_loss

        self.voltages = []
        return total_loss

    def compute_loss(self, x: torch.Tensor):
        if self.voltages:
            voltages = torch.cat(self.voltages, dim=-1)
            self.voltages = []

            target = torch.full_like(voltages, self.target_voltage)
            return self.metric_fn(voltages, target)
        else:
            return torch.tensor(
                0.0, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad
            )

    def forward(
        self, model: BaseModel, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.reset_activation_lists()
        model(x)
        if self.stabilize_mode == "gate_occupancy":
            return self._gate_occupancy_loss(x)
        return self.compute_loss(x)


class ExcitationInhibitionBalanceLoss(ForwardHookRemovalMixin, LossFunction):
    def __init__(self, loss_metric: str = "mse", reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Excitation-Inhibition Balance"
        self.loss_metric = loss_metric
        self.eps = torch.finfo(torch.float32).eps

        self.metric_fn = _build_pair_metric_fn(
            loss_metric,
            reduction=self.reduction,
            eps=self.eps,
            error_context="excitation-inhibition balance",
            reduce_mean_error_difference_first=False,
        )

    def attach_forward_hooks(self, model: BaseModel):
        return register_forward_hook_groups(
            model,
            DendriticBranchLayer,
            self._attach_branch_balance_hooks,
        )

    def _attach_branch_balance_hooks(
        self,
        module: DendriticBranchLayer,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        handles = []
        try:
            if (
                module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                handle = module.branch_excitation.register_forward_hook(
                    lambda module, inputs, outputs: self.forward_hook(
                        module, inputs, outputs, "exc"
                    )
                )
                handles.append(handle)
                handle = module.branch_inhibition.register_forward_hook(
                    lambda module, inputs, outputs: self.forward_hook(
                        module, inputs, outputs, "inh"
                    )
                )
                handles.append(handle)
            return handles
        except Exception:
            self.remove_forward_hooks(handles)
            raise

    def forward_hook(
        self,
        module: TopKLinear,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        activation: str,
    ):
        if activation == "exc":
            self.excitation.append(outputs)
        elif activation == "inh":
            self.inhibition.append(outputs)

    def reset_activation_lists(self):
        self.excitation = []
        self.inhibition = []

    def compute_loss(self, x: torch.Tensor):
        if self.excitation and self.inhibition:
            excitation = torch.cat(self.excitation, dim=-1)
            inhibition = torch.cat(self.inhibition, dim=-1)
            self.excitation = []
            self.inhibition = []

            return self.metric_fn(excitation, inhibition)
        else:
            return torch.tensor(
                0.0, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad
            )

    def forward(
        self, model: BaseModel, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.reset_activation_lists()
        model(x)
        return self.compute_loss(x)


class HomeostaticControlLoss(ForwardHookRemovalMixin, LossFunction):
    def __init__(
        self,
        stabilize_mode: str = "vinf",
        target_voltage: float = 0.5,
        target_saturation: float = 0.2,
        loss_metric: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self._loss_name = "Homeostatic Control"
        self.voltage_stabilization = VoltageStabilizationLoss(
            stabilize_mode=stabilize_mode,
            target_voltage=target_voltage,
            target_saturation=target_saturation,
            loss_metric=loss_metric,
            reduction=reduction,
        )
        self.ei_balance = ExcitationInhibitionBalanceLoss(
            loss_metric=loss_metric, reduction=reduction
        )

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        try:
            handles.extend(self.voltage_stabilization.attach_forward_hooks(model))
            handles.extend(self.ei_balance.attach_forward_hooks(model))
            return handles
        except Exception:
            self.remove_forward_hooks(handles)
            raise

    def forward(
        self, model: BaseModel, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.voltage_stabilization.reset_activation_lists()
        self.ei_balance.reset_activation_lists()
        model(x)
        loss = self.voltage_stabilization.compute_loss(x)
        loss = loss + self.ei_balance.compute_loss(x)
        return loss / 2.0


class EIWeightRatioLoss(LossFunction):
    """
    Loss function to enforce a target ratio between excitatory and inhibitory synaptic weights.

    This loss function computes the deviation from a target E/I weight ratio across
    dendritic branch layers and adds it as a regularization term to encourage
    biologically plausible E/I balance.
    """

    def __init__(
        self,
        target_ratio: float = 1.5,
        loss_weight: float = 0.1,
        scope: str = "per_branch",
        metric: str = "mean",
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self._loss_name = "E/I Weight Ratio"
        self.target_ratio = target_ratio
        self.loss_weight = loss_weight
        self.scope = scope  # "per_branch", "per_layer", "global"
        self.metric = metric  # "mean", "median", "rms"

        self.reduction_fn = _loss_reduction_fn(self.reduction)

    def forward(
        self, model: BaseModel, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute E/I weight ratio loss."""
        ratio_losses = []

        for module in iter_modules_of_type(model, DendriticBranchLayer):
            if (
                module.branch_excitation is not None
                and module.branch_inhibition is not None
                and hasattr(module.branch_excitation, "pruned_weight")
                and hasattr(module.branch_inhibition, "pruned_weight")
            ):
                try:
                    exc_weights = (
                        module.branch_excitation.pruned_weight()
                    )  # [n_branches, in_features]
                    inh_weights = (
                        module.branch_inhibition.pruned_weight()
                    )  # [n_branches, in_features]

                    if self.scope == "per_branch":
                        # Enforce ratio per branch
                        ratio_loss = self._compute_per_branch_ratio_loss(
                            exc_weights, inh_weights
                        )
                    elif self.scope == "per_layer":
                        # Enforce ratio per layer (average across branches)
                        ratio_loss = self._compute_per_layer_ratio_loss(
                            exc_weights, inh_weights
                        )
                    else:  # global
                        # Enforce global ratio across entire layer
                        ratio_loss = self._compute_global_ratio_loss(
                            exc_weights, inh_weights
                        )

                    if ratio_loss is not None:
                        ratio_losses.append(ratio_loss)

                except Exception:
                    # Skip layers that don't have proper weight structure
                    continue

        if ratio_losses:
            total_ratio_loss = self.reduction_fn(torch.stack(ratio_losses))
            return self.loss_weight * total_ratio_loss
        else:
            # Return zero loss if no valid layers found
            return torch.tensor(0.0, device=x.device, requires_grad=True)

    def _compute_per_branch_ratio_loss(
        self, exc_weights: torch.Tensor, inh_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute ratio loss per branch."""
        # Compute weight statistics per branch
        if self.metric == "mean":
            exc_stats = exc_weights.mean(dim=1)  # [n_branches]
            inh_stats = inh_weights.mean(dim=1)  # [n_branches]
        elif self.metric == "median":
            exc_stats = exc_weights.median(dim=1)[0]
            inh_stats = inh_weights.median(dim=1)[0]
        else:  # rms
            exc_stats = torch.sqrt(torch.mean(exc_weights**2, dim=1))
            inh_stats = torch.sqrt(torch.mean(inh_weights**2, dim=1))

        # Compute actual E/I ratio per branch (avoid division by zero)
        actual_ratios = exc_stats / (inh_stats + 1e-8)

        # Compute loss as squared deviation from target
        ratio_loss = torch.mean((actual_ratios - self.target_ratio) ** 2)
        return ratio_loss

    def _compute_per_layer_ratio_loss(
        self, exc_weights: torch.Tensor, inh_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute ratio loss per layer (average across branches first)."""
        # Average across branches, then compute ratio
        if self.metric == "mean":
            exc_layer_mean = exc_weights.mean()
            inh_layer_mean = inh_weights.mean()
        elif self.metric == "median":
            exc_layer_mean = exc_weights.median()
            inh_layer_mean = inh_weights.median()
        else:  # rms
            exc_layer_mean = torch.sqrt(torch.mean(exc_weights**2))
            inh_layer_mean = torch.sqrt(torch.mean(inh_weights**2))

        # Compute layer-level ratio
        actual_ratio = exc_layer_mean / (inh_layer_mean + 1e-8)

        # Compute loss
        ratio_loss = (actual_ratio - self.target_ratio) ** 2
        return ratio_loss

    def _compute_global_ratio_loss(
        self, exc_weights: torch.Tensor, inh_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute global ratio loss across entire network."""
        # Same as per_layer but will be aggregated across all layers by reduction_fn
        return self._compute_per_layer_ratio_loss(exc_weights, inh_weights)


def _register_builtin_loss_functions() -> None:
    register_loss_function(
        "mse",
        MSELoss,
        aliases=("mean_squared_error",),
    )
    register_loss_function(
        "log_mse",
        LogMSELoss,
        aliases=("log_mean_squared_error",),
    )
    register_loss_function(
        "ce",
        CrossEntropyLoss,
        aliases=("cross_entropy",),
    )
    register_loss_function(
        "bce",
        BCELoss,
        aliases=("binary_cross_entropy",),
    )
    register_loss_function(
        "cat_nll",
        CategoricalNLLLoss,
        aliases=("categorical_negative_log_likelihood",),
    )
    register_loss_function(
        "vae_elbo",
        VaeElboLoss,
        aliases=("variational_autoencoder_elbo",),
    )
    register_loss_function(
        "neg_cos_sim",
        NegativeCosineSimilarityLoss,
        aliases=("negative_cosine_similarity",),
    )
    register_loss_function(
        "log_adj_neg_cos_sim",
        LogAdjustedNegativeCosineSimilarityLoss,
        aliases=("log_adjusted_negative_cosine_similarity",),
    )


_register_builtin_loss_functions()


def get_loss_function(loss_function: str) -> LossFunction:
    builder = LOSS_FUNCTION_REGISTRY.get(loss_function)
    if builder is None:
        raise ValueError(f"Invalid loss function: {loss_function}")
    return builder()
