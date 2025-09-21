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
        if self.reduction == "mean":
            return torch.mean(nll)
        elif self.reduction == "sum":
            return torch.sum(nll)
        else:
            return nll


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

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


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
        if self.reduction == "mean":
            return torch.mean(-1 * cosine_similarity)
        elif self.reduction == "sum":
            return torch.sum(-1 * cosine_similarity)
        else:
            return -1 * cosine_similarity


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


class VoltageStabilizationLoss(LossFunction):
    def __init__(
        self,
        stabilize_mode: str = "vinf",
        target_voltage: float = 0.5,
        loss_metric: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self._loss_name = "Voltage Stabilization"
        self.stabilize_mode = stabilize_mode
        self.target_voltage = target_voltage

        self.save_inputs = False
        self.save_outputs = False
        if self.stabilize_mode not in ["vinf", "vout", "both"]:
            raise ValueError(f"Invalid stabilize mode: {stabilize_mode}")

        if self.stabilize_mode in ["vinf", "both"]:
            self.save_inputs = True
        if self.stabilize_mode in ["vout", "both"]:
            self.save_outputs = True

        if self.reduction == "mean":
            reduction_fn = torch.mean
        elif self.reduction == "sum":
            reduction_fn = torch.sum
        else:

            def reduction_fn(x):
                return x

        if loss_metric == "mse":
            self.metric_fn = lambda x, y: reduction_fn((x - y) ** 2)
        elif loss_metric == "log_mse":
            self.metric_fn = lambda x, y: torch.log(
                reduction_fn((x - y) ** 2) + self.eps
            )
        elif loss_metric == "mae":
            self.metric_fn = lambda x, y: reduction_fn(torch.abs(x - y))
        elif loss_metric == "log_mae":
            self.metric_fn = lambda x, y: torch.log(
                reduction_fn(torch.abs(x - y)) + self.eps
            )
        elif loss_metric == "squared_mean_error":
            self.metric_fn = lambda x, y: reduction_fn(x - y) ** 2
        elif loss_metric == "log_squared_mean_error":
            self.metric_fn = lambda x, y: torch.log(
                reduction_fn(x - y) ** 2 + self.eps
            )
        elif loss_metric == "abs_mean_error":
            self.metric_fn = lambda x, y: torch.abs(reduction_fn(x - y))
        elif loss_metric == "log_abs_mean_error":
            self.metric_fn = lambda x, y: torch.log(
                torch.abs(reduction_fn(x - y)) + self.eps
            )
        else:
            raise ValueError(f"Invalid metric for voltage stabilization: {loss_metric}")

        self.eps = torch.finfo(torch.float32).eps

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for module in model.modules():
            if isinstance(module, DendriticBranchLayer):
                handle = module.reactivation.register_forward_hook(self.forward_hook)
                handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()

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
        return self.compute_loss(x)


class ExcitationInhibitionBalanceLoss(LossFunction):
    def __init__(self, loss_metric: str = "mse", reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Excitation-Inhibition Balance"
        self.loss_metric = loss_metric

        if self.reduction == "mean":
            reduction_fn = torch.mean
        elif self.reduction == "sum":
            reduction_fn = torch.sum
        else:

            def reduction_fn(x):
                return x

        if loss_metric == "mse":
            self.metric_fn = lambda x, y: reduction_fn((x - y) ** 2)
        elif loss_metric == "log_mse":
            self.metric_fn = lambda x, y: torch.log(
                reduction_fn((x - y) ** 2) + self.eps
            )
        elif loss_metric == "mae":
            self.metric_fn = lambda x, y: reduction_fn(torch.abs(x - y))
        elif loss_metric == "log_mae":
            self.metric_fn = lambda x, y: torch.log(
                reduction_fn(torch.abs(x - y)) + self.eps
            )
        elif loss_metric == "squared_mean_error":
            self.metric_fn = lambda x, y: (reduction_fn(x) - reduction_fn(y)) ** 2
        elif loss_metric == "log_squared_mean_error":
            self.metric_fn = lambda x, y: torch.log(
                (reduction_fn(x) - reduction_fn(y)) ** 2 + self.eps
            )
        elif loss_metric == "abs_mean_error":
            self.metric_fn = lambda x, y: torch.abs(reduction_fn(x) - reduction_fn(y))
        elif loss_metric == "log_abs_mean_error":
            self.metric_fn = lambda x, y: torch.log(
                torch.abs(reduction_fn(x) - reduction_fn(y)) + self.eps
            )
        else:
            raise ValueError(
                f"Invalid metric for excitation-inhibition balance: {loss_metric}"
            )

        self.eps = torch.finfo(torch.float32).eps

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for module in model.modules():
            if isinstance(module, DendriticBranchLayer):
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

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()

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


class HomeostaticControlLoss(LossFunction):
    def __init__(
        self,
        stabilize_mode: str = "vinf",
        target_voltage: float = 0.5,
        loss_metric: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self._loss_name = "Homeostatic Control"
        self.voltage_stabilization = VoltageStabilizationLoss(
            stabilize_mode=stabilize_mode,
            target_voltage=target_voltage,
            loss_metric=loss_metric,
            reduction=reduction,
        )
        self.ei_balance = ExcitationInhibitionBalanceLoss(
            loss_metric=loss_metric, reduction=reduction
        )

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        handles.extend(self.voltage_stabilization.attach_forward_hooks(model))
        handles.extend(self.ei_balance.attach_forward_hooks(model))
        return handles

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()

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

        if self.reduction == "mean":
            self.reduction_fn = torch.mean
        elif self.reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            self.reduction_fn = lambda x: x

    def forward(
        self, model: BaseModel, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute E/I weight ratio loss."""
        ratio_losses = []

        for module in model.modules():
            if isinstance(module, DendriticBranchLayer):
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


def get_loss_function(loss_function: str) -> LossFunction:
    if loss_function in ["mse", "mean_squared_error"]:
        return MSELoss()
    elif loss_function in ["log_mse", "log_mean_squared_error"]:
        return LogMSELoss()
    elif loss_function in ["ce", "cross_entropy"]:
        return CrossEntropyLoss()
    elif loss_function in ["bce", "binary_cross_entropy"]:
        return BCELoss()
    elif loss_function in ["cat_nll", "categorical_negative_log_likelihood"]:
        return CategoricalNLLLoss()
    elif loss_function in ["vae_elbo", "variational_autoencoder_elbo"]:
        return VaeElboLoss()
    elif loss_function in ["neg_cos_sim", "negative_cosine_similarity"]:
        return NegativeCosineSimilarityLoss()
    elif loss_function in [
        "log_adj_neg_cos_sim",
        "log_adjusted_negative_cosine_similarity",
    ]:
        return LogAdjustedNegativeCosineSimilarityLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
