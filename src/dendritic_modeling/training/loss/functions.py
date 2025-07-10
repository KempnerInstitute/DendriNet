from typing import , , Optional

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.classical.autoencoder import BaseVariationalAutoencoder

class LossFunction(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self._loss_name = "Base Loss"

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

class MSELoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Mean Squared Error"
        self.mse = nn.MSELoss(reduction=self.reduction)

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(model(x), y)

class LogMSELoss(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Log Mean Squared Error"

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        mse = super().forward(model, x, y)
        return torch.log(mse + 1e-12)

class CrossEntropyLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Cross Entropy"
        self.ce = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.ce(model(x), y)

class BCELoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Binary Cross Entropy"
        self.bce = nn.BCELoss(reduction=self.reduction)

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(model(x), y)

class CategoricalNLLLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "Negative Log Likelihood"

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        logits = model(x)
        nll = -1 * Categorical(logits=logits).log_prob(y)
        if self.reduction == "mean":
            return torch.mean(nll)
        elif self.reduction == "sum":
            return torch.sum(nll)
        else:
            return nll

class VAE_ELBOLoss(LossFunction):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self._loss_name = "VAE ELBO"
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        model: BaseVariationalAutoencoder,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(model, BaseVariationalAutoencoder), \
            "Model must be an instance of BaseVariationalAutoencoder" + \
            "but got " + str(type(model))

        z_mu, z_sigma = model.encoder.latent_distribution(x)
        z = torch.sigmoid(Normal(z_mu, z_sigma).sample())
        x_reconstructed: torch.Tensor = model.decoder(z)

        mse = torch.mean(
            self.mse(x_reconstructed, y),
            dim=tuple(range(1, x_reconstructed.dim()))
        )

        z_var = z_sigma**2
        kl_div = 0.5 * torch.sum(
            z_mu**2 + z_var - torch.log(z_var) - 1,
            dim=tuple(range(1, z_mu.dim()))
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
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
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
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        neg_cos_sim = super().forward(model, x, y)
        return torch.log(neg_cos_sim + 1 + 1e-12)

class VoltageStabilizationLoss(LossFunction):
    def __init__(
        self,
        target_voltage: float = 0.5,
        reduction: str = "mean"
    ):
        super().__init__(reduction)
        self._loss_name = "Voltage Stabilization"
        self.target_voltage = target_voltage
        self.mse = nn.MSELoss(reduction=self.reduction)
        self.voltages = []

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for module in model.modules():
            if isinstance(module, DendriticBranchLayer):
                handle = module.register_forward_hook(self.forward_hook)
                handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor
    ):
        self.voltages.append(output)

    def forward(
        self,
        model: BaseModel,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        model(x)

        if self.voltages:
            voltages = torch.cat(self.voltages, dim=-1)
            self.voltages = []

            target = torch.full_like(voltages, self.target_voltage)
            return self.mse(voltages, target)
        else:
            return torch.tensor(
                0.0,
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad
            )

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
        return VAE_ELBOLoss()
    elif loss_function in ["neg_cos_sim", "negative_cosine_similarity"]:
        return NegativeCosineSimilarityLoss()
    elif loss_function in [
        "log_adj_neg_cos_sim", "log_adjusted_negative_cosine_similarity"
    ]:
        return LogAdjustedNegativeCosineSimilarityLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
