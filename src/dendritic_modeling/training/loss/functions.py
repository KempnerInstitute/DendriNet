from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.distributions import Categorical

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.layers import DendriticBranchLayer


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
    
    def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()
    
    def forward_hook(
        self, 
        module: DendriticBranchLayer, 
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
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
    if loss_function == "mse":
        return MSELoss()
    elif loss_function == "cross_entropy":
        return CrossEntropyLoss()
    elif loss_function == "bce":
        return BCELoss()
    elif loss_function == "categorical_nll":
        return CategoricalNLLLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
