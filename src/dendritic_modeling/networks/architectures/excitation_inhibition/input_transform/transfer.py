from typing import Any, Optional

import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.identity import (
    IdentityInputTransform,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import (
    IndexInputTransform,
)


class TransferLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        transfer_params: dict[str, Any],
        n_inhibitory_cells: Optional[int] = None,
        inhibitory_branch_factors: Optional[list[int]] = None,
        excitatory_synapses_per_branch: Optional[int] = None,
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        somatic_synapses: bool = True,
        topk_init_method: str = "xavier_normal",
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "conductance_dynamic",
        topk_noise_level: float = 0.0,
        topk_strategy: str = "conductance_dynamic",
        topk_type: str = "standard",
        print_hooks: bool = False,
        excitatory_target_density: float = 0.1,
        inhibitory_target_density: float = 0.1,
        use_noise: bool = True,
        sigma: float = 0.05,
        rewiring_mode: str = "global",
        rewire_frequency: int = 1,
        synapses_per_branch: Optional[list[int]] = None,
        freeze_excitatory_connectivity: bool = False,
        freeze_inhibitory_connectivity: bool = False,
        init_method: str = "xavier_normal",
        weight_threshold: float = 1e-6,
    ):
        super().__init__()

        input_mode = transfer_params.get("input_mode", 0)
        independent_pathways = transfer_params.get("independent_pathways", False)

        if independent_pathways:
            excitatory_input_dim = transfer_params.get(
                "excitatory_dim", int(input_dim / 2)
            )
            inhibitory_input_dim = transfer_params.get(
                "inhibitory_dim", int(input_dim - excitatory_input_dim)
            )
            self.pathway_split = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_input_dim,
                inhibitory_dim=inhibitory_input_dim,
            )
        else:
            excitatory_input_dim = input_dim
            inhibitory_input_dim = input_dim
            self.pathway_split = IdentityInputTransform()

        if input_mode == 0:
            self.inhibitory_cells = DendriNet(
                n_soma=n_inhibitory_cells,
                branch_factors=inhibitory_branch_factors,
                excitatory_input_dim=inhibitory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                inhibitory_input_dim=None,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                print_hooks=print_hooks,
                excitatory_target_density=excitatory_target_density,
                inhibitory_target_density=inhibitory_target_density,
                use_noise=use_noise,
                sigma=sigma,
                rewiring_mode=rewiring_mode,
                rewire_frequency=rewire_frequency,
                synapses_per_branch=synapses_per_branch,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
            )

        self.excitatory_dim = excitatory_input_dim

        if input_mode == 0:
            self.inhibitory_dim = n_inhibitory_cells
        elif input_mode == 1 and independent_pathways:
            self.inhibitory_dim = int(input_dim - self.excitatory_dim)
        elif input_mode == 1 and not independent_pathways:
            self.inhibitory_dim = self.excitatory_dim
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = self.pathway_split(x)

        if hasattr(self, "inhibitory_cells"):
            x2 = self.inhibitory_cells(x2)

        return x1, x2
