import torch
import numpy as np
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.dendrinet import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.information import compute_mi_cc, compute_mi_cd
from dendritic_modeling.config import AnalysisConfig
from typing import Tuple, Optional, List, Dict
import os
import logging


class EIMutualInfoAnalyzer:
    def __init__(
        self, 
        compute_mi_E_I: bool = False,
        compute_mi_EI_Vout: bool = False,
        compute_mi_EI_label: bool = False,
        n_neighbors: int = 10
    ):
        self.mi_E_I = compute_mi_E_I
        self.mi_EI_Vout = compute_mi_EI_Vout
        self.mi_EI_label = compute_mi_EI_label
        self.data_dict = {}
        self.n_neighbors = n_neighbors
        
    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None 
                    and module.branch_inhibition is not None
                ):
                    if name not in self.data_dict.keys():
                        self.data_dict[name] = {}
                        module._name = name
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
        excitatory_input, inhibitory_input, _ = input
        key = module._name
        self.data_dict[key]['excitation'] = module.branch_excitation(excitatory_input)
        self.data_dict[key]['inhibition'] = module.branch_inhibition(inhibitory_input)
        if self.mi_EI_Vout:
            self.data_dict[key]['Vout'] = output
    
    def compute_mi_E_I(self, excitation: np.ndarray, inhibition: np.ndarray) -> float:
        return compute_mi_cc(excitation, inhibition, self.n_neighbors)

    def compute_mi_EI_Vout(self, EI: np.ndarray, Vout: np.ndarray) -> float:
        return compute_mi_cc(EI, Vout, self.n_neighbors)

    def compute_mi_EI_label(self, EI: np.ndarray, labels: np.ndarray) -> float:
        return compute_mi_cd(EI, labels, self.n_neighbors)

    def analyze(
        self, 
        model: BaseModel,
        dataset: torch.utils.data.Dataset,
        device: str = 'cpu', # 'cuda' if torch.cuda.is_available() else 'cpu'
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork), \
                "model.net must be an instance of ExcitationInhibitionNetwork. " + \
                f"Got model.net type: {type(model.net)}"
        except AssertionError:
            return

        model = model.to(device)
        model.eval()

        inputs, labels = dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor

        handles = self.attach_forward_hooks(model)

        _ = model(inputs)

        excitation = []
        inhibition = []
        if self.mi_EI_Vout:
            Vout = []
        for branch_dict in self.data_dict.values():
            excitation.append(branch_dict['excitation'])
            inhibition.append(branch_dict['inhibition'])
            if self.mi_EI_Vout:
                Vout.append(branch_dict['Vout'])
        
        excitation = torch.cat(excitation, dim=-1).detach().cpu().numpy()
        inhibition = torch.cat(inhibition, dim=-1).detach().cpu().numpy()

        if self.mi_EI_label:
            labels = labels.detach().cpu().numpy()
            label_len = len(labels)
            for i, dim_size in enumerate(excitation.shape):
                if dim_size == label_len:
                    break
            new_shape = [1] * excitation.ndim
            new_shape[i] = label_len
            labels = labels.reshape(new_shape)
            labels = np.broadcast_to(labels, excitation.shape).flatten()

        excitation = excitation.flatten()[:, None]
        inhibition = inhibition.flatten()[:, None]

        if self.mi_EI_Vout or self.mi_EI_label:
            EI = np.concatenate([excitation, inhibition], axis=1)

        mi_dict = {}

        if self.mi_E_I:
            mi_dict['mi_E_I'] = self.compute_mi_E_I(excitation, inhibition)

        if self.mi_EI_Vout:
            Vout = torch.cat(Vout, dim=-1).detach().cpu().flatten().numpy()[:, None]
            mi_dict['mi_EI_Vout'] = self.compute_mi_EI_Vout(EI, Vout)

        if self.mi_EI_label:
            mi_dict['mi_EI_label'] = self.compute_mi_EI_label(EI, labels)

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(mi_dict, save_path, filename)
        else:
            return mi_dict


class AnalysisManager:
    def __init__(
        self,
        model: BaseModel,
        data: Dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str='cpu'
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.save_root, "analysis.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    
    def run_analysis(self, filename: str='final'):
        ei_mi_params = self.config.ei_mi_analysis_params
        if self.config.ei_mi_analysis_enabled:
            analyzer = EIMutualInfoAnalyzer(
                compute_mi_E_I=ei_mi_params.get('compute_mi_E_I', True),
                compute_mi_EI_Vout=ei_mi_params.get('compute_mi_EI_Vout', True),
                compute_mi_EI_label=ei_mi_params.get('compute_mi_EI_label', True),
                n_neighbors=ei_mi_params.get('n_neighbors', 10)
            )
            analyzer.analyze(
                model=self.model,
                dataset=self.data['test'],
                device=self.device,
                save_path=os.path.join(self.save_root, 'ei_mi_analysis'),
                filename=filename
            )
            
        # TODO: add other analysis classes here implemented same as above
                
            
        