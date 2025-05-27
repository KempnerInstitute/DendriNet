import torch
from torch.utils.data import TensorDataset
import numpy as np

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.dendrinet import DendriticBranchLayer
from dendritic_modeling.utils import (
    save_dict, 
    add_uniform_noise, 
    add_gaussian_noise, 
    evaluate_accuracy,
    evaluate_categorical_loglikelihood
)
from dendritic_modeling.information import (
    compute_mi_cc, 
    compute_mi_cd, 
    compute_pairwise_sr
)
from dendritic_modeling.config import AnalysisConfig

from typing import Tuple, Optional, List, Dict
import os
from copy import deepcopy
import logging


class EIInformationAnalyzer:
    def __init__(
        self, 
        compute_mi_E_I: bool = False,
        compute_mi_EI_Vout: bool = False,
        compute_sr_EI_Vout: bool = False,
        compute_mi_EI_label: bool = False,
        n_neighbors: int = 10
    ):
        self.mi_E_I = compute_mi_E_I
        self.mi_EI_Vout = compute_mi_EI_Vout
        self.sr_EI_Vout = compute_sr_EI_Vout
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

    def compute_sr_EI_Vout(
        self, 
        excitation: np.ndarray, 
        inhibition: np.ndarray, 
        Vout: np.ndarray,
        mi_EI_Vout: Optional[float] = None
    ) -> float:
        return compute_pairwise_sr(
            excitation, inhibition, Vout, self.n_neighbors, mi_EI_Vout
        )
        
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

        info_dict = {}

        if self.mi_E_I:
            info_dict['mi_E_I'] = self.compute_mi_E_I(excitation, inhibition)

        if self.mi_EI_Vout:
            Vout = torch.cat(Vout, dim=-1).detach().cpu().flatten().numpy()[:, None]
            info_dict['mi_EI_Vout'] = self.compute_mi_EI_Vout(EI, Vout)

        if self.sr_EI_Vout:
            mi_EI_Vout = None
            if self.mi_EI_Vout:
                mi_EI_Vout = info_dict['mi_EI_Vout']
            (
                si_EI_Vout, ri_EI_Vout, mi_EI_Vout, mi_E_Vout, mi_I_Vout
            ) = self.compute_sr_EI_Vout(
                excitation, inhibition, Vout, mi_EI_Vout
            )
            info_dict['si_EI_Vout'] = si_EI_Vout
            info_dict['ri_EI_Vout'] = ri_EI_Vout
            info_dict['mi_EI_Vout'] = mi_EI_Vout
            info_dict['mi_E_Vout'] = mi_E_Vout
            info_dict['mi_I_Vout'] = mi_I_Vout

        if self.mi_EI_label:
            info_dict['mi_EI_label'] = self.compute_mi_EI_label(EI, labels)

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(info_dict, save_path, filename)
        else:
            return info_dict


class SynapticTurnoverAnalyzer:
    def __init__(self, model: BaseModel):
        assert isinstance(model.net, ExcitationInhibitionNetwork), \
            "model.net must be an instance of ExcitationInhibitionNetwork. " + \
            f"Got model.net type: {type(model.net)}"
        
        self.data_dict_list = []
        self.analyze(model=model, training=True)

    def analyze(
        self, 
        model: BaseModel, 
        training: bool = True, 
        save_path: Optional[str] = None, 
        filename: str = 'final'
    ):
        if training:
            data_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    data_dict[name] = {}
                    if module.branch_excitation is not None:
                        data_dict[name]['exc_mask'] = deepcopy(
                            module.branch_excitation.weight_mask()).detach().cpu()
                    if module.branch_inhibition is not None:
                        data_dict[name]['inh_mask'] = deepcopy(
                            module.branch_inhibition.weight_mask()).detach().cpu()
            self.data_dict_list.append(data_dict)
        else:
            def turnover_helper(
                prev_mask: torch.Tensor, 
                curr_mask: torch.Tensor,
                total_active: int,
                total_stable: int,
                total_transient: int
            ) -> Tuple[int, int, int]:
                mask_diff = curr_mask - prev_mask
                curr_active = torch.sum(curr_mask).item()
                curr_transient = torch.sum(mask_diff > 1e-3).item()
                curr_stable = curr_active - curr_transient
                total_active += curr_active
                total_stable += curr_stable
                total_transient += curr_transient
                return total_active, total_stable, total_transient

            turnover_dict = {}
            for i in range(1, len(self.data_dict_list)):
                prev_data_dict: dict = self.data_dict_list[i - 1]
                curr_data_dict: dict = self.data_dict_list[i]

                exc_total_active = 0
                exc_total_stable = 0
                exc_total_transient = 0

                inh_total_active = 0
                inh_total_stable = 0
                inh_total_transient = 0

                for name in curr_data_dict.keys():
                    prev_module_dict: dict = prev_data_dict[name]
                    curr_module_dict: dict = curr_data_dict[name]

                    if 'exc_mask' in curr_module_dict.keys():
                        prev_exc_mask = prev_module_dict['exc_mask']
                        curr_exc_mask = curr_module_dict['exc_mask']
                        (
                            exc_total_active, exc_total_stable, exc_total_transient
                        ) = turnover_helper(
                            prev_exc_mask, curr_exc_mask, 
                            exc_total_active, exc_total_stable, exc_total_transient
                        )

                    if 'inh_mask' in curr_module_dict.keys():
                        prev_inh_mask = prev_module_dict['inh_mask']
                        curr_inh_mask = curr_module_dict['inh_mask']
                        (
                            inh_total_active, inh_total_stable, inh_total_transient
                        ) = turnover_helper(
                            prev_inh_mask, curr_inh_mask, 
                            inh_total_active, inh_total_stable, inh_total_transient
                        )

                turnover_dict[i] = {}
                turnover_dict[i]['exc'] = {
                    'active': exc_total_active,
                    'stable': exc_total_stable,
                    'transient': exc_total_transient
                }
                turnover_dict[i]['inh'] = {
                    'active': inh_total_active,
                    'stable': inh_total_stable,
                    'transient': inh_total_transient
                }
                turnover_dict[i]['both'] = {
                    'active': exc_total_active + inh_total_active,
                    'stable': exc_total_stable + inh_total_stable,
                    'transient': exc_total_transient + inh_total_transient
                }
            
            if save_path is not None:
                save_dict(turnover_dict, save_path, filename)
            else:
                return turnover_dict


class NoisePerturbationAnalyzer:
    def __init__(
        self,
        uniform_noise: bool = False,
        uniform_magnitudes: List[float] = [0.1],
        gaussian_noise: bool = False,
        gaussian_sdevs: List[float] = [0.1],
        n_samples: int = 10,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True
    ):
        self.uniform_noise = uniform_noise
        self.uniform_magnitudes = uniform_magnitudes
        self.gaussian_noise = gaussian_noise
        self.gaussian_sdevs = gaussian_sdevs
        self.n_samples = n_samples
        self.accuracy = accuracy
        self.categorical_ll = categorical_loglikelihood

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        model = model.to(device)
        x, y = test_dataset[:]
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)
        y = y[None, ...].expand(self.n_samples, *y.shape)

        noise_results = {}

        if self.uniform_noise:
            noise_results["uniform"] = {}
            for mag in self.uniform_magnitudes:
                noise_results["uniform"][f"{mag}"] = {}
                noisy_x = add_uniform_noise(x, spread=mag, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["categorical_ll"] = cat_lls
        
        if self.gaussian_noise:
            noise_results["gaussian"] = {}
            for sdev in self.gaussian_sdevs:
                noise_results["gaussian"][f"{sdev}"] = {}
                noisy_x = add_gaussian_noise(x, std=sdev, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["categorical_ll"] = cat_lls
        
        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results


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
        # Note: Logging should be configured externally at the application startup.
        self.logger = logging.getLogger(__name__)

        ei_info_config = self.config.ei_info_analysis
        self.ei_info_enabled = ei_info_config.get('enabled', False)
        self.ei_info_training = ei_info_config.get('training', False)
        if self.ei_info_enabled:
            ei_info_params: dict = ei_info_config.get('params', {})
            self.ei_info_analyzer = EIInformationAnalyzer(
                compute_mi_E_I=ei_info_params.get('compute_mi_E_I', True),
                compute_mi_EI_Vout=ei_info_params.get('compute_mi_EI_Vout', True),
                compute_sr_EI_Vout=ei_info_params.get('compute_sr_EI_Vout', True),
                compute_mi_EI_label=ei_info_params.get('compute_mi_EI_label', True),
                n_neighbors=ei_info_params.get('n_neighbors', 10)
            )
        
        syn_turnover_config = self.config.synapse_turnover_analysis
        self.syn_turnover_enabled = syn_turnover_config.get('enabled', False)
        self.syn_turnover_training = syn_turnover_config.get('training', False)
        if self.syn_turnover_enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)
        
        noise_config = self.config.noise_perturbation_analysis
        self.noise_enabled = noise_config.get('enabled', False)
        self.noise_training = noise_config.get('training', False)
        if self.noise_enabled:
            noise_params: dict = noise_config.get('params', {})
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                uniform_noise=noise_params.get('uniform_noise', True),
                uniform_magnitudes=noise_params.get('uniform_magnitudes', [0.1]),
                gaussian_noise=noise_params.get('gaussian_noise', True),
                gaussian_sdevs=noise_params.get('gaussian_sdevs', [0.1]),
                n_samples=noise_params.get('n_samples', 10),
                accuracy=noise_params.get('accuracy', True),
                categorical_loglikelihood=noise_params.get('categorical_loglikelihood', True)
            )
    
    def run_analysis(self, filename: str='final', training: bool=False):
        if (
            self.ei_info_enabled and 
            ((self.ei_info_training and training) or not training)
        ):
            self.ei_info_analyzer.analyze(
                model=self.model,
                dataset=self.data['test'],
                device=self.device,
                save_path=os.path.join(self.save_root, 'ei_info_analysis'),
                filename=filename
            )
        
        if (
            self.syn_turnover_enabled and
            ((self.syn_turnover_training and training) or not training)
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, 'synapse_turnover_analysis'),
                filename=filename
            )
        
        if (
            self.noise_enabled and
            ((self.noise_training and training) or not training)
        ):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data['test'],
                device=self.device,
                save_path=os.path.join(self.save_root, 'noise_perturbation_analysis'),
                filename=filename
            )
        
        # TODO: add other analysis classes here implemented same as above
                
            
        