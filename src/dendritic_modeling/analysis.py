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

        self.data_dict = {}

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


class AblationAnalyzer:
    def __init__(
        self,
        ablate_synapses: bool = True,
        ablate_inhibition: bool = True,
        ablate_branches: bool = True,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True
    ):
        self.ablate_synapses = ablate_synapses
        self.ablate_inhibition = ablate_inhibition
        self.ablate_branches = ablate_branches
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
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        state_dict = deepcopy(model.state_dict())

        results = {}
        if self.ablate_synapses:
            results['all synapses'] = {}
        if self.ablate_inhibition:
            results['inhibition'] = {}
        if self.ablate_branches:
            results['branches'] = {}

        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    self.ablate_synapses 
                    and module.branch_excitation is not None 
                    and module.branch_inhibition is not None
                ):
                    results['all synapses'][name] = {}
                    results['all synapses'][name]['layer_idx'] = module.layer_idx
                    
                    module.branch_excitation.pre_w.data.fill_(-100)
                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['all synapses'][name]['accuracy'] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['all synapses'][name]['log likelihood'] = loglikelihood
                    
                    model.load_state_dict(state_dict)
                
                if (
                    self.ablate_inhibition
                    and module.branch_inhibition is not None
                ):
                    results['inhibition'][name] = {}
                    results['inhibition'][name]['layer_idx'] = module.layer_idx
                    
                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['inhibition'][name]['accuracy'] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['inhibition'][name]['log likelihood'] = loglikelihood
                    
                    model.load_state_dict(state_dict)
                
                if (
                    self.ablate_branches
                    and hasattr(module, 'branches_to_output')
                ):
                    results['branches'][name] = {}
                    results['branches'][name]['layer_idx'] = module.layer_idx
                    
                    module.branches_to_output.log_weight.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['branches'][name]['accuracy'] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['branches'][name]['log likelihood'] = loglikelihood
                    
                    model.load_state_dict(state_dict)
        
        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


class WeightAnalyzer:
    def __init__(
        self,
        synapse_weights: bool = True,
        branch_weights: bool = True
    ):
        self.synapse_weights = synapse_weights
        self.branch_weights = branch_weights

    def analyze(
        self,
        model: BaseModel,
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        exc_weights = []
        inh_weights = []
        branch_weights = []
        
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if self.synapse_weights and module.branch_excitation is not None:
                    exc_w = module.branch_excitation.pruned_weight().flatten()
                    exc_w = exc_w[exc_w > 1e-3].cpu()
                    exc_weights.append(exc_w)
                if self.synapse_weights and module.branch_inhibition is not None:
                    inh_w = module.branch_inhibition.pruned_weight().flatten()
                    inh_w = inh_w[inh_w > 1e-3].cpu()
                    inh_weights.append(inh_w)
                if self.branch_weights and hasattr(module, 'branches_to_output'):
                    branch_w = module.branches_to_output.log_weight.exp().flatten()
                    branch_w = branch_w.cpu()
                    branch_weights.append(branch_w)
        
        weights = {}
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            weights['exc weights'] = {}
            weights['exc weights']['mean'] = exc_weights.mean().item()
            weights['exc weights']['std'] = exc_weights.std().item()
            weights['exc weights']['min'] = exc_weights.min().item()
            weights['exc weights']['max'] = exc_weights.max().item()
            weights['exc weights']['median'] = exc_weights.median().item()
            weights['exc weights']['q1'] = exc_weights.quantile(q=0.25).item()
            weights['exc weights']['q3'] = exc_weights.quantile(q=0.75).item()

        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            weights['inh weights'] = {}
            weights['inh weights']['mean'] = inh_weights.mean().item()
            weights['inh weights']['std'] = inh_weights.std().item()
            weights['inh weights']['min'] = inh_weights.min().item()
            weights['inh weights']['max'] = inh_weights.max().item()
            weights['inh weights']['median'] = inh_weights.median().item()
            weights['inh weights']['q1'] = inh_weights.quantile(q=0.25).item()
            weights['inh weights']['q3'] = inh_weights.quantile(q=0.75).item()

        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            weights['branch weights'] = {}
            weights['branch weights']['mean'] = branch_weights.mean().item()
            weights['branch weights']['std'] = branch_weights.std().item()
            weights['branch weights']['min'] = branch_weights.min().item()
            weights['branch weights']['max'] = branch_weights.max().item()
            weights['branch weights']['median'] = branch_weights.median().item()
            weights['branch weights']['q1'] = branch_weights.quantile(q=0.25).item()
            weights['branch weights']['q3'] = branch_weights.quantile(q=0.75).item()

        if save_path is not None:
            save_dict(weights, save_path, filename)
        else:
            return weights


class ActivationAnalyzer:
    def __init__(
        self,
        synapse_activations: bool = True,
        branch_activations: bool = True,
        network_summary: bool = True,
        layer_summary: bool = True,
        branch_summary: bool = True,
        downsample: bool = False,
        samples: int = 1000
    ):
        self.synapse_activations = synapse_activations
        self.branch_activations = branch_activations
        self.network_summary = network_summary
        self.layer_summary = layer_summary
        self.branch_summary = branch_summary
        self.downsample = downsample
        self.samples = samples

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
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

        if self.synapse_activations and module.branch_excitation is not None:
            self.data_dict[key]['exc'] = module.branch_excitation(excitatory_input)

        if self.synapse_activations and module.branch_inhibition is not None:
            self.data_dict[key]['inh'] = module.branch_inhibition(inhibitory_input)
        
        if self.branch_activations:
            self.data_dict[key]['Vout'] = output

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        self.data_dict = {}

        model = model.to(device)
        model.eval()

        inputs, _ = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)

        if self.downsample:
            ixs = torch.randperm(inputs.shape[0])[:self.samples]
            inputs = inputs[ixs]

        handles = self.attach_forward_hooks(model)

        _ = model(inputs)

        if self.network_summary:
            if self.synapse_activations:
                exc_net = []
                inh_net = []
            if self.branch_activations:
                Vout_net = []
        
        if self.layer_summary:
            layer_summary = {}
        
        if self.branch_summary:
            branch_summary = {}
        
        for name, module_dict in self.data_dict.items():
            print(name, flush=True)
            module_dict: dict
            if self.layer_summary:
                layer_summary[name] = {}
            
            if self.branch_summary:
                branch_summary[name] = {}

            if 'exc' in module_dict.keys():
                print('exc', flush=True)
                exc: torch.Tensor = module_dict['exc']
                # exc = exc.cpu()

                if self.network_summary:
                    print('network summary', flush=True)
                    if self.synapse_activations:
                        exc_net.append(exc.flatten())                
                
                if self.layer_summary:
                    print('layer summary', flush=True)
                    layer_summary[name]['exc'] = {}
                    if self.synapse_activations:
                        layer_summary[name]['exc']['individual'] = {}
                        layer_summary[name]['exc']['individual']['mean'] = exc_ind.mean().item()
                        layer_summary[name]['exc']['individual']['std'] = exc_ind.std().item()
                        # layer_summary[name]['exc']['individual']['min'] = exc_ind.min().item()
                        # layer_summary[name]['exc']['individual']['max'] = exc_ind.max().item()
                        # layer_summary[name]['exc']['individual']['median'] = exc_ind.median().item()
                        # layer_summary[name]['exc']['individual']['q1'] = exc_ind.quantile(q=0.25).item()
                        # layer_summary[name]['exc']['individual']['q3'] = exc_ind.quantile(q=0.75).item()
                    if self.aggregate_synapse_activations:
                        layer_summary[name]['exc']['aggregate'] = {}
                        layer_summary[name]['exc']['aggregate']['mean'] = exc_agg.mean().item()
                        layer_summary[name]['exc']['aggregate']['std'] = exc_agg.std().item()
                        # layer_summary[name]['exc']['aggregate']['min'] = exc_agg.min().item()
                        # layer_summary[name]['exc']['aggregate']['max'] = exc_agg.max().item()
                        # layer_summary[name]['exc']['aggregate']['median'] = exc_agg.median().item()
                        # layer_summary[name]['exc']['aggregate']['q1'] = exc_agg.quantile(q=0.25).item()
                        # layer_summary[name]['exc']['aggregate']['q3'] = exc_agg.quantile(q=0.75).item()
                
                if self.raw_summary:
                    print('raw summary', flush=True)
                    self.data_dict[name]['exc'] = exc.tolist()
            
            if 'inh' in module_dict.keys():
                print('inh', flush=True)
                inh: torch.Tensor = module_dict['inh']
                # inh = inh.cpu()

                if self.network_summary or self.layer_summary:
                    if self.individual_synapse_activations:
                        inh_ind = inh.flatten()
                        inh_ind = inh_ind[inh_ind > 1e-3]
                    if self.aggregate_synapse_activations:
                        inh_agg = inh.sum(dim=-2)
                        inh_agg = inh_agg.flatten()

                if self.network_summary:
                    if self.individual_synapse_activations:
                        inh_ind_net.append(inh_ind)
                    if self.aggregate_synapse_activations:
                        inh_agg_net.append(inh_agg)
                
                if self.layer_summary:
                    layer_summary[name]['inh'] = {}
                    if self.individual_synapse_activations:
                        layer_summary[name]['inh']['individual'] = {}
                        layer_summary[name]['inh']['individual']['mean'] = inh_ind.mean().item()
                        layer_summary[name]['inh']['individual']['std'] = inh_ind.std().item()
                        # layer_summary[name]['inh']['individual']['min'] = inh_ind.min().item()
                        # layer_summary[name]['inh']['individual']['max'] = inh_ind.max().item()
                        # layer_summary[name]['inh']['individual']['median'] = inh_ind.median().item()
                        # layer_summary[name]['inh']['individual']['q1'] = inh_ind.quantile(q=0.25).item()
                        # layer_summary[name]['inh']['individual']['q3'] = inh_ind.quantile(q=0.75).item()
                    if self.aggregate_synapse_activations:
                        layer_summary[name]['inh']['aggregate'] = {}
                        layer_summary[name]['inh']['aggregate']['mean'] = inh_agg.mean().item()
                        layer_summary[name]['inh']['aggregate']['std'] = inh_agg.std().item()
                        # layer_summary[name]['inh']['aggregate']['min'] = inh_agg.min().item()
                        # layer_summary[name]['inh']['aggregate']['max'] = inh_agg.max().item()
                        # layer_summary[name]['inh']['aggregate']['median'] = inh_agg.median().item()
                        # layer_summary[name]['inh']['aggregate']['q1'] = inh_agg.quantile(q=0.25).item()
                        # layer_summary[name]['inh']['aggregate']['q3'] = inh_agg.quantile(q=0.75).item()
                    
                    if self.raw_summary:
                        self.data_dict[name]['inh'] = inh.tolist()

            if 'Vout' in module_dict.keys():
                print('Vout', flush=True)
                Vout: torch.Tensor = module_dict['Vout']
                # Vout = Vout.cpu()

                if self.network_summary or self.layer_summary:
                    Vout_flat = Vout.flatten()
                
                if self.network_summary:
                    branch_net.append(Vout_flat)
                
                if self.layer_summary:
                    layer_summary[name]['Vout'] = {}
                    layer_summary[name]['Vout']['mean'] = Vout_flat.mean().item()
                    layer_summary[name]['Vout']['std'] = Vout_flat.std().item()
                    # layer_summary[name]['Vout']['min'] = Vout_flat.min().item()
                    # layer_summary[name]['Vout']['max'] = Vout_flat.max().item()
                    # layer_summary[name]['Vout']['median'] = Vout_flat.median().item()
                    # layer_summary[name]['Vout']['q1'] = Vout_flat.quantile(q=0.25).item()
                    # layer_summary[name]['Vout']['q3'] = Vout_flat.quantile(q=0.75).item()
            
                if self.raw_summary:
                    self.data_dict[name]['Vout'] = Vout.tolist()

        if self.network_summary:
            network_summary = {}
            if self.synapse_activations:
                network_summary['exc'] = {}
                network_summary['inh'] = {}
            if self.branch_activations:
                network_summary['Vout'] = {}
            if self.individual_synapse_activations:
                network_summary['exc']['individual'] = {}
                network_summary['inh']['individual'] = {}
                network_summary['exc']['individual']['mean'] = torch.cat(exc_ind_net).mean().item()
                network_summary['inh']['individual']['mean'] = torch.cat(inh_ind_net).mean().item()
                network_summary['exc']['individual']['std'] = torch.cat(exc_ind_net).std().item()
                network_summary['inh']['individual']['std'] = torch.cat(inh_ind_net).std().item()
                # network_summary['exc']['individual']['min'] = torch.cat(exc_ind_net).min().item()
                # network_summary['inh']['individual']['min'] = torch.cat(inh_ind_net).min().item()
                # network_summary['exc']['individual']['max'] = torch.cat(exc_ind_net).max().item()
                # network_summary['inh']['individual']['max'] = torch.cat(inh_ind_net).max().item()
                # network_summary['exc']['individual']['median'] = torch.cat(exc_ind_net).median().item()
                # network_summary['inh']['individual']['median'] = torch.cat(inh_ind_net).median().item()
                # network_summary['exc']['individual']['q1'] = torch.cat(exc_ind_net).quantile(q=0.25).item()
                # network_summary['inh']['individual']['q1'] = torch.cat(inh_ind_net).quantile(q=0.25).item()
                # network_summary['exc']['individual']['q3'] = torch.cat(exc_ind_net).quantile(q=0.75).item()
                # network_summary['inh']['individual']['q3'] = torch.cat(inh_ind_net).quantile(q=0.75).item()
            if self.aggregate_synapse_activations:
                network_summary['exc']['aggregate'] = {}
                network_summary['inh']['aggregate'] = {}
                network_summary['exc']['aggregate']['mean'] = torch.cat(exc_agg_net).mean().item()
                network_summary['inh']['aggregate']['mean'] = torch.cat(inh_agg_net).mean().item()
                network_summary['exc']['aggregate']['std'] = torch.cat(exc_agg_net).std().item()
                network_summary['inh']['aggregate']['std'] = torch.cat(inh_agg_net).std().item()
                # network_summary['exc']['aggregate']['min'] = torch.cat(exc_agg_net).min().item()
                # network_summary['inh']['aggregate']['min'] = torch.cat(inh_agg_net).min().item()
                # network_summary['exc']['aggregate']['max'] = torch.cat(exc_agg_net).max().item()
                # network_summary['inh']['aggregate']['max'] = torch.cat(inh_agg_net).max().item()
                # network_summary['exc']['aggregate']['median'] = torch.cat(exc_agg_net).median().item()
                # network_summary['inh']['aggregate']['median'] = torch.cat(inh_agg_net).median().item()
                # network_summary['exc']['aggregate']['q1'] = torch.cat(exc_agg_net).quantile(q=0.25).item()
                # network_summary['inh']['aggregate']['q1'] = torch.cat(inh_agg_net).quantile(q=0.25).item()
                # network_summary['exc']['aggregate']['q3'] = torch.cat(exc_agg_net).quantile(q=0.75).item()
                # network_summary['inh']['aggregate']['q3'] = torch.cat(inh_agg_net).quantile(q=0.75).item()
            
        results = {
            'network_summary': network_summary,
            'layer_summary': layer_summary,
            'data_dict': self.data_dict
        }

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results



# class ActivationAnalyzer:
#     def __init__(
#         self,
#         individual_synapse_activations: bool = True,
#         aggregate_synapse_activations: bool = True,
#         branch_activations: bool = True,
#         network_summary: bool = True,
#         layer_summary: bool = True,
#         raw_summary: bool = True,
#         downsample: bool = False,
#         samples: int = 1000
#     ):
#         self.synapse_activations = (
#             individual_synapse_activations or aggregate_synapse_activations
#         )
#         self.individual_synapse_activations = individual_synapse_activations
#         self.aggregate_synapse_activations = aggregate_synapse_activations
#         self.branch_activations = branch_activations
#         self.network_summary = network_summary
#         self.layer_summary = layer_summary
#         self.raw_summary = raw_summary
#         self.downsample = downsample
#         self.samples = samples

#     def attach_forward_hooks(self, model: BaseModel):
#         handles = []
#         for name, module in model.named_modules():
#             if isinstance(module, DendriticBranchLayer):
#                 if name not in self.data_dict.keys():
#                     self.data_dict[name] = {}
#                     module._name = name
#                     handle = module.register_forward_hook(self.forward_hook)
#                     handles.append(handle)
#         return handles
    
#     def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
#         for handle in handles:
#             handle.remove()
    
#     def forward_hook(
#         self, 
#         module: DendriticBranchLayer, 
#         input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
#         output: torch.Tensor
#     ):
#         excitatory_input, inhibitory_input, _ = input
#         excitatory_input = excitatory_input[..., None]
#         inhibitory_input = inhibitory_input[..., None]

#         key = module._name

#         if self.synapse_activations and module.branch_excitation is not None:
#             pruned_exc_syn = module.branch_excitation.pruned_weight().t()
#             for _ in range(excitatory_input.ndim - 2):
#                 pruned_exc_syn = pruned_exc_syn[None, ...]
#             self.data_dict[key]['exc'] = excitatory_input * pruned_exc_syn

#         if self.synapse_activations and module.branch_inhibition is not None:
#             pruned_inh_syn = module.branch_inhibition.pruned_weight().t()
#             for _ in range(inhibitory_input.ndim - 2):
#                 pruned_inh_syn = pruned_inh_syn[None, ...]
#             self.data_dict[key]['inh'] = inhibitory_input * pruned_inh_syn
        
#         if self.branch_activations:
#             self.data_dict[key]['Vout'] = output

#     def analyze(
#         self,
#         model: BaseModel,
#         test_dataset: torch.utils.data.Dataset,
#         device: str = 'cpu',
#         save_path: Optional[str] = None,
#         filename: str = 'final'
#     ):
#         self.data_dict = {}

#         model = model.to(device)
#         model.eval()

#         inputs, _ = test_dataset[:]
#         inputs: torch.Tensor = inputs.to(device)

#         if self.downsample:
#             ixs = torch.randperm(inputs.shape[0])[:self.samples]
#             inputs = inputs[ixs]

#         handles = self.attach_forward_hooks(model)

#         _ = model(inputs)

#         if self.network_summary:
#             if self.individual_synapse_activations:
#                 exc_ind_net = []
#                 inh_ind_net = []
#             if self.aggregate_synapse_activations:
#                 exc_agg_net = []
#                 inh_agg_net = []
#             if self.branch_activations:
#                 branch_net = []
        
#         if self.layer_summary:
#             layer_summary = {}
        
#         for name, module_dict in self.data_dict.items():
#             print(name, flush=True)
#             module_dict: dict
#             if self.layer_summary:
#                 layer_summary[name] = {}

#             if 'exc' in module_dict.keys():
#                 print('exc', flush=True)
#                 exc: torch.Tensor = module_dict['exc']
#                 # exc = exc.cpu()

#                 if self.network_summary or self.layer_summary:
#                     if self.individual_synapse_activations:
#                         exc_ind = exc.flatten()
#                         exc_ind = exc_ind[exc_ind > 1e-3]
#                     if self.aggregate_synapse_activations:
#                         exc_agg = exc.sum(dim=-2)
#                         exc_agg = exc_agg.flatten()

#                 if self.network_summary:
#                     print('network summary', flush=True)
#                     if self.individual_synapse_activations:
#                         exc_ind_net.append(exc_ind)
#                     if self.aggregate_synapse_activations:
#                         exc_agg_net.append(exc_agg)
                
#                 if self.layer_summary:
#                     print('layer summary', flush=True)
#                     layer_summary[name]['exc'] = {}
#                     if self.individual_synapse_activations:
#                         layer_summary[name]['exc']['individual'] = {}
#                         layer_summary[name]['exc']['individual']['mean'] = exc_ind.mean().item()
#                         layer_summary[name]['exc']['individual']['std'] = exc_ind.std().item()
#                         # layer_summary[name]['exc']['individual']['min'] = exc_ind.min().item()
#                         # layer_summary[name]['exc']['individual']['max'] = exc_ind.max().item()
#                         # layer_summary[name]['exc']['individual']['median'] = exc_ind.median().item()
#                         # layer_summary[name]['exc']['individual']['q1'] = exc_ind.quantile(q=0.25).item()
#                         # layer_summary[name]['exc']['individual']['q3'] = exc_ind.quantile(q=0.75).item()
#                     if self.aggregate_synapse_activations:
#                         layer_summary[name]['exc']['aggregate'] = {}
#                         layer_summary[name]['exc']['aggregate']['mean'] = exc_agg.mean().item()
#                         layer_summary[name]['exc']['aggregate']['std'] = exc_agg.std().item()
#                         # layer_summary[name]['exc']['aggregate']['min'] = exc_agg.min().item()
#                         # layer_summary[name]['exc']['aggregate']['max'] = exc_agg.max().item()
#                         # layer_summary[name]['exc']['aggregate']['median'] = exc_agg.median().item()
#                         # layer_summary[name]['exc']['aggregate']['q1'] = exc_agg.quantile(q=0.25).item()
#                         # layer_summary[name]['exc']['aggregate']['q3'] = exc_agg.quantile(q=0.75).item()
                
#                 if self.raw_summary:
#                     print('raw summary', flush=True)
#                     self.data_dict[name]['exc'] = exc.tolist()
            
#             if 'inh' in module_dict.keys():
#                 print('inh', flush=True)
#                 inh: torch.Tensor = module_dict['inh']
#                 # inh = inh.cpu()

#                 if self.network_summary or self.layer_summary:
#                     if self.individual_synapse_activations:
#                         inh_ind = inh.flatten()
#                         inh_ind = inh_ind[inh_ind > 1e-3]
#                     if self.aggregate_synapse_activations:
#                         inh_agg = inh.sum(dim=-2)
#                         inh_agg = inh_agg.flatten()

#                 if self.network_summary:
#                     if self.individual_synapse_activations:
#                         inh_ind_net.append(inh_ind)
#                     if self.aggregate_synapse_activations:
#                         inh_agg_net.append(inh_agg)
                
#                 if self.layer_summary:
#                     layer_summary[name]['inh'] = {}
#                     if self.individual_synapse_activations:
#                         layer_summary[name]['inh']['individual'] = {}
#                         layer_summary[name]['inh']['individual']['mean'] = inh_ind.mean().item()
#                         layer_summary[name]['inh']['individual']['std'] = inh_ind.std().item()
#                         # layer_summary[name]['inh']['individual']['min'] = inh_ind.min().item()
#                         # layer_summary[name]['inh']['individual']['max'] = inh_ind.max().item()
#                         # layer_summary[name]['inh']['individual']['median'] = inh_ind.median().item()
#                         # layer_summary[name]['inh']['individual']['q1'] = inh_ind.quantile(q=0.25).item()
#                         # layer_summary[name]['inh']['individual']['q3'] = inh_ind.quantile(q=0.75).item()
#                     if self.aggregate_synapse_activations:
#                         layer_summary[name]['inh']['aggregate'] = {}
#                         layer_summary[name]['inh']['aggregate']['mean'] = inh_agg.mean().item()
#                         layer_summary[name]['inh']['aggregate']['std'] = inh_agg.std().item()
#                         # layer_summary[name]['inh']['aggregate']['min'] = inh_agg.min().item()
#                         # layer_summary[name]['inh']['aggregate']['max'] = inh_agg.max().item()
#                         # layer_summary[name]['inh']['aggregate']['median'] = inh_agg.median().item()
#                         # layer_summary[name]['inh']['aggregate']['q1'] = inh_agg.quantile(q=0.25).item()
#                         # layer_summary[name]['inh']['aggregate']['q3'] = inh_agg.quantile(q=0.75).item()
                    
#                     if self.raw_summary:
#                         self.data_dict[name]['inh'] = inh.tolist()

#             if 'Vout' in module_dict.keys():
#                 print('Vout', flush=True)
#                 Vout: torch.Tensor = module_dict['Vout']
#                 # Vout = Vout.cpu()

#                 if self.network_summary or self.layer_summary:
#                     Vout_flat = Vout.flatten()
                
#                 if self.network_summary:
#                     branch_net.append(Vout_flat)
                
#                 if self.layer_summary:
#                     layer_summary[name]['Vout'] = {}
#                     layer_summary[name]['Vout']['mean'] = Vout_flat.mean().item()
#                     layer_summary[name]['Vout']['std'] = Vout_flat.std().item()
#                     # layer_summary[name]['Vout']['min'] = Vout_flat.min().item()
#                     # layer_summary[name]['Vout']['max'] = Vout_flat.max().item()
#                     # layer_summary[name]['Vout']['median'] = Vout_flat.median().item()
#                     # layer_summary[name]['Vout']['q1'] = Vout_flat.quantile(q=0.25).item()
#                     # layer_summary[name]['Vout']['q3'] = Vout_flat.quantile(q=0.75).item()
            
#                 if self.raw_summary:
#                     self.data_dict[name]['Vout'] = Vout.tolist()

#         if self.network_summary:
#             network_summary = {}
#             if self.synapse_activations:
#                 network_summary['exc'] = {}
#                 network_summary['inh'] = {}
#             if self.branch_activations:
#                 network_summary['Vout'] = {}
#             if self.individual_synapse_activations:
#                 network_summary['exc']['individual'] = {}
#                 network_summary['inh']['individual'] = {}
#                 network_summary['exc']['individual']['mean'] = torch.cat(exc_ind_net).mean().item()
#                 network_summary['inh']['individual']['mean'] = torch.cat(inh_ind_net).mean().item()
#                 network_summary['exc']['individual']['std'] = torch.cat(exc_ind_net).std().item()
#                 network_summary['inh']['individual']['std'] = torch.cat(inh_ind_net).std().item()
#                 # network_summary['exc']['individual']['min'] = torch.cat(exc_ind_net).min().item()
#                 # network_summary['inh']['individual']['min'] = torch.cat(inh_ind_net).min().item()
#                 # network_summary['exc']['individual']['max'] = torch.cat(exc_ind_net).max().item()
#                 # network_summary['inh']['individual']['max'] = torch.cat(inh_ind_net).max().item()
#                 # network_summary['exc']['individual']['median'] = torch.cat(exc_ind_net).median().item()
#                 # network_summary['inh']['individual']['median'] = torch.cat(inh_ind_net).median().item()
#                 # network_summary['exc']['individual']['q1'] = torch.cat(exc_ind_net).quantile(q=0.25).item()
#                 # network_summary['inh']['individual']['q1'] = torch.cat(inh_ind_net).quantile(q=0.25).item()
#                 # network_summary['exc']['individual']['q3'] = torch.cat(exc_ind_net).quantile(q=0.75).item()
#                 # network_summary['inh']['individual']['q3'] = torch.cat(inh_ind_net).quantile(q=0.75).item()
#             if self.aggregate_synapse_activations:
#                 network_summary['exc']['aggregate'] = {}
#                 network_summary['inh']['aggregate'] = {}
#                 network_summary['exc']['aggregate']['mean'] = torch.cat(exc_agg_net).mean().item()
#                 network_summary['inh']['aggregate']['mean'] = torch.cat(inh_agg_net).mean().item()
#                 network_summary['exc']['aggregate']['std'] = torch.cat(exc_agg_net).std().item()
#                 network_summary['inh']['aggregate']['std'] = torch.cat(inh_agg_net).std().item()
#                 # network_summary['exc']['aggregate']['min'] = torch.cat(exc_agg_net).min().item()
#                 # network_summary['inh']['aggregate']['min'] = torch.cat(inh_agg_net).min().item()
#                 # network_summary['exc']['aggregate']['max'] = torch.cat(exc_agg_net).max().item()
#                 # network_summary['inh']['aggregate']['max'] = torch.cat(inh_agg_net).max().item()
#                 # network_summary['exc']['aggregate']['median'] = torch.cat(exc_agg_net).median().item()
#                 # network_summary['inh']['aggregate']['median'] = torch.cat(inh_agg_net).median().item()
#                 # network_summary['exc']['aggregate']['q1'] = torch.cat(exc_agg_net).quantile(q=0.25).item()
#                 # network_summary['inh']['aggregate']['q1'] = torch.cat(inh_agg_net).quantile(q=0.25).item()
#                 # network_summary['exc']['aggregate']['q3'] = torch.cat(exc_agg_net).quantile(q=0.75).item()
#                 # network_summary['inh']['aggregate']['q3'] = torch.cat(inh_agg_net).quantile(q=0.75).item()
            
#         results = {
#             'network_summary': network_summary,
#             'layer_summary': layer_summary,
#             'data_dict': self.data_dict
#         }

#         self.remove_forward_hooks(handles)

#         if save_path is not None:
#             save_dict(results, save_path, filename)
#         else:
#             return results


class AnalysisManager:
    def __init__(
        self,
        model: BaseModel,
        data: Dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = 'cpu'
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
        
        ablation_config = self.config.ablation_analysis
        self.ablation_enabled = ablation_config.get('enabled', False)
        self.ablation_training = ablation_config.get('training', False)
        if self.ablation_enabled:
            ablation_params: dict = ablation_config.get('params', {})
            self.ablation_analyzer = AblationAnalyzer(
                ablate_synapses=ablation_params.get('ablate_synapses', True),
                ablate_inhibition=ablation_params.get('ablate_inhibition', True),
                ablate_branches=ablation_params.get('ablate_branches', True),
                accuracy=ablation_params.get('accuracy', True),
                categorical_loglikelihood=ablation_params.get('categorical_loglikelihood', True)
            )
        
        weight_config = self.config.weight_analysis
        self.weight_enabled = weight_config.get('enabled', False)
        self.weight_training = weight_config.get('training', False)
        if self.weight_enabled:
            weight_params: dict = weight_config.get('params', {})
            self.weight_analyzer = WeightAnalyzer(
                synapse_weights=weight_params.get('synapse_weights', True),
                branch_weights=weight_params.get('branch_weights', True)
            )

        activation_config = self.config.activation_analysis
        self.activation_enabled = activation_config.get('enabled', False)
        self.activation_training = activation_config.get('training', False)
        if self.activation_enabled:
            self.activation_analyzer = ActivationAnalyzer(
                individual_synapse_activations=activation_config.get('individual_synapse_activations', True),
                aggregate_synapse_activations=activation_config.get('aggregate_synapse_activations', True),
                branch_activations=activation_config.get('branch_activations', True),
                network_summary=activation_config.get('network_summary', True),
                layer_summary=activation_config.get('layer_summary', True),
                raw_summary=activation_config.get('raw_summary', True),
                downsample=activation_config.get('downsample', False),
                samples=activation_config.get('samples', 1000)
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
        
        if (
            self.ablation_enabled and
            ((self.ablation_training and training) or not training)
        ):
            self.ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data['test'],
                device=self.device,
                save_path=os.path.join(self.save_root, 'ablation_analysis'),
                filename=filename
            )
        
        if (
            self.weight_enabled and
            ((self.weight_training and training) or not training)
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, 'weight_analysis'),
                filename=filename
            )
        
        if (
            self.activation_enabled and
            ((self.activation_training and training) or not training)
        ):
            self.activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data['test'],
                device='cpu',
                save_path=os.path.join(self.save_root, 'activation_analysis'),
                filename=filename
            )

        # TODO: add other analysis classes here implemented same as above
                
            
        