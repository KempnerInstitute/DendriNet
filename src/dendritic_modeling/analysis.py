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
    evaluate_categorical_loglikelihood,
    evaluate_mse
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


class PerformanceAnalyzer:
    def __init__(
        self,
        accuracy: bool = False,
        categorical_loglikelihood: bool = False,
        mse: bool = False
    ):
        self.accuracy = accuracy
        self.categorical_loglikelihood = categorical_loglikelihood
        self.mse = mse

    def analyze(
        self,
        model: BaseModel,
        train_ds: Optional[torch.utils.data.Dataset] = None,
        valid_ds: Optional[torch.utils.data.Dataset] = None,
        test_ds: Optional[torch.utils.data.Dataset] = None,
        device: str = 'cpu',
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        results = {}
        if self.accuracy:
            results['accuracy'] = {}
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device
            )
            if train_ds is not None:
                results['accuracy']['train'] = train_acc
            if valid_ds is not None:
                results['accuracy']['valid'] = valid_acc
            if test_ds is not None:
                results['accuracy']['test'] = test_acc

        if self.categorical_loglikelihood:
            results['categorical_loglikelihood'] = {}
            train_catll, valid_catll, test_catll = evaluate_categorical_loglikelihood(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device
            )
            if train_ds is not None:
                results['categorical_loglikelihood']['train'] = train_catll
            if valid_ds is not None:
                results['categorical_loglikelihood']['valid'] = valid_catll
            if test_ds is not None:
                results['categorical_loglikelihood']['test'] = test_catll

        if self.mse:
            results['mse'] = {}
            train_mse, valid_mse, test_mse = evaluate_mse(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device
            )
            if train_ds is not None:
                results['mse']['train'] = train_mse
            if valid_ds is not None:
                results['mse']['valid'] = valid_mse
            if test_ds is not None:
                results['mse']['test'] = test_mse
        
        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


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
            assert isinstance(model.net, ExcitationInhibitionNetwork)
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
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
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

            turnover_dict = {
                'exc': {'active': [], 'stable': [], 'transient': []},
                'inh': {'active': [], 'stable': [], 'transient': []},
                'both': {'active': [], 'stable': [], 'transient': []}
            }
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

                turnover_dict['exc']['active'].append(exc_total_active)
                turnover_dict['exc']['stable'].append(exc_total_stable)
                turnover_dict['exc']['transient'].append(exc_total_transient)

                turnover_dict['inh']['active'].append(inh_total_active)
                turnover_dict['inh']['stable'].append(inh_total_stable)
                turnover_dict['inh']['transient'].append(inh_total_transient)

                turnover_dict['both']['active'].append(exc_total_active + inh_total_active)
                turnover_dict['both']['stable'].append(exc_total_stable + inh_total_stable)
                turnover_dict['both']['transient'].append(exc_total_transient + inh_total_transient)

                # turnover_dict[i] = {}
                # turnover_dict[i]['exc'] = {
                #     'active': exc_total_active,
                #     'stable': exc_total_stable,
                #     'transient': exc_total_transient
                # }
                # turnover_dict[i]['inh'] = {
                #     'active': inh_total_active,
                #     'stable': inh_total_stable,
                #     'transient': inh_total_transient
                # }
                # turnover_dict[i]['both'] = {
                #     'active': exc_total_active + inh_total_active,
                #     'stable': exc_total_stable + inh_total_stable,
                #     'transient': exc_total_transient + inh_total_transient
                # }
            
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
        ablate_all_synapses: bool = True,
        ablate_excitation: bool = True,
        ablate_inhibition: bool = True,
        ablate_branches: bool = True,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True
    ):
        self.ablate_all_synapses = ablate_all_synapses
        self.ablate_excitation = ablate_excitation
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
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
        model = model.to(device)
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        state_dict = deepcopy(model.state_dict())

        results = {}
        if self.ablate_all_synapses:
            results['all synapses'] = {}
        if self.ablate_excitation:
            results['excitation'] = {}
        if self.ablate_inhibition:
            results['inhibition'] = {}
        if self.ablate_branches:
            results['branches'] = {}

        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    self.ablate_all_synapses 
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
                    self.ablate_excitation
                    and module.branch_excitation is not None
                ):
                    results['excitation'][name] = {}
                    results['excitation'][name]['layer_idx'] = module.layer_idx
                    
                    module.branch_excitation.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['excitation'][name]['accuracy'] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model,
                            test_ds=test_dataset,
                            move_device=False
                        )[-1]
                        results['excitation'][name]['log likelihood'] = loglikelihood
                    
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
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
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


class SynapticActivationAnalyzer:
    def __init__(
        self,
        network_summary: bool = True,
        layer_summary: bool = True,
        branch_summary: bool = True,
        raw_summary: bool = True,
        logspace: bool = False,
        downsample: bool = False,
        samples: int = 1000
    ):
        self.network_summary = network_summary
        self.layer_summary = layer_summary
        self.branch_summary = branch_summary
        self.raw_summary = raw_summary
        self.logspace = logspace
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
        excitatory_input = excitatory_input[..., None]
        inhibitory_input = inhibitory_input[..., None]

        key = module._name

        if module.branch_excitation is not None:
            pruned_exc_syn = module.branch_excitation.pruned_weight().t()
            for _ in range(excitatory_input.ndim - 2):
                pruned_exc_syn = pruned_exc_syn[None, ...]
            self.data_dict[key]['exc'] = excitatory_input * pruned_exc_syn

        if module.branch_inhibition is not None:
            pruned_inh_syn = module.branch_inhibition.pruned_weight().t()
            for _ in range(inhibitory_input.ndim - 2):
                pruned_inh_syn = pruned_inh_syn[None, ...]
            self.data_dict[key]['inh'] = inhibitory_input * pruned_inh_syn
        
    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        save_path: Optional[str] = None,
        filename: str = 'final'
    ):
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
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
            exc_net = []
            inh_net = []
        
        if self.layer_summary:
            layer_summary = {}
        
        if self.branch_summary:
            branch_summary = {}

        for name, module_dict in self.data_dict.items():
            module_dict: dict
            if self.layer_summary:
                layer_summary[name] = {}
            if self.branch_summary:
                branch_summary[name] = {}

            if 'exc' in module_dict.keys():
                exc: torch.Tensor = module_dict['exc']
                if self.logspace:
                    exc = torch.log(exc+1e-9)
                
                if self.network_summary or self.layer_summary:
                    exc_flat = exc.flatten()
                    if self.logspace:
                        exc_flat = exc_flat[exc_flat > -20]
                    else:
                        exc_flat = exc_flat[exc_flat > 0]

                if self.network_summary:
                    exc_net.append(exc_flat)
                
                if self.layer_summary:
                    layer_summary[name]['exc'] = {}
                    layer_summary[name]['exc']['mean'] = exc_flat.mean().item()
                    layer_summary[name]['exc']['std'] = exc_flat.std().item()
                    # layer_summary[name]['exc']['min'] = exc_flat.min().item()
                    # layer_summary[name]['exc']['max'] = exc_flat.max().item()
                    # layer_summary[name]['exc']['median'] = exc_flat.median().item()
                    # layer_summary[name]['exc']['q1'] = exc_flat.quantile(q=0.25).item()
                    # layer_summary[name]['exc']['q3'] = exc_flat.quantile(q=0.75).item()
                
                if self.branch_summary:
                    branch_summary[name]['exc'] = {}
                    branch_summary[name]['exc']['mean'] = exc.mean(dim=0).tolist()
                    branch_summary[name]['exc']['std'] = exc.std(dim=0).tolist()
                    # branch_summary[name]['exc']['min'] = exc.min(dim=0).tolist()
                    # branch_summary[name]['exc']['max'] = exc.max(dim=0).tolist()
                    # branch_summary[name]['exc']['median'] = exc.median(dim=0).tolist()
                    # branch_summary[name]['exc']['q1'] = exc.quantile(q=0.25, dim=0).tolist()
                    # branch_summary[name]['exc']['q3'] = exc.quantile(q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]['exc'] = exc.tolist()
            
            if 'inh' in module_dict.keys():
                inh: torch.Tensor = module_dict['inh']
                if self.logspace:
                    inh = torch.log(inh+1e-9)
                
                if self.network_summary or self.layer_summary:
                    inh_flat = inh.flatten()
                    if self.logspace:
                        inh_flat = inh_flat[inh_flat > -20]
                    else:
                        inh_flat = inh_flat[inh_flat > 0]

                if self.network_summary:
                    inh_net.append(inh_flat)
                
                if self.layer_summary:
                    layer_summary[name]['inh'] = {}
                    layer_summary[name]['inh']['mean'] = inh_flat.mean().item()
                    layer_summary[name]['inh']['std'] = inh_flat.std().item()
                    # layer_summary[name]['inh']['min'] = inh_flat.min().item()
                    # layer_summary[name]['inh']['max'] = inh_flat.max().item()
                    # layer_summary[name]['inh']['median'] = inh_flat.median().item()
                    # layer_summary[name]['inh']['q1'] = inh_flat.quantile(q=0.25).item()
                    # layer_summary[name]['inh']['q3'] = inh_flat.quantile(q=0.75).item()
                
                if self.branch_summary:
                    branch_summary[name]['inh'] = {}
                    branch_summary[name]['inh']['mean'] = inh.mean(dim=0).tolist()
                    branch_summary[name]['inh']['std'] = inh.std(dim=0).tolist()
                    # branch_summary[name]['inh']['min'] = inh.min(dim=0).tolist()
                    # branch_summary[name]['inh']['max'] = inh.max(dim=0).tolist()
                    # branch_summary[name]['inh']['median'] = inh.median(dim=0).tolist()
                    # branch_summary[name]['inh']['q1'] = inh.quantile(q=0.25, dim=0).tolist()
                    # branch_summary[name]['inh']['q3'] = inh.quantile(q=0.75, dim=0).tolist()
                
                if self.raw_summary:
                    self.data_dict[name]['inh'] = inh.tolist()

        results = {'logspace': self.logspace}

        if self.network_summary:
            network_summary = {}
            network_summary['exc'] = {}
            network_summary['inh'] = {}
            network_summary['exc']['mean'] = torch.cat(exc_net).mean().item()
            network_summary['inh']['mean'] = torch.cat(inh_net).mean().item()
            network_summary['exc']['std'] = torch.cat(exc_net).std().item()
            network_summary['inh']['std'] = torch.cat(inh_net).std().item()
            # network_summary['exc']['min'] = torch.cat(exc_net).min().item()
            # network_summary['inh']['min'] = torch.cat(inh_net).min().item()
            # network_summary['exc']['max'] = torch.cat(exc_net).max().item()
            # network_summary['inh']['max'] = torch.cat(inh_net).max().item()
            # network_summary['exc']['median'] = torch.cat(exc_net).median().item()
            # network_summary['inh']['median'] = torch.cat(inh_net).median().item()
            # network_summary['exc']['q1'] = torch.cat(exc_net).quantile(q=0.25).item()
            # network_summary['inh']['q1'] = torch.cat(inh_net).quantile(q=0.25).item()
            # network_summary['exc']['q3'] = torch.cat(exc_net).quantile(q=0.75).item()
            # network_summary['inh']['q3'] = torch.cat(inh_net).quantile(q=0.75).item()
            
            results['network_summary'] = network_summary

        if self.layer_summary:
            results['layer_summary'] = layer_summary
        
        if self.branch_summary:
            results['branch_summary'] = branch_summary

        if self.raw_summary:
            results['raw_summary'] = self.data_dict

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


class BranchActivationAnalyzer:
    def __init__(
        self,
        synapse_activations: bool = True,
        branch_activations: bool = True,
        network_summary: bool = True,
        layer_summary: bool = True,
        branch_summary: bool = True,
        raw_summary: bool = True,
        logspace: bool = False,
        downsample: bool = False,
        samples: int = 1000
    ):
        self.synapse_activations = synapse_activations
        self.branch_activations = branch_activations
        self.network_summary = network_summary
        self.layer_summary = layer_summary
        self.branch_summary = branch_summary
        self.raw_summary = raw_summary
        self.logspace = logspace
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
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
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
                
                if self.logspace:
                    exc = torch.log(exc)

                if self.network_summary:
                    print('network summary', flush=True)
                    exc_net.append(exc.flatten())                
                
                if self.layer_summary:
                    print('layer summary', flush=True)
                    layer_summary[name]['exc'] = {}
                    layer_summary[name]['exc']['mean'] = torch.mean(exc).item()
                    layer_summary[name]['exc']['std'] = torch.std(exc).item()
                    # layer_summary[name]['exc']['min'] = exc.min().item()
                    # layer_summary[name]['exc']['max'] = exc.max().item()
                    # layer_summary[name]['exc']['median'] = exc.median().item()
                    # layer_summary[name]['exc']['q1'] = exc.quantile(q=0.25).item()
                    # layer_summary[name]['exc']['q3'] = exc.quantile(q=0.75).item()
                
                if self.branch_summary:
                    print('branch summary', flush=True)
                    branch_summary[name]['exc'] = {}
                    branch_summary[name]['exc']['mean'] = torch.mean(exc, dim=0).tolist()
                    branch_summary[name]['exc']['std'] = torch.std(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['min'] = torch.min(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['max'] = torch.max(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['median'] = torch.median(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['q1'] = torch.quantile(exc, q=0.25, dim=0).tolist()
                    # branch_summary[name]['exc']['q3'] = torch.quantile(exc, q=0.75, dim=0).tolist()
                
                if self.raw_summary:
                    self.data_dict[name]['exc'] = exc.tolist()
            if 'inh' in module_dict.keys():
                print('inh', flush=True)
                inh: torch.Tensor = module_dict['inh']
                
                if self.logspace:
                    inh = torch.log(inh)

                if self.network_summary:
                    print('network summary', flush=True)
                    inh_net.append(inh.flatten())
                
                if self.layer_summary:
                    print('layer summary', flush=True)
                    layer_summary[name]['inh'] = {}
                    layer_summary[name]['inh']['mean'] = torch.mean(inh).item()
                    layer_summary[name]['inh']['std'] = torch.std(inh).item()
                    # layer_summary[name]['inh']['min'] = inh.min().item()
                    # layer_summary[name]['inh']['max'] = inh.max().item()
                    # layer_summary[name]['inh']['median'] = inh.median().item()
                    # layer_summary[name]['inh']['q1'] = inh.quantile(q=0.25).item()
                    # layer_summary[name]['inh']['q3'] = inh.quantile(q=0.75).item()
                    
                if self.branch_summary:
                    print('branch summary', flush=True)
                    branch_summary[name]['inh'] = {}
                    branch_summary[name]['inh']['mean'] = torch.mean(inh, dim=0).tolist()
                    branch_summary[name]['inh']['std'] = torch.std(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['min'] = torch.min(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['max'] = torch.max(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['median'] = torch.median(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['q1'] = torch.quantile(inh, q=0.25, dim=0).tolist()
                    # branch_summary[name]['inh']['q3'] = torch.quantile(inh, q=0.75, dim=0).tolist()
                
                if self.raw_summary:
                    self.data_dict[name]['inh'] = inh.tolist()

            if 'Vout' in module_dict.keys():
                print('Vout', flush=True)
                Vout: torch.Tensor = module_dict['Vout']

                if self.logspace:
                    Vout = torch.log(Vout)

                if self.network_summary:
                    print('network summary', flush=True)
                    Vout_net.append(Vout.flatten())
                
                if self.layer_summary:
                    print('layer summary', flush=True)
                    layer_summary[name]['Vout'] = {}
                    layer_summary[name]['Vout']['mean'] = torch.mean(Vout).item()
                    layer_summary[name]['Vout']['std'] = torch.std(Vout).item()
                    # layer_summary[name]['Vout']['min'] = Vout.min().item()
                    # layer_summary[name]['Vout']['max'] = Vout.max().item()
                    # layer_summary[name]['Vout']['median'] = Vout.median().item()
                    # layer_summary[name]['Vout']['q1'] = Vout.quantile(q=0.25).item()
                    # layer_summary[name]['Vout']['q3'] = Vout.quantile(q=0.75).item()
                
                if self.branch_summary:
                    print('branch summary', flush=True)
                    branch_summary[name]['Vout'] = {}
                    branch_summary[name]['Vout']['mean'] = torch.mean(Vout, dim=0).tolist()
                    branch_summary[name]['Vout']['std'] = torch.std(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['min'] = torch.min(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['max'] = torch.max(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['median'] = torch.median(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['q1'] = torch.quantile(Vout, q=0.25, dim=0).tolist()
                    # branch_summary[name]['Vout']['q3'] = torch.quantile(Vout, q=0.75, dim=0).tolist()
                
                if self.raw_summary:
                    self.data_dict[name]['Vout'] = Vout.tolist()

        results = {'logspace': self.logspace}

        if self.network_summary:
            network_summary = {}

            if self.synapse_activations:
                exc_net = torch.cat(exc_net, dim=0)
                network_summary['exc'] = {}
                network_summary['exc']['mean'] = torch.mean(exc_net).item()
                network_summary['exc']['std'] = torch.std(exc_net).item()
                # network_summary['exc']['min'] = torch.min(exc_net).item()
                # network_summary['exc']['max'] = torch.max(exc_net).item()
                # network_summary['exc']['median'] = torch.median(exc_net).item()
                # network_summary['exc']['q1'] = torch.quantile(exc_net, q=0.25).item()
                # network_summary['exc']['q3'] = torch.quantile(exc_net, q=0.75).item()

                inh_net = torch.cat(inh_net, dim=0)
                network_summary['inh'] = {}
                network_summary['inh']['mean'] = torch.mean(inh_net).item()
                network_summary['inh']['std'] = torch.std(inh_net).item()
                # network_summary['inh']['min'] = torch.min(inh_net).item()
                # network_summary['inh']['max'] = torch.max(inh_net).item()
                # network_summary['inh']['median'] = torch.median(inh_net).item()
                # network_summary['inh']['q1'] = torch.quantile(inh_net, q=0.25).item()
                # network_summary['inh']['q3'] = torch.quantile(inh_net, q=0.75).item()

            if self.branch_activations:
                Vout_net = torch.cat(Vout_net, dim=0)
                network_summary['Vout'] = {}
                network_summary['Vout']['mean'] = torch.mean(Vout_net).item()
                network_summary['Vout']['std'] = torch.std(Vout_net).item()
                # network_summary['Vout']['min'] = torch.min(Vout_net).item()
                # network_summary['Vout']['max'] = torch.max(Vout_net).item()
                # network_summary['Vout']['median'] = torch.median(Vout_net).item()
                # network_summary['Vout']['q1'] = torch.quantile(Vout_net, q=0.25).item()
                # network_summary['Vout']['q3'] = torch.quantile(Vout_net, q=0.75).item()
            
            results['network_summary'] = network_summary

        if self.layer_summary:
            results['layer_summary'] = layer_summary

        if self.branch_summary:
            results['branch_summary'] = branch_summary

        if self.raw_summary:
            results['raw_summary'] = self.data_dict

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


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

        performance_config = self.config.performance_analysis
        self.performance_enabled = performance_config.get('enabled', False)
        self.performance_training = performance_config.get('training', False)
        if self.performance_enabled:
            performance_params: dict = performance_config.get('params', {})
            self.performance_analyzer = PerformanceAnalyzer(
                accuracy=performance_params.get('accuracy', False),
                categorical_loglikelihood=performance_params.get('categorical_loglikelihood', False),
                mse=performance_params.get('mse', False)
            )
                
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
                ablate_all_synapses=ablation_params.get('ablate_all_synapses', True),
                ablate_excitation=ablation_params.get('ablate_excitation', True),
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

        synaptic_activation_config = self.config.synaptic_activation_analysis
        self.synaptic_activation_enabled = synaptic_activation_config.get('enabled', False)
        self.synaptic_activation_training = synaptic_activation_config.get('training', False)
        if self.synaptic_activation_enabled:
            synaptic_activation_params: dict = synaptic_activation_config.get('params', {})
            self.synaptic_activation_analyzer = SynapticActivationAnalyzer(
                network_summary=synaptic_activation_params.get('network_summary', True),
                layer_summary=synaptic_activation_params.get('layer_summary', True),
                branch_summary=synaptic_activation_params.get('branch_summary', True),
                raw_summary=synaptic_activation_params.get('raw_summary', True),
                logspace=synaptic_activation_params.get('logspace', False),
                downsample=synaptic_activation_params.get('downsample', False),
                samples=synaptic_activation_params.get('samples', 1000)
            )
        
        branch_activation_config = self.config.branch_activation_analysis
        self.branch_activation_enabled = branch_activation_config.get('enabled', False)
        self.branch_activation_training = branch_activation_config.get('training', False)
        if self.branch_activation_enabled:
            branch_activation_params: dict = branch_activation_config.get('params', {})
            self.branch_activation_analyzer = BranchActivationAnalyzer(
                synapse_activations=branch_activation_params.get('synapse_activations', True),
                branch_activations=branch_activation_params.get('branch_activations', True),
                network_summary=branch_activation_params.get('network_summary', True),
                layer_summary=branch_activation_params.get('layer_summary', True),
                branch_summary=branch_activation_params.get('branch_summary', True),
                raw_summary=branch_activation_params.get('raw_summary', True),
                logspace=branch_activation_params.get('logspace', False),
                downsample=branch_activation_params.get('downsample', False),
                samples=branch_activation_params.get('samples', 1000)
            )
    
    def run_analysis(self, filename: str='final', training: bool=False):
        if (
            self.performance_enabled and
            ((self.performance_training and training) or not training)
        ):
            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data['train'],
                valid_ds=self.data['valid'],
                test_ds=self.data['test'],
                device=self.device,
                save_path=os.path.join(self.save_root, 'performance'),
                filename=filename
            )
        
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
            self.synaptic_activation_enabled and
            ((self.synaptic_activation_training and training) or not training)
        ):
            self.synaptic_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data['test'],
                device='cpu',
                save_path=os.path.join(self.save_root, 'synaptic_activation_analysis'),
                filename=filename
            )
        
        if (
            self.branch_activation_enabled and
            ((self.branch_activation_training and training) or not training)
        ):
            self.branch_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data['test'],
                device='cpu',
                save_path=os.path.join(self.save_root, 'branch_activation_analysis'),
                filename=filename
            )


        # TODO: add other analysis classes here implemented same as above
                
            
        