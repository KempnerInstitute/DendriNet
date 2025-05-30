import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.dendrinet import DendriticBranchLayer
from dendritic_modeling.information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.utils import (
    add_gaussian_noise,
    add_uniform_noise,
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    evaluate_mse,
    save_dict,
)


class PerformanceAnalyzer:
    def __init__(
        self,
        accuracy: bool = False,
        categorical_loglikelihood: bool = False,
        mse: bool = False,
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
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        results = {}
        if self.accuracy:
            results["accuracy"] = {}
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["accuracy"]["train"] = train_acc
            if valid_ds is not None:
                results["accuracy"]["valid"] = valid_acc
            if test_ds is not None:
                results["accuracy"]["test"] = test_acc

        if self.categorical_loglikelihood:
            results["categorical_loglikelihood"] = {}
            train_catll, valid_catll, test_catll = evaluate_categorical_loglikelihood(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["categorical_loglikelihood"]["train"] = train_catll
            if valid_ds is not None:
                results["categorical_loglikelihood"]["valid"] = valid_catll
            if test_ds is not None:
                results["categorical_loglikelihood"]["test"] = test_catll

        if self.mse:
            results["mse"] = {}
            train_mse, valid_mse, test_mse = evaluate_mse(
                classifier=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                device=device,
            )
            if train_ds is not None:
                results["mse"]["train"] = train_mse
            if valid_ds is not None:
                results["mse"]["valid"] = valid_mse
            if test_ds is not None:
                results["mse"]["test"] = test_mse

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
        n_neighbors: int = 10,
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
        output: torch.Tensor,
    ):
        excitatory_input, inhibitory_input, _ = input
        key = module._name
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        if self.mi_EI_Vout:
            self.data_dict[key]["Vout"] = output

    def compute_mi_E_I(self, excitation: np.ndarray, inhibition: np.ndarray) -> float:
        return compute_mi_cc(excitation, inhibition, self.n_neighbors)

    def compute_mi_EI_Vout(self, EI: np.ndarray, Vout: np.ndarray) -> float:
        return compute_mi_cc(EI, Vout, self.n_neighbors)

    def compute_sr_EI_Vout(
        self,
        excitation: np.ndarray,
        inhibition: np.ndarray,
        Vout: np.ndarray,
        mi_EI_Vout: Optional[float] = None,
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
        device: str = "cpu",  # 'cuda' if torch.cuda.is_available() else 'cpu'
        save_path: Optional[str] = None,
        filename: str = "final",
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
            excitation.append(branch_dict["excitation"])
            inhibition.append(branch_dict["inhibition"])
            if self.mi_EI_Vout:
                Vout.append(branch_dict["Vout"])

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
            info_dict["mi_E_I"] = self.compute_mi_E_I(excitation, inhibition)

        if self.mi_EI_Vout:
            Vout = torch.cat(Vout, dim=-1).detach().cpu().flatten().numpy()[:, None]
            info_dict["mi_EI_Vout"] = self.compute_mi_EI_Vout(EI, Vout)

        if self.sr_EI_Vout:
            mi_EI_Vout = None
            if self.mi_EI_Vout:
                mi_EI_Vout = info_dict["mi_EI_Vout"]
            (
                si_EI_Vout,
                ri_EI_Vout,
                mi_EI_Vout,
                mi_E_Vout,
                mi_I_Vout,
            ) = self.compute_sr_EI_Vout(excitation, inhibition, Vout, mi_EI_Vout)
            info_dict["si_EI_Vout"] = si_EI_Vout
            info_dict["ri_EI_Vout"] = ri_EI_Vout
            info_dict["mi_EI_Vout"] = mi_EI_Vout
            info_dict["mi_E_Vout"] = mi_E_Vout
            info_dict["mi_I_Vout"] = mi_I_Vout

        if self.mi_EI_label:
            info_dict["mi_EI_label"] = self.compute_mi_EI_label(EI, labels)

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(info_dict, save_path, filename)
        else:
            return info_dict


class SynapticTurnoverAnalyzer:
    def __init__(self, model: BaseModel):
        assert isinstance(model.net, ExcitationInhibitionNetwork), (
            "model.net must be an instance of ExcitationInhibitionNetwork. "
            + f"Got model.net type: {type(model.net)}"
        )

        self.data_dict_list = []
        self.analyze(model=model, training=True)

    def analyze(
        self,
        model: BaseModel,
        training: bool = True,
        save_path: Optional[str] = None,
        filename: str = "final",
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
                        data_dict[name]["exc_mask"] = (
                            deepcopy(module.branch_excitation.weight_mask())
                            .detach()
                            .cpu()
                        )
                    if module.branch_inhibition is not None:
                        data_dict[name]["inh_mask"] = (
                            deepcopy(module.branch_inhibition.weight_mask())
                            .detach()
                            .cpu()
                        )
            self.data_dict_list.append(data_dict)
        else:

            def turnover_helper(
                prev_mask: torch.Tensor,
                curr_mask: torch.Tensor,
                total_active: int,
                total_stable: int,
                total_transient: int,
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
                "exc": {"active": [], "stable": [], "transient": []},
                "inh": {"active": [], "stable": [], "transient": []},
                "both": {"active": [], "stable": [], "transient": []},
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

                    if "exc_mask" in curr_module_dict.keys():
                        prev_exc_mask = prev_module_dict["exc_mask"]
                        curr_exc_mask = curr_module_dict["exc_mask"]
                        (
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        ) = turnover_helper(
                            prev_exc_mask,
                            curr_exc_mask,
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        )

                    if "inh_mask" in curr_module_dict.keys():
                        prev_inh_mask = prev_module_dict["inh_mask"]
                        curr_inh_mask = curr_module_dict["inh_mask"]
                        (
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        ) = turnover_helper(
                            prev_inh_mask,
                            curr_inh_mask,
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        )

                turnover_dict["exc"]["active"].append(exc_total_active)
                turnover_dict["exc"]["stable"].append(exc_total_stable)
                turnover_dict["exc"]["transient"].append(exc_total_transient)

                turnover_dict["inh"]["active"].append(inh_total_active)
                turnover_dict["inh"]["stable"].append(inh_total_stable)
                turnover_dict["inh"]["transient"].append(inh_total_transient)

                turnover_dict["both"]["active"].append(
                    exc_total_active + inh_total_active
                )
                turnover_dict["both"]["stable"].append(
                    exc_total_stable + inh_total_stable
                )
                turnover_dict["both"]["transient"].append(
                    exc_total_transient + inh_total_transient
                )

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
        categorical_loglikelihood: bool = True,
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
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
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
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
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
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["categorical_ll"] = cat_lls

        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results


class BranchInformationAnalyzer:
    """
    Analyzes mutual information between branch components (E, I, V_O, C).
    Uses copula entropy from DVC_tensorflow package for MI computation.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        use_dvc: bool = True,  # Whether to use DVC_tensorflow or fallback to sklearn
        save_plots: bool = True,
        save_csv: bool = True,
        copula_params: Dict = None,  # Additional parameters for copula MI estimation
    ):
        self.n_neighbors = n_neighbors
        self.use_dvc = use_dvc
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

        # Default copula parameters
        self.copula_params = copula_params or {
            "cases": 1000,  # Number of samples for entropy estimation
            "iterations": 10,  # Number of iterations
            "alpha": 0.05,  # Confidence level
            "vine_type": "c-vine",  # Type of vine copula
            "families": "kercop",  # Copula families
            "knots": 30,  # Number of knots for non-parametric copula
        }

        # Initialize DVC components if requested
        if self.use_dvc:
            try:
                import os
                import sys

                # Get the path to the external/DVC_tensorflow directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                dvc_path = os.path.join(project_root, "external", "DVC_tensorflow")

                # Store original paths and working directory
                original_cwd = os.getcwd()
                original_path = sys.path.copy()

                try:
                    # Change to DVC directory for proper relative imports
                    os.chdir(dvc_path)

                    # Clear sys.path and add only DVC path at beginning
                    # This ensures relative imports within DVC work correctly
                    sys.path.insert(0, dvc_path)

                    # Import required modules with proper path context
                    import tensorflow as tf

                    # Import the modules that cop_eval.py needs
                    import utils.tensor_op  # Import this first to ensure it's available

                    # Now import the main modules we need
                    from classes.objects import margin_obj, vine_obj_bin
                    from info.info_estimation import vine_entropy
                    from pre_proc.preparation import prep_cop

                    self.vine_entropy = vine_entropy
                    self.vine_obj_bin = vine_obj_bin
                    self.margin_obj = margin_obj
                    self.prep_cop = prep_cop
                    self.tf = tf

                    self.logger.info(
                        "Successfully loaded DVC_tensorflow for copula-based MI"
                    )

                finally:
                    # Always restore original working directory and path
                    os.chdir(original_cwd)
                    # Don't remove from sys.path since the modules are now loaded

            except ImportError as e:
                self.logger.warning(f"Failed to import DVC_tensorflow: {e}")
                self.logger.warning("Falling back to sklearn-based MI computation")
                # Disable DVC and fall back to sklearn
                self.use_dvc = False
                self.vine_entropy = None
                self.vine_obj_bin = None
                self.margin_obj = None
                self.prep_cop = None
                self.tf = None
        else:
            self.vine_entropy = None
            self.vine_obj_bin = None
            self.margin_obj = None
            self.prep_cop = None
            self.tf = None

        # Initialize vine parameters
        self.copula_params = copula_params or {}

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to collect branch data (reusing EI analyzer pattern)"""
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
        """Remove forward hooks"""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data during forward pass"""
        excitatory_input, inhibitory_input, _ = input
        key = module._name

        # Ensure the key exists in data_dict (might be cleared between batches)
        if key not in self.data_dict:
            self.data_dict[key] = {}

        # Collect the same data as EIInformationAnalyzer
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        self.data_dict[key]["output"] = output

        # Store layer index for organization
        if hasattr(module, "layer_idx"):
            self.data_dict[key]["layer_idx"] = module.layer_idx

    def compute_mi_continuous(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI between continuous variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.use_dvc:
            # Use copula entropy approach
            mi = self._compute_copula_mi(x, y)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, y, self.n_neighbors)

    def compute_mi_discrete(self, x: np.ndarray, c: np.ndarray) -> float:
        """Compute MI between continuous and discrete variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        # Add uniform noise to discrete labels as suggested
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula entropy approach with noisy labels
            mi = self._compute_copula_mi(x, c_noisy)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, c_noisy, self.n_neighbors)

    def compute_cmi(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
        """Compute conditional MI: I(X;Y|C)"""
        # Add uniform noise to discrete condition variable
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula approach for conditional MI
            # I(X;Y|C) = H(X|C) + H(Y|C) - H(X,Y|C)
            # We compute this per class and average
            unique_classes = np.unique(c)
            cmi_values = []

            for class_val in unique_classes:
                mask = c == class_val
                if np.sum(mask) < 10:  # Skip if too few samples
                    continue

                x_class = x[mask]
                y_class = y[mask]

                # Compute MI for this class
                mi_class = self._compute_copula_mi(x_class, y_class)
                cmi_values.append(mi_class)

            # Average over classes
            return np.mean(cmi_values) if cmi_values else 0.0

        # Fallback approach: compute per class and average
        unique_classes = np.unique(c)
        cmi_values = []

        for class_val in unique_classes:
            mask = c == class_val
            if np.sum(mask) < self.n_neighbors + 1:
                continue

            x_class = x[mask]
            y_class = y[mask]

            # Compute MI for this class
            mi_class = compute_mi_cc(x_class, y_class, self.n_neighbors)
            cmi_values.append(mi_class)

        return np.mean(cmi_values) if cmi_values else 0.0

    def _compute_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MI using copula-based dependence estimation from DVC_tensorflow
        Uses a simplified robust approach
        """
        try:
            # Force TensorFlow to use CPU to avoid GPU compilation issues
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")

            # Ensure inputs are 2D
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Combine data for joint distribution
            data_joint = np.hstack([x, y])
            n_samples, n_dims = data_joint.shape

            # Ensure we have enough samples and only 2D
            if n_samples < 50:
                raise ValueError(f"Too few samples ({n_samples}) for copula estimation")

            if n_dims != 2:
                raise ValueError("Copula method only supports 2D MI estimation")

            # Convert to float32 and normalize to [0,1] range for copula
            # Transform to uniform margins using empirical CDF
            from scipy import stats

            # Transform each dimension to uniform [0,1] using empirical CDF
            u = np.zeros_like(data_joint)
            for i in range(n_dims):
                # Rank transform to uniform
                ranks = stats.rankdata(data_joint[:, i])
                u[:, i] = ranks / (len(ranks) + 1)  # Avoid 0 and 1

            u = u.astype(np.float32)

            # Create a basic vine copula for dependence estimation
            # Use simple parametric approach
            margin_list = []
            for i in range(n_dims):
                # Use uniform margins since we already transformed
                mar_p = self.margin_obj("unif", [0, 1], True)
                margin_list.append(mar_p)

            # Create vine with minimal configuration
            vine = self.vine_obj_bin(
                "c-vine",  # C-vine structure
                "gaussian",  # Simple parametric family
                n_dims,
                margin_list,
                10,  # Minimal knots
                "data",  # Data mode
                u,  # Uniform data
            )

            if vine is None:
                raise ValueError("Failed to create vine object")

            # Simple fitting with minimal configuration
            gen_dict = {
                "parallel": False,
                "binning": False,  # Disable binning for simplicity
                "param": True,  # Use parametric approach
                "vine_depth": 2,
                "fitted": False,
            }
            par_dict = {"param_families": ["gaussian"]}  # Only Gaussian copula
            npc_dict = {"opt_method": "LL1", "batch_paral": 1}
            bin_dict = {"n_bin": 3}

            # Fit the vine
            vine.fit(u, gen_dict, npc_dict, par_dict, bin_dict)

            # For Gaussian copula, we can extract the correlation parameter
            # and convert it to MI using the known formula: MI = -0.5 * log(1 - rho^2)

            # Try to extract correlation parameter from fitted vine
            if hasattr(vine, "copula_structure") and hasattr(
                vine.copula_structure, "parameters"
            ):
                # Extract correlation parameter
                try:
                    # For Gaussian copula, the parameter is the correlation
                    rho = vine.copula_structure.parameters[0]
                    if isinstance(rho, (list, np.ndarray)):
                        rho = rho[0]

                    # Convert correlation to MI: MI = -0.5 * log(1 - rho^2)
                    rho = float(rho)
                    rho = np.clip(rho, -0.99, 0.99)  # Avoid numerical issues
                    mi = -0.5 * np.log(1 - rho**2)
                    mi = max(0.0, mi)  # Ensure non-negative

                    self.logger.debug(
                        f"Copula MI computed from correlation: rho={rho:.4f}, MI={mi:.4f}"
                    )
                    return float(mi)
                except:
                    pass

            # If parameter extraction fails, use a simple dependence measure
            # Based on the copula likelihood or kendall's tau
            try:
                # Compute Kendall's tau and convert to MI approximation
                from scipy.stats import kendalltau

                tau, _ = kendalltau(x.flatten(), y.flatten())

                # Convert Kendall's tau to approximate MI
                # This is a rough approximation: MI ≈ -0.5 * log(1 - tau^2) for elliptical copulas
                tau = np.clip(tau, -0.99, 0.99)
                mi_approx = (
                    -0.5 * np.log(1 - tau**2) * 0.5
                )  # Scale down for robustness
                mi_approx = max(0.0, mi_approx)

                self.logger.debug(
                    f"Copula MI computed from Kendall's tau: tau={tau:.4f}, MI={mi_approx:.4f}"
                )
                return float(mi_approx)

            except Exception as e2:
                raise ValueError(f"Failed to compute dependence measure: {e2}")

        except Exception as e:
            self.logger.warning(f"Copula MI computation failed: {e}")
            self.logger.warning("Falling back to sklearn for this computation")
            # Fallback to sklearn
            from dendritic_modeling.information import compute_mi_cc

            return compute_mi_cc(x, y, self.n_neighbors)

    def analyze(
        self, model: BaseModel, test_loader: DataLoader, output_dir: str, epoch: int
    ) -> Dict[str, Any]:
        """Analyze branch information metrics using the improved hook-based approach"""
        self.logger.info("Starting branch information analysis...")

        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data collection
        self.data_dict = {}

        # Store all collected data
        all_data = {
            "E": [],  # Excitatory inputs
            "I": [],  # Inhibitory inputs
            "V": [],  # Branch outputs
            "C": [],  # Class labels
            "layer_idx": [],  # Layer indices
        }

        model.eval()
        with torch.no_grad():
            # Attach hooks to collect branch data (reusing EI analyzer pattern)
            handles = self.attach_forward_hooks(model)

            try:
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit samples for efficiency
                        break

                    inputs = inputs.to(next(model.parameters()).device)

                    # Run standard forward pass - hooks will collect data automatically
                    _ = model(inputs)

                    # Process collected data from hooks
                    for layer_name, layer_data in self.data_dict.items():
                        if "excitation" in layer_data and "inhibition" in layer_data:
                            # Get the collected tensors
                            excitation = layer_data[
                                "excitation"
                            ]  # Shape: [batch, n_branches]
                            inhibition = layer_data[
                                "inhibition"
                            ]  # Shape: [batch, n_branches]
                            output = layer_data["output"]  # Shape: [batch, n_branches]

                            layer_idx = layer_data.get("layer_idx", 0)
                            n_samples = excitation.shape[0]
                            n_branches = (
                                excitation.shape[1] if excitation.ndim > 1 else 1
                            )

                            # Flatten across branches if needed
                            if excitation.ndim > 1:
                                for branch_idx in range(n_branches):
                                    all_data["E"].extend(
                                        excitation[:, branch_idx].cpu().numpy()
                                    )
                                    all_data["I"].extend(
                                        inhibition[:, branch_idx].cpu().numpy()
                                    )
                                    all_data["V"].extend(
                                        output[:, branch_idx].cpu().numpy()
                                    )
                                    all_data["C"].extend(targets.cpu().numpy())
                                    all_data["layer_idx"].extend(
                                        [layer_idx] * n_samples
                                    )
                            else:
                                all_data["E"].extend(excitation.cpu().numpy())
                                all_data["I"].extend(inhibition.cpu().numpy())
                                all_data["V"].extend(output.cpu().numpy())
                                all_data["C"].extend(targets.cpu().numpy())
                                all_data["layer_idx"].extend([layer_idx] * n_samples)

                    # Clear collected data for next batch to save memory
                    self.data_dict.clear()

            finally:
                # Always remove hooks
                self.remove_forward_hooks(handles)

        # Convert to numpy arrays
        for key in all_data:
            all_data[key] = np.array(all_data[key])

        # Check if we have data
        if len(all_data["E"]) == 0:
            self.logger.warning("No branch data collected. Skipping analysis.")
            return {}

        # Compute information metrics
        results = {
            "layer": [],
            "I_E_C": [],  # I(E;C)
            "I_I_C": [],  # I(I;C)
            "I_V_C": [],  # I(V;C)
            "I_E_I": [],  # I(E;I)
            "I_E_V": [],  # I(E;V)
            "I_I_V": [],  # I(I;V)
            "I_E_I_given_C": [],  # I(E;I|C)
            "I_E_V_given_C": [],  # I(E;V|C)
            "I_I_V_given_C": [],  # I(I;V|C)
        }

        # Compute metrics per layer
        unique_layers = np.unique(all_data["layer_idx"])
        for layer_idx in unique_layers:
            mask = all_data["layer_idx"] == layer_idx

            E_layer = all_data["E"][mask]
            I_layer = all_data["I"][mask]
            V_layer = all_data["V"][mask]
            C_layer = all_data["C"][mask[: len(all_data["C"])]]

            # Skip if insufficient data
            if len(E_layer) < 100:
                self.logger.warning(
                    f"Insufficient data for layer {layer_idx}, skipping"
                )
                continue

            self.logger.info(f"Computing MI metrics for layer {layer_idx}")

            # Compute mutual information metrics
            results["layer"].append(int(layer_idx))

            # MI with class labels
            results["I_E_C"].append(self.compute_mi_discrete(E_layer, C_layer))
            results["I_I_C"].append(self.compute_mi_discrete(I_layer, C_layer))
            results["I_V_C"].append(self.compute_mi_discrete(V_layer, C_layer))

            # MI between branch components
            results["I_E_I"].append(self.compute_mi_continuous(E_layer, I_layer))
            results["I_E_V"].append(self.compute_mi_continuous(E_layer, V_layer))
            results["I_I_V"].append(self.compute_mi_continuous(I_layer, V_layer))

            # Conditional MI given class
            results["I_E_I_given_C"].append(self.compute_cmi(E_layer, I_layer, C_layer))
            results["I_E_V_given_C"].append(self.compute_cmi(E_layer, V_layer, C_layer))
            results["I_I_V_given_C"].append(self.compute_cmi(I_layer, V_layer, C_layer))

        # Create plots
        if self.save_plots and results["layer"]:
            self._plot_results(results, output_dir, epoch)

        # Save CSV
        if self.save_csv and results["layer"]:
            self._save_csv(results, output_dir, epoch)

        return results

    def _plot_results(self, results: Dict, output_dir: str, epoch: int):
        """Create plots for branch information metrics"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"Branch Information Analysis - Epoch {epoch}", fontsize=16)

        # Flatten axes for easier indexing
        axes = axes.flatten()

        # Define metrics to plot
        metrics = [
            ("I_E_C", "I(E;C)", "Excitatory-Class MI"),
            ("I_I_C", "I(I;C)", "Inhibitory-Class MI"),
            ("I_V_C", "I(V;C)", "Output-Class MI"),
            ("I_E_I", "I(E;I)", "Excitatory-Inhibitory MI"),
            ("I_E_V", "I(E;V)", "Excitatory-Output MI"),
            ("I_I_V", "I(I;V)", "Inhibitory-Output MI"),
            ("I_E_I_given_C", "I(E;I|C)", "Conditional E-I MI"),
            ("I_E_V_given_C", "I(E;V|C)", "Conditional E-V MI"),
            ("I_I_V_given_C", "I(I;V|C)", "Conditional I-V MI"),
        ]

        # Create plots
        for idx, (key, ylabel, title) in enumerate(metrics):
            ax = axes[idx]
            if key in results and results[key]:
                ax.plot(results["layer"], results[key], "o-", linewidth=2, markersize=8)
                ax.set_xlabel("Layer")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"branch_information_epoch_{epoch}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved branch information plot to {plot_path}")

    def _save_csv(self, results: Dict, output_dir: str, epoch: int):
        """Save results to CSV"""
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f"branch_information_epoch_{epoch}.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved branch information data to {csv_path}")


class SingleLayerContributionAnalyzer:
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
        self.batch_size = batch_size
        self.device = device
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        save_path: Optional[str] = None,
        filename: str = "single_layer_contribution",
    ):
        """Analyze each layer's contribution in isolation"""
        import matplotlib.pyplot as plt
        import pandas as pd

        from dendritic_modeling.utils import measure_performance

        self.logger.info("Starting single layer contribution analysis...")

        # Ensure model is on the correct device
        model = model.to(self.device)
        # Get the underlying network
        if hasattr(model, "net"):
            net = model.net
        else:
            net = model

        # Debug logging
        self.logger.info(f"Model type: {type(model)}")
        self.logger.info(f"Net type: {type(net)}")
        self.logger.info(
            f"Net attributes: {[attr for attr in dir(net) if 'branch' in attr]}"
        )

        # Check layers structure
        if hasattr(net, "layers"):
            self.logger.info(f"Number of EI layers: {len(net.layers)}")
            for i, layer in enumerate(net.layers):
                self.logger.info(
                    f"Layer {i}: has excitatory_cells={hasattr(layer, 'excitatory_cells')}, has inhibitory_cells={hasattr(layer, 'inhibitory_cells')}"
                )
                if (
                    hasattr(layer, "excitatory_cells")
                    and layer.excitatory_cells is not None
                ):
                    self.logger.info(
                        f"  Excitatory cells type: {type(layer.excitatory_cells)}"
                    )
                    self.logger.info(
                        f"  Excitatory has branch_layers: {hasattr(layer.excitatory_cells, 'branch_layers')}"
                    )
                if (
                    hasattr(layer, "inhibitory_cells")
                    and layer.inhibitory_cells is not None
                ):
                    self.logger.info(
                        f"  Inhibitory cells type: {type(layer.inhibitory_cells)}"
                    )
                    self.logger.info(
                        f"  Inhibitory has branch_layers: {hasattr(layer.inhibitory_cells, 'branch_layers')}"
                    )

        # Get branch layers
        try:
            branch_layers = net.branch_layers
            self.logger.info(
                f"Successfully got branch_layers: {len(branch_layers)} layers"
            )
        except Exception as e:
            self.logger.error(f"Error accessing branch_layers: {type(e).__name__}: {e}")
            return

        # Get baseline performance with all layers active
        baseline_acc = measure_performance(net, test_dataset, device=self.device)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Log layer structure for debugging
        self.logger.info(f"Total branch layers: {len(branch_layers)}")
        for i, layer in enumerate(branch_layers):
            has_exc = (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            )
            has_inh = (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            )
            self.logger.info(
                f"Layer {i}: has_excitation={has_exc}, has_inhibition={has_inh}"
            )

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results
        results = {
            "layer_idx": [],
            "isolated_baseline_acc": [],
            "isolated_exc_only_acc": [],
            "isolated_inh_only_acc": [],
            "isolated_neither_acc": [],
        }

        # Analyze each layer in isolation
        for layer_idx in range(len(branch_layers)):
            self.logger.info(f"Analyzing layer {layer_idx} in isolation...")

            # First, ablate all OTHER layers (set to very negative values)
            for i in range(len(branch_layers)):
                if i != layer_idx:
                    if (
                        hasattr(branch_layers[i], "branch_excitation")
                        and branch_layers[i].branch_excitation is not None
                    ):
                        branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                    if (
                        hasattr(branch_layers[i], "branch_inhibition")
                        and branch_layers[i].branch_inhibition is not None
                    ):
                        branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)

            # Measure with only this layer active (both E and I)
            isolated_baseline = measure_performance(
                net, test_dataset, device=self.device
            )

            # Test with only excitatory active
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
                exc_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore inhibitory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_inhibition" in n:
                        p.data.copy_(original_params[n])
            else:
                exc_only_acc = isolated_baseline

            # Test with only inhibitory active
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
                inh_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore excitatory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_excitation" in n:
                        p.data.copy_(original_params[n])
            else:
                inh_only_acc = isolated_baseline

            # Test with neither (both ablated)
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
            neither_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore all parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["isolated_baseline_acc"].append(isolated_baseline)
            results["isolated_exc_only_acc"].append(exc_only_acc)
            results["isolated_inh_only_acc"].append(inh_only_acc)
            results["isolated_neither_acc"].append(neither_acc)

            self.logger.info(
                f"Layer {layer_idx}: baseline={isolated_baseline:.4f}, "
                f"exc_only={exc_only_acc:.4f}, inh_only={inh_only_acc:.4f}, "
                f"neither={neither_acc:.4f}"
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            if self.save_csv:
                df = pd.DataFrame(results)
                df["full_network_baseline"] = baseline_acc
                csv_path = os.path.join(save_path, f"{filename}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved results to {csv_path}")

            if self.save_plots:
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Absolute accuracy values
                x = np.array(reversed_layer_idx)

                # Reverse the data to match the x positions
                baseline_acc_reversed = [
                    results["isolated_baseline_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                exc_only_acc_reversed = [
                    results["isolated_exc_only_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                inh_only_acc_reversed = [
                    results["isolated_inh_only_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                neither_acc_reversed = [
                    results["isolated_neither_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]

                ax1.plot(
                    x,
                    baseline_acc_reversed,
                    "o-",
                    label="Both E+I",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax1.plot(
                    x,
                    exc_only_acc_reversed,
                    "s-",
                    label="Excitatory only",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax1.plot(
                    x,
                    inh_only_acc_reversed,
                    "^-",
                    label="Inhibitory only",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )
                ax1.plot(
                    x,
                    neither_acc_reversed,
                    "v-",
                    label="Neither",
                    linewidth=3,
                    markersize=10,
                    color="gray",
                )
                ax1.axhline(
                    y=baseline_acc,
                    color="k",
                    linestyle="--",
                    linewidth=2,
                    label=f"Full network baseline ({baseline_acc:.3f})",
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Classification Accuracy", fontsize=14)
                ax1.set_title(
                    "Single Layer Contribution Analysis\n(Each layer tested in isolation)",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(
                    [0, max(max(results["isolated_baseline_acc"]), baseline_acc) * 1.1]
                )

                # Right plot: Contribution breakdown (bar chart)
                width = 0.25

                # Calculate contributions
                exc_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_exc_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                inh_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_inh_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                synergy = [
                    full - max(e, i)
                    for full, e, i in zip(
                        results["isolated_baseline_acc"],
                        results["isolated_exc_only_acc"],
                        results["isolated_inh_only_acc"],
                    )
                ]

                # Debug: Print contributions before reversal
                self.logger.info(
                    f"Before reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"Before reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"Before reversal - Synergy: {synergy}")

                # Reverse the contributions too
                exc_contrib = [exc_contrib[i] for i in range(num_layers - 1, -1, -1)]
                inh_contrib = [inh_contrib[i] for i in range(num_layers - 1, -1, -1)]
                synergy = [synergy[i] for i in range(num_layers - 1, -1, -1)]

                # Debug: Print contributions after reversal
                self.logger.info(
                    f"After reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"After reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"After reversal - Synergy: {synergy}")
                self.logger.info(f"Plotting x positions: {x.tolist()}")

                ax2.bar(
                    x - width,
                    exc_contrib,
                    width,
                    label="Excitatory contribution",
                    color="red",
                    alpha=0.7,
                )
                ax2.bar(
                    x,
                    inh_contrib,
                    width,
                    label="Inhibitory contribution",
                    color="blue",
                    alpha=0.7,
                )
                ax2.bar(
                    x + width,
                    synergy,
                    width,
                    label="E+I synergy",
                    color="purple",
                    alpha=0.7,
                )

                # Set x-axis
                ax2.set_xticks([i * 3 for i in range(len(reversed_layer_idx))])
                # Create position-based labels
                x_labels = [
                    "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
                ]
                ax2.set_xticklabels(x_labels)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)

                ax2.set_ylabel("Accuracy Contribution", fontsize=14)
                ax2.set_title(
                    "Layer Contributions Breakdown\n(When tested in isolation)",
                    fontsize=16,
                )
                ax2.legend(fontsize=12)
                ax2.grid(True, axis="y", alpha=0.3)
                ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                self.logger.info(f"Saved plot to {plot_path}")

        return results


class AblationAnalyzer:
    def __init__(
        self,
        ablate_all_synapses: bool = True,
        ablate_excitation: bool = True,
        ablate_inhibition: bool = True,
        ablate_branches: bool = True,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True,
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
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
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
            results["all synapses"] = {}
        if self.ablate_excitation:
            results["excitation"] = {}
        if self.ablate_inhibition:
            results["inhibition"] = {}
        if self.ablate_branches:
            results["branches"] = {}

        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    self.ablate_all_synapses
                    and module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    results["all synapses"][name] = {}
                    results["all synapses"][name]["layer_idx"] = module.layer_idx

                    module.branch_excitation.pre_w.data.fill_(-100)
                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["all synapses"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["all synapses"][name]["log likelihood"] = loglikelihood

                    model.load_state_dict(state_dict)

                if self.ablate_excitation and module.branch_excitation is not None:
                    results["excitation"][name] = {}
                    results["excitation"][name]["layer_idx"] = module.layer_idx

                    module.branch_excitation.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["excitation"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["excitation"][name]["log likelihood"] = loglikelihood

                    model.load_state_dict(state_dict)

                if self.ablate_inhibition and module.branch_inhibition is not None:
                    results["inhibition"][name] = {}
                    results["inhibition"][name]["layer_idx"] = module.layer_idx

                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["inhibition"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["inhibition"][name]["log likelihood"] = loglikelihood

                    model.load_state_dict(state_dict)

                if self.ablate_branches and hasattr(module, "branches_to_output"):
                    results["branches"][name] = {}
                    results["branches"][name]["layer_idx"] = module.layer_idx

                    module.branches_to_output.log_weight.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["branches"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["branches"][name]["log likelihood"] = loglikelihood

                    model.load_state_dict(state_dict)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


class WeightAnalyzer:
    def __init__(self, synapse_weights: bool = True, branch_weights: bool = True):
        self.synapse_weights = synapse_weights
        self.branch_weights = branch_weights

    def analyze(
        self, model: BaseModel, save_path: Optional[str] = None, filename: str = "final"
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
                if self.branch_weights and hasattr(module, "branches_to_output"):
                    branch_w = module.branches_to_output.log_weight.exp().flatten()
                    branch_w = branch_w.cpu()
                    branch_weights.append(branch_w)

        weights = {}
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            weights["exc weights"] = {}
            weights["exc weights"]["mean"] = exc_weights.mean().item()
            weights["exc weights"]["std"] = exc_weights.std().item()
            weights["exc weights"]["min"] = exc_weights.min().item()
            weights["exc weights"]["max"] = exc_weights.max().item()
            weights["exc weights"]["median"] = exc_weights.median().item()
            weights["exc weights"]["q1"] = exc_weights.quantile(q=0.25).item()
            weights["exc weights"]["q3"] = exc_weights.quantile(q=0.75).item()

        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            weights["inh weights"] = {}
            weights["inh weights"]["mean"] = inh_weights.mean().item()
            weights["inh weights"]["std"] = inh_weights.std().item()
            weights["inh weights"]["min"] = inh_weights.min().item()
            weights["inh weights"]["max"] = inh_weights.max().item()
            weights["inh weights"]["median"] = inh_weights.median().item()
            weights["inh weights"]["q1"] = inh_weights.quantile(q=0.25).item()
            weights["inh weights"]["q3"] = inh_weights.quantile(q=0.75).item()

        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            weights["branch weights"] = {}
            weights["branch weights"]["mean"] = branch_weights.mean().item()
            weights["branch weights"]["std"] = branch_weights.std().item()
            weights["branch weights"]["min"] = branch_weights.min().item()
            weights["branch weights"]["max"] = branch_weights.max().item()
            weights["branch weights"]["median"] = branch_weights.median().item()
            weights["branch weights"]["q1"] = branch_weights.quantile(q=0.25).item()
            weights["branch weights"]["q3"] = branch_weights.quantile(q=0.75).item()

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
        samples: int = 1000,
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
        output: torch.Tensor,
    ):
        excitatory_input, inhibitory_input, _ = input
        excitatory_input = excitatory_input[..., None]
        inhibitory_input = inhibitory_input[..., None]

        key = module._name

        if module.branch_excitation is not None:
            pruned_exc_syn = module.branch_excitation.pruned_weight().t()
            for _ in range(excitatory_input.ndim - 2):
                pruned_exc_syn = pruned_exc_syn[None, ...]
            self.data_dict[key]["exc"] = excitatory_input * pruned_exc_syn

        if module.branch_inhibition is not None:
            pruned_inh_syn = module.branch_inhibition.pruned_weight().t()
            for _ in range(inhibitory_input.ndim - 2):
                pruned_inh_syn = pruned_inh_syn[None, ...]
            self.data_dict[key]["inh"] = inhibitory_input * pruned_inh_syn

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
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
            ixs = torch.randperm(inputs.shape[0])[: self.samples]
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

            if "exc" in module_dict.keys():
                exc: torch.Tensor = module_dict["exc"]
                if self.logspace:
                    exc = torch.log(exc + 1e-9)

                if self.network_summary or self.layer_summary:
                    exc_flat = exc.flatten()
                    if self.logspace:
                        exc_flat = exc_flat[exc_flat > -20]
                    else:
                        exc_flat = exc_flat[exc_flat > 0]

                if self.network_summary:
                    exc_net.append(exc_flat)

                if self.layer_summary:
                    layer_summary[name]["exc"] = {}
                    layer_summary[name]["exc"]["mean"] = exc_flat.mean().item()
                    layer_summary[name]["exc"]["std"] = exc_flat.std().item()
                    # layer_summary[name]['exc']['min'] = exc_flat.min().item()
                    # layer_summary[name]['exc']['max'] = exc_flat.max().item()
                    # layer_summary[name]['exc']['median'] = exc_flat.median().item()
                    # layer_summary[name]['exc']['q1'] = exc_flat.quantile(q=0.25).item()
                    # layer_summary[name]['exc']['q3'] = exc_flat.quantile(q=0.75).item()

                if self.branch_summary:
                    branch_summary[name]["exc"] = {}
                    branch_summary[name]["exc"]["mean"] = exc.mean(dim=0).tolist()
                    branch_summary[name]["exc"]["std"] = exc.std(dim=0).tolist()
                    # branch_summary[name]['exc']['min'] = exc.min(dim=0).tolist()
                    # branch_summary[name]['exc']['max'] = exc.max(dim=0).tolist()
                    # branch_summary[name]['exc']['median'] = exc.median(dim=0).tolist()
                    # branch_summary[name]['exc']['q1'] = exc.quantile(q=0.25, dim=0).tolist()
                    # branch_summary[name]['exc']['q3'] = exc.quantile(q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["exc"] = exc.tolist()

            if "inh" in module_dict.keys():
                inh: torch.Tensor = module_dict["inh"]
                if self.logspace:
                    inh = torch.log(inh + 1e-9)

                if self.network_summary or self.layer_summary:
                    inh_flat = inh.flatten()
                    if self.logspace:
                        inh_flat = inh_flat[inh_flat > -20]
                    else:
                        inh_flat = inh_flat[inh_flat > 0]

                if self.network_summary:
                    inh_net.append(inh_flat)

                if self.layer_summary:
                    layer_summary[name]["inh"] = {}
                    layer_summary[name]["inh"]["mean"] = inh_flat.mean().item()
                    layer_summary[name]["inh"]["std"] = inh_flat.std().item()
                    # layer_summary[name]['inh']['min'] = inh_flat.min().item()
                    # layer_summary[name]['inh']['max'] = inh_flat.max().item()
                    # layer_summary[name]['inh']['median'] = inh_flat.median().item()
                    # layer_summary[name]['inh']['q1'] = inh_flat.quantile(q=0.25).item()
                    # layer_summary[name]['inh']['q3'] = inh_flat.quantile(q=0.75).item()

                if self.branch_summary:
                    branch_summary[name]["inh"] = {}
                    branch_summary[name]["inh"]["mean"] = inh.mean(dim=0).tolist()
                    branch_summary[name]["inh"]["std"] = inh.std(dim=0).tolist()
                    # branch_summary[name]['inh']['min'] = inh.min(dim=0).tolist()
                    # branch_summary[name]['inh']['max'] = inh.max(dim=0).tolist()
                    # branch_summary[name]['inh']['median'] = inh.median(dim=0).tolist()
                    # branch_summary[name]['inh']['q1'] = inh.quantile(q=0.25, dim=0).tolist()
                    # branch_summary[name]['inh']['q3'] = inh.quantile(q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["inh"] = inh.tolist()

        results = {"logspace": self.logspace}

        if self.network_summary:
            network_summary = {}
            network_summary["exc"] = {}
            network_summary["inh"] = {}
            network_summary["exc"]["mean"] = torch.cat(exc_net).mean().item()
            network_summary["inh"]["mean"] = torch.cat(inh_net).mean().item()
            network_summary["exc"]["std"] = torch.cat(exc_net).std().item()
            network_summary["inh"]["std"] = torch.cat(inh_net).std().item()
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

            results["network_summary"] = network_summary

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

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
        samples: int = 1000,
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
        output: torch.Tensor,
    ):
        excitatory_input, inhibitory_input, _ = input

        key = module._name

        if self.synapse_activations and module.branch_excitation is not None:
            self.data_dict[key]["exc"] = module.branch_excitation(excitatory_input)

        if self.synapse_activations and module.branch_inhibition is not None:
            self.data_dict[key]["inh"] = module.branch_inhibition(inhibitory_input)

        if self.branch_activations:
            self.data_dict[key]["Vout"] = output

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
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
            ixs = torch.randperm(inputs.shape[0])[: self.samples]
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

            if "exc" in module_dict.keys():
                print("exc", flush=True)
                exc: torch.Tensor = module_dict["exc"]

                if self.logspace:
                    exc = torch.log(exc)

                if self.network_summary:
                    print("network summary", flush=True)
                    exc_net.append(exc.flatten())

                if self.layer_summary:
                    print("layer summary", flush=True)
                    layer_summary[name]["exc"] = {}
                    layer_summary[name]["exc"]["mean"] = torch.mean(exc).item()
                    layer_summary[name]["exc"]["std"] = torch.std(exc).item()
                    # layer_summary[name]['exc']['min'] = exc.min().item()
                    # layer_summary[name]['exc']['max'] = exc.max().item()
                    # layer_summary[name]['exc']['median'] = exc.median().item()
                    # layer_summary[name]['exc']['q1'] = exc.quantile(q=0.25).item()
                    # layer_summary[name]['exc']['q3'] = exc.quantile(q=0.75).item()

                if self.branch_summary:
                    print("branch summary", flush=True)
                    branch_summary[name]["exc"] = {}
                    branch_summary[name]["exc"]["mean"] = torch.mean(
                        exc, dim=0
                    ).tolist()
                    branch_summary[name]["exc"]["std"] = torch.std(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['min'] = torch.min(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['max'] = torch.max(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['median'] = torch.median(exc, dim=0).tolist()
                    # branch_summary[name]['exc']['q1'] = torch.quantile(exc, q=0.25, dim=0).tolist()
                    # branch_summary[name]['exc']['q3'] = torch.quantile(exc, q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["exc"] = exc.tolist()
            if "inh" in module_dict.keys():
                print("inh", flush=True)
                inh: torch.Tensor = module_dict["inh"]

                if self.logspace:
                    inh = torch.log(inh)

                if self.network_summary:
                    print("network summary", flush=True)
                    inh_net.append(inh.flatten())

                if self.layer_summary:
                    print("layer summary", flush=True)
                    layer_summary[name]["inh"] = {}
                    layer_summary[name]["inh"]["mean"] = torch.mean(inh).item()
                    layer_summary[name]["inh"]["std"] = torch.std(inh).item()
                    # layer_summary[name]['inh']['min'] = inh.min().item()
                    # layer_summary[name]['inh']['max'] = inh.max().item()
                    # layer_summary[name]['inh']['median'] = inh.median().item()
                    # layer_summary[name]['inh']['q1'] = inh.quantile(q=0.25).item()
                    # layer_summary[name]['inh']['q3'] = inh.quantile(q=0.75).item()

                if self.branch_summary:
                    print("branch summary", flush=True)
                    branch_summary[name]["inh"] = {}
                    branch_summary[name]["inh"]["mean"] = torch.mean(
                        inh, dim=0
                    ).tolist()
                    branch_summary[name]["inh"]["std"] = torch.std(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['min'] = torch.min(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['max'] = torch.max(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['median'] = torch.median(inh, dim=0).tolist()
                    # branch_summary[name]['inh']['q1'] = torch.quantile(inh, q=0.25, dim=0).tolist()
                    # branch_summary[name]['inh']['q3'] = torch.quantile(inh, q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["inh"] = inh.tolist()

            if "Vout" in module_dict.keys():
                print("Vout", flush=True)
                Vout: torch.Tensor = module_dict["Vout"]

                if self.logspace:
                    Vout = torch.log(Vout)

                if self.network_summary:
                    print("network summary", flush=True)
                    Vout_net.append(Vout.flatten())

                if self.layer_summary:
                    print("layer summary", flush=True)
                    layer_summary[name]["Vout"] = {}
                    layer_summary[name]["Vout"]["mean"] = torch.mean(Vout).item()
                    layer_summary[name]["Vout"]["std"] = torch.std(Vout).item()
                    # layer_summary[name]['Vout']['min'] = Vout.min().item()
                    # layer_summary[name]['Vout']['max'] = Vout.max().item()
                    # layer_summary[name]['Vout']['median'] = Vout.median().item()
                    # layer_summary[name]['Vout']['q1'] = Vout.quantile(q=0.25).item()
                    # layer_summary[name]['Vout']['q3'] = Vout.quantile(q=0.75).item()

                if self.branch_summary:
                    print("branch summary", flush=True)
                    branch_summary[name]["Vout"] = {}
                    branch_summary[name]["Vout"]["mean"] = torch.mean(
                        Vout, dim=0
                    ).tolist()
                    branch_summary[name]["Vout"]["std"] = torch.std(
                        Vout, dim=0
                    ).tolist()
                    # branch_summary[name]['Vout']['min'] = torch.min(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['max'] = torch.max(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['median'] = torch.median(Vout, dim=0).tolist()
                    # branch_summary[name]['Vout']['q1'] = torch.quantile(Vout, q=0.25, dim=0).tolist()
                    # branch_summary[name]['Vout']['q3'] = torch.quantile(Vout, q=0.75, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["Vout"] = Vout.tolist()

        results = {"logspace": self.logspace}

        if self.network_summary:
            network_summary = {}

            if self.synapse_activations:
                exc_net = torch.cat(exc_net, dim=0)
                network_summary["exc"] = {}
                network_summary["exc"]["mean"] = torch.mean(exc_net).item()
                network_summary["exc"]["std"] = torch.std(exc_net).item()
                # network_summary['exc']['min'] = torch.min(exc_net).item()
                # network_summary['exc']['max'] = torch.max(exc_net).item()
                # network_summary['exc']['median'] = torch.median(exc_net).item()
                # network_summary['exc']['q1'] = torch.quantile(exc_net, q=0.25).item()
                # network_summary['exc']['q3'] = torch.quantile(exc_net, q=0.75).item()

                inh_net = torch.cat(inh_net, dim=0)
                network_summary["inh"] = {}
                network_summary["inh"]["mean"] = torch.mean(inh_net).item()
                network_summary["inh"]["std"] = torch.std(inh_net).item()
                # network_summary['inh']['min'] = torch.min(inh_net).item()
                # network_summary['inh']['max'] = torch.max(inh_net).item()
                # network_summary['inh']['median'] = torch.median(inh_net).item()
                # network_summary['inh']['q1'] = torch.quantile(inh_net, q=0.25).item()
                # network_summary['inh']['q3'] = torch.quantile(inh_net, q=0.75).item()

            if self.branch_activations:
                Vout_net = torch.cat(Vout_net, dim=0)
                network_summary["Vout"] = {}
                network_summary["Vout"]["mean"] = torch.mean(Vout_net).item()
                network_summary["Vout"]["std"] = torch.std(Vout_net).item()
                # network_summary['Vout']['min'] = torch.min(Vout_net).item()
                # network_summary['Vout']['max'] = torch.max(Vout_net).item()
                # network_summary['Vout']['median'] = torch.median(Vout_net).item()
                # network_summary['Vout']['q1'] = torch.quantile(Vout_net, q=0.25).item()
                # network_summary['Vout']['q3'] = torch.quantile(Vout_net, q=0.75).item()

            results["network_summary"] = network_summary

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results


class SynapticAblationAnalyzer:
    """
    Performs synaptic ablation analysis - removes specific synapses while keeping
    the rest of the network intact. This is the renamed version of "branch_ablation".
    """

    def __init__(
        self,
        ablation_type: str = "whole_layer",  # "whole_layer" or "single_branch"
        layers_to_analyze: List[int] = [],  # Empty means all layers
        mode: str = "direct",  # "direct" or "full"
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
        self.ablation_type = ablation_type
        self.layers_to_analyze = layers_to_analyze
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        save_path: Optional[str] = None,
        filename: str = "synaptic_ablation",
    ):
        """Perform synaptic ablation analysis"""
        import matplotlib.pyplot as plt
        import pandas as pd

        from dendritic_modeling.utils import ablate_branch, measure_performance

        self.logger.info(
            f"Starting synaptic ablation analysis (type: {self.ablation_type})..."
        )

        # Ensure model is on the correct device
        model = model.to(self.device)
        # Get the underlying network
        if hasattr(model, "net"):
            net = model.net
        else:
            net = model

        # Debug logging
        self.logger.info(f"Model type: {type(model)}")
        self.logger.info(f"Net type: {type(net)}")
        self.logger.info(
            f"Net attributes: {[attr for attr in dir(net) if 'branch' in attr]}"
        )

        # Check layers structure
        if hasattr(net, "layers"):
            self.logger.info(f"Number of EI layers: {len(net.layers)}")
            for i, layer in enumerate(net.layers):
                self.logger.info(
                    f"Layer {i}: has excitatory_cells={hasattr(layer, 'excitatory_cells')}, has inhibitory_cells={hasattr(layer, 'inhibitory_cells')}"
                )
                if (
                    hasattr(layer, "excitatory_cells")
                    and layer.excitatory_cells is not None
                ):
                    self.logger.info(
                        f"  Excitatory cells type: {type(layer.excitatory_cells)}"
                    )
                    self.logger.info(
                        f"  Excitatory has branch_layers: {hasattr(layer.excitatory_cells, 'branch_layers')}"
                    )
                if (
                    hasattr(layer, "inhibitory_cells")
                    and layer.inhibitory_cells is not None
                ):
                    self.logger.info(
                        f"  Inhibitory cells type: {type(layer.inhibitory_cells)}"
                    )
                    self.logger.info(
                        f"  Inhibitory has branch_layers: {hasattr(layer.inhibitory_cells, 'branch_layers')}"
                    )

        # Get branch layers
        try:
            branch_layers = net.branch_layers
            self.logger.info(
                f"Successfully got branch_layers: {len(branch_layers)} layers"
            )
        except Exception as e:
            self.logger.error(f"Error accessing branch_layers: {type(e).__name__}: {e}")
            return

        # Get baseline performance
        baseline_acc = measure_performance(net, test_dataset, device=self.device)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Log layer structure for debugging
        self.logger.info(f"Total branch layers: {len(branch_layers)}")
        for i, layer in enumerate(branch_layers):
            has_exc = (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            )
            has_inh = (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            )
            self.logger.info(
                f"Layer {i}: has_excitation={has_exc}, has_inhibition={has_inh}"
            )

        # Determine which layers to analyze
        if self.layers_to_analyze:
            layer_indices = self.layers_to_analyze
        else:
            layer_indices = list(range(len(branch_layers)))

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results based on ablation type
        if self.ablation_type == "whole_layer":
            results = self._analyze_whole_layers(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )
        else:  # single_branch
            results = self._analyze_single_branches(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self._save_results(results, baseline_acc, save_path, filename)

        return results

    def _analyze_whole_layers(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of whole layers"""
        from dendritic_modeling.utils import measure_performance

        results = {
            "layer_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]
            self.logger.info(f"Ablating layer {layer_idx}...")

            # Ablate both E and I
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
            both_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only excitatory
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
                exc_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                exc_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only inhibitory
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
                inh_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                inh_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["both_ablated_acc"].append(both_acc)
            results["exc_ablated_acc"].append(exc_acc)
            results["inh_ablated_acc"].append(inh_acc)
            results["both_drop"].append(baseline_acc - both_acc)
            results["exc_drop"].append(baseline_acc - exc_acc)
            results["inh_drop"].append(baseline_acc - inh_acc)

            self.logger.info(
                f"Layer {layer_idx}: both_drop={baseline_acc - both_acc:.4f}, "
                f"exc_drop={baseline_acc - exc_acc:.4f}, inh_drop={baseline_acc - inh_acc:.4f}"
            )

        return results

    def _analyze_single_branches(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of individual branches"""
        from dendritic_modeling.utils import ablate_branch, measure_performance

        results = {
            "branch_id": [],
            "layer_idx": [],
            "branch_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]

            # Determine number of branches
            n_branches = 0
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
                and hasattr(layer.branch_excitation, "pre_w")
            ):
                n_branches = layer.branch_excitation.pre_w.shape[0]
            elif (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
                and hasattr(layer.branch_inhibition, "pre_w")
            ):
                n_branches = layer.branch_inhibition.pre_w.shape[0]

            if n_branches == 0:
                continue

            self.logger.info(f"Analyzing {n_branches} branches in layer {layer_idx}...")

            for branch_idx in range(n_branches):
                # Ablate both E and I for this branch
                ablate_branch(net, layer_idx, [branch_idx], mode=self.mode)
                both_acc = measure_performance(net, test_dataset, device=self.device)

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only excitatory
                if (
                    hasattr(layer, "branch_excitation")
                    and layer.branch_excitation is not None
                ):
                    layer.branch_excitation.pre_w.data[branch_idx].zero_()
                    exc_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    exc_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only inhibitory
                if (
                    hasattr(layer, "branch_inhibition")
                    and layer.branch_inhibition is not None
                ):
                    layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                    inh_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    inh_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Store results
                branch_id = f"{layer_idx}-{branch_idx}"
                results["branch_id"].append(branch_id)
                results["layer_idx"].append(layer_idx)
                results["branch_idx"].append(branch_idx)
                results["both_ablated_acc"].append(both_acc)
                results["exc_ablated_acc"].append(exc_acc)
                results["inh_ablated_acc"].append(inh_acc)
                results["both_drop"].append(baseline_acc - both_acc)
                results["exc_drop"].append(baseline_acc - exc_acc)
                results["inh_drop"].append(baseline_acc - inh_acc)

        return results

    def _save_results(self, results, baseline_acc, save_path, filename):
        """Save ablation results"""
        import matplotlib.pyplot as plt
        import pandas as pd

        if self.save_csv:
            df = pd.DataFrame(results)
            df["baseline_acc"] = baseline_acc
            csv_path = os.path.join(save_path, f"{filename}_{self.ablation_type}.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved results to {csv_path}")

        if self.save_plots:
            if self.ablation_type == "whole_layer":
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Reverse the results too
                both_drop_reversed = [
                    results["both_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                exc_drop_reversed = [
                    results["exc_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                inh_drop_reversed = [
                    results["inh_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Accuracy drop (bar chart)
                x = np.array(reversed_layer_idx)
                width = 0.25

                ax1.bar(
                    x - width,
                    both_drop_reversed,
                    width,
                    label="Both E+I ablated",
                    color="purple",
                    alpha=0.7,
                )
                ax1.bar(
                    x,
                    exc_drop_reversed,
                    width,
                    label="Excitatory ablated",
                    color="red",
                    alpha=0.7,
                )
                ax1.bar(
                    x + width,
                    inh_drop_reversed,
                    width,
                    label="Inhibitory ablated",
                    color="blue",
                    alpha=0.7,
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Accuracy Drop", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation Analysis - Whole Layers\n(Baseline accuracy: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)
                ax1.set_ylim([0, max(max(both_drop_reversed), 0.1) * 1.2])

                # Right plot: Remaining accuracy after ablation
                both_acc_reversed = [
                    results["both_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_acc_reversed = [
                    results["exc_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_acc_reversed = [
                    results["inh_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax2.plot(
                    x,
                    [baseline_acc] * len(x),
                    "k--",
                    linewidth=2,
                    label=f"Baseline ({baseline_acc:.3f})",
                )
                ax2.plot(
                    x,
                    both_acc_reversed,
                    "o-",
                    label="Both E+I ablated",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax2.plot(
                    x,
                    exc_acc_reversed,
                    "s-",
                    label="Excitatory ablated",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax2.plot(
                    x,
                    inh_acc_reversed,
                    "^-",
                    label="Inhibitory ablated",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )

                # Set x-axis ticks and labels
                ax2.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax2.set_xticklabels(x_labels)

                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Classification Accuracy", fontsize=14)
                ax2.set_title("Remaining Accuracy After Layer Ablation", fontsize=16)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, baseline_acc * 1.1])

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_whole_layer.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

            else:  # single_branch
                # Calculate layer-wise statistics
                df = pd.DataFrame(results)
                layer_stats = (
                    df.groupby("layer_idx")
                    .agg(
                        {
                            "both_drop": ["mean", "std"],
                            "exc_drop": ["mean", "std"],
                            "inh_drop": ["mean", "std"],
                        }
                    )
                    .reset_index()
                )

                # Reverse layer indices
                num_layers = len(layer_stats)
                reversed_layer_idx = [num_layers - 1 - idx for idx in range(num_layers)]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Mean accuracy drop per layer
                x = np.array(reversed_layer_idx)
                width = 0.25

                # Reverse the statistics
                both_mean = [
                    layer_stats["both_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                both_std = [
                    layer_stats["both_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_mean = [
                    layer_stats["exc_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_std = [
                    layer_stats["exc_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_mean = [
                    layer_stats["inh_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_std = [
                    layer_stats["inh_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax1.bar(
                    x - width,
                    both_mean,
                    width,
                    yerr=both_std,
                    label="Both E+I",
                    color="purple",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x,
                    exc_mean,
                    width,
                    yerr=exc_std,
                    label="Excitatory",
                    color="red",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x + width,
                    inh_mean,
                    width,
                    yerr=inh_std,
                    label="Inhibitory",
                    color="blue",
                    alpha=0.7,
                    capsize=5,
                )

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Mean Accuracy Drop (± std)", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation - Single Branches\n(Baseline: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)

                # Right plot: Distribution of drops across branches
                # Create box plots for each layer
                positions = []
                data_to_plot = []
                labels = []
                colors = []

                for i, layer_idx in enumerate(reversed_layer_idx):
                    original_idx = num_layers - 1 - layer_idx
                    layer_data = df[df["layer_idx"] == original_idx]

                    positions.extend([i * 3 - 0.6, i * 3, i * 3 + 0.6])
                    data_to_plot.extend(
                        [
                            layer_data["both_drop"].values,
                            layer_data["exc_drop"].values,
                            layer_data["inh_drop"].values,
                        ]
                    )
                    if i == 0:  # Only add labels once
                        labels.extend(["Both E+I", "Excitatory", "Inhibitory"])
                    colors.extend(["purple", "red", "blue"])

                bp = ax2.boxplot(
                    data_to_plot, positions=positions, widths=0.5, patch_artist=True
                )

                # Color the boxes
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Set x-axis
                ax2.set_xticks([i * 3 for i in range(len(reversed_layer_idx))])
                # Create position-based labels
                x_labels = [
                    "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
                ]
                ax2.set_xticklabels(x_labels)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Accuracy Drop Distribution", fontsize=14)
                ax2.set_title("Branch-wise Accuracy Drop Distribution", fontsize=16)
                ax2.grid(True, axis="y", alpha=0.3)

                # Add a custom legend
                legend_elements = [
                    Patch(facecolor="purple", alpha=0.7, label="Both E+I"),
                    Patch(facecolor="red", alpha=0.7, label="Excitatory"),
                    Patch(facecolor="blue", alpha=0.7, label="Inhibitory"),
                ]
                ax2.legend(handles=legend_elements, fontsize=12)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_single_branch.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Save summary statistics with reversed indices
                if self.save_csv:
                    summary_df = pd.DataFrame(
                        {
                            "layer_idx_reversed": reversed_layer_idx,
                            "layer_idx_original": [
                                num_layers - 1 - idx for idx in reversed_layer_idx
                            ],
                            "both_drop_mean": both_mean,
                            "both_drop_std": both_std,
                            "exc_drop_mean": exc_mean,
                            "exc_drop_std": exc_std,
                            "inh_drop_mean": inh_mean,
                            "inh_drop_std": inh_std,
                        }
                    )
                    summary_path = os.path.join(
                        save_path, f"{filename}_single_branch_summary.csv"
                    )
                    summary_df.to_csv(summary_path, index=False)

            self.logger.info(f"Saved plots to {save_path}")


class AnalysisManager:
    def __init__(
        self,
        model: BaseModel,
        data: Dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = "cpu",
        train_config: Optional[Dict] = None,
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.train_config = train_config or {}
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        # Note: Logging should be configured externally at the application startup.
        self.logger = logging.getLogger(__name__)

        performance_config = self.config.performance_analysis
        self.performance_enabled = performance_config.get("enabled", False)
        self.performance_training = performance_config.get("training", False)
        if self.performance_enabled:
            performance_params: dict = performance_config.get("params", {})
            self.performance_analyzer = PerformanceAnalyzer(
                accuracy=performance_params.get("accuracy", False),
                categorical_loglikelihood=performance_params.get(
                    "categorical_loglikelihood", False
                ),
                mse=performance_params.get("mse", False),
            )

        ei_info_config = self.config.ei_info_analysis
        self.ei_info_enabled = ei_info_config.get("enabled", False)
        self.ei_info_training = ei_info_config.get("training", False)
        if self.ei_info_enabled:
            ei_info_params: dict = ei_info_config.get("params", {})
            self.ei_info_analyzer = EIInformationAnalyzer(
                compute_mi_E_I=ei_info_params.get("compute_mi_E_I", True),
                compute_mi_EI_Vout=ei_info_params.get("compute_mi_EI_Vout", True),
                compute_sr_EI_Vout=ei_info_params.get("compute_sr_EI_Vout", True),
                compute_mi_EI_label=ei_info_params.get("compute_mi_EI_label", True),
                n_neighbors=ei_info_params.get("n_neighbors", 10),
            )

        syn_turnover_config = self.config.synapse_turnover_analysis
        self.syn_turnover_enabled = syn_turnover_config.get("enabled", False)
        self.syn_turnover_training = syn_turnover_config.get("training", False)
        if self.syn_turnover_enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)

        noise_config = self.config.noise_perturbation_analysis
        self.noise_enabled = noise_config.get("enabled", False)
        self.noise_training = noise_config.get("training", False)
        if self.noise_enabled:
            noise_params: dict = noise_config.get("params", {})
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                uniform_noise=noise_params.get("uniform_noise", True),
                uniform_magnitudes=noise_params.get("uniform_magnitudes", [0.1]),
                gaussian_noise=noise_params.get("gaussian_noise", True),
                gaussian_sdevs=noise_params.get("gaussian_sdevs", [0.1]),
                n_samples=noise_params.get("n_samples", 10),
                accuracy=noise_params.get("accuracy", True),
                categorical_loglikelihood=noise_params.get(
                    "categorical_loglikelihood", True
                ),
            )

        # Single layer contribution analysis (formerly layer_ablation)
        single_layer_config = self.config.single_layer_contribution
        self.single_layer_enabled = single_layer_config.get("enabled", False)
        self.single_layer_training = single_layer_config.get("training", False)
        if self.single_layer_enabled:
            params = single_layer_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.single_layer_analyzer = SingleLayerContributionAnalyzer(
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

        # Synaptic ablation analysis (formerly branch_ablation)
        synaptic_ablation_config = self.config.synaptic_ablation
        self.synaptic_ablation_enabled = synaptic_ablation_config.get("enabled", False)
        self.synaptic_ablation_training = synaptic_ablation_config.get(
            "training", False
        )
        if self.synaptic_ablation_enabled:
            params = synaptic_ablation_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.synaptic_ablation_analyzer = SynapticAblationAnalyzer(
                ablation_type=params.get("ablation_type", "whole_layer"),
                layers_to_analyze=params.get("layers_to_analyze", []),
                mode=params.get("mode", "direct"),
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

        # Branch information analysis
        branch_info_config = self.config.branch_information
        self.branch_info_enabled = branch_info_config.get("enabled", False)
        self.branch_info_training = branch_info_config.get("training", False)
        if self.branch_info_enabled:
            params = branch_info_config.get("params", {})
            self.branch_info_analyzer = BranchInformationAnalyzer(
                n_neighbors=params.get("n_neighbors", 10),
                use_dvc=params.get("use_dvc", True),
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
                copula_params=params.get("copula_params", None),
            )

        ablation_config = self.config.ablation_analysis
        self.ablation_enabled = ablation_config.get("enabled", False)
        self.ablation_training = ablation_config.get("training", False)
        if self.ablation_enabled:
            ablation_params: dict = ablation_config.get("params", {})
            self.ablation_analyzer = AblationAnalyzer(
                ablate_all_synapses=ablation_params.get("ablate_all_synapses", True),
                ablate_excitation=ablation_params.get("ablate_excitation", True),
                ablate_inhibition=ablation_params.get("ablate_inhibition", True),
                ablate_branches=ablation_params.get("ablate_branches", True),
                accuracy=ablation_params.get("accuracy", True),
                categorical_loglikelihood=ablation_params.get(
                    "categorical_loglikelihood", True
                ),
            )

        weight_config = self.config.weight_analysis
        self.weight_enabled = weight_config.get("enabled", False)
        self.weight_training = weight_config.get("training", False)
        if self.weight_enabled:
            weight_params: dict = weight_config.get("params", {})
            self.weight_analyzer = WeightAnalyzer(
                synapse_weights=weight_params.get("synapse_weights", True),
                branch_weights=weight_params.get("branch_weights", True),
            )

        synaptic_activation_config = self.config.synaptic_activation_analysis
        self.synaptic_activation_enabled = synaptic_activation_config.get(
            "enabled", False
        )
        self.synaptic_activation_training = synaptic_activation_config.get(
            "training", False
        )
        if self.synaptic_activation_enabled:
            synaptic_activation_params: dict = synaptic_activation_config.get(
                "params", {}
            )
            self.synaptic_activation_analyzer = SynapticActivationAnalyzer(
                network_summary=synaptic_activation_params.get("network_summary", True),
                layer_summary=synaptic_activation_params.get("layer_summary", True),
                branch_summary=synaptic_activation_params.get("branch_summary", True),
                raw_summary=synaptic_activation_params.get("raw_summary", True),
                logspace=synaptic_activation_params.get("logspace", False),
                downsample=synaptic_activation_params.get("downsample", False),
                samples=synaptic_activation_params.get("samples", 1000),
            )

        branch_activation_config = self.config.branch_activation_analysis
        self.branch_activation_enabled = branch_activation_config.get("enabled", False)
        self.branch_activation_training = branch_activation_config.get(
            "training", False
        )
        if self.branch_activation_enabled:
            branch_activation_params: dict = branch_activation_config.get("params", {})
            self.branch_activation_analyzer = BranchActivationAnalyzer(
                synapse_activations=branch_activation_params.get(
                    "synapse_activations", True
                ),
                branch_activations=branch_activation_params.get(
                    "branch_activations", True
                ),
                network_summary=branch_activation_params.get("network_summary", True),
                layer_summary=branch_activation_params.get("layer_summary", True),
                branch_summary=branch_activation_params.get("branch_summary", True),
                raw_summary=branch_activation_params.get("raw_summary", True),
                logspace=branch_activation_params.get("logspace", False),
                downsample=branch_activation_params.get("downsample", False),
                samples=branch_activation_params.get("samples", 1000),
            )

    def run_analysis(self, filename: str = "final", training: bool = False):
        if self.performance_enabled and (
            (self.performance_training and training) or not training
        ):
            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "performance"),
                filename=filename,
            )

        if self.ei_info_enabled and (
            (self.ei_info_training and training) or not training
        ):
            self.ei_info_analyzer.analyze(
                model=self.model,
                dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ei_info_analysis"),
                filename=filename,
            )

        if self.syn_turnover_enabled and (
            (self.syn_turnover_training and training) or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename,
            )

        if self.noise_enabled and ((self.noise_training and training) or not training):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "noise_perturbation_analysis"),
                filename=filename,
            )

        if self.single_layer_enabled and (
            (self.single_layer_training and training) or not training
        ):
            self.single_layer_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "single_layer_contribution"),
                filename=filename,
            )

        if self.synaptic_ablation_enabled and (
            (self.synaptic_ablation_training and training) or not training
        ):
            self.synaptic_ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "synaptic_ablation"),
                filename=filename,
            )

        if self.branch_info_enabled and (
            (self.branch_info_training and training) or not training
        ):
            test_loader = DataLoader(
                self.data["test"],
                batch_size=self.train_config.get("batch_size", 32),
                shuffle=False,
                num_workers=0,
            )

            # Extract epoch number from filename if possible
            try:
                epoch = int(filename.split("_")[-1]) if "_" in filename else 0
            except:
                epoch = 0

            self.branch_info_analyzer.analyze(
                model=self.model,
                test_loader=test_loader,
                output_dir=os.path.join(self.save_root, "branch_information"),
                epoch=epoch,
            )

        if self.ablation_enabled and (
            (self.ablation_training and training) or not training
        ):
            self.ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ablation_analysis"),
                filename=filename,
            )

        if self.weight_enabled and (
            (self.weight_training and training) or not training
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename,
            )

        if self.synaptic_activation_enabled and (
            (self.synaptic_activation_training and training) or not training
        ):
            self.synaptic_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "synaptic_activation_analysis"),
                filename=filename,
            )

        if self.branch_activation_enabled and (
            (self.branch_activation_training and training) or not training
        ):
            self.branch_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "branch_activation_analysis"),
                filename=filename,
            )

        # TODO: add other analysis classes here implemented same as above
