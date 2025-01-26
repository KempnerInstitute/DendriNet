"""
visualization_manager.py
========================
Contains a PlotManager class that holds references to all visualization
and analysis functions in one place.

New plots or analyses can be added by creating another method and
registering it in the plot_methods dictionary.
"""

import os
import logging

import torch
from dendritic_modeling.utils import split_MNIST_inputs
from dendritic_modeling.plot_utils import (
    plot_einet_params,
    plot_einet_activations
)
from dendritic_modeling.utils import (
    Shaper,
    analyze_branch_ablation,
    analyze_branch_information
)

class PlotManager:
    """
    A manager that stores all known plot/analysis methods in a dictionary.
    The main experiment only calls PlotManager.run_plot(plot_name, ...),
    so no need to modify train_experiments.py each time a new plot is added.
    """

    def __init__(self, model, data, config, device='cpu'):
        """
        model : the trained classifier (with .net for EINet)
        data : dict of 'train', 'valid', 'test' datasets
        config : the loaded config
        device : string, 'cuda' or 'cpu'
        """

        self.model = model
        self.data = data
        self.config = config
        self.device = device

        vis_cfg = self.config.visualization
        self.save_path = vis_cfg.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.plot_methods = {
            "weights": self._plot_weights,
            "activations": self._plot_activations,
            "gradients": self._plot_gradients,
            "branch_info": self._branch_info,
            "ablation": self._ablation,
        }

    def run_plot(self, plot_name):
        """
        Calls the method associated with plot_name if it exists.
        """
        func = self.plot_methods.get(plot_name, None)
        if func is not None:
            logging.info(f"Running plot/analysis '{plot_name}' in PlotManager...")
            func()
        else:
            logging.info(f"Plot name '{plot_name}' not recognized in PlotManager.")

    def _plot_weights(self):
        """
        Plots or saves parameter visuals for an EINet if it exists.
        """
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "param_visuals")
            shaper = Shaper(shape=(28, 28))
            plot_einet_params(
                einet=self.model.net,
                save_root=save_dir,
                reshape_fn=shaper.reshape,
                reshape_exc_syn=True,
            )
        else:
            logging.info("No 'net' attribute found on model. Skipping _plot_weights.")

    def _plot_activations(self):
        """
        Plots or saves activation distributions for an EINet if it exists.
        """
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "activation_visuals")
            train_input, train_label = self.data['train'][:]
            input_list = split_MNIST_inputs(train_input, train_label)
            plot_einet_activations(
                einet=self.model.net,
                input_list=input_list,
                save_root=save_dir
            )
        else:
            logging.info("No 'net' attribute on model. Skipping _plot_activations.")

    def _plot_gradients(self):
        """
        Placeholder for gradient visualization if needed.
        """
        logging.info("No explicit gradient plotting function here. Could implement if needed.")

    def _branch_info(self):
        """
        Computes mutual information or branch-based info
        if forward_with_branch_outputs is available.
        """
        if hasattr(self.model.net, 'forward_with_branch_outputs'):
            branch_info = analyze_branch_information(
                self.model.net,
                self.data['train'],
                batch_size=self.config.train.get('batch_size', 64),
                device=self.device,
                n_bins=10
            )
            logging.info(f"Branch Info (MI) per layer: {branch_info}")
        else:
            logging.info("No forward_with_branch_outputs found. Skipping branch_info analysis.")

    def _ablation(self):
        """
        Example ablation test on layer 0 branches.
        """
        if hasattr(self.model.net, 'forward_with_branch_outputs'):
            baseline_acc, ablation_res = analyze_branch_ablation(
                self.model.net,
                self.data['test'],
                layer_idx=0,
                branch_ids=[0,1],
                mode='direct',
                device=self.device
            )
            logging.info(f"Ablation baseline acc on test = {baseline_acc}")
            for b_id, (acc_after, delta) in ablation_res.items():
                logging.info(f"Layer 0, branch {b_id}: acc={acc_after:.3f}, drop={delta:.3f}")
        else:
            logging.info("No forward_with_branch_outputs found. Skipping ablation.")