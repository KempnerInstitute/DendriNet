"""
visualization_manager.py
========================
Contains a PlotManager class that holds references to all visualization
and analysis functions in one place. We expand it to produce actual
visualizations for "gradients", "branch_info", and "ablation" using
plot_utils.py functions like plot_einet_gradients, plot_depth_info,
visualize_dendritic_tree, etc.
"""

import os
import logging

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dendritic_modeling.utils import split_MNIST_inputs
from dendritic_modeling.utils import (
    Shaper,
    analyze_branch_ablation,
    analyze_branch_information,
    measure_performance
)
from dendritic_modeling.plot_utils import (
    plot_einet_params,
    plot_einet_activations,
    plot_einet_gradients,
    plot_depth_info,
    # build_dendritic_tree_graph,         # we'll omit the tree graph
    # visualize_dendritic_tree            # no dendritic tree plotting
)

class PlotManager:
    """
    A manager that stores all known plot/analysis methods in a dictionary.
    The main experiment code can call PlotManager.run_plot(plot_name),
    without needing to manually code each function call in train_experiments.py.
    """

    def __init__(self, model, data, config, device='cpu'):
        """
        model : the trained classifier (with .net for EINet)
        data : dict of 'train', 'valid', 'test' datasets
        config : the loaded config (with .visualization section)
        device : 'cuda' or 'cpu'
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
            os.makedirs(save_dir, exist_ok=True)
            # Possibly reshape weights for e.g. MNIST
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
            os.makedirs(save_dir, exist_ok=True)
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
        Generates gradient visualizations by calling plot_einet_gradients.
        """
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "grad_visuals")
            os.makedirs(save_dir, exist_ok=True)

            train_input, train_label = self.data['train'][:]

            plot_einet_gradients(
                model=self.model,
                inputs=train_input,
                labels=train_label,
                save_root=save_dir,
                reshape_fn=None,       # if we want to reshape gradients, we can pass a Shaper
                reshape_exc_syn=True, 
                save_in_dir=True,
                filename="gradients"
            )
            logging.info(f"Saved gradient visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient plotting.")

    def _branch_info(self):
        """
        Computes mutual information across layers/branches, then plots
        or visualizes the results (depth plot).
        """
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping branch_info.")
            return
        net = self.model.net
        
        print(hasattr(net, 'forward_with_branch_outputs'))
        # Must have forward_with_branch_outputs for analyze_branch_information
        if not hasattr(net, 'forward_with_branch_outputs'):
            logging.info("No forward_with_branch_outputs found. Skipping branch_info analysis.")
            return

        logging.info("Computing branch mutual info ...")
        branch_info = analyze_branch_information(
            net,
            self.data['train'],  # or 'valid'
            batch_size=getattr(self.config.train, 'batch_size', 64),
            device=self.device,
            n_bins=10
        )
        logging.info(f"Branch Info (MI) per layer: {branch_info}")

        print(888888444444888888444444888888444444888888444444888888444444888888444444)
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)

        # 1) Plot MI vs. layer depth
        logging.info("Plotting MI vs. layer depth ...")
        plot_depth_info(branch_info)  
        plt.savefig(os.path.join(save_dir, "branch_info_depth_plot.png"))
        plt.close()
        logging.info(f"Saved branch_info_depth_plot.png in {save_dir}")

        # 2) We remove the dendritic tree graph calls, so no build_dendritic_tree_graph

    def _ablation(self):
        """
        Layer-wise ablation approach: zero out all dendrites in each layer 
        and measure performance drop. Then plot performance change vs layer index.
        """
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping ablation.")
            return
        net = self.model.net

        if not hasattr(net, 'forward_with_branch_outputs'):
            logging.info("No forward_with_branch_outputs found. Skipping ablation.")
            return

        logging.info("Performing layer-wise ablation ...")

        # measure baseline
        baseline_acc = measure_performance(net, self.data['test'], device=self.device)
        logging.info(f"Ablation baseline acc on test = {baseline_acc}")

        # store original weights
        original = {}
        for n, p in net.named_parameters():
            original[n] = p.detach().clone()

        layer_idxs = []
        perf_drops = []

        # Loop over each layer index in net.n_branch_layers
        # The user wants to set *all* dendrites in the same depth to zero
        for i in range(net.n_branch_layers):
            logging.info(f"Ablating layer {i} ...")
            # zero out entire excitatory sub-net (and optionally inhibitory)
            # We'll do excitatory + inhibitory for that layer
            ei_layer = net.layers[i]  # ExcitationInhibitionLayer

            # excitatory sub-net
            for dend_layer in ei_layer.excitatory_cells.branch_layers:
                if hasattr(dend_layer, 'branch_excitation'):
                    dend_layer.branch_excitation.pre_w.data.zero_()
                if hasattr(dend_layer, 'branch_inhibition'):
                    dend_layer.branch_inhibition.pre_w.data.zero_()

            # inhibitory sub-net
            for dend_layer in ei_layer.inhibitory_cells.branch_layers:
                if hasattr(dend_layer, 'branch_excitation'):
                    dend_layer.branch_excitation.pre_w.data.zero_()
                if hasattr(dend_layer, 'branch_inhibition'):
                    dend_layer.branch_inhibition.pre_w.data.zero_()

            acc_after = measure_performance(net, self.data['test'], device=self.device)
            delta = baseline_acc - acc_after
            logging.info(f"Layer {i}: new acc={acc_after:.3f}, drop={delta:.3f}")

            layer_idxs.append(i)
            perf_drops.append(delta)

            # restore weights
            for n, p in net.named_parameters():
                p.data.copy_(original[n])

        # Now plot performance drop vs. layer index
        save_dir = os.path.join(self.save_path, "ablation_plots")
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(layer_idxs, perf_drops, marker='o', color='orange')
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Accuracy Drop")
        ax.set_title("Layer-wise Ablation: Performance Drop vs. Layer Index")
        fig.savefig(os.path.join(save_dir, "ablation_vs_layer_index.png"))
        plt.close(fig)
        logging.info(f"Saved ablation_vs_layer_index.png in {save_dir}")

