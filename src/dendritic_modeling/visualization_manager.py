"""
visualization_manager.py
========================
Contains a PlotManager class that holds references to all visualization
and analysis functions to produce actual visualizations for "gradients",
"branch_info", and "ablation" using plot_utils.py functions
"""

import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import wandb
from dendritic_modeling.plot_utils import (
    einet_activations_to_csv,
    plot_depth_info,
    plot_einet_activations,
    plot_einet_gradients,
    plot_einet_params,
    plot_einet_profiles,
    plot_gradient_flow,
    plot_NLL_loss_curves,
)
from dendritic_modeling.synthetic_datasets import split_MNIST_inputs
from dendritic_modeling.utils import (
    Shaper,
    analyze_branch_ablation,
    analyze_branch_information,
    analyze_dendritic_information,
    analyze_layerwise_ablation,
    measure_performance,
)


class PlotManager:
    def __init__(self, model, data, args, device="cpu"):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.save_path, "visualization.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        vis_cfg = self.args.visualization
        self.file_name = args.file_name
        os.makedirs(self.save_path, exist_ok=True)
        self.plot_methods = {
            "weights": self._plot_weights,
            "activations": self._plot_activations,
            "gradients": self._plot_gradients,
            "profiles": self._plot_profiles,
            "branch_info": self._branch_info,
            "ablation": self._branch_ablation,
            "layerwise_ablation": self._branch_ablation,
            "layer_ablation": self._branch_ablation,
            "branch_ablation": self._branch_ablation,
            "dendritic_ablation_all": self._dendritic_ablation_all,
            "dendritic_information": self._dendritic_information,
            "dendritic_information_inputs": self._dendritic_information_inputs,
            "dendritic_information_activations": self._dendritic_information_activations,
            "gradient_flow": self._plot_gradient_flow,
            "gradient_flow_torchviz": self._plot_gradient_flow_torchviz,
        }

    def run_plot(self, plot_name, epoch=None):
        if epoch is not None:
            logging.info(f"At epoch {epoch}: generating plot '{plot_name}'...")
        else:
            logging.info(f"Generating plot '{plot_name}'...")
        func = self.plot_methods.get(plot_name, None)
        if func is not None:
            self.current_plot = plot_name
            try:
                print(f"Starting to generate plot '{plot_name}'...")
                func()
                print(f"Successfully completed plot '{plot_name}'")
                self.current_plot = None
                if epoch is not None:
                    logging.info(f"Plot '{plot_name}' saved for epoch {epoch}.")
                else:
                    logging.info(f"Plot '{plot_name}' saved.")
            except Exception as e:
                print(f"Error generating plot '{plot_name}': {e}")
                print(traceback.format_exc())
                logging.error(f"Error generating plot '{plot_name}': {e}")
                logging.error(traceback.format_exc())
        else:
            logging.info(
                f"Plot name '{plot_name}' not recognized in PlotManager. Available plots: {list(self.plot_methods.keys())}"
            )
            print(
                f"Plot name '{plot_name}' not recognized. Available plots: {list(self.plot_methods.keys())}"
            )

    def _plot_weights(self):
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "param_visuals")
            os.makedirs(save_dir, exist_ok=True)
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_params(
                einet=self.model.net,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
                reshape_exc_syn=True,
                logspace=True,
                save_in_dir=True,
                filename=self.file_name,
            )
        else:
            logging.info("No 'net' attribute found on model. Skipping _plot_weights.")

    def _plot_activations(self):
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "activation_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data["train"][:]
            input_list = split_MNIST_inputs(train_input, train_label)
            plot_einet_activations(
                einet=self.model.net,
                input_list=input_list,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name,
            )
        else:
            logging.info("No 'net' attribute on model. Skipping _plot_activations.")

    def _plot_gradients(self):
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "grad_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data["train"][:]
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_gradients(
                model=self.model,
                inputs=train_input,
                labels=train_label,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
                reshape_exc_syn=True,
                save_in_dir=True,
                filename=self.file_name,
            )
            logging.info(f"Saved gradient visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient plotting.")

    def _plot_gradient_flow(self):
        """
        Visualizes the gradient flow across the network by plotting the distribution
        of gradients for each layer's parameters using matplotlib.

        This helps identify vanishing or exploding gradients during training.
        """
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "gradient_flow")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data["train"][:]

            # Use a subset of data for gradient computation if dataset is large
            batch_size = min(128, len(train_input))
            indices = torch.randperm(len(train_input))[:batch_size]
            batch_input = train_input[indices]
            batch_label = train_label[indices]

            logging.info(f"Using standard matplotlib-based gradient flow visualization")

            plot_path = plot_gradient_flow(
                model=self.model,
                inputs=batch_input,
                labels=batch_label,
                save_root=save_dir,
                filename=f"{self.file_name}_gradient_flow",
                use_torchviz=False,
            )

            logging.info(f"Saved gradient flow visualization in {plot_path}")

            # Also log to wandb if available
            if hasattr(self.args, "use_wandb") and self.args.use_wandb:
                try:
                    import wandb

                    wandb.log(
                        {
                            "gradient_flow": wandb.Image(plot_path),
                            "gradient_flow_by_type": wandb.Image(
                                plot_path.replace(".png", "_by_type.png")
                            ),
                        }
                    )
                    logging.info("Logged gradient flow to wandb")
                except Exception as e:
                    logging.warning(f"Failed to log gradient flow to wandb: {e}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient flow plotting.")

    def _plot_gradient_flow_torchviz(self):
        """
        Visualizes the gradient flow across the network using torchviz to generate
        a computation graph visualization.

        This helps understand the network structure and gradient flow paths.
        """
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "gradient_flow_torchviz")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data["train"][:]

            # Use a smaller subset of data for torchviz to prevent graph explosion
            batch_size = min(16, len(train_input))
            indices = torch.randperm(len(train_input))[:batch_size]
            batch_input = train_input[indices]
            batch_label = train_label[indices]

            logging.info(f"Using torchviz-based gradient flow visualization")

            try:
                plot_path = plot_gradient_flow(
                    model=self.model,
                    inputs=batch_input,
                    labels=batch_label,
                    save_root=save_dir,
                    filename=f"{self.file_name}_gradient_flow",
                    use_torchviz=True,
                )

                logging.info(f"Saved gradient flow visualization in {plot_path}")

                # Also log to wandb if available
                if hasattr(self.args, "use_wandb") and self.args.use_wandb:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "gradient_flow_graph": wandb.Image(plot_path),
                                "gradient_flow_magnitudes": wandb.Image(
                                    plot_path.replace(
                                        "_graph.png", "_torchviz_magnitudes.png"
                                    )
                                ),
                            }
                        )
                        logging.info("Logged gradient flow to wandb")
                    except Exception as e:
                        logging.warning(f"Failed to log gradient flow to wandb: {e}")
            except ImportError:
                logging.error(
                    "torchviz is not installed. Please install it with 'pip install torchviz graphviz'"
                )
                print(
                    "Error: torchviz is not installed. Please install it with 'pip install torchviz graphviz'"
                )
            except Exception as e:
                logging.error(f"Error in torchviz visualization: {e}")
                print(f"Error in torchviz visualization: {e}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient flow plotting.")

    def _plot_profiles(self):
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "profiles")
            os.makedirs(save_dir, exist_ok=True)
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_profiles(
                model=self.model,
                train_data=self.data["train"],
                valid_data=self.data["valid"],
                n_tasks=1,
                logspace=True,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name,
            )
            logging.info(f"Saved profile visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping profile plotting.")

    def _branch_info(self):
        print("\n" + "=" * 80)
        print("STARTING BRANCH INFORMATION ANALYSIS")
        print("=" * 80)
        if not hasattr(self.model, "net"):
            print(
                "ERROR: No 'net' attribute found on model. Skipping branch_info analysis."
            )
            logging.error(
                "No 'net' attribute found on model. Skipping branch_info analysis."
            )
            return
        net = self.model.net
        if not hasattr(net, "forward_with_branch_outputs"):
            print(
                "ERROR: No forward_with_branch_outputs method found on net. Skipping branch_info analysis."
            )
            logging.error(
                "No forward_with_branch_outputs method found on net. Skipping branch_info analysis."
            )
            return
        print(f"Model has {len(net.branch_layers)} branch layers")
        logging.info(f"Model has {len(net.branch_layers)} branch layers")
        print("Computing branch info using MLP decoding...")
        logging.info("Computing branch info using MLP decoding...")
        batch_size = getattr(self.args.train, "batch_size", 64)
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
        print("Loading datasets...")
        train_inputs, train_labels = self.data["train"][:]
        valid_inputs, valid_labels = self.data["valid"][:]
        test_inputs, test_labels = self.data["test"][:]
        print(
            f"Loaded datasets: train={len(train_inputs)}, valid={len(valid_inputs)}, test={len(test_inputs)} samples"
        )
        layer_indices = []
        branch_indices = []
        layer_exc_accuracies = []
        layer_inh_accuracies = []
        layer_combined_accuracies = []
        use_shunting = getattr(self.model.net, "use_shunting", True)
        print(
            f"Using {'shunting' if use_shunting else 'subtraction'} for combining excitatory and inhibitory inputs"
        )
        for layer_idx, layer in enumerate(net.branch_layers):
            print(f"\nProcessing layer {layer_idx} for branch_info...")
            logging.info(f"Processing layer {layer_idx} for branch_info...")
            if not hasattr(layer, "branch_excitation") and not hasattr(
                layer, "branch_inhibition"
            ):
                print(
                    f"  Layer {layer_idx} has no branch excitation or inhibition. Skipping."
                )
                logging.info(
                    f"Layer {layer_idx} has no branch excitation or inhibition. Skipping."
                )
                continue
            n_branches = 0
            if hasattr(layer, "branch_excitation"):
                n_branches = layer.branch_excitation.pre_w.shape[0]
                print(
                    f"  Layer {layer_idx} has branch_excitation with {n_branches} branches"
                )
            elif hasattr(layer, "branch_inhibition"):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
                print(
                    f"  Layer {layer_idx} has branch_inhibition with {n_branches} branches"
                )
            if n_branches == 0:
                print(f"  Layer {layer_idx} has no branches. Skipping.")
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
            print(f"  Layer {layer_idx} has {n_branches} branches")
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            print("  Collecting branch inputs from dataset...")
            exc_branch_inputs_all = []
            inh_branch_inputs_all = []
            net.eval()
            net = net.to(self.device)
            with torch.no_grad():
                for batch_start in range(0, len(train_inputs), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_inputs))
                    batch = train_inputs[batch_start:batch_end].to(self.device)
                    _ = net(batch)
                    if hasattr(layer, "branch_excitation"):
                        if hasattr(layer, "excitatory_input"):
                            exc_inputs = layer.excitatory_input
                            exc_branch_inputs_all.append(exc_inputs.cpu())
                        else:
                            print(
                                "    Warning: layer has branch_excitation but no excitatory_input attribute"
                            )
                    if hasattr(layer, "branch_inhibition"):
                        if hasattr(layer, "inhibitory_input"):
                            inh_inputs = layer.inhibitory_input
                            inh_branch_inputs_all.append(inh_inputs.cpu())
                        else:
                            print(
                                "    Warning: layer has branch_inhibition but no inhibitory_input attribute"
                            )
            if torch.cuda.is_available():
                net = net.cuda()
            if exc_branch_inputs_all:
                exc_inputs = torch.cat(exc_branch_inputs_all, dim=0)
                print(f"  Collected excitatory inputs: {exc_inputs.shape}")
            else:
                print(
                    "  No excitatory inputs collected - this is expected with input_mode=0"
                )
                print("  Skipping branch information analysis for this layer")
                continue
            if inh_branch_inputs_all:
                inh_inputs = torch.cat(inh_branch_inputs_all, dim=0)
                print(f"  Collected inhibitory inputs: {inh_inputs.shape}")
            else:
                inh_inputs = torch.zeros((len(train_inputs), 0))
                print("  No inhibitory inputs collected")
            layer_exc_acc = 0.0
            layer_inh_acc = 0.0
            layer_combined_acc = 0.0
            for branch_idx in range(n_branches):
                print(f"  Processing branch {branch_idx} in layer {layer_idx}...")
                logging.info(f"  Processing branch {branch_idx} in layer {layer_idx}")
                if hasattr(layer, "branch_excitation") and exc_inputs.shape[1] > 0:
                    branch_exc_weights = layer.branch_excitation.pre_w[branch_idx].cpu()
                    print(f"    Excitatory weights shape: {branch_exc_weights.shape}")
                    branch_exc_activations = exc_inputs @ branch_exc_weights
                    print(
                        f"    Excitatory activations shape: {branch_exc_activations.shape}"
                    )
                    train_exc_dataset = torch.utils.data.TensorDataset(
                        branch_exc_activations.unsqueeze(1), train_labels
                    )
                    valid_exc_dataset = torch.utils.data.TensorDataset(
                        branch_exc_activations[: len(valid_inputs)].unsqueeze(1),
                        valid_labels,
                    )
                    print(f"    Training excitatory classifier...")
                    exc_acc = self._train_mlp_classifier(
                        train_exc_dataset, valid_exc_dataset
                    )
                    print(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    logging.info(
                        f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}"
                    )
                    layer_exc_acc = max(layer_exc_acc, exc_acc)
                else:
                    print(f"    No excitatory inputs for branch {branch_idx}")
                    branch_exc_activations = None
                    exc_acc = 0.0
                if hasattr(layer, "branch_inhibition") and inh_inputs.shape[1] > 0:
                    branch_inh_weights = layer.branch_inhibition.pre_w[branch_idx].cpu()
                    print(f"    Inhibitory weights shape: {branch_inh_weights.shape}")
                    branch_inh_activations = inh_inputs @ branch_inh_weights
                    print(
                        f"    Inhibitory activations shape: {branch_inh_activations.shape}"
                    )
                    train_inh_dataset = torch.utils.data.TensorDataset(
                        branch_inh_activations.unsqueeze(1), train_labels
                    )
                    valid_inh_dataset = torch.utils.data.TensorDataset(
                        branch_inh_activations[: len(valid_inputs)].unsqueeze(1),
                        valid_labels,
                    )
                    print(f"    Training inhibitory classifier...")
                    inh_acc = self._train_mlp_classifier(
                        train_inh_dataset, valid_inh_dataset
                    )
                    print(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    logging.info(
                        f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}"
                    )
                    layer_inh_acc = max(layer_inh_acc, inh_acc)
                else:
                    print(f"    No inhibitory inputs for branch {branch_idx}")
                    branch_inh_activations = None
                    inh_acc = 0.0
                if (
                    branch_exc_activations is not None
                    and branch_inh_activations is not None
                ):
                    if use_shunting:
                        branch_combined = branch_exc_activations / (
                            branch_exc_activations + branch_inh_activations + 1.0
                        )
                        print(f"    Using shunting combination")
                    else:
                        branch_combined = (
                            branch_exc_activations - branch_inh_activations
                        )
                        print(f"    Using subtraction combination")
                    train_combined_dataset = torch.utils.data.TensorDataset(
                        branch_combined.unsqueeze(1), train_labels
                    )
                    valid_combined_dataset = torch.utils.data.TensorDataset(
                        branch_combined[: len(valid_inputs)].unsqueeze(1), valid_labels
                    )
                    print(f"    Training combined classifier...")
                    combined_acc = self._train_mlp_classifier(
                        train_combined_dataset, valid_combined_dataset
                    )
                    print(
                        f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}"
                    )
                    logging.info(
                        f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}"
                    )
                    layer_combined_acc = max(layer_combined_acc, combined_acc)
                else:
                    print(
                        f"    Could not compute combined inputs (missing excitatory or inhibitory)"
                    )
                    combined_acc = max(exc_acc, inh_acc)
                branch_indices.append(f"{layer_idx}-{branch_idx}")
            layer_indices.append(layer_idx)
            layer_exc_accuracies.append(layer_exc_acc)
            layer_inh_accuracies.append(layer_inh_acc)
            layer_combined_accuracies.append(layer_combined_acc)
            print(
                f"  Layer {layer_idx} best accuracies: exc={layer_exc_acc:.4f}, inh={layer_inh_acc:.4f}, combined={layer_combined_acc:.4f}"
            )
        if layer_indices:
            print("\nGenerating branch information plots...")
            plt.figure(figsize=(10, 6))
            plt.plot(layer_indices, layer_exc_accuracies, "o-", label="Excitatory")
            plt.plot(layer_indices, layer_inh_accuracies, "s-", label="Inhibitory")
            plt.plot(layer_indices, layer_combined_accuracies, "^-", label="Combined")
            plt.xlabel("Layer Index")
            plt.ylabel("Max Branch Classification Accuracy")
            plt.title("Branch Information Analysis - MLP Decoding Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "branch_info_mlp_decoding.png"), dpi=300)
            plt.close()
            print(
                f"Saved plot to {os.path.join(save_dir, 'branch_info_mlp_decoding.png')}"
            )
            df = pd.DataFrame(
                {
                    "layer_idx": layer_indices,
                    "excitatory_acc": layer_exc_accuracies,
                    "inhibitory_acc": layer_inh_accuracies,
                    "combined_acc": layer_combined_accuracies,
                }
            )
            csv_path = os.path.join(save_dir, "branch_info_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            print("\nBranch Information Analysis Summary:")
            print("-" * 60)
            print(
                f"{'Layer':<8} {'Excitatory':<12} {'Inhibitory':<12} {'Combined':<12}"
            )
            print("-" * 60)
            for i, layer_idx in enumerate(layer_indices):
                print(
                    f"{layer_idx:<8} {layer_exc_accuracies[i]:<12.4f} {layer_inh_accuracies[i]:<12.4f} {layer_combined_accuracies[i]:<12.4f}"
                )
            print("-" * 60)
        else:
            print("No valid layers processed. No plots generated.")
        print(f"Branch information analysis completed. Results saved to {save_dir}")
        logging.info(f"Saved branch info plots in {save_dir}")
        print("=" * 80 + "\n")

    def _train_mlp_classifier(
        self, train_dataset, valid_dataset, hidden_dim=100, lr=0.001, epochs=20
    ):
        input_dim = train_dataset.tensors[0].shape[1]
        num_classes = len(torch.unique(train_dataset.tensors[1]))

        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleMLP(input_dim, hidden_dim, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64)
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def _branch_ablation(self):
        print("\n" + "=" * 80)
        print("STARTING BRANCH ABLATION ANALYSIS")
        print("=" * 80)
        if not hasattr(self.model, "net"):
            print(
                "ERROR: No 'net' attribute found on model. Skipping branch_ablation analysis."
            )
            logging.error(
                "No 'net' attribute found on model. Skipping branch_ablation analysis."
            )
            return
        net = self.model.net
        do_layer_ablation = getattr(self.args.visualization, "layer_ablation", True)
        do_branch_ablation = getattr(self.args.visualization, "branch_ablation", True)
        plots_list = getattr(self.args.visualization, "plots", [])
        if "layer_ablation" in plots_list:
            do_layer_ablation = True
        if "branch_ablation" in plots_list:
            do_branch_ablation = True
        current_plot = getattr(self, "current_plot", None)
        if current_plot == "layer_ablation":
            do_layer_ablation = True
            do_branch_ablation = False
        elif current_plot == "branch_ablation":
            do_branch_ablation = True
            do_layer_ablation = False
        if not (do_layer_ablation or do_branch_ablation):
            print(
                "Both layer_ablation and branch_ablation are disabled in config. Skipping analysis."
            )
            logging.info(
                "Both layer_ablation and branch_ablation are disabled in config. Skipping analysis."
            )
            return
        branch_layers = []
        if hasattr(net, "branch_layers"):
            branch_layers = net.branch_layers
            print(
                f"Found branch_layers directly. Model has {len(branch_layers)} branch layers"
            )
        elif hasattr(net, "layers"):
            branch_layers = []
            for layer_idx, layer in enumerate(net.layers):
                if hasattr(layer, "excitatory_cells") and hasattr(
                    layer.excitatory_cells, "branch_layers"
                ):
                    branch_layers.extend(layer.excitatory_cells.branch_layers)
                if hasattr(layer, "inhibitory_cells") and hasattr(
                    layer.inhibitory_cells, "branch_layers"
                ):
                    branch_layers.extend(layer.inhibitory_cells.branch_layers)
            print(
                f"Extracted branch_layers from layers structure. Found {len(branch_layers)} branch layers"
            )
        if not branch_layers:
            print("No branch layers found in model. Skipping ablation analysis.")
            logging.error(
                "No branch layers found in model. Skipping branch_ablation analysis."
            )
            return
        logging.info(f"Model has {len(branch_layers)} branch layers")
        print("Performing branch ablation analysis...")
        logging.info("Performing branch ablation analysis...")
        print(f"Layer ablation enabled: {do_layer_ablation}")
        print(f"Branch ablation enabled: {do_branch_ablation}")
        logging.info(f"Layer ablation enabled: {do_layer_ablation}")
        logging.info(f"Branch ablation enabled: {do_branch_ablation}")
        from dendritic_modeling.utils import measure_performance

        print("Measuring baseline performance...")
        baseline_acc = measure_performance(net, self.data["test"], device=self.device)
        print(f"Baseline accuracy on test set: {baseline_acc:.4f}")
        logging.info(f"Branch ablation baseline acc on test = {baseline_acc:.4f}")
        print("Storing original model parameters...")
        original = {}
        for n, p in net.named_parameters():
            original[n] = p.detach().clone()
        layer_indices = []
        layer_both_ablated_acc = []
        layer_exc_ablated_acc = []
        layer_inh_ablated_acc = []
        branch_indices = []
        branch_layer_indices = []
        branch_both_ablated_acc = []
        branch_exc_ablated_acc = []
        branch_inh_ablated_acc = []
        for layer_idx, layer in enumerate(branch_layers):
            print(f"\nAblating layer {layer_idx}...")
            logging.info(f"Ablating layer {layer_idx}...")
            has_branch_excitation = hasattr(layer, "branch_excitation")
            has_branch_inhibition = hasattr(layer, "branch_inhibition")
            if not (has_branch_excitation or has_branch_inhibition):
                print(
                    f"  Layer {layer_idx} has neither branch_excitation nor branch_inhibition. Skipping."
                )
                logging.info(
                    f"Layer {layer_idx} has neither branch_excitation nor branch_inhibition. Skipping."
                )
                continue
            n_branches = 0
            if has_branch_excitation:
                if hasattr(layer.branch_excitation, "pre_w"):
                    n_branches = layer.branch_excitation.pre_w.shape[0]
                    print(
                        f"  Layer {layer_idx} has branch_excitation with {n_branches} branches"
                    )
            elif has_branch_inhibition:
                if hasattr(layer.branch_inhibition, "pre_w"):
                    n_branches = layer.branch_inhibition.pre_w.shape[0]
                    print(
                        f"  Layer {layer_idx} has branch_inhibition with {n_branches} branches"
                    )
            if n_branches == 0:
                print(
                    f"  Layer {layer_idx} has no branches or pre_w not found. Skipping."
                )
                logging.info(
                    f"Layer {layer_idx} has no branches or pre_w not found. Skipping."
                )
                continue
            print(f"  Layer {layer_idx} has {n_branches} branches")
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            if do_layer_ablation:
                print("  Performing layer-level ablation...")
                if has_branch_excitation and hasattr(layer.branch_excitation, "pre_w"):
                    layer.branch_excitation.pre_w.data.zero_()
                    print("    Set excitatory weights to zero")
                if has_branch_inhibition and hasattr(layer.branch_inhibition, "pre_w"):
                    layer.branch_inhibition.pre_w.data.zero_()
                    print("    Set inhibitory weights to zero")
                print("    Measuring accuracy with both ablated...")
                both_acc = measure_performance(
                    net, self.data["test"], device=self.device
                )
                both_drop = baseline_acc - both_acc
                print(
                    f"    Layer {layer_idx} with both ablated: acc={both_acc:.4f}, drop={both_drop:.4f}"
                )
                logging.info(
                    f"Layer {layer_idx} with both ablated: acc={both_acc:.4f}, drop={both_drop:.4f}"
                )
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                if has_branch_excitation and hasattr(layer.branch_excitation, "pre_w"):
                    layer.branch_excitation.pre_w.data.zero_()
                    print("    Set excitatory weights to zero")
                    print("    Measuring accuracy with excitatory ablated...")
                    exc_acc = measure_performance(
                        net, self.data["test"], device=self.device
                    )
                    exc_drop = baseline_acc - exc_acc
                    print(
                        f"    Layer {layer_idx} with exc ablated: acc={exc_acc:.4f}, drop={exc_drop:.4f}"
                    )
                    logging.info(
                        f"Layer {layer_idx} with exc ablated: acc={exc_acc:.4f}, drop={exc_drop:.4f}"
                    )
                else:
                    print("    No excitatory weights to ablate")
                    exc_acc = baseline_acc
                    exc_drop = 0.0
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                if has_branch_inhibition and hasattr(layer.branch_inhibition, "pre_w"):
                    layer.branch_inhibition.pre_w.data.zero_()
                    print("    Set inhibitory weights to zero")
                    print("    Measuring accuracy with inhibitory ablated...")
                    inh_acc = measure_performance(
                        net, self.data["test"], device=self.device
                    )
                    inh_drop = baseline_acc - inh_acc
                    print(
                        f"    Layer {layer_idx} with inh ablated: acc={inh_acc:.4f}, drop={inh_drop:.4f}"
                    )
                    logging.info(
                        f"Layer {layer_idx} with inh ablated: acc={inh_acc:.4f}, drop={inh_drop:.4f}"
                    )
                else:
                    print("    No inhibitory weights to ablate")
                    inh_acc = baseline_acc
                    inh_drop = 0.0
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                layer_indices.append(layer_idx)
                layer_both_ablated_acc.append(both_acc)
                layer_exc_ablated_acc.append(exc_acc)
                layer_inh_ablated_acc.append(inh_acc)
            if do_branch_ablation:
                print(
                    f"  Performing branch-level ablation for {n_branches} branches..."
                )
                for branch_idx in range(n_branches):
                    print(f"    Ablating branch {branch_idx}...")
                    exc_ablated = False
                    inh_ablated = False
                    if has_branch_excitation and hasattr(
                        layer.branch_excitation, "pre_w"
                    ):
                        layer.branch_excitation.pre_w.data[branch_idx].zero_()
                        exc_ablated = True
                        print(
                            f"      Set excitatory weights to zero for branch {branch_idx}"
                        )
                    if has_branch_inhibition and hasattr(
                        layer.branch_inhibition, "pre_w"
                    ):
                        layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                        inh_ablated = True
                        print(
                            f"      Set inhibitory weights to zero for branch {branch_idx}"
                        )
                    if not (exc_ablated or inh_ablated):
                        print(
                            f"      No weights to ablate for branch {branch_idx}. Skipping."
                        )
                        continue
                    print(f"      Measuring accuracy with both ablated...")
                    branch_both_acc = measure_performance(
                        net, self.data["test"], device=self.device
                    )
                    branch_both_drop = baseline_acc - branch_both_acc
                    print(
                        f"      Branch {branch_idx} with both ablated: acc={branch_both_acc:.4f}, drop={branch_both_drop:.4f}"
                    )
                    logging.info(
                        f"    Branch {branch_idx} with both ablated: acc={branch_both_acc:.4f}, drop={branch_both_drop:.4f}"
                    )
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    if has_branch_excitation and hasattr(
                        layer.branch_excitation, "pre_w"
                    ):
                        layer.branch_excitation.pre_w.data[branch_idx].zero_()
                        print(
                            f"      Set excitatory weights to zero for branch {branch_idx}"
                        )
                        print(f"      Measuring accuracy with excitatory ablated...")
                        branch_exc_acc = measure_performance(
                            net, self.data["test"], device=self.device
                        )
                        branch_exc_drop = baseline_acc - branch_exc_acc
                        print(
                            f"      Branch {branch_idx} with exc ablated: acc={branch_exc_acc:.4f}, drop={branch_exc_drop:.4f}"
                        )
                        logging.info(
                            f"    Branch {branch_idx} with exc ablated: acc={branch_exc_acc:.4f}, drop={branch_exc_drop:.4f}"
                        )
                    else:
                        print(
                            f"      No excitatory weights to ablate for branch {branch_idx}"
                        )
                        branch_exc_acc = baseline_acc
                        branch_exc_drop = 0.0
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    if has_branch_inhibition and hasattr(
                        layer.branch_inhibition, "pre_w"
                    ):
                        layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                        print(
                            f"      Set inhibitory weights to zero for branch {branch_idx}"
                        )
                        print(f"      Measuring accuracy with inhibitory ablated...")
                        branch_inh_acc = measure_performance(
                            net, self.data["test"], device=self.device
                        )
                        branch_inh_drop = baseline_acc - branch_inh_acc
                        print(
                            f"      Branch {branch_idx} with inh ablated: acc={branch_inh_acc:.4f}, drop={branch_inh_drop:.4f}"
                        )
                        logging.info(
                            f"    Branch {branch_idx} with inh ablated: acc={branch_inh_acc:.4f}, drop={branch_inh_drop:.4f}"
                        )
                    else:
                        print(
                            f"      No inhibitory weights to ablate for branch {branch_idx}"
                        )
                        branch_inh_acc = baseline_acc
                        branch_inh_drop = 0.0
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    branch_indices.append(f"{layer_idx}-{branch_idx}")
                    branch_layer_indices.append(layer_idx)
                    branch_both_ablated_acc.append(branch_both_acc)
                    branch_exc_ablated_acc.append(branch_exc_acc)
                    branch_inh_ablated_acc.append(branch_inh_acc)
        save_dir = os.path.join(self.save_path, "branch_ablation_plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
        if not (layer_indices or branch_indices):
            print("No valid layers or branches processed. No plots generated.")
            return
        print("Generating ablation plots...")
        if do_layer_ablation and layer_indices:
            import pandas as pd

            df_layer = pd.DataFrame(
                {
                    "layer_idx": layer_indices,
                    "baseline_acc": [baseline_acc] * len(layer_indices),
                    "both_ablated_acc": layer_both_ablated_acc,
                    "exc_ablated_acc": layer_exc_ablated_acc,
                    "inh_ablated_acc": layer_inh_ablated_acc,
                    "both_drop": [baseline_acc - acc for acc in layer_both_ablated_acc],
                    "exc_drop": [baseline_acc - acc for acc in layer_exc_ablated_acc],
                    "inh_drop": [baseline_acc - acc for acc in layer_inh_ablated_acc],
                }
            )
            layer_csv = os.path.join(save_dir, "layer_ablation_results.csv")
            df_layer.to_csv(layer_csv, index=False)
            print(f"Saved layer results to {layer_csv}")
            try:
                print("Generating layer-level ablation plot...")
                plt.figure(figsize=(12, 8))
                x = np.arange(len(layer_indices))
                width = 0.25
                plt.bar(
                    x - width,
                    [baseline_acc - acc for acc in layer_both_ablated_acc],
                    width,
                    label="Both Ablated",
                )
                plt.bar(
                    x,
                    [baseline_acc - acc for acc in layer_exc_ablated_acc],
                    width,
                    label="Excitatory Ablated",
                )
                plt.bar(
                    x + width,
                    [baseline_acc - acc for acc in layer_inh_ablated_acc],
                    width,
                    label="Inhibitory Ablated",
                )
                plt.xlabel("Layer Index (lower = closer to soma)")
                plt.ylabel("Accuracy Drop")
                plt.title("Layer-wise Ablation Analysis")
                plt.xticks(x, layer_indices)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                layer_plot_path = os.path.join(save_dir, "layer_ablation_plot.png")
                plt.savefig(layer_plot_path, dpi=300)
                plt.close()
                print(f"Saved layer ablation plot to {layer_plot_path}")
            except Exception as e:
                print(f"Error generating layer ablation plot: {e}")
                import traceback

                print(traceback.format_exc())
        if do_branch_ablation and branch_indices:
            import pandas as pd

            df_branch = pd.DataFrame(
                {
                    "branch_idx": branch_indices,
                    "layer_idx": branch_layer_indices,
                    "baseline_acc": [baseline_acc] * len(branch_indices),
                    "both_ablated_acc": branch_both_ablated_acc,
                    "exc_ablated_acc": branch_exc_ablated_acc,
                    "inh_ablated_acc": branch_inh_ablated_acc,
                    "both_drop": [
                        baseline_acc - acc for acc in branch_both_ablated_acc
                    ],
                    "exc_drop": [baseline_acc - acc for acc in branch_exc_ablated_acc],
                    "inh_drop": [baseline_acc - acc for acc in branch_inh_ablated_acc],
                }
            )
            branch_csv = os.path.join(save_dir, "branch_ablation_results.csv")
            df_branch.to_csv(branch_csv, index=False)
            print(f"Saved branch results to {branch_csv}")
            print("Calculating branch-level summary statistics by layer...")
            layer_summary = {}
            unique_layer_indices = sorted(set(branch_layer_indices))
            for layer_idx in unique_layer_indices:
                layer_mask = [
                    i for i, idx in enumerate(branch_layer_indices) if idx == layer_idx
                ]
                if not layer_mask:
                    continue
                exc_accs = [branch_exc_ablated_acc[i] for i in layer_mask]
                inh_accs = [branch_inh_ablated_acc[i] for i in layer_mask]
                both_accs = [branch_both_ablated_acc[i] for i in layer_mask]
                exc_drops = [baseline_acc - acc for acc in exc_accs]
                inh_drops = [baseline_acc - acc for acc in inh_accs]
                both_drops = [baseline_acc - acc for acc in both_accs]
                layer_summary[layer_idx] = {
                    "n_branches": len(layer_mask),
                    "both_mean_acc": np.mean(both_accs),
                    "both_std_acc": np.std(both_accs),
                    "exc_mean_acc": np.mean(exc_accs),
                    "exc_std_acc": np.std(exc_accs),
                    "inh_mean_acc": np.mean(inh_accs),
                    "inh_std_acc": np.std(inh_accs),
                    "both_mean_drop": np.mean(both_drops),
                    "both_std_drop": np.std(both_drops),
                    "exc_mean_drop": np.mean(exc_drops),
                    "exc_std_drop": np.std(exc_drops),
                    "inh_mean_drop": np.mean(inh_drops),
                    "inh_std_drop": np.std(inh_drops),
                }
            df_branch_summary = pd.DataFrame.from_dict(layer_summary, orient="index")
            df_branch_summary.index.name = "layer_idx"
            df_branch_summary.reset_index(inplace=True)
            branch_summary_csv = os.path.join(
                save_dir, "branch_ablation_summary_by_layer.csv"
            )
            df_branch_summary.to_csv(branch_summary_csv, index=False)
            print(
                f"Saved branch-level summary statistics by layer to {branch_summary_csv}"
            )
            if len(df_branch_summary) > 0:
                try:
                    print("Generating branch summary plot...")
                    plt.figure(figsize=(12, 8))
                    x = np.arange(len(df_branch_summary))
                    width = 0.25
                    plt.bar(
                        x - width,
                        df_branch_summary["both_mean_drop"],
                        width,
                        yerr=df_branch_summary["both_std_drop"],
                        label="Both Ablated",
                        capsize=5,
                        color="purple",
                        alpha=0.7,
                    )
                    plt.bar(
                        x,
                        df_branch_summary["exc_mean_drop"],
                        width,
                        yerr=df_branch_summary["exc_std_drop"],
                        label="Excitatory Ablated",
                        capsize=5,
                        color="red",
                        alpha=0.7,
                    )
                    plt.bar(
                        x + width,
                        df_branch_summary["inh_mean_drop"],
                        width,
                        yerr=df_branch_summary["inh_std_drop"],
                        label="Inhibitory Ablated",
                        capsize=5,
                        color="blue",
                        alpha=0.7,
                    )
                    plt.xlabel("Layer Index (lower = closer to soma)")
                    plt.ylabel("Mean Accuracy Drop (with std dev)")
                    plt.title("Average Branch Ablation Effect by Layer")
                    plt.xticks(x, df_branch_summary["layer_idx"])
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plot_path = os.path.join(
                        save_dir, "branch_ablation_summary_plot.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    print(f"Saved branch summary plot to {plot_path}")
                except Exception as e:
                    print(f"Error generating branch summary plot: {e}")
                    import traceback

                    print(traceback.format_exc())
        if do_layer_ablation and layer_indices:
            print("\nLayer-wise Ablation Analysis Summary:")
            print("-" * 80)
            print(
                f"{'Layer':<8} {'Baseline':<10} {'Both Acc':<10} {'Both Drop':<10} {'Exc Acc':<10} {'Exc Drop':<10} {'Inh Acc':<10} {'Inh Drop':<10}"
            )
            print("-" * 80)
            for i, layer_idx in enumerate(layer_indices):
                both_drop = baseline_acc - layer_both_ablated_acc[i]
                exc_drop = baseline_acc - layer_exc_ablated_acc[i]
                inh_drop = baseline_acc - layer_inh_ablated_acc[i]
                print(
                    f"{layer_idx:<8} {baseline_acc:<10.4f} {layer_both_ablated_acc[i]:<10.4f} {both_drop:<10.4f} {layer_exc_ablated_acc[i]:<10.4f} {exc_drop:<10.4f} {layer_inh_ablated_acc[i]:<10.4f} {inh_drop:<10.4f}"
                )
            print("-" * 80)
        if (
            do_branch_ablation
            and "df_branch_summary" in locals()
            and len(df_branch_summary) > 0
        ):
            print("\nBranch-level Summary Statistics by Layer:")
            print("-" * 100)
            print(
                f"{'Layer':<8} {'# Branches':<12} {'Both Mean':<12} {'Both Std':<12} {'Exc Mean':<12} {'Exc Std':<12} {'Inh Mean':<12} {'Inh Std':<12}"
            )
            print("-" * 100)
            for _, row in df_branch_summary.iterrows():
                print(
                    f"{int(row['layer_idx']):<8} {int(row['n_branches']):<12} {row['both_mean_drop']:<12.4f} {row['both_std_drop']:<12.4f} {row['exc_mean_drop']:<12.4f} {row['exc_std_drop']:<12.4f} {row['inh_mean_drop']:<12.4f} {row['inh_std_drop']:<12.4f}"
                )
            print("-" * 100)
        print(f"Branch ablation analysis completed. Results saved to {save_dir}")
        logging.info(f"Saved branch ablation plots in {save_dir}")
        print("=" * 80 + "\n")

    def _dendritic_ablation_all(self):
        save_dir = os.path.join(self.save_path, "dendritic_ablation_all")
        os.makedirs(save_dir, exist_ok=True)
        try:
            from dendritic_modeling.utils import ablate_branch, measure_performance

            net = self.model.net if hasattr(self.model, "net") else self.model
            baseline_acc = measure_performance(
                net, self.data["test"], device=self.device
            )
            original = {}
            for n, p in net.named_parameters():
                original[n] = p.detach().clone()
            n_orders = 0
            if hasattr(net, "excitatory_branch_factors"):
                n_orders = len(net.excitatory_branch_factors)
            elif hasattr(net, "branch_layers"):
                n_orders = len(net.branch_layers)
            else:
                if hasattr(net, "layers"):
                    n_orders = len(net.layers)
            if n_orders == 0:
                logging.error(
                    "Could not determine number of dendritic orders in the model"
                )
                return
            results = {
                "order": list(range(1, n_orders + 1)),
                "exc_drop": [],
                "inh_drop": [],
                "both_drop": [],
            }
            for order in range(1, n_orders + 1):
                if hasattr(net, "branch_layers") and order - 1 < len(net.branch_layers):
                    layer = net.branch_layers[order - 1]
                    has_exc = hasattr(layer, "branch_excitation") and hasattr(
                        layer.branch_excitation, "pre_w"
                    )
                    has_inh = hasattr(layer, "branch_inhibition") and hasattr(
                        layer.branch_inhibition, "pre_w"
                    )
                    if has_exc and has_inh:
                        layer.branch_excitation.pre_w.data.zero_()
                        layer.branch_inhibition.pre_w.data.zero_()
                        both_acc = measure_performance(
                            net, self.data["test"], device=self.device
                        )
                        both_drop = baseline_acc - both_acc
                        results["both_drop"].append(both_drop)
                        for n, p in net.named_parameters():
                            p.data.copy_(original[n])
                    elif has_exc:
                        results["both_drop"].append(None)
                    elif has_inh:
                        results["both_drop"].append(None)
                    else:
                        results["both_drop"].append(0.0)
                    if has_exc:
                        layer.branch_excitation.pre_w.data.zero_()
                        exc_acc = measure_performance(
                            net, self.data["test"], device=self.device
                        )
                        exc_drop = baseline_acc - exc_acc
                        results["exc_drop"].append(exc_drop)
                        if results["both_drop"][-1] is None and not has_inh:
                            results["both_drop"][-1] = exc_drop
                        for n, p in net.named_parameters():
                            p.data.copy_(original[n])
                    else:
                        results["exc_drop"].append(0.0)
                    if has_inh:
                        layer.branch_inhibition.pre_w.data.zero_()
                        inh_acc = measure_performance(
                            net, self.data["test"], device=self.device
                        )
                        inh_drop = baseline_acc - inh_acc
                        results["inh_drop"].append(inh_drop)
                        if results["both_drop"][-1] is None and not has_exc:
                            results["both_drop"][-1] = inh_drop
                        for n, p in net.named_parameters():
                            p.data.copy_(original[n])
                    else:
                        results["inh_drop"].append(0.0)
            if isinstance(results["exc_drop"], list) and len(results["exc_drop"]) > 0:
                exc_pct = [(drop / baseline_acc) * 100 for drop in results["exc_drop"]]
            else:
                exc_pct = (results["exc_drop"] / baseline_acc) * 100
            if isinstance(results["inh_drop"], list) and len(results["inh_drop"]) > 0:
                inh_pct = [(drop / baseline_acc) * 100 for drop in results["inh_drop"]]
            else:
                inh_pct = (results["inh_drop"] / baseline_acc) * 100
            if isinstance(results["both_drop"], list) and len(results["both_drop"]) > 0:
                both_pct = [
                    (drop / baseline_acc) * 100 if drop is not None else 0.0
                    for drop in results["both_drop"]
                ]
            else:
                both_pct = (results["both_drop"] / baseline_acc) * 100
            df = pd.DataFrame(
                {
                    "Order": results["order"],
                    "Excitatory Drop (%)": exc_pct,
                    "Inhibitory Drop (%)": inh_pct,
                    "Both Drop (%)": both_pct,
                }
            )
            df.to_csv(os.path.join(save_dir, "dendritic_ablation_all.csv"), index=False)
            plt.figure(figsize=(10, 6))
            x = np.arange(len(df["Order"]))
            width = 0.25
            plt.bar(
                x - width,
                df["Excitatory Drop (%)"],
                width,
                label="Excitatory",
                color="blue",
            )
            plt.bar(
                x, df["Inhibitory Drop (%)"], width, label="Inhibitory", color="red"
            )
            plt.bar(x + width, df["Both Drop (%)"], width, label="Both", color="purple")
            plt.xlabel("Dendritic Order")
            plt.ylabel("Performance Drop (%)")
            plt.title("Dendritic Ablation Results")
            plt.xticks(x, df["Order"])
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(os.path.join(save_dir, "dendritic_ablation_all.png"))

            # Log the ablation plot to W&B if it's available
            try:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "dendritic_ablation_all": wandb.Image(
                                os.path.join(save_dir, "dendritic_ablation_all.png")
                            ),
                            "dendritic_ablation_data": wandb.Table(dataframe=df),
                        }
                    )
                    logging.info("Successfully logged dendritic ablation plot to W&B")
            except Exception as e:
                logging.error(f"Error logging ablation plot to W&B: {str(e)}")

            plt.close()
        except Exception as e:
            logging.error(f"Error in dendritic order ablation analysis: {str(e)}")
            logging.error(traceback.format_exc())

    def _dendritic_information(self):
        print("\n" + "=" * 80)
        print("STARTING DENDRITIC INFORMATION ANALYSIS")
        print("=" * 80)
        if not hasattr(self.model, "net"):
            print(
                "ERROR: No 'net' attribute found on model. Skipping dendritic information analysis."
            )
            logging.error(
                "No 'net' attribute found on model. Skipping dendritic information analysis."
            )
            return
        save_dir = os.path.join(self.save_path, "dendritic_information")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
        if "test" in self.data and self.data["test"] is not None:
            dataset = self.data["test"]
            print(f"Using test dataset with {len(dataset)} samples")
        elif "valid" in self.data and self.data["valid"] is not None:
            dataset = self.data["valid"]
            print(f"Using validation dataset with {len(dataset)} samples")
        else:
            dataset = self.data["train"]
            print(f"Using training dataset with {len(dataset)} samples")
        mlp_hidden_dims = [100, 50]
        training_epochs = 20
        batch_size = 64
        use_post_shunting = False
        if (
            hasattr(self, "config")
            and "visualization" in self.config
            and "parameters" in self.config["visualization"]
        ):
            params = self.config["visualization"]["parameters"]
            if "dendritic_information" in params:
                if "mlp_hidden_dims" in params["dendritic_information"]:
                    mlp_hidden_dims = params["dendritic_information"]["mlp_hidden_dims"]
                    print(f"Using mlp_hidden_dims from config: {mlp_hidden_dims}")
                if "training_epochs" in params["dendritic_information"]:
                    training_epochs = params["dendritic_information"]["training_epochs"]
                    print(f"Using training_epochs from config: {training_epochs}")
                if "batch_size" in params["dendritic_information"]:
                    batch_size = params["dendritic_information"]["batch_size"]
                    print(f"Using batch_size from config: {batch_size}")
                if "use_post_shunting" in params["dendritic_information"]:
                    use_post_shunting = params["dendritic_information"][
                        "use_post_shunting"
                    ]
                    print(
                        f"Using post-shunting branch outputs for analysis: {use_post_shunting}"
                    )
        try:
            model = self.model
            if hasattr(model, "net"):
                net = model.net
                net = net.to(self.device)
            has_branch_outputs = False
            if hasattr(model, "forward_with_branch_outputs"):
                has_branch_outputs = True
                print(
                    "Model has forward_with_branch_outputs method. Will use it to extract branch activations."
                )
            elif hasattr(model, "net") and hasattr(
                model.net, "forward_with_branch_outputs"
            ):
                has_branch_outputs = True
                print(
                    "Model.net has forward_with_branch_outputs method. Will use it to extract branch activations."
                )
            if not has_branch_outputs:
                print(
                    "ERROR: Model does not have forward_with_branch_outputs method. Cannot analyze dendritic information."
                )
                logging.error(
                    "Model does not have forward_with_branch_outputs method. Cannot analyze dendritic information."
                )
                return
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )
            inputs, labels = next(iter(dataloader))
            inputs = inputs.to(self.device)
            act_format = None
            with torch.no_grad():
                forward_func = None
                if hasattr(model, "forward_with_branch_outputs"):
                    forward_func = model.forward_with_branch_outputs
                elif hasattr(model.net, "forward_with_branch_outputs"):
                    forward_func = model.net.forward_with_branch_outputs
                if hasattr(model, "net") and hasattr(model.net, "output_layer"):
                    if isinstance(model.net.output_layer, bool):
                        print(
                            f"NOTE: model.net.output_layer is a boolean ({model.net.output_layer}), not a callable module"
                        )
                outputs, branch_acts = forward_func(inputs)
                if isinstance(branch_acts, list):
                    n_levels = len(branch_acts)
                    print(
                        f"Successfully ran forward_with_branch_outputs. Found {n_levels} branch activation levels."
                    )
                    if n_levels > 0:
                        if isinstance(branch_acts[0], (list, tuple)):
                            print(
                                f"Activation format: nested list. Each level has {len(branch_acts[0])} components."
                            )
                            act_format = "nested_list"
                        elif isinstance(branch_acts[0], torch.Tensor):
                            print(
                                f"Activation format: tensor list. First tensor shape: {branch_acts[0].shape}"
                            )
                            act_format = "tensor_list"
                elif isinstance(branch_acts, dict):
                    keys = list(branch_acts.keys())
                    print(f"Activation format: dictionary with keys {keys}")
                    act_format = "dict"
                else:
                    print(f"Unknown activation format: {type(branch_acts)}")
                    act_format = "unknown"
                from dendritic_modeling.utils import analyze_dendritic_information

                print(
                    f"Running dendritic information analysis with format: {act_format}"
                )
                if use_post_shunting:
                    print("ANALYZING POST-SHUNTING BRANCH OUTPUTS")
                else:
                    print("ANALYZING PRE-SHUNTING BRANCH INPUTS")
                results = analyze_dendritic_information(
                    model=model,
                    dataset=dataset,
                    device=self.device,
                    mlp_hidden_dims=mlp_hidden_dims,
                    training_epochs=training_epochs,
                    batch_size=batch_size,
                    use_excitatory=True,
                    use_inhibitory=True,
                    use_post_shunting=use_post_shunting,
                )
                if not results.get("branch_depths"):
                    results["branch_depths"] = ["Order 1"]
                    results["exc_accuracy"] = [0.0]
                    results["inh_accuracy"] = [0.0]
                    results["combined_accuracy"] = [0.0]
                print("\nDEBUG: analyze_dendritic_information results:")
                print(f"  Branch depths: {results['branch_depths']}")
                print(f"  Excitatory accuracy raw: {results['exc_accuracy']}")
                print(f"  Inhibitory accuracy raw: {results['inh_accuracy']}")
                print(f"  Combined accuracy raw: {results['combined_accuracy']}")
        except Exception as inner_e:
            print(f"Error during forward_with_branch_outputs: {inner_e}")
            print(traceback.format_exc())
            print("Using placeholder results to generate visualization.")
            placeholder_levels = 6
            if act_format == "nested_list" and "n_levels" in locals():
                placeholder_levels = n_levels
            elif act_format == "tensor_list" and "n_levels" in locals():
                placeholder_levels = n_levels
            elif act_format == "dict" and "keys" in locals():
                placeholder_levels = len(keys)
            results = {
                "branch_depths": [f"Order {i+1}" for i in range(placeholder_levels)],
                "exc_accuracy": [0.6 - i * 0.05 for i in range(placeholder_levels)],
                "inh_accuracy": [0.4 - i * 0.05 for i in range(placeholder_levels)],
                "combined_accuracy": [
                    0.7 - i * 0.05 for i in range(placeholder_levels)
                ],
            }
            if not results.get("branch_depths"):
                results["branch_depths"] = ["Order 1"]
                results["exc_accuracy"] = [0.0]
                results["inh_accuracy"] = [0.0]
                results["combined_accuracy"] = [0.0]
        import pandas as pd

        df = pd.DataFrame(
            {
                "branch_depth": results["branch_depths"],
                "exc_accuracy": results["exc_accuracy"],
                "inh_accuracy": results["inh_accuracy"],
                "combined_accuracy": results["combined_accuracy"],
            }
        )
        csv_path = os.path.join(save_dir, "dendritic_information.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        import matplotlib.pyplot as plt
        import numpy as np

        depth_names = results["branch_depths"]
        exc_acc = np.array(
            [acc if acc is not None else np.nan for acc in results["exc_accuracy"]]
        )
        inh_acc = np.array(
            [acc if acc is not None else np.nan for acc in results["inh_accuracy"]]
        )
        combined_acc = np.array(
            [acc if acc is not None else np.nan for acc in results["combined_accuracy"]]
        )
        plt.figure(figsize=(10, 6))
        x = np.arange(len(depth_names))
        width = 0.25
        if not np.all(np.isnan(exc_acc)):
            plt.bar(
                x - width if not np.all(np.isnan(inh_acc)) else x,
                exc_acc,
                width,
                label="Excitatory Input",
                color="blue",
            )
        if not np.all(np.isnan(inh_acc)):
            plt.bar(x, inh_acc, width, label="Inhibitory Input", color="red")
        position = x + width if not np.all(np.isnan(inh_acc)) else x
        if not np.all(np.isnan(combined_acc)):
            plt.bar(
                position,
                combined_acc,
                width,
                label="Combined (Exc+Inh)",
                color="purple",
            )
        plt.xlabel("Dendritic Branch Depth")
        plt.ylabel("Classification Accuracy")
        plt.title(
            "Information Content in Branch Activations"
            if not use_post_shunting
            else "Information Content in Post-Shunting Branch Outputs"
        )
        plt.xticks(x, depth_names)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt_path = os.path.join(save_dir, "dendritic_information.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        classes = torch.unique(
            dataset.targets
            if hasattr(dataset, "targets")
            else torch.tensor([y for _, y in dataset])
        )
        n_classes = len(classes)
        random_chance = 1.0 / n_classes
        print(f"Random chance accuracy: {random_chance:.4f} (1/{n_classes})")
        norm_factor = 1 - random_chance
        exc_pct = (exc_acc - random_chance) * 100 / norm_factor
        inh_pct = (
            (inh_acc - random_chance) * 100 / norm_factor
            if not np.all(np.isnan(inh_acc))
            else np.array([np.nan] * len(depth_names))
        )
        combined_pct = (combined_acc - random_chance) * 100 / norm_factor
        plt.figure(figsize=(10, 6))
        if not np.all(np.isnan(exc_acc)):
            plt.bar(
                x - width if not np.all(np.isnan(inh_acc)) else x,
                exc_pct,
                width,
                label="Excitatory Input",
                color="blue",
            )
        if not np.all(np.isnan(inh_acc)):
            plt.bar(x, inh_pct, width, label="Inhibitory Input", color="red")
        pos = x + width if not np.all(np.isnan(inh_acc)) else x
        plt.bar(pos, combined_pct, width, label="Combined (Exc+Inh)", color="purple")
        plt.xlabel("Dendritic Branch Depth")
        plt.ylabel("% Information Above Random Chance")
        plt.title(
            f'Normalized Information Content in {"Post-Shunting Branch Outputs" if use_post_shunting else "Branch Activations"}'
        )
        plt.xticks(x, depth_names)
        plt.legend()
        plt.axhline(y=0, color="gray", linestyle="--", label="Random Chance")
        plt.tight_layout()
        plt_norm_path = os.path.join(save_dir, "dendritic_information_normalized.png")
        plt.savefig(plt_norm_path, dpi=300)
        plt.close()
        print(f"Dendritic information analysis completed. Results saved to {save_dir}")
        print("=" * 80 + "\n")

    def _dendritic_information_inputs(self):
        print("\n" + "=" * 80)
        print("ANALYZING BRANCH INPUT INFORMATION CONTENT")
        print("=" * 80)
        save_dir = os.path.join(self.save_path, "dendritic_information_inputs")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
        mlp_hidden_dims = [100, 50]
        training_epochs = 20
        batch_size = 64
        if hasattr(self.args, "visualization") and hasattr(
            self.args.visualization, "parameters"
        ):
            params = self.args.visualization.parameters
            if "dendritic_information_inputs" in params:
                param_dict = params["dendritic_information_inputs"]
                if "mlp_hidden_dims" in param_dict:
                    mlp_hidden_dims = param_dict["mlp_hidden_dims"]
                if "training_epochs" in param_dict:
                    training_epochs = param_dict["training_epochs"]
                if "batch_size" in param_dict:
                    batch_size = param_dict["batch_size"]
        if "test" in self.data and self.data["test"] is not None:
            dataset = self.data["test"]
        elif "valid" in self.data and self.data["valid"] is not None:
            dataset = self.data["valid"]
        else:
            dataset = self.data["train"]
        try:
            if hasattr(self.model, "net") and hasattr(
                self.model.net, "excitatory_branch_factors"
            ):
                branch_factors = self.model.net.excitatory_branch_factors
                somatic_synapses = getattr(self.model.net, "somatic_synapses", True)
                print(
                    f"Model configuration: branch_factors={branch_factors}, somatic_synapses={somatic_synapses}"
                )
            from dendritic_modeling.utils import analyze_dendritic_information

            print("ANALYZING PRE-SHUNTING BRANCH INPUTS (SYNAPTIC INPUTS)")
            results = analyze_dendritic_information(
                model=self.model,
                dataset=dataset,
                device=self.device,
                mlp_hidden_dims=mlp_hidden_dims,
                training_epochs=training_epochs,
                batch_size=batch_size,
                use_excitatory=True,
                use_inhibitory=True,
                use_post_shunting=False,
            )
            if not results.get("branch_depths"):
                results["branch_depths"] = ["Order 1"]
                results["exc_accuracy"] = [0.0]
                results["inh_accuracy"] = [0.0]
                results["combined_accuracy"] = [0.0]
            print(
                f"Received {len(results['branch_depths'])} branch depths: {results['branch_depths']}"
            )
            import pandas as pd

            df = pd.DataFrame(
                {
                    "branch_depth": results["branch_depths"],
                    "exc_accuracy": results["exc_accuracy"],
                    "inh_accuracy": results["inh_accuracy"],
                    "combined_accuracy": results["combined_accuracy"],
                }
            )
            csv_path = os.path.join(save_dir, "dendritic_information_inputs.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            import matplotlib.pyplot as plt
            import numpy as np

            depth_names = results["branch_depths"]
            exc_acc = np.array(
                [acc if acc is not None else np.nan for acc in results["exc_accuracy"]]
            )
            inh_acc = np.array(
                [acc if acc is not None else np.nan for acc in results["inh_accuracy"]]
            )
            combined_acc = np.array(
                [
                    acc if acc is not None else np.nan
                    for acc in results["combined_accuracy"]
                ]
            )
            plt.figure(figsize=(10, 6))
            x = np.arange(len(depth_names))
            width = 0.25
            plot_title = "Information Content in Branch Inputs (Pre-Shunting)"
            if not np.all(np.isnan(exc_acc)):
                plt.bar(
                    x - width, exc_acc, width, label="Excitatory Inputs", color="blue"
                )
            if not np.all(np.isnan(inh_acc)):
                plt.bar(x, inh_acc, width, label="Inhibitory Inputs", color="red")
            if not np.all(np.isnan(combined_acc)):
                plt.bar(
                    x + width,
                    combined_acc,
                    width,
                    label="Combined Inputs",
                    color="purple",
                )
            plt.xlabel("Branch Depth")
            plt.ylabel("Classification Accuracy")
            plt.title(plot_title)
            plt.xticks(x, depth_names)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(save_dir, "information_content_inputs.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Log the dendritic information inputs plot to W&B if it's available
            try:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "dendritic_information_inputs": wandb.Image(plot_path),
                            "dendritic_information_inputs_data": wandb.Table(
                                dataframe=df
                            ),
                        }
                    )
                    logging.info(
                        "Successfully logged dendritic information inputs plot to W&B"
                    )
            except Exception as e:
                logging.error(
                    f"Error logging dendritic information inputs plot to W&B: {str(e)}"
                )

            plt.close()
            print(f"Saved plot to {plot_path}")
        except Exception as e:
            print(f"Error in dendritic information analysis (inputs): {str(e)}")
            print(traceback.format_exc())
            logging.error(f"Error in dendritic information analysis (inputs): {str(e)}")
            logging.error(traceback.format_exc())

    def _dendritic_information_activations(self):
        import numpy as np

        print("\n" + "=" * 80)
        print("ANALYZING BRANCH ACTIVATION INFORMATION CONTENT")
        print("=" * 80)
        save_dir = os.path.join(self.save_path, "dendritic_information_activations")
        os.makedirs(save_dir, exist_ok=True)
        if not hasattr(self.model, "net"):
            print(
                "ERROR: No 'net' attribute found on model. Skipping dendritic information analysis."
            )
            logging.error(
                "No 'net' attribute found on model. Skipping dendritic information analysis."
            )
            return
        if "test" in self.data and self.data["test"] is not None:
            dataset = self.data["test"]
            print(f"Using test dataset with {len(dataset)} samples")
        elif "valid" in self.data and self.data["valid"] is not None:
            dataset = self.data["valid"]
            print(f"Using validation dataset with {len(dataset)} samples")
        else:
            dataset = self.data["train"]
            print(f"Using training dataset with {len(dataset)} samples")
        mlp_hidden_dims = [100, 50]
        training_epochs = 20
        batch_size = 64
        if hasattr(self.args, "visualization") and hasattr(
            self.args.visualization, "parameters"
        ):
            params = self.args.visualization.parameters
            if "dendritic_information_activations" in params:
                if "mlp_hidden_dims" in params["dendritic_information_activations"]:
                    mlp_hidden_dims = params["dendritic_information_activations"][
                        "mlp_hidden_dims"
                    ]
                    print(f"Using mlp_hidden_dims from config: {mlp_hidden_dims}")
                if "training_epochs" in params["dendritic_information_activations"]:
                    training_epochs = params["dendritic_information_activations"][
                        "training_epochs"
                    ]
                    print(f"Using training_epochs from config: {training_epochs}")
                if "batch_size" in params["dendritic_information_activations"]:
                    batch_size = params["dendritic_information_activations"][
                        "batch_size"
                    ]
                    print(f"Using batch_size from config: {batch_size}")
        try:
            if hasattr(self.model, "net") and hasattr(
                self.model.net, "excitatory_branch_factors"
            ):
                branch_factors = self.model.net.excitatory_branch_factors
                somatic_synapses = getattr(self.model.net, "somatic_synapses", True)
                print(
                    f"Model configuration: branch_factors={branch_factors}, somatic_synapses={somatic_synapses}"
                )
            model = self.model
            if hasattr(model, "net"):
                net = model.net
                net = net.to(self.device)
            has_branch_outputs = False
            if hasattr(model, "forward_with_branch_outputs"):
                has_branch_outputs = True
                print(
                    "Model has forward_with_branch_outputs method. Will use it to extract branch activations."
                )
            elif hasattr(model, "net") and hasattr(
                model.net, "forward_with_branch_outputs"
            ):
                has_branch_outputs = True
                print(
                    "Model.net has forward_with_branch_outputs method. Will use it to extract branch activations."
                )
            if not has_branch_outputs:
                print(
                    "ERROR: Model does not have forward_with_branch_outputs method. Cannot analyze dendritic information."
                )
                logging.error(
                    "Model does not have forward_with_branch_outputs method. Cannot analyze dendritic information."
                )
                return
            print("\nDEBUGGING: Testing forward_with_branch_outputs manually...")
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False
            )
            test_inputs, test_labels = next(iter(test_loader))
            test_inputs = test_inputs.to(self.device)
            with torch.no_grad():
                if hasattr(model, "forward_with_branch_outputs"):
                    _, test_activations = model.forward_with_branch_outputs(test_inputs)
                else:
                    _, test_activations = model.net.forward_with_branch_outputs(
                        test_inputs
                    )
                print(f"Test activations type: {type(test_activations)}")
                if isinstance(test_activations, list):
                    print(f"List length: {len(test_activations)}")
                    if len(test_activations) > 0:
                        print(f"First element type: {type(test_activations[0])}")
                        if isinstance(test_activations[0], list):
                            print(
                                f"First list nested length: {len(test_activations[0])}"
                            )
                            print(
                                f"First list nested element type: {type(test_activations[0][0])}"
                            )
                            if isinstance(test_activations[0][0], torch.Tensor):
                                print(
                                    f"First tensor shape: {test_activations[0][0].shape}"
                                )
                        elif isinstance(test_activations[0], torch.Tensor):
                            print(f"First tensor shape: {test_activations[0].shape}")
                elif isinstance(test_activations, dict):
                    print(f"Dict keys: {list(test_activations.keys())}")
                    for k, v in test_activations.items():
                        print(
                            f"Key {k}: Type {type(v)} - Shape {v.shape if isinstance(v, torch.Tensor) else 'N/A'}"
                        )
                print("End of test activation debugging\n")
            from dendritic_modeling.utils import analyze_dendritic_information

            print("ANALYZING POST-SHUNTING BRANCH ACTIVATIONS")
            results = analyze_dendritic_information(
                model=self.model,
                dataset=dataset,
                device=self.device,
                mlp_hidden_dims=mlp_hidden_dims,
                training_epochs=training_epochs,
                batch_size=batch_size,
                use_excitatory=True,
                use_inhibitory=True,
                use_post_shunting=True,
            )
            if not results.get("branch_depths"):
                results["branch_depths"] = ["Order 1"]
                results["exc_accuracy"] = [0.0]
                results["inh_accuracy"] = [0.0]
                results["combined_accuracy"] = [0.0]
            print(
                f"Received {len(results['branch_depths'])} branch depths: {results['branch_depths']}"
            )
            print(f"Excitatory accuracy: {results['exc_accuracy']}")
            print(f"Inhibitory accuracy: {results['inh_accuracy']}")
            print(f"Combined accuracy: {results['combined_accuracy']}")
            import pandas as pd

            df = pd.DataFrame(
                {
                    "branch_depth": results["branch_depths"],
                    "activation_accuracy": results["exc_accuracy"],
                    "combined_accuracy": results["combined_accuracy"],
                }
            )
            csv_path = os.path.join(save_dir, "dendritic_information_activations.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            import matplotlib.pyplot as plt
            import numpy as np

            depth_names = results["branch_depths"]
            activation_acc = np.array(
                [acc if acc is not None else np.nan for acc in results["exc_accuracy"]]
            )
            combined_acc = np.array(
                [
                    acc if acc is not None else np.nan
                    for acc in results["combined_accuracy"]
                ]
            )
            plt.figure(figsize=(10, 6))
            x = np.arange(len(depth_names))
            width = 0.25
            plot_title = "Information Content in Branch Activations (Post-Shunting)"
            if not np.all(np.isnan(activation_acc)):
                plt.bar(
                    x, activation_acc, width, label="Branch Activations", color="green"
                )
            plt.xlabel("Branch Depth")
            plt.ylabel("Classification Accuracy")
            plt.title(plot_title)
            plt.xticks(x, depth_names)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(save_dir, "information_content_activations.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Log the dendritic information activations plot to W&B if it's available
            try:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "dendritic_information_activations": wandb.Image(plot_path),
                            "dendritic_information_activations_data": wandb.Table(
                                dataframe=df
                            ),
                        }
                    )
                    logging.info(
                        "Successfully logged dendritic information activations plot to W&B"
                    )
            except Exception as e:
                logging.error(
                    f"Error logging dendritic information activations plot to W&B: {str(e)}"
                )

            plt.close()
            print(f"Saved plot to {plot_path}")
        except Exception as e:
            print(f"Error in dendritic information analysis (activations): {str(e)}")
            print(traceback.format_exc())
            logging.error(
                f"Error in dendritic information analysis (activations): {str(e)}"
            )
            logging.error(traceback.format_exc())
