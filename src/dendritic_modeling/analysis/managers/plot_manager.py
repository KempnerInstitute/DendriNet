"""
Refactored plot manager for dendritic models.

This module contains a slimmed-down PlotManager class that delegates to the
organized analysis and plotting modules, providing backward compatibility
while maintaining clean separation of concerns.
"""

import logging
import os
import traceback

from dendritic_modeling.plotting.analysis import plot_einet_activations
from dendritic_modeling.plotting.network import (
    plot_dendrinet_gradients,
    plot_einet_params,
)

# from dendritic_modeling.analysis.branch_interactive import InteractiveBranchAnalyzer
# from dendritic_modeling.analysis.dendrite_comprehensive import ComprehensiveDendriticAnalyzer

logger = logging.getLogger(__name__)

class PlotManager:
    """
    Refactored plot manager for dendritic model analysis and visualization.

    This class provides a unified interface for all visualization and analysis
    functions, while delegating actual implementation to organized modules.
    Maintains backward compatibility with the original monolithic PlotManager.

    Args:
        model: The trained dendritic network model
        data: Dictionary containing 'train', 'valid', 'test' datasets
        args: Configuration arguments
        device: Device to run analysis on
    """

    def __init__(self, model, data, args, device="cpu"):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.save_path = args.save_path
        self.file_name = getattr(args, 'file_name', None)

        os.makedirs(self.save_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.save_path, "visualization.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Initialize specialized analyzers
        # self.branch_analyzer = InteractiveBranchAnalyzer(device=device)
        # self.dendritic_analyzer = ComprehensiveDendriticAnalyzer(device=device)

        # Define available plot methods
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
        }

    def run_plot(self, plot_name, epoch=None):
        """
        Execute a specific analysis or visualization.

        Args:
            plot_name: Name of the analysis/plot to run
            epoch: Optional epoch number for logging
        """
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

    # Plotting methods - delegate to plotting package
    def _plot_weights(self):
        """Generate weight visualizations - delegates to plotting.network module."""
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "param_visuals")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{self.file_name}_params.png" if self.file_name else "params.png")
            plot_einet_params(
                model=self.model.net,
                save_path=save_path,
                show=False
            )
            logging.info(f"Saved parameter visualizations in {save_path}")
        else:
            logging.info("No 'net' attribute found on model. Skipping _plot_weights.")

    def _plot_activations(self):
        """Generate activation visualizations - delegates to plotting.analysis module."""
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "activation_visuals")
            os.makedirs(save_dir, exist_ok=True)

            train_input, train_label = self.data["train"][:]
            input_list = split_mnist_inputs(train_input, train_label)

            plot_einet_activations(
                einet=self.model.net,
                input_list=input_list,
                save_path=save_dir,
                filename=self.file_name if self.file_name else "activations"
            )
            logging.info(f"Saved activation visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute on model. Skipping _plot_activations.")

    def _plot_gradients(self):
        """Generate gradient visualizations - delegates to plotting.network module."""
        if hasattr(self.model, "net"):
            save_dir = os.path.join(self.save_path, "grad_visuals")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{self.file_name}_gradients.png" if self.file_name else "gradients.png")
            plot_dendrinet_gradients(
                model=self.model,
                save_path=save_path,
                show=False
            )
            logging.info(f"Saved gradient visualizations in {save_path}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient plotting.")

    def _plot_profiles(self):
        """Generate profile visualizations - NOT IMPLEMENTED."""
        logging.info("Profile plotting is not currently implemented.")
        # TODO: Implement profile visualization if needed

    # Branch analysis methods - delegate to InteractiveBranchAnalyzer
    def _branch_info(self):
        """Analyze branch information content - delegates to analysis.branches.interactive module."""
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        self.branch_analyzer.analyze_branch_info(
            model=self.model,
            data=self.data,
            save_path=save_dir,
            file_name=self.file_name
        )

    def _branch_ablation(self):
        """Perform branch ablation analysis - delegates to analysis.branches.interactive module."""
        save_dir = os.path.join(self.save_path, "branch_ablation_plots")
        self.branch_analyzer.analyze_branch_ablation(
            model=self.model,
            data=self.data,
            save_path=save_dir,
            args=self.args
        )

    # Dendritic analysis methods - delegate to ComprehensiveDendriticAnalyzer
    def _dendritic_ablation_all(self):
        """Perform comprehensive dendritic ablation - delegates to analysis.dendritic.comprehensive module."""
        save_dir = os.path.join(self.save_path, "dendritic_ablation_plots")
        self.dendritic_analyzer.analyze_dendritic_ablation_all(
            model=self.model,
            data=self.data,
            save_path=save_dir
        )

    def _dendritic_information(self):
        """Analyze dendritic information processing - delegates to analysis.dendritic.comprehensive module."""
        save_dir = os.path.join(self.save_path, "dendritic_information_plots")
        self.dendritic_analyzer.analyze_dendritic_information(
            model=self.model,
            data=self.data,
            save_path=save_dir
        )

    def _dendritic_information_inputs(self):
        """Analyze dendritic input information - delegates to analysis.dendritic.comprehensive module."""
        save_dir = os.path.join(self.save_path, "dendritic_input_info_plots")
        self.dendritic_analyzer.analyze_dendritic_information_inputs(
            model=self.model,
            data=self.data,
            save_path=save_dir
        )

    def _dendritic_information_activations(self):
        """Analyze dendritic activation information - delegates to analysis.dendritic.comprehensive module."""
        save_dir = os.path.join(self.save_path, "dendritic_activation_info_plots")
        self.dendritic_analyzer.analyze_dendritic_information_activations(
            model=self.model,
            data=self.data,
            save_path=save_dir
        )

__all__ = ["PlotManager"]
