"""
Analysis manager module.

The AnalysisManager coordinates analysis tools for performance metrics,
information analysis, and visualization generation.
"""

import logging
import os

import numpy as np

from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
from dendritic_modeling.analysis.core.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.metrics.layer_contribution import (
    SingleLayerContributionAnalyzer,
)
from dendritic_modeling.analysis.metrics.noise_perturbation import (
    NoisePerturbationAnalyzer,
)
from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.analysis.tools.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_performance_metrics,
    plot_synapse_turnover,
    plot_weight_distributions,
)
from dendritic_modeling.plotting.visualizations.information_analysis import (
    plot_information_analysis,
    InformationAnalysisPlotter,
)


class AnalysisManager:
    """Manager for running multiple analyses on models."""

    def __init__(
        self, model, data, analysis_config, save_root, device="cpu", train_config=None
    ):
        """Initialize with model, data, and configs."""
        self.model = model
        self.data = data
        self.config = analysis_config
        self.train_config = train_config or {}
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        # Note: Logging should be configured externally at the application startup.
        self.logger = logging.getLogger(__name__)

        # Store pruning results if available
        self.pruning_results = None

        # Initialize Performance Analysis
        if self.config.performance_analysis.enabled:
            self.performance_analyzer = PerformanceAnalyzer(
                self.config.performance_analysis.params
            )

        # Initialize Integrated Information Analysis
        if self.config.information_analysis.enabled:
            self.information_analyzer = InformationAnalyzer(
                self.config.information_analysis.params
            )

        # Initialize Synaptic Turnover Analysis
        if self.config.synapse_turnover_analysis.enabled:
            # Get snapshot interval from config (default to every 5 epochs for efficiency)
            snapshot_interval = getattr(
                self.config.synapse_turnover_analysis, "snapshot_interval", 5
            )
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(
                model=model, snapshot_interval=snapshot_interval
            )
            self.logger.info(
                f"Synaptic turnover analyzer initialized with snapshot interval: {snapshot_interval} epochs"
            )

        # Initialize Noise Perturbation Analysis
        if self.config.noise_perturbation_analysis.enabled:
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                self.config.noise_perturbation_analysis.params
            )

        # Initialize Single Layer Contribution Analysis (formerly layer_ablation)
        if self.config.single_layer_contribution.enabled:
            self.single_layer_analyzer = SingleLayerContributionAnalyzer(
                self.config.single_layer_contribution.params
            )

        # Initialize Ablation Analysis
        if self.config.ablation_analysis.enabled:
            self.ablation_analyzer = AblationAnalyzer(
                self.config.ablation_analysis.params
            )

        # Initialize Weight Analysis
        if self.config.weight_analysis.enabled:
            self.weight_analyzer = WeightAnalyzer(self.config.weight_analysis.params)

        # Initialize Synaptic Activation Analysis
        if self.config.synaptic_activation_analysis.enabled:
            self.synaptic_activation_analyzer = SynapticActivationAnalyzer(
                self.config.synaptic_activation_analysis.params
            )

        # Initialize Branch Activation Analysis
        if self.config.branch_activation_analysis.enabled:
            self.branch_activation_analyzer = BranchActivationAnalyzer(
                self.config.branch_activation_analysis.params
            )

        # Initialize Correlation Analysis
        if (
            hasattr(self.config, "correlation_analysis")
            and self.config.correlation_analysis.enabled
        ):
            # Note: CorrelationAnalyzer is initialized differently, needs dataset
            self.correlation_analysis_enabled = True
        else:
            self.correlation_analysis_enabled = False

    def set_pruning_results(self, pruning_results: dict):
        """Set pruning results for inclusion in performance plots."""
        self.pruning_results = pruning_results

    def run_analysis(self, filename: str = "final", training: bool = False):
        """Run configured analyses and generate plots."""
        # Only log for final analysis, not every training epoch
        if not training:
            self.logger.info(
                f"Starting analysis run - filename: {filename}, training: {training}"
            )

        # Run analyses and get results
        if self.config.performance_analysis.enabled and (
            (self.config.performance_analysis.training and training) or not training
        ):
            # Only log for final analysis, not every training epoch
            if not training:
                self.logger.info(
                    f"Running performance analysis - enabled: {self.config.performance_analysis.enabled}, training_mode: {self.config.performance_analysis.training}, is_training: {training}"
                )

            # Determine save path based on whether this is training or final analysis
            if training:
                # Save epoch results in a subfolder
                epoch_save_path = os.path.join(self.save_root, "performance", "epochs")
                os.makedirs(epoch_save_path, exist_ok=True)
            else:
                # Save final results in main performance folder
                epoch_save_path = os.path.join(self.save_root, "performance")

            perf_results = self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=epoch_save_path,
                filename=filename,
            )

            # Only generate plots when training is complete
            if not training:
                # Generate bar plots only for final results (when training=false)
                if not self.config.performance_analysis.training:
                    try:
                        plot_performance_metrics(
                            perf_results,
                            save_path=os.path.join(self.save_root, "performance"),
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error generating performance metrics plots: {e}"
                        )

                # Generate epoch evolution plots if training analysis was enabled
                if self.config.performance_analysis.training:
                    try:
                        from dendritic_modeling.plotting.visualizations.performance_plots import (
                            PerformancePlotter,
                        )

                        plotter = PerformancePlotter()

                        # Get pruning performance data if available
                        pruning_performance = self.pruning_results
                        final_epoch = None
                        if (
                            self.train_config
                            and hasattr(self.train_config, "main")
                            and hasattr(self.train_config.main, "trainer")
                            and hasattr(self.train_config.main.trainer, "epochs")
                        ):
                            final_epoch = self.train_config.main.trainer.epochs

                        # Ensure performance directory exists
                        performance_dir = os.path.join(
                            self.save_root, "performance", "epochs"
                        )
                        if os.path.exists(performance_dir):
                            plotter.plot_performance_evolution(
                                performance_dir=performance_dir,
                                save_path=os.path.join(self.save_root, "performance"),
                                pruning_performance=pruning_performance,
                                final_epoch=final_epoch,
                            )
                        else:
                            self.logger.warning(
                                f"Performance directory {performance_dir} does not exist, skipping evolution plots"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Error generating performance evolution plots: {e}"
                        )
                        # Continue with other analyses even if plotting fails

        # Information Analysis
        if self.config.information_analysis.enabled and (
            (self.config.information_analysis.training and training) or not training
        ):
            self.logger.info("Starting information analysis...")
            self.logger.info(f"Information analysis config: enabled={self.config.information_analysis.enabled}, training={self.config.information_analysis.training}, current_training_mode={training}")
            self.logger.info(f"Information analyzer type: {type(self.information_analyzer)}")
            self.logger.info(f"Model type: {type(self.model)}")
            self.logger.info(f"Test dataset length: {len(self.data['test'])}")
            
            try:
                info_results = self.information_analyzer.analyze(
                    model=self.model,
                    test_dataset=self.data["test"],
                    device=self.device,
                    save_path=os.path.join(self.save_root, "information_analysis"),
                    filename=filename,
                )
                self.logger.info("Information analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Information analysis failed: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                info_results = None
            if not training and info_results:
                plot_information_analysis(
                    info_results,
                    save_path=os.path.join(self.save_root, "information_analysis"),
                )

                # Initialize plotter for enhanced visualizations
                plotter = InformationAnalysisPlotter()

                # Generate dendritic depth plots based on enabled analysis modes
                if info_results and "layer_statistics" in info_results:
                    per_layer_enabled = info_results.get("per_layer_analysis_enabled", False)
                    per_einet_enabled = info_results.get("per_einet_analysis_enabled", False)
                    
                    if per_layer_enabled and per_einet_enabled:
                        self.logger.info("Both per_layer and per_einet analysis enabled - generating detailed comparison plots...")
                        # Generate structured plots grouped by major layer and network type
                        self._plot_structured_ei_network_analysis(
                        info_results["layer_statistics"],
                        save_path=os.path.join(self.save_root, "information_analysis"),
                        )
                    elif per_layer_enabled and not per_einet_enabled:
                        self.logger.info("per_layer analysis enabled, per_einet disabled - generating layer-wise plots...")
                        # Aggregate E/I networks, but keep layers separate
                        self._plot_layer_wise_aggregated_analysis(
                            info_results["layer_statistics"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                        )
                    elif not per_layer_enabled and per_einet_enabled:
                        self.logger.info("per_einet analysis enabled, per_layer disabled - generating network-wise plots...")
                        # Aggregate layers, but keep E/I networks separate
                        self._plot_network_wise_aggregated_analysis(
                            info_results["layer_statistics"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                        )
                    else:
                        self.logger.info("Both per_layer and per_einet analysis disabled - generating overall aggregated plots...")
                        # Aggregate both layers and E/I networks
                        self._plot_fully_aggregated_analysis(
                            info_results["layer_statistics"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                        )

                # Remove per_layer_enhanced plots as they're redundant and empty
                # The structured plots above provide the proper dendritic depth analysis

                # Plot per-EI-network analysis only if per_einet_analysis was actually enabled
                if (info_results and "per_einet_analysis" in info_results and 
                    info_results.get("per_einet_analysis_enabled", False)):
                    self.logger.info("per_einet_analysis is enabled - generating per-EI-network analysis plots...")
                    # Create separate plots for excitatory and inhibitory networks
                    ei_results = info_results["per_einet_analysis"]
                    
                    # Plot excitatory network results
                    if "excitatory_results" in ei_results:
                        plot_information_analysis(
                            ei_results["excitatory_results"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                            filename_prefix="excitatory_network",
                        )
                    
                    # Plot inhibitory network results
                    if "inhibitory_results" in ei_results:
                        plot_information_analysis(
                            ei_results["inhibitory_results"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                            filename_prefix="inhibitory_network",
                        )
                    
                    # Plot combined EI network results
                    if "combined_results" in ei_results:
                        plot_information_analysis(
                            ei_results["combined_results"],
                            save_path=os.path.join(self.save_root, "information_analysis"),
                            filename_prefix="combined_ei_network",
                        )

                # Plot individual layer EI-network analysis if available in per_layer_analysis
                if info_results and "per_layer_analysis" in info_results:
                    per_layer_data = info_results["per_layer_analysis"]
                    for layer_name, layer_results in per_layer_data.items():
                        if "per_einet_analysis" in layer_results:
                            self.logger.info(f"Generating EI-network plots for layer: {layer_name}")
                            layer_ei_results = layer_results["per_einet_analysis"]
                            
                            # Clean layer name for filename (remove special characters)
                            clean_layer_name = layer_name.replace(".", "_").replace("/", "_")
                            
                            # Plot excitatory network for this layer
                            if "excitatory_results" in layer_ei_results:
                                plot_information_analysis(
                                    layer_ei_results["excitatory_results"],
                                    save_path=os.path.join(self.save_root, "information_analysis"),
                                    filename_prefix=f"layer_{clean_layer_name}_excitatory",
                                )
                            
                            # Plot inhibitory network for this layer
                            if "inhibitory_results" in layer_ei_results:
                                plot_information_analysis(
                                    layer_ei_results["inhibitory_results"],
                                    save_path=os.path.join(self.save_root, "information_analysis"),
                                    filename_prefix=f"layer_{clean_layer_name}_inhibitory",
                                )
                            
                            # Plot combined network for this layer
                            if "combined_results" in layer_ei_results:
                                plot_information_analysis(
                                    layer_ei_results["combined_results"],
                                    save_path=os.path.join(self.save_root, "information_analysis"),
                                    filename_prefix=f"layer_{clean_layer_name}_combined",
                                )
                elif info_results and "per_einet_analysis" in info_results:
                    self.logger.info("per_einet_analysis is disabled - skipping per-EI-network analysis plots")

    def _plot_structured_ei_network_analysis(
        self, 
        layer_statistics: dict, 
        save_path: str
    ):
        """Generate structured plots comparing different layers with color-coded lines.
        
        Creates 2 plots:
        - Excitatory Networks: Layer 0 vs Layer 1 as different colored lines
        - Inhibitory Networks: Layer 0 vs Layer 1 as different colored lines
        X-axis: [Soma, Layer 1, Layer 2] (dendritic depth)
        """
        self.logger.info("Generating structured EI network plots with layer comparison...")
        
        # Group layers by major layer and network type
        layer_groups = {}
        for layer_name in layer_statistics.keys():
            parts = layer_name.split('.')
            if len(parts) >= 6:
                major_layer = parts[2]  # layers.0 or layers.1
                network_type = parts[3]  # inhibitory_cells or excitatory_cells  
                branch_layer = parts[5]  # branch_layers.0, branch_layers.1, branch_layers.2
                
                group_key = f"{major_layer}_{network_type}"
                if group_key not in layer_groups:
                    layer_groups[group_key] = {}
                
                layer_groups[group_key][branch_layer] = layer_statistics[layer_name]
        
        self.logger.info(f"Found {len(layer_groups)} major layer groups: {list(layer_groups.keys())}")
        
        # Reorganize by network type instead of individual layers
        network_comparison_data = {
            "excitatory": {},  # Will contain Layer 0 and Layer 1 data
            "inhibitory": {}   # Will contain Layer 0 and Layer 1 data
        }
        
        # Process each group and organize by network type
        for group_key, group_data in layer_groups.items():
            major_layer, network_type = group_key.split('_', 1)
            
            # Create ordered layer data for this group (only include existing branches)
            ordered_data = {}
            branch_to_label = {
                "0": "Soma",
                "1": "Layer 1", 
                "2": "Layer 2"
            }
            
            # Only process branches that actually exist in the data
            for branch_idx in sorted(group_data.keys()):  # Use available branches only
                if branch_idx in branch_to_label:
                    label = branch_to_label[branch_idx]
                    ordered_data[label] = group_data[branch_idx]
            
            if ordered_data:
                # Get clean network type name
                net_type_clean = network_type.split('_')[0]  # "excitatory" or "inhibitory"
                layer_label = f"Layer {major_layer[-1]}"  # "Layer 0" or "Layer 1"
                
                # Store data organized for comparison plotting
                if net_type_clean not in network_comparison_data:
                    network_comparison_data[net_type_clean] = {}
                
                network_comparison_data[net_type_clean][layer_label] = ordered_data
        
        # Generate comparison plots
        self._generate_layer_comparison_plots(network_comparison_data, save_path)

    def _generate_layer_comparison_plots(self, network_comparison_data: dict, save_path: str):
        """Generate plots comparing Layer 0 vs Layer 1 for each network type."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Define colors for different layers
        layer_colors = {
            "Layer 0": "#1f77b4",  # Blue
            "Layer 1": "#ff7f0e",  # Orange
        }
        
        # Information metrics to plot
        metric_types = [
            ("basic_mi", ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E,I;C)"]),
            ("pairwise_mi", ["I(E;I)", "I(E;Vout)", "I(I;Vout)"]),
            ("conditional_mi", ["I(E;I|C)", "I(E;Vout|C)", "I(I;Vout|C)"])
        ]
        
        for network_type, layer_data in network_comparison_data.items():
            self.logger.info(f"Generating comparison plots for {network_type} networks")
            
            for metric_category, metrics in metric_types:
                # Create figure with subplots for each metric
                n_metrics = len(metrics)
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, metric in enumerate(metrics):
                    ax = axes[idx] if idx < len(axes) else axes[-1]
                    
                    # First, determine all available depths from all layers
                    all_available_depths = set()
                    for layer_name, depth_data in layer_data.items():
                        all_available_depths.update(depth_data.keys())
                    
                    # Sort depths in logical order
                    depth_order_map = {"Soma": 0, "Layer 1": 1, "Layer 2": 2}
                    sorted_depths = sorted(all_available_depths, key=lambda x: depth_order_map.get(x, 99))
                    
                    # Plot each layer as a different colored line
                    for layer_name, depth_data in layer_data.items():
                        if not depth_data:
                            continue
                            
                        # Extract dendritic depths and metric values
                        depths = []
                        values = []
                        errors = []
                        
                        # Use the sorted depths that were determined for all layers
                        for depth_label in sorted_depths:
                            if depth_label in depth_data:
                                depth_stats = depth_data[depth_label]
                                if f"{metric}_mean" in depth_stats:
                                    depths.append(depth_label)
                                    values.append(depth_stats[f"{metric}_mean"])
                                    # Use std/5 for smaller error bars
                                    errors.append(depth_stats.get(f"{metric}_std", 0) / 5.0)
                        
                        if depths and values:
                            x_pos = np.arange(len(depths))
                            color = layer_colors.get(layer_name, "#808080")
                            
                            # Plot line with error bars
                            ax.errorbar(
                                x_pos, values, yerr=errors,
                                marker='o', linewidth=2, markersize=6,
                                label=layer_name, color=color, capsize=3
                            )
                            
                            # Fill between for variance
                            ax.fill_between(
                                x_pos,
                                [v - e for v, e in zip(values, errors)],
                                [v + e for v, e in zip(values, errors)],
                                alpha=0.2, color=color
                            )
                    
                    # Customize subplot using the actual depths found in the data
                    if sorted_depths:  # Only customize if we have data
                        ax.set_xlabel("Dendritic Depth")
                        ax.set_ylabel(f"{metric} (bits)")
                        ax.set_title(f"{metric}")
                        ax.set_xticks(range(len(sorted_depths)))
                        ax.set_xticklabels(sorted_depths, rotation=0)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                    else:
                        ax.text(0.5, 0.5, f"No data for {metric}", 
                               ha='center', va='center', transform=ax.transAxes)
                
                # Remove empty subplots
                for idx in range(len(metrics), len(axes)):
                    fig.delaxes(axes[idx])
                
                # Set overall title and save
                fig.suptitle(f"{network_type.title()} Networks: {metric_category.replace('_', ' ').title()} vs Dendritic Depth", fontsize=16)
                plt.tight_layout()
                
                filename = f"layer_comparison_{network_type}_{metric_category}_depth.png"
                filepath = os.path.join(save_path, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved comparison plot: {filename}")

    def _plot_fully_aggregated_analysis(self, layer_statistics: dict, save_path: str):
        """Generate fully aggregated plots when both per_layer and per_einet analysis are disabled.
        
        This creates plots that:
        - Aggregate across all layers (Layer 0 + Layer 1)
        - Aggregate across E/I networks (Excitatory + Inhibitory)
        - Show dendritic depth (Soma, Layer 1, Layer 2) on x-axis
        - Display one line representing the overall network
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        self.logger.info("Generating fully aggregated dendritic depth plots...")
        
        # Aggregate data across all layers and E/I networks
        aggregated_data = {}
        
        # First, collect all depth labels that exist across all entries
        all_depths = set()
        for layer_name, layer_data in layer_statistics.items():
            if isinstance(layer_data, dict):
                # Extract depth labels - convert from branch_layers.X to readable names
                for depth_key in layer_data.keys():
                    if depth_key.endswith('_mean'):
                        metric_name = depth_key.replace('_mean', '')
                        # Determine depth label from layer structure
                        if 'branch_layers.0' in layer_name:
                            depth_label = 'Soma'
                        elif 'branch_layers.1' in layer_name:
                            depth_label = 'Layer 1'
                        elif 'branch_layers.2' in layer_name:
                            depth_label = 'Layer 2'
                        else:
                            continue
                        
                        all_depths.add(depth_label)
                        
                        # Initialize aggregated structure
                        if depth_label not in aggregated_data:
                            aggregated_data[depth_label] = {}
                        if metric_name not in aggregated_data[depth_label]:
                            aggregated_data[depth_label][metric_name] = {'values': [], 'stds': []}
                        
                        # Collect values and stds for averaging
                        mean_val = layer_data.get(f'{metric_name}_mean', 0)
                        std_val = layer_data.get(f'{metric_name}_std', 0)
                        
                        aggregated_data[depth_label][metric_name]['values'].append(mean_val)
                        aggregated_data[depth_label][metric_name]['stds'].append(std_val)
        
        # Compute final aggregated statistics
        final_aggregated = {}
        for depth_label, depth_data in aggregated_data.items():
            final_aggregated[depth_label] = {}
            for metric_name, metric_data in depth_data.items():
                # Average across all layers and networks
                final_aggregated[depth_label][f'{metric_name}_mean'] = np.mean(metric_data['values'])
                # Combine standard deviations (root mean square)
                final_aggregated[depth_label][f'{metric_name}_std'] = np.sqrt(np.mean(np.array(metric_data['stds'])**2))
        
        # Sort depths in logical order
        depth_order_map = {"Soma": 0, "Layer 1": 1, "Layer 2": 2}
        sorted_depths = sorted(all_depths, key=lambda x: depth_order_map.get(x, 99))
        
        # Define metric categories
        metric_types = [
            ("basic_mi", ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E,I;C)"]),
            ("pairwise_mi", ["I(E;I)", "I(E;Vout)", "I(I;Vout)"]),
            ("conditional_mi", ["I(E;I|C)", "I(E;Vout|C)", "I(I;Vout|C)"])
        ]
        
        # Generate plots for each metric category
        for metric_category, metrics in metric_types:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx] if idx < len(axes) else axes[-1]
                
                # Extract values for this metric across depths
                depths = []
                values = []
                errors = []
                
                for depth_label in sorted_depths:
                    if depth_label in final_aggregated:
                        depth_stats = final_aggregated[depth_label]
                        if f"{metric}_mean" in depth_stats:
                            depths.append(depth_label)
                            values.append(depth_stats[f"{metric}_mean"])
                            errors.append(depth_stats.get(f"{metric}_std", 0) / 5.0)  # Smaller error bars
                
                if depths and values:
                    x_pos = np.arange(len(depths))
                    
                    # Plot single line representing aggregated network
                    ax.errorbar(
                        x_pos, values, yerr=errors,
                        marker='o', linewidth=2, markersize=6,
                        label='All Networks', color='#1f77b4', capsize=3
                    )
                    
                    # Fill between for variance
                    ax.fill_between(
                        x_pos,
                        [v - e for v, e in zip(values, errors)],
                        [v + e for v, e in zip(values, errors)],
                        alpha=0.2, color='#1f77b4'
                    )
                    
                    ax.set_xlabel("Dendritic Depth")
                    ax.set_ylabel(f"{metric} (bits)")
                    ax.set_title(f"{metric}")
                    ax.set_xticks(range(len(depths)))
                    ax.set_xticklabels(depths, rotation=0)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f"No data for {metric}",
                           ha='center', va='center', transform=ax.transAxes)
            
            # Remove empty subplots
            for idx in range(len(metrics), len(axes)):
                fig.delaxes(axes[idx])
            
            fig.suptitle(f"Information Analysis: {metric_category.replace('_', ' ').title()} vs Dendritic Depth", fontsize=16)
            plt.tight_layout()
            
            filename = f"aggregated_{metric_category}_depth.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved aggregated plot: {filename}")
