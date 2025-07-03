"""
Correlation analysis for studying how EI networks learn input correlations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
import os
import pandas as pd
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze synaptic correlations and tuning properties in EI networks."""
    
    def __init__(self, model, dataset, device='cpu'):
        """
        Initialize correlation analyzer.
        
        Args:
            model: Trained EINet model
            dataset: Dataset with correlation structure
            device: Device to run computations on
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def extract_synaptic_inputs(self, batch_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Extract synaptic inputs for all samples in the dataset.
        
        Returns:
            Dictionary with keys 'excitatory_inputs', 'inhibitory_inputs', 'labels'
        """
        all_exc_inputs = []
        all_inh_inputs = []
        all_labels = []
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                
                # Get transfer function outputs
                if hasattr(self.model.core_network, 'transfer_fn'):
                    exc_out, inh_out = self.model.core_network.transfer_fn(inputs)
                else:
                    # For models without explicit transfer function
                    exc_out = inputs
                    inh_out = inputs
                
                all_exc_inputs.append(exc_out.cpu().numpy())
                all_inh_inputs.append(inh_out.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return {
            'excitatory_inputs': np.vstack(all_exc_inputs),
            'inhibitory_inputs': np.vstack(all_inh_inputs),
            'labels': np.hstack(all_labels)
        }
    
    def compute_synaptic_correlations(self, synaptic_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute correlation matrices for E-E, E-I, and I-I synaptic inputs.
        Computes both total correlations and noise correlations (within-class).
        
        Args:
            synaptic_data: Dictionary with excitatory_inputs, inhibitory_inputs, labels
            
        Returns:
            Dictionary with correlation matrices (total and noise)
        """
        exc_inputs = synaptic_data['excitatory_inputs']
        inh_inputs = synaptic_data['inhibitory_inputs']
        labels = synaptic_data['labels']
        
        # Compute TOTAL correlation matrices (across all data)
        ee_corr_total = np.corrcoef(exc_inputs.T)  # E-E correlations
        ii_corr_total = np.corrcoef(inh_inputs.T)  # I-I correlations
        
        # E-I correlations (cross-correlation)
        ei_corr_total = np.zeros((exc_inputs.shape[1], inh_inputs.shape[1]))
        for i in range(exc_inputs.shape[1]):
            for j in range(inh_inputs.shape[1]):
                ei_corr_total[i, j] = pearsonr(exc_inputs[:, i], inh_inputs[:, j])[0]
        
        # Compute NOISE correlations (within-class correlations, averaged across classes)
        class0_mask = labels == 0
        class1_mask = labels == 1
        
        # E-E noise correlations
        ee_corr_class0 = np.corrcoef(exc_inputs[class0_mask].T)
        ee_corr_class1 = np.corrcoef(exc_inputs[class1_mask].T)
        ee_corr_noise = (ee_corr_class0 + ee_corr_class1) / 2
        
        # I-I noise correlations
        ii_corr_class0 = np.corrcoef(inh_inputs[class0_mask].T)
        ii_corr_class1 = np.corrcoef(inh_inputs[class1_mask].T)
        ii_corr_noise = (ii_corr_class0 + ii_corr_class1) / 2
        
        # E-I noise correlations
        ei_corr_noise = np.zeros((exc_inputs.shape[1], inh_inputs.shape[1]))
        for i in range(exc_inputs.shape[1]):
            for j in range(inh_inputs.shape[1]):
                corr0 = pearsonr(exc_inputs[class0_mask, i], inh_inputs[class0_mask, j])[0]
                corr1 = pearsonr(exc_inputs[class1_mask, i], inh_inputs[class1_mask, j])[0]
                ei_corr_noise[i, j] = (corr0 + corr1) / 2
        
        return {
            # Total correlations
            'ee_correlation_total': ee_corr_total,
            'ei_correlation_total': ei_corr_total,
            'ii_correlation_total': ii_corr_total,
            # Noise correlations
            'ee_correlation_noise': ee_corr_noise,
            'ei_correlation_noise': ei_corr_noise,
            'ii_correlation_noise': ii_corr_noise,
            # Per-class correlations for detailed analysis
            'ee_correlation_class0': ee_corr_class0,
            'ee_correlation_class1': ee_corr_class1,
            'ii_correlation_class0': ii_corr_class0,
            'ii_correlation_class1': ii_corr_class1
        }
    
    def compute_tuning_curves(self, synaptic_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute tuning curves (class selectivity) for each synapse.
        
        Tuning = mean(response|class1) - mean(response|class2)
        
        Returns:
            Dictionary with tuning curves for excitatory and inhibitory synapses
        """
        exc_inputs = synaptic_data['excitatory_inputs']
        inh_inputs = synaptic_data['inhibitory_inputs']
        labels = synaptic_data['labels']
        
        # Separate by class
        class0_mask = labels == 0
        class1_mask = labels == 1
        
        # Compute tuning for excitatory synapses
        exc_tuning = (
            np.mean(exc_inputs[class1_mask], axis=0) - 
            np.mean(exc_inputs[class0_mask], axis=0)
        )
        
        # Compute tuning for inhibitory synapses
        inh_tuning = (
            np.mean(inh_inputs[class1_mask], axis=0) - 
            np.mean(inh_inputs[class0_mask], axis=0)
        )
        
        return {
            'excitatory_tuning': exc_tuning,
            'inhibitory_tuning': inh_tuning
        }
    
    def get_learned_connectivity(self) -> Dict[str, torch.Tensor]:
        """
        Extract the learned synaptic weights from the model.
        
        Returns:
            Dictionary with weight matrices and connectivity patterns
        """
        weights = {}
        
        # Get EI layer (first layer after transfer)
        if hasattr(self.model.core_network, 'network_layers'):
            ei_layer = self.model.core_network.network_layers[0]
            
            # Extract TopK weights
            if hasattr(ei_layer, 'excitatory_cells'):
                # E-E connections
                if hasattr(ei_layer.excitatory_cells, 'excitatory_synapses'):
                    ee_topk = ei_layer.excitatory_cells.excitatory_synapses
                    weights['ee_weights'] = ee_topk.weight.detach().cpu()
                    weights['ee_indices'] = ee_topk.indices.detach().cpu()
                
                # E-I connections
                if hasattr(ei_layer.excitatory_cells, 'inhibitory_synapses'):
                    ei_topk = ei_layer.excitatory_cells.inhibitory_synapses
                    weights['ei_weights'] = ei_topk.weight.detach().cpu()
                    weights['ei_indices'] = ei_topk.indices.detach().cpu()
            
            # I-E connections
            if hasattr(ei_layer, 'inhibitory_cells') and ei_layer.inhibitory_cells is not None:
                if hasattr(ei_layer.inhibitory_cells, 'excitatory_synapses'):
                    ie_topk = ei_layer.inhibitory_cells.excitatory_synapses
                    weights['ie_weights'] = ie_topk.weight.detach().cpu()
                    weights['ie_indices'] = ie_topk.indices.detach().cpu()
        
        return weights
    
    def analyze_weight_input_correlations(
        self, 
        weights: Dict[str, torch.Tensor],
        input_corr_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze how learned weights relate to input correlations.
        
        Args:
            weights: Dictionary of weight matrices
            input_corr_matrix: Input correlation matrix
            
        Returns:
            Dictionary of correlation statistics
        """
        results = {}
        
        # For E-E connections
        if 'ee_weights' in weights and 'ee_indices' in weights:
            ee_weights = weights['ee_weights'].numpy()
            ee_indices = weights['ee_indices'].numpy()
            
            # Compute average correlation strength for connected vs unconnected pairs
            connected_corrs = []
            for neuron_idx in range(ee_indices.shape[0]):
                for synapse_idx, input_idx in enumerate(ee_indices[neuron_idx]):
                    if input_idx >= 0:  # Valid connection
                        # Get correlations between this input and all other connected inputs
                        for other_idx in ee_indices[neuron_idx]:
                            if other_idx >= 0 and other_idx != input_idx:
                                connected_corrs.append(
                                    input_corr_matrix[input_idx, other_idx]
                                )
            
            if connected_corrs:
                results['ee_mean_connected_correlation'] = np.mean(connected_corrs)
                results['ee_std_connected_correlation'] = np.std(connected_corrs)
        
        return results
    
    def plot_correlation_matrices(
        self, 
        correlations: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """Plot correlation matrices for E-E, E-I, and I-I connections (both total and noise)."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Total correlations
        # E-E total correlations
        sns.heatmap(
            correlations['ee_correlation_total'][:50, :50],  # Show subset
            ax=axes[0, 0],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[0, 0].set_title('E-E Total Correlations (subset)')
        axes[0, 0].set_xlabel('Excitatory Input')
        axes[0, 0].set_ylabel('Excitatory Input')
        
        # E-I total correlations
        sns.heatmap(
            correlations['ei_correlation_total'][:50, :50],
            ax=axes[0, 1],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[0, 1].set_title('E-I Total Correlations (subset)')
        axes[0, 1].set_xlabel('Inhibitory Input')
        axes[0, 1].set_ylabel('Excitatory Input')
        
        # I-I total correlations
        sns.heatmap(
            correlations['ii_correlation_total'][:50, :50],
            ax=axes[0, 2],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[0, 2].set_title('I-I Total Correlations (subset)')
        axes[0, 2].set_xlabel('Inhibitory Input')
        axes[0, 2].set_ylabel('Inhibitory Input')
        
        # Row 2: Noise correlations
        # E-E noise correlations
        sns.heatmap(
            correlations['ee_correlation_noise'][:50, :50],
            ax=axes[1, 0],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[1, 0].set_title('E-E Noise Correlations (subset)')
        axes[1, 0].set_xlabel('Excitatory Input')
        axes[1, 0].set_ylabel('Excitatory Input')
        
        # E-I noise correlations
        sns.heatmap(
            correlations['ei_correlation_noise'][:50, :50],
            ax=axes[1, 1],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[1, 1].set_title('E-I Noise Correlations (subset)')
        axes[1, 1].set_xlabel('Inhibitory Input')
        axes[1, 1].set_ylabel('Excitatory Input')
        
        # I-I noise correlations
        sns.heatmap(
            correlations['ii_correlation_noise'][:50, :50],
            ax=axes[1, 2],
            cmap='coolwarm',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            cbar_kws={'label': 'Correlation'}
        )
        axes[1, 2].set_title('I-I Noise Correlations (subset)')
        axes[1, 2].set_xlabel('Inhibitory Input')
        axes[1, 2].set_ylabel('Inhibitory Input')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tuning_curves(
        self,
        tuning: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """Plot tuning curves for excitatory and inhibitory synapses."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Excitatory tuning histogram
        axes[0, 0].hist(tuning['excitatory_tuning'], bins=50, alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Tuning (Class 1 - Class 0)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Excitatory Synapse Tuning Distribution')
        axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Inhibitory tuning histogram
        axes[0, 1].hist(tuning['inhibitory_tuning'], bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Tuning (Class 1 - Class 0)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Inhibitory Synapse Tuning Distribution')
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Scatter plot of E vs I tuning
        axes[1, 0].scatter(
            tuning['excitatory_tuning'][:100],
            tuning['inhibitory_tuning'][:100],
            alpha=0.5
        )
        axes[1, 0].set_xlabel('Excitatory Tuning')
        axes[1, 0].set_ylabel('Inhibitory Tuning')
        axes[1, 0].set_title('E vs I Tuning Correlation')
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # Tuning strength by input dimension
        d_third = len(tuning['excitatory_tuning']) // 3
        exc_tuning_by_region = [
            tuning['excitatory_tuning'][:d_third],
            tuning['excitatory_tuning'][d_third:2*d_third],
            tuning['excitatory_tuning'][2*d_third:]
        ]
        
        axes[1, 1].violinplot(exc_tuning_by_region, positions=[1, 2, 3])
        axes[1, 1].set_xlabel('Input Region')
        axes[1, 1].set_ylabel('Tuning Strength')
        axes[1, 1].set_title('Tuning by Input Region')
        axes[1, 1].set_xticks([1, 2, 3])
        axes[1, 1].set_xticklabels(['First Third\n(μ₁=-0.5, μ₂=0.5)', 
                                    'Second Third\n(μ₁=0, μ₂=0)', 
                                    'Last Third\n(μ₁=0.5, μ₂=-0.5)'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full correlation analysis.
        
        Args:
            save_dir: Directory to save results and plots
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting correlation analysis...")
        
        # Extract synaptic inputs
        logger.info("Extracting synaptic inputs...")
        synaptic_data = self.extract_synaptic_inputs()
        
        # Compute correlations
        logger.info("Computing synaptic correlations...")
        correlations = self.compute_synaptic_correlations(synaptic_data)
        
        # Compute tuning curves
        logger.info("Computing tuning curves...")
        tuning = self.compute_tuning_curves(synaptic_data)
        
        # Get learned weights
        logger.info("Extracting learned connectivity...")
        weights = self.get_learned_connectivity()
        
        # Analyze weight-correlation relationship
        if hasattr(self.dataset, 'get_correlation_matrix'):
            input_corr = self.dataset.get_correlation_matrix()
            weight_corr_stats = self.analyze_weight_input_correlations(weights, input_corr)
        else:
            weight_corr_stats = {}
        
        # Compile results
        results = {
            'correlations': correlations,
            'tuning': tuning,
            'weights': weights,
            'weight_correlation_stats': weight_corr_stats,
            'synaptic_data': synaptic_data
        }
        
        # Save results and plots if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save numerical results
            np.savez(
                os.path.join(save_dir, 'correlation_analysis_results.npz'),
                **{k: v for k, v in results.items() if isinstance(v, (np.ndarray, dict))}
            )
            
            # Generate plots
            self.plot_correlation_matrices(
                correlations,
                os.path.join(save_dir, 'correlation_matrices.png')
            )
            
            self.plot_tuning_curves(
                tuning,
                os.path.join(save_dir, 'tuning_curves.png')
            )
            
            # Save summary statistics
            summary = {
                'mean_ee_correlation_total': np.mean(np.abs(correlations['ee_correlation_total'][np.triu_indices_from(correlations['ee_correlation_total'], k=1)])),
                'mean_ei_correlation_total': np.mean(np.abs(correlations['ei_correlation_total'])),
                'mean_ii_correlation_total': np.mean(np.abs(correlations['ii_correlation_total'][np.triu_indices_from(correlations['ii_correlation_total'], k=1)])),
                'mean_ee_correlation_noise': np.mean(np.abs(correlations['ee_correlation_noise'][np.triu_indices_from(correlations['ee_correlation_noise'], k=1)])),
                'mean_ei_correlation_noise': np.mean(np.abs(correlations['ei_correlation_noise'])),
                'mean_ii_correlation_noise': np.mean(np.abs(correlations['ii_correlation_noise'][np.triu_indices_from(correlations['ii_correlation_noise'], k=1)])),
                'mean_exc_tuning': np.mean(np.abs(tuning['excitatory_tuning'])),
                'mean_inh_tuning': np.mean(np.abs(tuning['inhibitory_tuning'])),
                **weight_corr_stats
            }
            
            pd.DataFrame([summary]).to_csv(
                os.path.join(save_dir, 'correlation_summary.csv'),
                index=False
            )
            
            logger.info(f"Results saved to {save_dir}")
        
        return results 