"""
Base training strategy classes.

This module implements the core Trainer class used across all training
strategies. It provides standard training loops, optimization, and optional
advanced features like regularization and pruning.
"""

import logging
import os
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
import torch.profiler as torch_profiler
import torch.utils.data
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dendritic_modeling.analysis.tools.synaptic_pruning import (
    SynapticPruningAnalyzer,
    integrate_pruning_with_turnover_analysis,
)
from dendritic_modeling.config.training import (
    OptimizerConfig,
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_loss_curves as plot_training_loss_curves,
)
from dendritic_modeling.training.loss.functions import LossFunction, get_loss_function
from dendritic_modeling.training.optimizers.factory import create_optimizer
from dendritic_modeling.training.regularization import (
    PruningManager,
    RegularizationManager,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_function: str = "mse",
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value=5,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        save_every_n_epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        enable_adaptive_clipping: bool = False,
        disable_hooks_with_amp: bool = False,
        # NEW: Optional enhanced features (all default to disabled)
        optimizer_config: Optional[OptimizerConfig] = None,
        regularization_config: Optional[RegularizationConfig] = None,
        pruning_config: Optional[PruningConfig] = None,
        reporting_config: Optional[ReportingConfig] = None,
        # Parameter groups configuration for differential learning rates
        param_groups_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        # Handle optimizer configuration (backward compatibility)
        if optimizer_config is not None:
            # Enhanced mode: create optimizer from config during training
            if optimizer is not None:
                logger.warning(
                    "Both optimizer and optimizer_config provided. Using optimizer_config."
                )
            self.optimizer_config = optimizer_config
            self.optimizer = None  # Will be created in train()
        else:
            # Standard mode: use provided optimizer
            if optimizer is None:
                raise ValueError(
                    "Either optimizer or optimizer_config must be provided"
                )
            self.optimizer = optimizer
            self.optimizer_config = None

        self.loss_function: LossFunction = get_loss_function(loss_function)
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grad_clip_value = grad_clip_value
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager

        # Initialize enhanced feature configs with safe defaults
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        self.reporting_config = reporting_config or ReportingConfig()
        self.param_groups_config = param_groups_config

        # Initialize managers only if features are enabled
        if (
            self.regularization_config.l1_weight > 0
            or self.regularization_config.l2_weight > 0
        ):
            self.regularization_manager = RegularizationManager(
                self.regularization_config
            )
            logger.info(
                f"Regularization enabled: L1={self.regularization_config.l1_weight}, L2={self.regularization_config.l2_weight}"
            )
        else:
            self.regularization_manager = None

        if self.pruning_config.enabled:
            self.pruning_manager = PruningManager(self.pruning_config)
            logger.info(
                f"Pruning enabled with threshold {self.pruning_config.threshold}"
            )
        else:
            self.pruning_manager = None

        # Initialize synaptic pruning analyzer
        self.synaptic_pruning_analyzer = None

        # Track original performance for comparison (needed for pruning)
        self.pre_pruning_performance = {}
        self.post_pruning_performance = {}

        # Profiling Support
        self.enable_profiling = kwargs.get("enable_profiling", False)
        self.profiler = None

        if self.enable_profiling:
            # Auto-disable profiling for small-scale/CPU runs to improve speed
            if not torch.cuda.is_available():
                logger.info("Profiling auto-disabled for CPU run to improve speed")
                self.enable_profiling = False
            elif self.batch_size < 32:
                logger.info(
                    "Profiling auto-disabled for small batch size to reduce overhead"
                )
                self.enable_profiling = False
            else:
                try:
                    profiling_output_dir = kwargs.get(
                        "profiling_output_dir", "outputs/profiling/"
                    )
                    os.makedirs(profiling_output_dir, exist_ok=True)

                    self.profiler = torch_profiler.profile(
                        activities=[
                            torch_profiler.ProfilerActivity.CPU,
                            torch_profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch_profiler.schedule(
                            wait=1, warmup=1, active=3, repeat=2
                        ),
                        on_trace_ready=torch_profiler.tensorboard_trace_handler(
                            profiling_output_dir
                        ),
                        record_shapes=True,
                        with_stack=True,
                    )
                    logger.info(
                        f"Profiling enabled - output directory: {profiling_output_dir}"
                    )
                except ImportError:
                    logger.warning("torch.profiler not available, profiling disabled")
                    self.enable_profiling = False

        # Mixed Precision Training Support
        self.enable_amp = kwargs.get("enable_amp", torch.cuda.is_available())
        if self.enable_amp and torch.cuda.is_available():
            try:
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("AMP not available, falling back to standard precision")
                self.enable_amp = False
        else:
            self.enable_amp = False

        # Hook Management for Performance
        self.enable_hooks = kwargs.get("enable_hooks", True)
        if not self.enable_hooks:
            logger.info("Hooks manually disabled")

        # Store hook handles for cleanup
        self.hook_handles = []

        # Adaptive gradient clipping
        self._enable_adaptive_clipping = enable_adaptive_clipping
        if self._enable_adaptive_clipping:
            logger.info("Adaptive gradient clipping enabled")
            # Store original learning rates for adaptive adjustment
            for param_group in self.optimizer.param_groups:
                param_group["_original_lr"] = param_group["lr"]

        # AMP compatibility for hooks
        self.disable_hooks_with_amp = disable_hooks_with_amp
        if self.disable_hooks_with_amp and self.enable_amp:
            logger.info("Hooks will be disabled when AMP is enabled for compatibility")

        # Checkpoint configuration
        self.save_every_n_epochs = save_every_n_epochs

        # Set up checkpoint directory structure
        if (
            checkpoint_dir is not None
            and save_every_n_epochs is not None
            and save_every_n_epochs > 0
        ):
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

        self.filename_prefix = ""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def register_conditional_hook(self, module, hook_fn, hook_type="forward"):
        """
        Register a hook only if hooks are enabled. This allows disabling hooks
        for maximum performance during large-scale experiments.

        Args:
            module: PyTorch module to attach hook to
            hook_fn: Hook function
            hook_type: 'forward', 'backward', or 'full_backward'

        Returns:
            Hook handle if registered, None otherwise
        """
        if not self.enable_hooks:
            return None

        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = module.register_backward_hook(hook_fn)
        elif hook_type == "full_backward":
            handle = module.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

        self.hook_handles.append(handle)
        return handle

    def remove_all_hooks(self):
        """Remove all registered hooks to clean up resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("All hooks removed")

    def _dataloaders(
        self, train_data: torch.utils.data.Dataset, valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        # Adaptive DataLoader settings based on dataset size and available resources
        dataset_size = len(train_data)

        # Optimize for small datasets (reduce multiprocessing overhead)
        if dataset_size < 10000:
            num_workers_train = 0  # No multiprocessing for small datasets
            num_workers_valid = 0
            pin_memory = False  # Skip pinning for small data
            persistent_workers = False
            logger.info("Using optimized DataLoader settings for small dataset")
        else:
            num_workers_train = 4  # Parallel data loading for large datasets
            num_workers_valid = 2
            pin_memory = torch.cuda.is_available()  # Only pin if GPU available
            persistent_workers = True

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers_train,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers_valid,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return train_loader, valid_loader

    def _initialize_attributes(self, model: BaseModel, total_epochs: int) -> None:
        if not hasattr(self, "device"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self.device)
        if not hasattr(self, "best_state_dict"):
            self.best_state_dict = deepcopy(model.state_dict())
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "best_performance"):
            self.best_performance = {}
        if not hasattr(self, "last_epoch_performance"):
            self.last_epoch_performance = {}
        if not hasattr(self, "epoch_counter"):
            self.epoch_counter = 0
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "total_epochs"):
            self.total_epochs = total_epochs
        self.patience_counter = 0

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        # Create optimizer from config if needed (enhanced mode)
        if self.optimizer_config is not None:
            # Check if we have param_groups configuration for differential learning rates
            param_groups_config = getattr(self, "param_groups_config", None)
            if param_groups_config and param_groups_config.get("split_params", False):
                # Use parameter groups with different learning rates (preserves original behavior)
                param_groups = model.get_param_groups(param_groups_config)
                self.optimizer = self._create_optimizer_from_param_groups(
                    param_groups, self.optimizer_config
                )
                logger.info(
                    f"Created optimizer from config with parameter groups: {self.optimizer_config.name}"
                )
            else:
                # Use single learning rate for all parameters
                self.optimizer = create_optimizer(
                    model.parameters(), self.optimizer_config
                )
                logger.info(
                    f"Created optimizer from config: {self.optimizer_config.name}"
                )

        # Initialize synaptic pruning analyzer if needed
        logger.info("Checking synaptic pruning analyzer initialization...")
        logger.info(f"  pruning_config.enabled: {self.pruning_config.enabled}")
        logger.info(f"  analysis_manager exists: {self.analysis_manager is not None}")

        if (
            self.pruning_config.enabled
            and self.analysis_manager
            and hasattr(self.analysis_manager, "config")
            and hasattr(self.analysis_manager.config, "synapse_turnover_analysis")
        ):
            turnover_config = self.analysis_manager.config.synapse_turnover_analysis
            synaptic_analysis_enabled = getattr(
                turnover_config, "enabled", False
            ) and getattr(turnover_config, "synaptic_analysis", False)

            logger.info(
                f"  turnover_config.enabled: {getattr(turnover_config, 'enabled', False)}"
            )
            logger.info(
                f"  turnover_config.synaptic_analysis: {getattr(turnover_config, 'synaptic_analysis', False)}"
            )
            logger.info(
                f"  detailed_branch_report: {self.reporting_config.detailed_branch_report}"
            )
            logger.info(f"  synaptic_analysis_enabled: {synaptic_analysis_enabled}")

            if (
                synaptic_analysis_enabled
                and self.reporting_config.detailed_branch_report
            ):
                try:
                    self.synaptic_pruning_analyzer = SynapticPruningAnalyzer(
                        model, snapshot_interval=1
                    )
                    logger.info("Successfully initialized synaptic pruning analyzer")
                except Exception as e:
                    logger.warning(
                        f"Could not initialize synaptic pruning analyzer: {e}"
                    )
                    self.synaptic_pruning_analyzer = None
            else:
                logger.info(
                    "  Synaptic pruning analyzer not initialized - conditions not met"
                )
                self.synaptic_pruning_analyzer = None
        else:
            logger.info(
                "  Synaptic pruning analyzer not initialized - basic conditions not met"
            )
            self.synaptic_pruning_analyzer = None

        # Log enhanced configuration if features are enabled
        self._log_enhanced_config()

        # Run training with optional regularization
        if self.regularization_manager:
            training_results = self._train_with_regularization(
                model, train_data, valid_data
            )
        else:
            training_results = self._train_standard(model, train_data, valid_data)

        # Apply pruning if enabled
        if self.pruning_config.enabled:
            pruning_results = self._apply_pruning_and_evaluate(
                model, train_data, valid_data
            )
            training_results.update(pruning_results)

        return training_results

    def _train_standard(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        """Standard training without regularization (original behavior)."""
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self.epoch_counter = 0
        train_losses = []
        valid_losses = []
        train_accs = []  # Add accuracy tracking
        valid_accs = []  # Add accuracy tracking

        self._initialize_attributes(model, self.epochs)

        # Start profiler if enabled
        if self.profiler is not None:
            self.profiler.start()

        train_losses, valid_losses, train_accs, valid_accs = self._run_training_loop(
            n_epochs=self.epochs,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_losses=train_losses,
            valid_losses=valid_losses,
            train_accs=train_accs,
            valid_accs=valid_accs,
            logger_info="[Standard] ",
            wandb_info={},
        )

        # Stop profiler if enabled
        if self.profiler is not None:
            self.profiler.stop()
            logger.info(
                "Profiling complete. View in TensorBoard: tensorboard --logdir=outputs/profiling/"
            )

        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)

        logger.info(
            f"The best validation loss was {self.best_loss} at epoch {self.best_epoch}"
        )

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accs": train_accs,  # Add to results
            "valid_accs": valid_accs,  # Add to results
        }

        self._loss_plotting(train_losses, valid_losses)

        return results

    def _train_with_regularization(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict[str, Any]:
        """Train the model with regularization."""
        # Store original loss function
        original_loss_function = self.loss_function

        # Wrap loss function to include regularization
        self.loss_function = self._create_regularized_loss_function(
            original_loss_function, model
        )

        # Run standard training
        results = self._train_standard(model, train_data, valid_data)

        # Restore original loss function
        self.loss_function = original_loss_function

        return results

    def _create_regularized_loss_function(self, original_loss_fn, model):
        """Create a loss function that includes regularization."""

        def regularized_loss(model_input, x_batch, y_batch):
            # Compute base loss
            base_loss = original_loss_fn(model_input, x_batch, y_batch)

            # Add regularization (pass parameter groups if available)
            param_groups = (
                getattr(self.optimizer, "param_groups", None)
                if hasattr(self, "optimizer") and self.optimizer
                else None
            )
            reg_loss = self.regularization_manager.compute_regularization_loss(
                model, param_groups
            )

            return base_loss + reg_loss

        return regularized_loss

    def _apply_pruning_and_evaluate(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict[str, Any]:
        """Apply pruning and evaluate performance."""
        logger.info("Starting pruning phase...")

        # Store pre-pruning performance
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        pre_pruning_loss, pre_pruning_acc = self._epoch_valid(model, valid_loader)
        self.pre_pruning_performance = {
            "loss": pre_pruning_loss,
            "accuracy": pre_pruning_acc,
        }

        # Record pre-pruning synaptic state
        if self.synaptic_pruning_analyzer:
            logger.info("Recording pre-pruning synaptic state...")
            self.synaptic_pruning_analyzer.record_pruning_event(
                model=model,
                epoch=self.epochs,  # Use final epoch
                pruning_stats={},  # Will be filled after pruning
                pre_pruning=True,
            )
            logger.info("Pre-pruning state recorded")
        else:
            logger.warning(
                "Synaptic pruning analyzer not available for pre-pruning recording"
            )

        # Apply pruning
        pruning_stats = self.pruning_manager.prune_model(model)

        # Record post-pruning synaptic state
        if self.synaptic_pruning_analyzer:
            logger.info("Recording post-pruning synaptic state...")
            self.synaptic_pruning_analyzer.record_pruning_event(
                model=model,
                epoch=self.epochs,
                pruning_stats=pruning_stats,
                pre_pruning=False,
            )
            logger.info("Post-pruning state recorded")
            logger.info(
                f"Total pruning events: {len(self.synaptic_pruning_analyzer.pruning_events)}"
            )
        else:
            logger.warning(
                "Synaptic pruning analyzer not available for post-pruning recording"
            )

        # Report pruning statistics
        if self.reporting_config.report_non_pruned:
            self.pruning_manager.print_pruning_report()

        # Save pruning statistics if requested
        if self.reporting_config.save_pruning_stats and self.save_path:
            stats_path = Path(self.save_path).parent / "pruning_stats.json"
            self.pruning_manager.save_pruning_stats(stats_path)

        # Re-evaluate after pruning
        if self.pruning_config.re_evaluate_after:
            post_pruning_loss, post_pruning_acc = self._epoch_valid(model, valid_loader)
            self.post_pruning_performance = {
                "loss": post_pruning_loss,
                "accuracy": post_pruning_acc,
            }

            # Log performance comparison
            logger.info(
                f"Pre-pruning performance: Loss={pre_pruning_loss:.4f}, Acc={pre_pruning_acc:.4f}"
            )
            logger.info(
                f"Post-pruning performance: Loss={post_pruning_loss:.4f}, Acc={post_pruning_acc:.4f}"
            )

            # Integrate with synapse turnover analysis if available
            if self.synaptic_pruning_analyzer and self.analysis_manager:
                integrate_pruning_with_turnover_analysis(
                    model, self.pruning_manager, self.analysis_manager, self.epochs
                )

                # Generate synaptic pruning plots if detailed reporting is enabled
                logger.info("Checking if pruning plots should be generated...")
                logger.info(
                    f"  detailed_branch_report: {self.reporting_config.detailed_branch_report}"
                )
                logger.info(f"  save_path exists: {self.save_path is not None}")

                if self.reporting_config.detailed_branch_report and self.save_path:
                    plots_path = Path(self.save_path).parent / "pruning_plots"
                    plots_path.mkdir(exist_ok=True)
                    logger.info(f"  Creating plots in: {plots_path}")

                    try:
                        # Generate the original per-branch plot
                        logger.info("  Generating per-branch plot...")
                        result1 = self.synaptic_pruning_analyzer.plot_active_synapses_per_branch(
                            save_path=str(plots_path),
                            filename="active_synapses_per_branch",
                        )
                        logger.info(f"  Per-branch plot result: {result1 is not None}")

                        # Generate the new summary plot with mean/std by layer
                        logger.info("  Generating summary plot...")
                        result2 = self.synaptic_pruning_analyzer.plot_active_synapses_summary_by_layer(
                            save_path=str(plots_path),
                            filename="active_synapses_summary_by_layer",
                        )
                        logger.info(f"  Summary plot result: {result2 is not None}")

                        logger.info(f"Pruning analysis plots saved to {plots_path}")
                    except Exception as e:
                        logger.error(f"Error generating pruning plots: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.info("  Pruning plots not generated - conditions not met")

            # If we have synaptic pruning analyzer but no analysis manager, still generate plots
            elif (
                self.synaptic_pruning_analyzer
                and self.reporting_config.detailed_branch_report
                and self.save_path
            ):
                plots_path = Path(self.save_path).parent / "pruning_plots"
                plots_path.mkdir(exist_ok=True)

                # Generate the plots
                self.synaptic_pruning_analyzer.plot_active_synapses_per_branch(
                    save_path=str(plots_path), filename="active_synapses_per_branch"
                )

                self.synaptic_pruning_analyzer.plot_active_synapses_summary_by_layer(
                    save_path=str(plots_path),
                    filename="active_synapses_summary_by_layer",
                )

                logger.info(f"Pruning analysis plots saved to {plots_path}")

            # Generate independent performance comparison plot
            if self.save_path:
                from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
                    plot_pruning_performance_comparison,
                )

                performance_comparison = {
                    "before_pruning": self.pre_pruning_performance,
                    "after_pruning": self.post_pruning_performance,
                }
                try:
                    plot_pruning_performance_comparison(
                        performance_comparison,
                        save_path=str(Path(self.save_path).parent),
                        filename="pruning_performance_comparison",
                    )
                    logger.info(
                        "Generated independent pruning performance comparison plot"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate independent performance plot: {e}"
                    )

            # Redo analysis after pruning if requested
            if (
                self.reporting_config.redo_analysis_after_pruning
                and self.analysis_manager
                and self.save_path
            ):
                self._redo_analysis_after_pruning(model, train_data, valid_data)

        return {
            "pruning_stats": pruning_stats,
            "pre_pruning_performance": self.pre_pruning_performance,
            "post_pruning_performance": self.post_pruning_performance,
        }

    def _redo_analysis_after_pruning(self, model: BaseModel, train_data, valid_data):
        """Redo all analysis after pruning for comprehensive comparison."""
        logger.info("Starting post-pruning analysis...")

        try:
            # Create a separate save path for post-pruning analysis
            post_pruning_analysis_path = Path(self.save_path).parent / "pruned_analysis"
            post_pruning_analysis_path.mkdir(exist_ok=True)

            # Collect performance data for comparison
            performance_comparison = {
                "before_pruning": getattr(self, "pre_pruning_performance", {}),
                "best_performance": getattr(self, "best_performance", {}),
                "last_epoch": getattr(self, "last_epoch_performance", {}),
                "after_pruning": getattr(self, "post_pruning_performance", {}),
            }

            # Add any additional metrics from the analysis manager if available
            if hasattr(self.analysis_manager, "performance_analyzer"):
                try:
                    # Get latest performance metrics
                    latest_performance = (
                        self.analysis_manager.performance_analyzer.get_latest_performance()
                    )
                    if latest_performance:
                        for key, value in latest_performance.items():
                            if key not in performance_comparison["after_pruning"]:
                                performance_comparison["after_pruning"][key] = value
                except Exception as e:
                    logger.warning(
                        f"Could not get latest performance from analyzer: {e}"
                    )

            # Generate performance comparison plot
            from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
                plot_pruning_performance_comparison,
            )

            performance_plot = plot_pruning_performance_comparison(
                performance_comparison,
                save_path=str(post_pruning_analysis_path),
                filename="performance_comparison",
            )

            if performance_plot:
                logger.info("Generated performance comparison plot")

            # Run all enabled analyses on the pruned model
            # This will create new analysis results in the pruned_analysis folder
            original_save_root = self.analysis_manager.save_root
            self.analysis_manager.save_root = str(post_pruning_analysis_path)

            try:
                # Run post-pruning analysis (final mode, not training mode)
                self.analysis_manager.run_analysis(
                    filename="post_pruning", training=False
                )
                logger.info(
                    f"Pruned model analysis completed and saved to {post_pruning_analysis_path}"
                )
            finally:
                # Restore original save root
                self.analysis_manager.save_root = original_save_root

        except Exception as e:
            logger.error(f"Error during post-pruning analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _log_enhanced_config(self):
        """Log enhanced configuration if any features are enabled."""
        if self.regularization_manager or self.pruning_manager:
            logger.info("Enhanced training features active:")
            if self.regularization_manager:
                logger.info(
                    f"  - Regularization: L1={self.regularization_config.l1_weight}, L2={self.regularization_config.l2_weight}"
                )
            if self.pruning_manager:
                logger.info(
                    f"  - Pruning: threshold={self.pruning_config.threshold}, dynamic={self.pruning_config.dynamic_pruning}"
                )
            if self.optimizer_config:
                logger.info(
                    f"  - Optimizer: {self.optimizer_config.name} (lr={self.optimizer_config.lr})"
                )

    def _create_optimizer_from_param_groups(
        self, param_groups: list, optimizer_config: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """Create optimizer from parameter groups with config settings."""
        optimizer_name = optimizer_config.name.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                param_groups,
                lr=optimizer_config.lr,  # This will be overridden by individual group lrs
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                param_groups,
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                momentum=optimizer_config.momentum,
            )
        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                param_groups,
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                momentum=optimizer_config.momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_accs: list[float],  # Added train_accs
        valid_accs: list[float],  # Added valid_accs
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> tuple[
        list[float], list[float], list[float], list[float]
    ]:  # Added train_accs, valid_accs
        if wandb_info is None:
            wandb_info = {}
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses, train_accs, valid_accs = (
                self._run_epoch(  # Updated return
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    train_accs=train_accs,  # Passed train_accs
                    valid_accs=valid_accs,  # Passed valid_accs
                    logger_info=logger_info,
                    wandb_info=wandb_info,
                )
            )

            if self._check_early_stopping():
                break

        return (
            train_losses,
            valid_losses,
            train_accs,
            valid_accs,
        )  # Added train_accs, valid_accs

    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_accs: list[float],
        valid_accs: list[float],
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        self.epoch_counter += 1

        train_loss, train_acc = self._epoch_train(model, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_loss, valid_acc = self._epoch_valid(model, valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Update last epoch performance (will be overwritten each epoch)
        self.last_epoch_performance = {"loss": valid_loss, "accuracy": valid_acc}

        self._update_best(model, valid_loss, valid_acc)
        self._analyze()
        self._save_checkpoint(model)
        self._tracking(train_loss, valid_loss, logger_info, wandb_info)
        return train_losses, valid_losses, train_accs, valid_accs

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float]:
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            if self.enable_amp:
                # Mixed precision forward pass
                with autocast():
                    loss = self.loss_function(model, x_batch, y_batch)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Check for NaNs in gradients after backward pass
                nan_detected = self._check_and_handle_nan_gradients(model)
                if nan_detected:
                    logger.warning(
                        "NaNs detected and handled in gradients during AMP training"
                    )

                if self.grad_clip_value:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Apply adaptive gradient clipping
                    self._apply_adaptive_gradient_clipping(model)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                loss = self.loss_function(model, x_batch, y_batch)

                self.optimizer.zero_grad()
                loss.backward()

                # Check for NaNs in gradients after backward pass
                nan_detected = self._check_and_handle_nan_gradients(model)
                if nan_detected:
                    logger.warning(
                        "NaNs detected and handled in gradients during standard training"
                    )

                # Apply adaptive gradient clipping
                self._apply_adaptive_gradient_clipping(model)
                self.optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy for classification tasks
            # Only compute accuracy if y_batch contains class labels (not images for autoencoder)
            if (
                hasattr(model, "forward") and y_batch.dim() == 1
            ):  # y_batch is 1D tensor of labels
                with torch.no_grad():
                    if self.enable_amp:
                        with autocast():
                            outputs = model(x_batch)
                    else:
                        outputs = model(x_batch)
                    if (
                        outputs.dim() > 1 and outputs.size(1) > 1
                    ):  # Multi-class classification
                        predicted = outputs.argmax(dim=1)
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

            # Step profiler if enabled
            if self.profiler is not None:
                self.profiler.step()

        avg_loss = train_loss / len(train_loader)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return avg_loss, accuracy

    def _check_and_handle_nan_gradients(self, model: BaseModel) -> bool:
        """
        Check for NaN values in gradients and handle them to prevent training crashes.

        Args:
            model: The model to check gradients for

        Returns:
            bool: True if NaNs were detected and handled, False otherwise
        """
        nan_detected = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_detected = True
                    # Replace NaNs with zeros to prevent propagation
                    param.grad = torch.nan_to_num(
                        param.grad, nan=0.0, posinf=1.0, neginf=-1.0
                    )

                    if (
                        hasattr(self, "print_every")
                        and self.epoch_counter % self.print_every == 0
                    ):
                        logger.warning(
                            f"NaNs detected in gradients for parameter: {name}"
                        )

        return nan_detected

    def _apply_adaptive_gradient_clipping(self, model: BaseModel) -> float:
        """
        Apply adaptive gradient clipping with dynamic learning rate adjustment.

        Args:
            model: The model to apply gradient clipping to

        Returns:
            float: The gradient norm before clipping
        """
        if not self.grad_clip_value:
            return 0.0

        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.grad_clip_value
        )

        # Adaptive gradient clipping and learning rate adjustment
        if (
            hasattr(self, "_enable_adaptive_clipping")
            and self._enable_adaptive_clipping
        ):
            exploding_threshold = (
                self.grad_clip_value * 2.0
            )  # Threshold for "exploding gradients"

            if grad_norm > exploding_threshold:
                # Reduce learning rate dynamically for exploding gradients
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= 0.5

                if (
                    hasattr(self, "print_every")
                    and self.epoch_counter % self.print_every == 0
                ):
                    logger.info(
                        f"High gradient norm {grad_norm:.4f} detected. Reducing learning rate."
                    )

            elif grad_norm < self.grad_clip_value * 0.1:
                # Slightly increase learning rate for very small gradients (vanishing)
                for param_group in self.optimizer.param_groups:
                    original_lr = getattr(
                        param_group, "_original_lr", param_group["lr"]
                    )
                    param_group["lr"] = min(param_group["lr"] * 1.01, original_lr)

        return grad_norm

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                if self.enable_amp:
                    with autocast():
                        loss = self.loss_function(model, x_batch, y_batch)
                        outputs = model(x_batch)
                else:
                    loss = self.loss_function(model, x_batch, y_batch)
                    outputs = model(x_batch)

                valid_loss += loss.item()

                # Calculate accuracy for classification tasks
                # Only compute accuracy if y_batch contains class labels (not images for autoencoder)
                if y_batch.dim() == 1:  # y_batch is 1D tensor of labels
                    if (
                        outputs.dim() > 1 and outputs.size(1) > 1
                    ):  # Multi-class classification
                        predicted = outputs.argmax(dim=1)
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

        avg_loss = valid_loss / len(valid_loader)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return avg_loss, accuracy

    def _update_best(
        self, model: BaseModel, valid_loss: float, valid_acc: Optional[float] = None
    ) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_dict = deepcopy(model.state_dict())

            # Update best performance metrics
            self.best_performance = {
                "loss": valid_loss,
                "accuracy": valid_acc if valid_acc is not None else 0.0,
            }

            self.patience_counter = 0
            if self.save_path and os.path.exists(self.save_path):
                torch.save(
                    self.best_state_dict,
                    os.path.join(
                        self.save_path, f"{self.filename_prefix}best_model.pt"
                    ),
                )
        else:
            self.patience_counter += 1

    def _analyze(self):
        if self.analysis_manager is not None:
            # Conditional import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager

            if isinstance(self.analysis_manager, AnalysisManager):
                # Check if any analysis is enabled for training
                config = self.analysis_manager.config
                training_analysis_enabled = any(
                    [
                        config.performance_analysis.enabled
                        and config.performance_analysis.training,
                        config.information_analysis.enabled
                        and config.information_analysis.training,
                        config.synapse_turnover_analysis.enabled
                        and config.synapse_turnover_analysis.training,
                        config.noise_perturbation_analysis.enabled
                        and config.noise_perturbation_analysis.training,
                        config.single_layer_contribution.enabled
                        and config.single_layer_contribution.training,
                        config.ablation_analysis.enabled
                        and config.ablation_analysis.training,
                        config.weight_analysis.enabled
                        and config.weight_analysis.training,
                        config.synaptic_activation_analysis.enabled
                        and config.synaptic_activation_analysis.training,
                        config.branch_activation_analysis.enabled
                        and config.branch_activation_analysis.training,
                        hasattr(config, "correlation_analysis")
                        and config.correlation_analysis.enabled
                        and config.correlation_analysis.training,
                    ]
                )

                if training_analysis_enabled:
                    self.analysis_manager.run_analysis(
                        filename=f"epoch{self.epoch_counter}", training=True
                    )

    def _save_checkpoint(self, model: BaseModel) -> None:
        """Save checkpoint if configured to do so."""
        if (
            self.checkpoint_dir is not None
            and self.save_every_n_epochs is not None
            and self.save_every_n_epochs > 0
            and self.epoch_counter % self.save_every_n_epochs == 0
        ):

            checkpoint = {
                "epoch": self.epoch_counter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "loss_function": self.loss_function._loss_name,
            }

            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.filename_prefix}checkpoint_epoch_{self.epoch_counter}.pt",
            )
            torch.save(checkpoint, checkpoint_path)

            if not self.suppress_prints:
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> None:
        if wandb_info is None:
            wandb_info = {}
        elapsed = time.time() - self.start_time
        if not self.suppress_prints and (self.epoch_counter % self.print_every == 0):
            logger_info = (
                logger_info
                + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                + f"train_loss={train_loss:.4f}, "
                + f"valid_loss={valid_loss:.4f}, "
                + f"elapsed={elapsed:.1f}s"
            )
            logger.info(logger_info)

        if wandb.run is not None:
            wandb_info["epoch"] = self.epoch_counter
            wandb_info["train_loss"] = train_loss
            wandb_info["valid_loss"] = valid_loss
            wandb_info["elapsed"] = elapsed
            wandb.log(wandb_info)

    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self, train_losses: list[float], valid_losses: list[float]
    ) -> None:
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                try:
                    # plot_training_loss_curves already imported at top
                    plot_training_loss_curves(
                        train_losses=train_losses,
                        valid_losses=valid_losses,
                        loss_name=self.loss_function._loss_name,
                        save_dir=self.save_path,
                        filename_prefix=self.filename_prefix,
                    )
                except Exception as e:
                    logger.warning(f"Could not plot loss curves: {e}")


__all__ = ["Trainer"]
