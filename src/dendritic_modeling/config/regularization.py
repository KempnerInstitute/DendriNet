"""
regularization.py
-----------------
Configuration classes for regularization and pruning.
"""

from dataclasses import dataclass, field

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.choices import normalize_choice

EI_RATIO_SCOPES = ("per_branch", "per_layer", "global")
EI_RATIO_METRICS = ("mean", "median", "rms")


@dataclass
class SelectiveConfig(BaseConfig):
    """Configuration for selective application of regularization/pruning."""

    layers: list[int] = field(
        default_factory=list
    )  # List of layer indices (empty = all)
    branches: list[str] = field(
        default_factory=list
    )  # List of branch names (empty = all)
    weight_types: list[str] = field(default_factory=lambda: ["all"])  # Types of weights


@dataclass
class RegularizationConfig(BaseConfig):
    """Configuration for all regularization types."""

    # L1/L2 regularization
    l1_weight: float = 0.0  # Strength of L1 regularization (0 to disable)
    l2_weight: float = 0.0  # Strength of L2 regularization (0 to disable)
    selective: SelectiveConfig = field(default_factory=SelectiveConfig)

    # Parameter group specific regularization
    split_params: bool = (
        False  # Split parameters into groups for differential regularization
    )
    param_group_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "topk": {"l1_weight": 0.0, "l2_weight": 0.0},
            "blocklinear": {"l1_weight": 0.0, "l2_weight": 0.0},
            "reactivation": {"l1_weight": 0.0, "l2_weight": 0.0},
            "decoder": {"l1_weight": 0.0, "l2_weight": 0.0},
        }
    )

    # E/I Weight Ratio Regularization
    enforce_ei_weight_ratio: bool = False  # Enable E/I weight ratio regularization
    target_ei_weight_ratio: float = 0.5  # Target excitatory/inhibitory weight ratio
    ei_ratio_loss_weight: float = (
        0.5  # Regularization strength (fraction of total loss)
    )
    # Statistics run over each synapse bank's dense weight matrix, inactive
    # candidates included as zeros. "per_branch" compares each branch's E and I
    # statistic; "per_layer" each layer's; "global" is the per-layer ratio
    # averaged over layers (the same computation as "per_layer").
    ei_ratio_scope: str = "per_branch"  # "per_branch", "per_layer", "global"
    ei_ratio_metric: str = "mean"  # "mean", "median", "rms"

    def __post_init__(self) -> None:
        self.ei_ratio_scope = normalize_choice(
            self.ei_ratio_scope, EI_RATIO_SCOPES, field="ei_ratio_scope"
        )
        self.ei_ratio_metric = normalize_choice(
            self.ei_ratio_metric, EI_RATIO_METRICS, field="ei_ratio_metric"
        )


@dataclass
class PruningConfig(BaseConfig):
    """Configuration for weight pruning."""

    enabled: bool = False
    threshold: float = 2  # Prune weights with abs(value) < threshold
    selective: SelectiveConfig = field(
        default_factory=lambda: SelectiveConfig(
            weight_types=["excitatory", "inhibitory"]
        )
    )
    re_evaluate_after: bool = False  # Whether to re-evaluate after pruning

    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = False  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting
    redo_analysis_after_pruning: bool = (
        False  # Redo all analyses after pruning for comparison
    )
