"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""

import json
import os
import re
from copy import deepcopy
from typing import Callable, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    evaluation_kwargs_from_runtime,
    subset_dataset_for_runtime,
)
from dendritic_modeling.config.analysis import (
    AblationAnalysisParams,
    EvaluationRuntimeConfig,
)
from dendritic_modeling.models import BaseModel, Classifier, Regressor
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.architectures import DendriticBranchLayer
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_ablation_results,
    plot_layer_ablation,
)
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
    evaluate_pred_label_mi,
)
from dendritic_modeling.utils import save_dict

# ---------------------------------------------------------------------------
# Ablation schema constants
# ---------------------------------------------------------------------------
_KNOWN_TARGETS = frozenset({"all_synapses", "excitation", "inhibition", "upstream"})


def flatten_3level_ablation(nested_results: dict) -> dict:
    """Convert 3-level ablation results to 2-level for legacy consumers.

    Input:  ``{"lesion": {"excitation": {module: metrics, ...}, ...}, ...}``
    Output: ``{"lesion_excitation": {module: metrics, ...}, ...}``

    If the dict is already 2-level (values contain module-level dicts with a
    ``"depth"`` key), it is returned unchanged.

    .. note:: New code should use the canonical 3-level schema directly.
              This helper exists only for backward compatibility with old
              plotting code.
    """
    flat: dict = {}
    for method, targets in nested_results.items():
        if not isinstance(targets, dict):
            flat[method] = targets
            continue
        if set(targets.keys()) & _KNOWN_TARGETS:
            for target, modules in targets.items():
                flat[f"{method}_{target}"] = modules
        else:
            flat[method] = targets
    return flat


class AblationAnalyzer(AbstractAnalyzer):
    """Analyzer for performing ablation studies on model components."""

    def __init__(self, params: AblationAnalysisParams):
        """Initialize with ablation parameters."""
        super().__init__("AblationAnalyzer")

        def _norm(x: str) -> str:
            return str(x).strip().lower().replace("-", "_").replace(" ", "_")

        selection_cfg = getattr(params, "selection", None)
        methods_cfg = getattr(params, "methods", None)

        # Method settings
        self.ablation_methods = list(
            getattr(
                methods_cfg,
                "ablation_methods",
                getattr(params, "ablation_methods", ["lesion"]),
            )
        )
        if not self.ablation_methods:
            self.ablation_methods = ["lesion"]
        self.shuffle_seed = int(
            getattr(methods_cfg, "shuffle_seed", getattr(params, "shuffle_seed", 0))
        )
        self.clamp_statistic = str(
            getattr(
                methods_cfg,
                "clamp_statistic",
                getattr(params, "clamp_statistic", "mean"),
            )
        )

        # Selection: levels, targets, metrics
        levels_raw = list(
            getattr(selection_cfg, "levels", getattr(params, "levels", [])) or []
        )
        targets_raw = list(
            getattr(selection_cfg, "targets", getattr(params, "targets", [])) or []
        )
        metrics_raw = list(
            getattr(selection_cfg, "metrics", getattr(params, "metrics", [])) or []
        )

        levels = {_norm(x) for x in levels_raw if str(x).strip()}
        targets = {_norm(x) for x in targets_raw if str(x).strip()}
        metrics = {_norm(x) for x in metrics_raw if str(x).strip()}

        if levels:
            self.layer_ablation = "layer" in levels
            self.compartment_ablation = "compartment" in levels
        else:
            self.layer_ablation = bool(getattr(params, "layer_ablation", False))
            self.compartment_ablation = bool(
                getattr(params, "compartment_ablation", False)
            )

        if targets:
            self.ablate_all_synapses = bool(
                {"all_synapses", "all"}.intersection(targets)
            )
            self.ablate_excitation = bool({"excitation", "exc"}.intersection(targets))
            self.ablate_inhibition = bool({"inhibition", "inh"}.intersection(targets))
            self.ablate_upstream = bool(
                {"upstream", "vb", "branch_input"}.intersection(targets)
            )
        else:
            self.ablate_all_synapses = bool(
                getattr(params, "ablate_all_synapses", False)
            )
            self.ablate_excitation = bool(getattr(params, "ablate_excitation", False))
            self.ablate_inhibition = bool(getattr(params, "ablate_inhibition", False))
            self.ablate_upstream = bool(getattr(params, "ablate_upstream", False))

        if metrics:
            self.accuracy = "accuracy" in metrics
            self.auc = "auc" in metrics
            self.categorical_ll = bool(
                {
                    "categorical_loglikelihood",
                    "categorical_ll",
                    "loglikelihood",
                    "log_likelihood",
                }.intersection(metrics)
            )
            self.mse = "mse" in metrics
            self.cosine_similarity = bool(
                {"cosine_similarity", "cosine"}.intersection(metrics)
            )
            self.compute_pred_label_mi = bool(
                {"pred_label_mi_bits", "pred_label_mi", "pred_mi"}.intersection(metrics)
            )
        else:
            self.accuracy = bool(getattr(params, "accuracy", False))
            self.auc = bool(getattr(params, "auc", False))
            self.categorical_ll = bool(
                getattr(params, "categorical_loglikelihood", False)
            )
            self.mse = bool(getattr(params, "mse", False))
            self.cosine_similarity = bool(getattr(params, "cosine_similarity", False))
            self.compute_pred_label_mi = bool(
                getattr(params, "pred_label_mi_bits", False)
                or getattr(params, "pred_label_mi", False)
            )

    @staticmethod
    def _normalize_ablation_target(target: str) -> str:
        """Normalize ablation target names (spaces -> underscores)."""
        return str(target).replace(" ", "_")

    def _make_signal_intervention_pre_hook(
        self,
        *,
        ablation_type: str,
        method: str,
    ):
        """Forward pre-hook that intervenes on (E, I, Vb) *signals*.

        - method="shuffle": permutes the targeted signal across samples in a batch.
        - method="mean_clamp": replaces the targeted signal with its batch mean.
        """
        call_idx = 0

        def _hook(_module, inputs):
            nonlocal call_idx
            if not inputs:
                return None

            inputs_list = list(inputs)
            x = inputs_list[0]
            inhibitory_input = inputs_list[1] if len(inputs_list) > 1 else None
            branch_input = inputs_list[2] if len(inputs_list) > 2 else None

            mod_exc = ablation_type in {"excitation", "all_synapses"}
            mod_inh = ablation_type in {"inhibition", "all_synapses"}
            mod_vb = ablation_type == "upstream"

            if method == "shuffle":
                # Deterministic per-call seed so runs are reproducible.
                seed = self.shuffle_seed + call_idx
                call_idx += 1

                gen = (
                    torch.Generator(device=x.device)
                    if isinstance(x, torch.Tensor) and x.is_cuda
                    else torch.Generator()
                )
                gen.manual_seed(seed)
                perm = torch.randperm(x.shape[0], generator=gen, device=x.device)

                if mod_exc and isinstance(x, torch.Tensor):
                    x = x[perm]
                if mod_inh and isinstance(inhibitory_input, torch.Tensor):
                    inhibitory_input = inhibitory_input[perm]
                if mod_vb and isinstance(branch_input, torch.Tensor):
                    branch_input = branch_input[perm]

            elif method == "mean_clamp":
                if self.clamp_statistic != "mean":
                    raise ValueError(
                        f"Unsupported clamp_statistic: {self.clamp_statistic}"
                    )

                if mod_exc and isinstance(x, torch.Tensor):
                    x = x.mean(dim=0, keepdim=True).expand_as(x)
                if mod_inh and isinstance(inhibitory_input, torch.Tensor):
                    inhibitory_input = inhibitory_input.mean(
                        dim=0, keepdim=True
                    ).expand_as(inhibitory_input)
                if mod_vb and isinstance(branch_input, torch.Tensor):
                    branch_input = branch_input.mean(dim=0, keepdim=True).expand_as(
                        branch_input
                    )
            else:
                return None

            # Write back modified values, preserving original tuple structure.
            inputs_list[0] = x
            if len(inputs_list) > 1:
                inputs_list[1] = inhibitory_input
            if len(inputs_list) > 2:
                inputs_list[2] = branch_input
            return tuple(inputs_list)

        return _hook

    def ablate_layer_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        ablation_method: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict,
        eval_kwargs: Optional[dict] = None,
    ):
        """Ablate a single layer and evaluate its impact.

        Args:
            ablation_type: The ablation target, e.g. ``"excitation"``,
                ``"inhibition"``, ``"all_synapses"``, ``"upstream"``.
                Legacy composite keys like ``"lesion_excitation"`` are also
                accepted and the method prefix is stripped automatically.
            ablation_method: ``"lesion"`` | ``"shuffle"`` | ``"mean_clamp"``.
            results_dict: Dict to write into at ``results_dict[ablation_type][module_name]``.
        """
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx
        results_dict[ablation_type][module_name]["ablation_method"] = ablation_method
        if eval_kwargs is None:
            eval_kwargs = {}

        # Allow ablation_type keys to be namespaced, e.g. "lesion_excitation".
        target_type = ablation_type
        prefix = f"{ablation_method}_"
        if isinstance(ablation_type, str) and ablation_type.startswith(prefix):
            target_type = ablation_type[len(prefix) :]
        target_type_norm = self._normalize_ablation_target(target_type)

        # Check if this is a soma layer without somatic synapses (using config)
        somatic_synapses_enabled = getattr(model.core_network, "somatic_synapses", True)
        # If somatic synapses are disabled, the soma layer has no E/I synapses.
        # We only short-circuit to "zero drop" for synapse-based targets; upstream
        # (branch input / branches_to_output) is still meaningful and must be computed.
        is_soma_without_synapses = (
            module.layer_idx == 0
            and not somatic_synapses_enabled
            and target_type_norm in {"all_synapses", "excitation", "inhibition"}
        )

        if is_soma_without_synapses:
            # For soma without synapses, ablation has no effect (zero drop)
            def _set_zero_metric(metric_key: str):
                results_dict[ablation_type][module_name][f"{metric_key}_drop"] = 0.0

            if self.accuracy:
                _set_zero_metric("accuracy")
            if self.auc:
                _set_zero_metric("auc")
            if self.categorical_ll:
                _set_zero_metric("categorical_loglikelihood")
            if self.mse:
                _set_zero_metric("mse")
            if self.cosine_similarity:
                _set_zero_metric("cosine_similarity")
            if self.compute_pred_label_mi:
                _set_zero_metric("pred_label_mi_bits")
        else:
            if ablation_method == "lesion":
                # Structural lesion: remove the relevant connection weights.
                if target_type_norm in {"all_synapses", "excitation"}:
                    if module.branch_excitation is not None and hasattr(
                        module.branch_excitation, "pre_w"
                    ):
                        module.branch_excitation.pre_w.data.fill_(-1000)
                if target_type_norm in {"all_synapses", "inhibition"}:
                    if module.branch_inhibition is not None and hasattr(
                        module.branch_inhibition, "pre_w"
                    ):
                        module.branch_inhibition.pre_w.data.fill_(-1000)
                if target_type_norm == "upstream":
                    if hasattr(module, "branches_to_output") and hasattr(
                        module.branches_to_output, "log_weight"
                    ):
                        module.branches_to_output.log_weight.data.fill_(-1000)

            elif ablation_method in {"shuffle", "mean_clamp"}:
                # Signal-level intervention: manipulate the input signals, not weights.
                # Hooks are registered *per metric* so all metrics see the same
                # deterministic intervention sequence.
                pass
            else:
                raise ValueError(f"Unknown ablation_method: {ablation_method}")

            def _evaluate_metric(metric_func: Callable, metric_key: str):
                local_hook = None
                try:
                    if ablation_method in {"shuffle", "mean_clamp"}:
                        pre_hook = self._make_signal_intervention_pre_hook(
                            ablation_type=target_type_norm, method=ablation_method
                        )
                        local_hook = module.register_forward_pre_hook(pre_hook)

                    metric_score = metric_func(
                        model,
                        test_ds=test_dataset,
                        move_device=False,
                        **eval_kwargs,
                    )[-1]
                finally:
                    if local_hook is not None:
                        local_hook.remove()

                results_dict[ablation_type][module_name][f"{metric_key}_drop"] = (
                    baseline_dict[metric_key] - metric_score
                )

            if self.accuracy:
                _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")
            if self.auc:
                _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")
            if self.categorical_ll:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="categorical_loglikelihood",
                )
            if self.mse:
                _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine_similarity",
                )
            if self.compute_pred_label_mi:
                _evaluate_metric(
                    metric_func=evaluate_pred_label_mi, metric_key="pred_label_mi_bits"
                )

        model.load_state_dict(state_dict)

        return results_dict

    def ablate_compartment_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        ablation_method: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict,
        eval_kwargs: Optional[dict] = None,
    ):
        """Ablate a single compartment and evaluate its impact."""
        if eval_kwargs is None:
            eval_kwargs = {}
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx
        results_dict[ablation_type][module_name]["ablation_method"] = ablation_method

        # Allow ablation_type keys to be namespaced, e.g. "lesion_excitation".
        target_type = ablation_type
        prefix = f"{ablation_method}_"
        if isinstance(ablation_type, str) and ablation_type.startswith(prefix):
            target_type = ablation_type[len(prefix) :]
        target_type_norm = self._normalize_ablation_target(target_type)

        if module.branch_excitation is not None:
            n_branches = module.branch_excitation.pre_w.shape[0]
        elif module.branch_inhibition is not None:
            n_branches = module.branch_inhibition.pre_w.shape[0]
        elif hasattr(module, "branches_to_output"):
            n_branches = module.branches_to_output.log_weight.shape[0]
        else:
            raise ValueError(
                f"Number of compartments cannot be determined for {module_name}"
            )

        if self.accuracy:
            accuracy_list = []
        if self.auc:
            auc_list = []
        if self.categorical_ll:
            loglikelihood_list = []
        if self.mse:
            mse_list = []
        if self.cosine_similarity:
            cosine_similarity_list = []

        def _evaluate_metric(metric_func: Callable, metric_key: str, metric_list: list):
            metric_score = metric_func(
                model,
                test_ds=test_dataset,
                move_device=False,
                **eval_kwargs,
            )[-1]
            metric_list.append(baseline_dict[metric_key] - metric_score)

        for branch_ix in range(n_branches):
            if target_type_norm in {"all_synapses", "excitation"}:
                module.branch_excitation.pre_w.data[branch_ix, :] = -1000
            if target_type_norm in {"all_synapses", "inhibition"}:
                module.branch_inhibition.pre_w.data[branch_ix, :] = -1000
            if target_type_norm == "upstream":
                module.branches_to_output.log_weight.data[branch_ix, :] = -1000

            if self.accuracy:
                _evaluate_metric(
                    metric_func=evaluate_accuracy,
                    metric_key="accuracy",
                    metric_list=accuracy_list,
                )
            if self.auc:
                _evaluate_metric(
                    metric_func=evaluate_auc, metric_key="auc", metric_list=auc_list
                )
            if self.categorical_ll:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="categorical_loglikelihood",
                    metric_list=loglikelihood_list,
                )
            if self.mse:
                _evaluate_metric(
                    metric_func=evaluate_mse, metric_key="mse", metric_list=mse_list
                )
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine_similarity",
                    metric_list=cosine_similarity_list,
                )

            model.load_state_dict(state_dict)

        if self.accuracy:
            results_dict[ablation_type][module_name]["accuracy_drop"] = accuracy_list
        if self.auc:
            results_dict[ablation_type][module_name]["auc_drop"] = auc_list
        if self.categorical_ll:
            results_dict[ablation_type][module_name][
                "categorical_loglikelihood_drop"
            ] = loglikelihood_list
        if self.mse:
            results_dict[ablation_type][module_name]["mse_drop"] = mse_list
        if self.cosine_similarity:
            results_dict[ablation_type][module_name][
                "cosine_similarity_drop"
            ] = cosine_similarity_list

        return results_dict

    def _plot_ablation_results(
        self,
        results: dict,
        save_path: str,
        filename: str,
        somatic_synapses: bool = True,
    ):
        """Generate comprehensive ablation plots similar to information analysis."""
        try:
            # Plot layer ablation results if available
            if results.get("layer"):
                plot_layer_ablation(
                    layer_results=results["layer"],
                    baselines=results["baseline"],
                    save_path=save_path,
                    filename=filename,
                    somatic_synapses=somatic_synapses,
                )

                if os.path.exists(os.path.join(save_path, "epochs")):
                    print("Creating video for ablation vs training", flush=True)
                    create_video_layer_ablation_vs_training(
                        save_path=save_path,
                        metric="accuracy",
                        video_format="mp4",
                    )

            # Plot compartment ablation results if available
            if results.get("compartment"):
                plot_ablation_results(
                    results["compartment"],
                    save_path=save_path,
                    filename=f"{filename}_compartment",
                )
        except ImportError as e:
            print(f"Warning: Ablation plotting functions not found: {e}")
        except Exception as e:
            print(f"Warning: Could not generate ablation plots: {e}")

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """Perform ablation analysis and return results."""
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        eval_kwargs = evaluation_kwargs_from_runtime(runtime)
        test_dataset = subset_dataset_for_runtime(test_dataset, runtime)
        model = model.to(device)
        model.eval()

        state_dict = deepcopy(model.state_dict())

        # Calculate baseline performance using correct function signatures
        baselines = {}
        if self.accuracy:
            if isinstance(model, Classifier):
                baselines["accuracy"] = evaluate_accuracy(
                    classifier=model,
                    test_ds=test_dataset,
                    move_device=False,
                    **eval_kwargs,
                )[-1]
            else:
                self.accuracy = False

        if self.auc:
            if isinstance(model, Classifier):
                baselines["auc"] = evaluate_auc(
                    classifier=model,
                    test_ds=test_dataset,
                    move_device=False,
                    **eval_kwargs,
                )[-1]
            else:
                self.auc = False

        if self.categorical_ll:
            if isinstance(model, Classifier):
                baselines["categorical_loglikelihood"] = (
                    evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=test_dataset,
                        move_device=False,
                        **eval_kwargs,
                    )[-1]
                )
            else:
                self.categorical_ll = False

        if self.mse:
            if isinstance(model, Regressor):
                baselines["mse"] = evaluate_mse(
                    regressor=model,
                    test_ds=test_dataset,
                    move_device=False,
                    **eval_kwargs,
                )[-1]
            else:
                self.mse = False

        if self.cosine_similarity:
            if isinstance(model, Regressor):
                baselines["cosine_similarity"] = evaluate_cosine_similarity(
                    regressor=model,
                    test_ds=test_dataset,
                    move_device=False,
                    **eval_kwargs,
                )[-1]
            else:
                self.cosine_similarity = False

        if self.compute_pred_label_mi:
            if isinstance(model, Classifier):
                baselines["pred_label_mi_bits"] = evaluate_pred_label_mi(
                    classifier=model,
                    test_ds=test_dataset,
                    move_device=False,
                    **eval_kwargs,
                )[-1]
            else:
                self.compute_pred_label_mi = False

        # Build canonical 3-level schema: method -> target -> modules
        # Internal 2-level keys (method_target) are kept as aliases for the
        # evaluation functions, but the top-level containers are 3-level.
        if self.layer_ablation:
            layer_results: dict[str, dict] = {}
            for method in self.ablation_methods:
                layer_results.setdefault(method, {})
                if self.ablate_all_synapses:
                    layer_results[method]["all_synapses"] = {}
                if self.ablate_excitation:
                    layer_results[method]["excitation"] = {}
                if self.ablate_inhibition:
                    layer_results[method]["inhibition"] = {}
                if self.ablate_upstream:
                    layer_results[method]["upstream"] = {}

        if self.compartment_ablation:
            compartment_results: dict[str, dict] = {}
            for method in self.ablation_methods:
                if method != "lesion":
                    continue
                compartment_results.setdefault(method, {})
                if self.ablate_all_synapses:
                    compartment_results[method]["all_synapses"] = {}
                if self.ablate_excitation:
                    compartment_results[method]["excitation"] = {}
                if self.ablate_inhibition:
                    compartment_results[method]["inhibition"] = {}
                if self.ablate_upstream:
                    compartment_results[method]["upstream"] = {}

        for _name, module in model.named_modules():
            if not (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                continue

            for method in self.ablation_methods:
                if self.ablate_all_synapses:
                    target = "all_synapses"
                    if self.layer_ablation:
                        self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=layer_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )
                    if self.compartment_ablation and method == "lesion":
                        self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=compartment_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                if self.ablate_excitation:
                    target = "excitation"
                    if self.layer_ablation:
                        self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=layer_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )
                    if self.compartment_ablation and method == "lesion":
                        self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=compartment_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                if self.ablate_inhibition:
                    target = "inhibition"
                    if self.layer_ablation:
                        self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=layer_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )
                    if self.compartment_ablation and method == "lesion":
                        self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=compartment_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                if self.ablate_upstream and hasattr(module, "branches_to_output"):
                    target = "upstream"
                    if self.layer_ablation:
                        self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=layer_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )
                    if self.compartment_ablation and method == "lesion":
                        self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type=target,
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=compartment_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

        # ------------------------------------------------------------------
        # Ensure upstream (branch input / branches_to_output) is evaluated at the
        # soma even when somatic_synapses=False.
        #
        # When somatic synapses are disabled, the soma DendriticBranchLayer exists
        # (layer_idx==0) but has no E/I synapse modules, so it is excluded by the
        # main loop filter above. Upstream, however, is still meaningful at soma
        # (it is the only remaining input pathway there), so we compute it here.
        # ------------------------------------------------------------------
        somatic_synapses_enabled = getattr(model.core_network, "somatic_synapses", True)
        if self.ablate_upstream and not somatic_synapses_enabled:
            for _name, module in model.named_modules():
                if not isinstance(module, DendriticBranchLayer):
                    continue
                if getattr(module, "layer_idx", None) != 0:
                    continue
                if not hasattr(module, "branches_to_output"):
                    continue

                for method in self.ablation_methods:
                    if self.layer_ablation:
                        # Avoid duplicate entries if somehow already computed.
                        if _name in layer_results.get(method, {}).get("upstream", {}):
                            continue
                        self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=layer_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                    if self.compartment_ablation and method == "lesion":
                        if _name in compartment_results.get(method, {}).get(
                            "upstream", {}
                        ):
                            continue
                        self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            ablation_method=method,
                            baseline_dict=baselines,
                            results_dict=compartment_results[method],
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

        results = {"baseline": baselines}
        if self.layer_ablation:
            results["layer"] = layer_results
        if self.compartment_ablation:
            results["compartment"] = compartment_results

        # Add synthetic soma entry when somatic_synapses=False for uniform plotting.
        #
        # IMPORTANT: We only add placeholders for synapse-based ablations when soma
        # has no E/I synapses. Upstream is computed explicitly above and must NOT be
        # forced to zero by a placeholder.
        if not somatic_synapses_enabled and "layer" in results:
            for _method, targets in results["layer"].items():
                if not isinstance(targets, dict):
                    continue
                for target_name, modules in targets.items():
                    if target_name == "upstream":
                        continue
                    if not isinstance(modules, dict):
                        continue
                    soma_entry: dict = {"depth": 0}
                    if self.accuracy:
                        soma_entry["accuracy_drop"] = 0.0
                    if self.auc:
                        soma_entry["auc_drop"] = 0.0
                    if self.categorical_ll:
                        soma_entry["categorical_loglikelihood_drop"] = 0.0
                    if self.mse:
                        soma_entry["mse_drop"] = 0.0
                    if self.cosine_similarity:
                        soma_entry["cosine_similarity_drop"] = 0.0
                    if self.compute_pred_label_mi:
                        soma_entry["pred_label_mi_bits_drop"] = 0.0
                    modules["synthetic_soma"] = soma_entry

        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")

            # Results are already in canonical 3-level schema
            # (method -> target -> modules). Save directly.
            save_dict(results, save_path, f"{filename}.json")

            if not training:
                # Plotting functions expect 2-level keys (method_target).
                # Flatten from 3-level for backward-compatible plotting.
                plot_results = dict(results)
                if "layer" in plot_results:
                    plot_results["layer"] = flatten_3level_ablation(
                        plot_results["layer"]
                    )
                if "compartment" in plot_results:
                    plot_results["compartment"] = flatten_3level_ablation(
                        plot_results["compartment"]
                    )
                self._plot_ablation_results(
                    plot_results, save_path, filename, somatic_synapses_enabled
                )
        else:
            return results


def create_video_layer_ablation_vs_training(
    save_path: str,
    metric: str,
    video_format: str = "mp4",
    method: Optional[str] = None,
):
    """
    Create a video showing layer ablation results evolving over training epochs.

    The video will be 20 seconds long, with fps automatically calculated based on
    the number of epochs.

    Args:
        save_path: Path to the directory containing the epochs folder
        metric: Metric to visualize (e.g., 'accuracy', 'auc', 'categorical_loglikelihood', 'mse', 'cosine_similarity')
        video_format: Video format ('mp4' or 'gif')
    """
    epochs_dir = os.path.join(save_path, "epochs")

    if not os.path.exists(epochs_dir):
        raise ValueError(f"Epochs directory not found: {epochs_dir}")

    # Collect and sort epoch JSON files
    epoch_files = []
    for filename in os.listdir(epochs_dir):
        if filename.startswith("epoch") and filename.endswith(".json"):
            # Extract epoch number from filename (e.g., "epoch0.json" -> 0)
            match = re.match(r"epoch(\d+)\.json", filename)
            if match:
                epoch_number = int(match.group(1))
                epoch_files.append((epoch_number, filename))

    if not epoch_files:
        raise ValueError(f"No epoch JSON files found in {epochs_dir}")

    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])

    # Load all epoch data (flatten 3-level schema to 2-level for plotting)
    epoch_data_list = []
    for epoch_number, filename in epoch_files:
        file_path = os.path.join(epochs_dir, filename)
        with open(file_path) as f:
            data = json.load(f)
            if "layer" in data:
                data["layer"] = flatten_3level_ablation(data["layer"])
            epoch_data_list.append((epoch_number, data))

    if not epoch_data_list:
        raise ValueError("No valid epoch data loaded")

    num_epochs = len(epoch_data_list)
    fps = int(num_epochs / 30.0)
    fps = min(max(fps, 2), 10)

    # Determine available metrics and ablation types from first epoch
    first_epoch_data = epoch_data_list[0][1]
    if "layer" not in first_epoch_data:
        raise ValueError("No layer ablation data found in epoch files")

    layer_results = first_epoch_data["layer"]
    # baselines = first_epoch_data.get("baseline", {})

    # Determine available metrics
    metric_mapping = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "categorical_loglikelihood": "Log Likelihood",
        "mse": "MSE",
        "cosine_similarity": "Cosine Similarity",
        "pred_label_mi_bits": "I(C; Ĉ) [bits]",
    }

    def has_metric_nested(dictionary, metric):
        """Check if key exists at any level in a nested dictionary"""
        if isinstance(dictionary, dict):
            key = f"{metric}_drop"
            if key in dictionary:
                return True
            return any(
                has_metric_nested(value, metric) for value in dictionary.values()
            )
        elif isinstance(dictionary, (list, tuple)):
            return any(has_metric_nested(item, metric) for item in dictionary)
        return False

    # Validate that the requested metric exists
    if metric not in metric_mapping:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of: {list(metric_mapping.keys())}"
        )

    if not has_metric_nested(layer_results, metric):
        available_metrics = []
        for m in metric_mapping.keys():
            if has_metric_nested(layer_results, m):
                available_metrics.append(m)
        raise ValueError(
            f"Metric '{metric}' not found in ablation data. "
            f"Available metrics: {available_metrics}"
        )

    # Determine ablation types (supports method-prefixed keys like "lesion_excitation")
    base_targets = ["excitation", "inhibition", "all_synapses", "upstream"]

    available_methods = set()
    for k in layer_results.keys():
        k_norm = str(k).replace(" ", "_")
        for base in base_targets:
            if k_norm.endswith(f"_{base}"):
                available_methods.add(k_norm[: -(len(base) + 1)])

    selected_method = method
    if selected_method is None:
        selected_method = "lesion" if "lesion" in available_methods else None
    if selected_method is None and available_methods:
        selected_method = sorted(available_methods)[0]

    synapse_ablation_types = []
    if selected_method is not None:
        for base in base_targets:
            key = f"{selected_method}_{base}"
            if key in layer_results:
                synapse_ablation_types.append(key)
    else:
        # Legacy (non-prefixed) keys
        for legacy_key in (
            "excitation",
            "inhibition",
            "all synapses",
            "all_synapses",
            "upstream",
        ):
            if legacy_key in layer_results:
                synapse_ablation_types.append(legacy_key)

    if not synapse_ablation_types:
        raise ValueError("No ablation types found in data")

    # Create a video for the specified metric
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine depth range from first epoch (assume consistent across epochs)
    first_layer_results = epoch_data_list[0][1]["layer"]
    first_type_dict = first_layer_results[synapse_ablation_types[0]]
    depths = sorted({layer_dict["depth"] for layer_dict in first_type_dict.values()})

    # Create depth labels
    depth_labels = []
    for depth in depths:
        if depth == min(depths):
            depth_labels.append("Soma")
        else:
            distal_layer = depth - min(depths)
            depth_labels.append(f"Distal {distal_layer}")

    # Calculate global min and max values across all epochs for consistent y-axis
    all_values = []
    for _epoch_number, epoch_data in epoch_data_list:
        layer_results = epoch_data["layer"]
        for ablation_type in synapse_ablation_types:
            type_dict = layer_results.get(ablation_type, {})
            for layer_dict in type_dict.values():
                metric_key = f"{metric}_drop"
                decrease_value = layer_dict.get(metric_key, 0.0)
                # Handle list values (from compartment ablation)
                if isinstance(decrease_value, list):
                    if decrease_value:
                        all_values.extend([float(v) for v in decrease_value])
                else:
                    all_values.append(float(decrease_value))

    if not all_values:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = min(all_values)
        y_max = max(all_values)
        # Add small padding (5% of range)
        y_range = y_max - y_min
        if y_range == 0:
            y_min -= 0.1
            y_max += 0.1
        else:
            padding = y_range * 0.05
            # y_min = max(0.0, y_min - padding)  # Don't go below 0 for drop values
            y_min = y_min - padding
            y_max = y_max + padding

    # Create evenly spaced y-ticks rounded to 1 decimal place
    num_ticks = 6  # Number of ticks desired
    y_ticks = np.linspace(y_min, y_max, num_ticks)
    y_ticks = np.round(y_ticks, 1)  # Round to 1 decimal place
    y_ticks = np.unique(y_ticks)  # Remove duplicates

    # Set up plot elements that won't change
    ax.set_xticks(depths)
    ax.set_xticklabels(depth_labels, rotation=45)
    ax.set_xlabel("Branch Depth (Soma to Distal)")
    ax.set_ylabel(f"{metric_mapping[metric]} Drop Decrease")
    # ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Store bar containers for updating
    n_ablation_types = len(synapse_ablation_types)
    width = 0.25

    def animate(frame):
        """Animation function that updates the plot for each epoch"""
        # Clear previous bars but keep axes setup
        ax.clear()

        # Reload axes setup
        ax.set_xticks(depths)
        ax.set_xticklabels(depth_labels, rotation=45)
        ax.set_xlabel("Branch Depth (Soma to Distal)")
        ax.set_ylabel(f"{metric_mapping[metric]} Drop Decrease")
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, axis="y", alpha=0.3)

        # Get data for this frame
        epoch_number, epoch_data = epoch_data_list[frame]
        layer_results = epoch_data["layer"]
        baselines = epoch_data.get("baseline", {})

        # Prepare data for plotting (matching plot_layer_ablation logic exactly)
        results = {}
        for ablation_type in synapse_ablation_types:
            data_dict = {"depth": [], "decrease": []}
            type_dict = layer_results.get(ablation_type, {})
            for layer_dict in type_dict.values():
                data_dict["depth"].append(layer_dict["depth"])
                metric_key = f"{metric}_drop"
                decrease_value = layer_dict.get(metric_key, 0.0)
                # Handle list values (from compartment ablation)
                if isinstance(decrease_value, list):
                    decrease_value = np.mean(decrease_value) if decrease_value else 0.0
                data_dict["decrease"].append(float(decrease_value))
            results[ablation_type] = data_dict

        # Use consistent depths (from first epoch) for x-axis positioning
        # Match values to depths in correct order
        for ablation_type in synapse_ablation_types:
            if ablation_type in results:
                # Create mapping from depth to decrease value
                depth_to_decrease = dict(
                    zip(
                        results[ablation_type]["depth"],
                        results[ablation_type]["decrease"],
                    )
                )
                # Map decreases to consistent depth order
                ordered_decreases = [depth_to_decrease.get(d, 0.0) for d in depths]
                results[ablation_type]["decrease"] = ordered_decreases

        # Plot bars for each ablation type (matching plot_layer_ablation exactly)
        for i, ablation_type in enumerate(synapse_ablation_types):
            if ablation_type in results:
                x = np.array(depths) + width * (i - (n_ablation_types - 1) / 2)
                ax.bar(
                    x,
                    results[ablation_type]["decrease"],
                    width=width,
                    label=ablation_type.replace("_", " ").title(),
                )

        # Update title with epoch number and baseline
        baseline_value = baselines.get(metric, 0.0)
        title = f"{metric_mapping[metric]} Drop Decrease vs. Branch Depth"
        if selected_method is not None:
            title += f"\nMethod: {selected_method}"
        title += f"\nEpoch {epoch_number} (Baseline: {baseline_value:.3f})"
        ax.set_title(title)
        ax.legend()

        return []

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(epoch_data_list),
        interval=1000 / fps,  # Convert fps to interval in milliseconds
        repeat=True,
        blit=True,
    )

    # Save video
    method_tag = selected_method if selected_method is not None else "legacy"
    video_filename = f"layer_ablation_{method_tag}_{metric}_evolution"
    print(f"Creating {num_epochs}-frame video at {fps:.2f} fps (20 seconds total)")
    if video_format == "mp4":
        if animation.writers.is_available("ffmpeg"):
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                video_path = os.path.join(save_path, f"{video_filename}.mp4")
                ani.save(video_path, writer=writer)
                print(f"Video saved to {video_path}")
            except Exception as e:
                print(f"Error saving MP4 video: {e}")
                print("Falling back to GIF format...")
                writer = animation.PillowWriter(fps=fps)
                video_path = os.path.join(save_path, f"{video_filename}.gif")
                ani.save(video_path, writer=writer)
                print(f"GIF saved to {video_path}")
        else:
            print("ffmpeg writer not available; saving GIF format...")
            writer = animation.PillowWriter(fps=fps)
            video_path = os.path.join(save_path, f"{video_filename}.gif")
            ani.save(video_path, writer=writer)
            print(f"GIF saved to {video_path}")
    else:
        writer = animation.PillowWriter(fps=fps)
        video_path = os.path.join(save_path, f"{video_filename}.gif")
        ani.save(video_path, writer=writer)
        print(f"GIF saved to {video_path}")

    plt.close(fig)
