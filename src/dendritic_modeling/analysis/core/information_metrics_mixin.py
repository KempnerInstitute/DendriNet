"""Metric computation helpers for information analysis."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_helpers import (
    convert_information_units_in_place,
    discrete_entropy_nats,
    ensure_2d,
    extract_metric_tokens,
    is_discrete_class_labels,
    keep_metric_key_given_availability,
    metric_targets_class,
    rename_metric_key_tokens,
    should_convert_information_key_to_bits,
)
from dendritic_modeling.analysis.core.information_pid_mixin import (
    InformationPidMetricsMixin,
)
from dendritic_modeling.analysis.core.information_results import (
    average_mi_results,
    compute_layer_statistics,
    compute_variance_analysis,
    filter_result_metrics_in_place,
)

ParsedMiKey = tuple[list[str], str, list[str]]


def _parse_mi_key_for_null_baseline(mi_key: Any) -> ParsedMiKey | None:
    if not isinstance(mi_key, str):
        return None
    key = mi_key.strip().replace(" ", "")
    if not (key.startswith("I(") and key.endswith(")")):
        return None
    inside = key[2:-1]
    if ";" not in inside:
        return None
    left, right = inside.split(";", 1)
    if "|" in right:
        y, z = right.split("|", 1)
        z_vars = [t for t in z.split(",") if t]
    else:
        y = right
        z_vars = []
    x_vars = [t for t in left.split(",") if t]
    return x_vars, y, z_vars


def _metric_variable_for_null_baseline(
    token: str,
    *,
    E: np.ndarray | None,
    I_var: np.ndarray | None,
    Vout: np.ndarray | None,
    C_col: np.ndarray,
    Vinf: np.ndarray | None,
    Vb: np.ndarray | None,
    has_exc_synapses: bool,
    has_inh_synapses: bool,
    has_branch_input: bool,
) -> np.ndarray | None:
    base = token.split("_", 1)[0]
    if base == "C":
        return C_col
    if base == "E":
        return E if has_exc_synapses else None
    if base == "I":
        return I_var if has_inh_synapses else None
    if base == "Vb":
        return Vb if has_branch_input else None
    if base == "Vout":
        return Vout
    if base == "Vinf":
        return Vinf
    return None


class InformationMetricsMixin(InformationPidMetricsMixin):
    """Computes MI/CMI/PID/Fisher metrics and result summaries."""

    def _estimate_mi(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute one MI metric with the existing debug timing format."""
        self.logger.debug(f"  Computing {metric_key}...")
        start_time = time.time()
        value = self.estimators[self.method].mutual_information(x, y)
        self.logger.debug(
            f"  {metric_key} = {value:.4f} (took {time.time() - start_time:.1f}s)"
        )
        return value

    def _estimate_cmi(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
        condition: np.ndarray,
    ) -> float:
        """Compute one conditional MI metric with the existing debug timing format."""
        self.logger.debug(f"  Computing {metric_key}...")
        start_time = time.time()
        value = self.estimators[self.method].conditional_mutual_information(
            x, y, condition
        )
        self.logger.debug(
            f"  {metric_key} = {value:.4f} (took {time.time() - start_time:.1f}s)"
        )
        return value

    def _estimate_cmi_or_zero(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
        condition: np.ndarray,
    ) -> float:
        """Compute one conditional MI metric, returning zero on estimator failure."""
        self.logger.debug(f"  Computing {metric_key}...")
        start_time = time.time()
        try:
            value = self.estimators[self.method].conditional_mutual_information(
                x, y, condition
            )
        except Exception as e:
            self.logger.warning(f"  Failed to compute {metric_key}: {e}")
            value = 0.0
        self.logger.debug(
            f"  {metric_key} = {value:.4f} (took {time.time() - start_time:.1f}s)"
        )
        return value

    def _compute_basic_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute mutual information between each variable and C."""
        self.logger.debug(
            f"Computing basic MI: E{E.shape}, I{I_var.shape}, Vout{Vout.shape}, C{C.shape}"
        )
        metrics = {}

        # I(E;C)
        metrics["I(E;C)"] = self._estimate_mi("I(E;C)", E, C)

        # I(I;C)
        metrics["I(I;C)"] = self._estimate_mi("I(I;C)", I_var, C)

        # I(Vb;C) if available
        if Vb is not None:
            metrics["I(Vb;C)"] = self._estimate_mi("I(Vb;C)", Vb, C)

        # I(Vout;C)
        metrics["I(Vout;C)"] = self._estimate_mi("I(Vout;C)", Vout, C)

        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            metrics["I(Vinf;C)"] = self._estimate_mi("I(Vinf;C)", Vinf, C)

        # Joint MI: I(E,I;C)
        EI = np.hstack([E, I_var])
        metrics["I(E,I;C)"] = self._estimate_mi("I(E,I;C)", EI, C)

        # Optionally normalize by entropy
        if self.normalize_mi:
            self.logger.debug("  Computing entropy H(C) for normalization...")
            start_time = time.time()
            H_C = self.estimators[self.method].entropy(C.reshape(-1, 1))
            self.logger.debug(
                f"  H(C) = {H_C:.4f} (took {time.time() - start_time:.1f}s)"
            )
            # Create a list of keys to avoid modifying dict during iteration
            keys_to_normalize = list(metrics.keys())
            for key in keys_to_normalize:
                metrics[f"{key}_normalized"] = metrics[key] / H_C if H_C > 0 else 0.0

        self.logger.debug("  Basic MI computation completed.")
        return metrics

    def _compute_pairwise_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute pairwise mutual information between variables."""
        self.logger.debug("Computing pairwise MI between neural variables...")
        metrics = {}

        # I(E;I)
        metrics["I(E;I)"] = self._estimate_mi("I(E;I)", E, I_var)

        # I(E;Vout)
        metrics["I(E;Vout)"] = self._estimate_mi("I(E;Vout)", E, Vout)

        # I(I;Vout)
        metrics["I(I;Vout)"] = self._estimate_mi("I(I;Vout)", I_var, Vout)

        # Pairwise MI with Vb if available
        if Vb is not None:
            # I(E;Vb)
            metrics["I(E;Vb)"] = self._estimate_mi("I(E;Vb)", E, Vb)

            # I(I;Vb)
            metrics["I(I;Vb)"] = self._estimate_mi("I(I;Vb)", I_var, Vb)

            # I(Vb;Vout)
            metrics["I(Vb;Vout)"] = self._estimate_mi("I(Vb;Vout)", Vb, Vout)

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            metrics["I(E;Vinf)"] = self._estimate_mi("I(E;Vinf)", E, Vinf)

            # I(I;Vinf)
            metrics["I(I;Vinf)"] = self._estimate_mi("I(I;Vinf)", I_var, Vinf)

            # I(Vout;Vinf)
            metrics["I(Vout;Vinf)"] = self._estimate_mi("I(Vout;Vinf)", Vout, Vinf)

        self.logger.debug("  Pairwise MI computation completed.")
        return metrics

    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute conditional mutual information given C."""
        self.logger.debug(
            "Computing conditional MI metrics (conditioned on class labels)..."
        )
        metrics = {}
        C_col = C.reshape(-1, 1)

        # I(E;I|C)
        metrics["I(E;I|C)"] = self._estimate_cmi("I(E;I|C)", E, I_var, C_col)

        # I(E;Vout|C)
        metrics["I(E;Vout|C)"] = self._estimate_cmi("I(E;Vout|C)", E, Vout, C_col)

        # I(I;Vout|C)
        metrics["I(I;Vout|C)"] = self._estimate_cmi("I(I;Vout|C)", I_var, Vout, C_col)

        # Conditional MI with Vb if available
        if Vb is not None:
            # I(E;Vb|C)
            metrics["I(E;Vb|C)"] = self._estimate_cmi("I(E;Vb|C)", E, Vb, C_col)

            # I(I;Vb|C)
            metrics["I(I;Vb|C)"] = self._estimate_cmi("I(I;Vb|C)", I_var, Vb, C_col)

            # I(Vb;Vout|C)
            metrics["I(Vb;Vout|C)"] = self._estimate_cmi(
                "I(Vb;Vout|C)", Vb, Vout, C_col
            )

        # ------------------------------------------------------------------
        # Ablation-aligned conditional MI proxies
        #
        # These measure *unique* class information that each component provides
        # beyond other components, which is often closer to what layer ablations
        # probe than raw I(E;C) / I(I;C).
        # ------------------------------------------------------------------
        if self.compute_ablation_aligned_mi:
            # One-sided conditional class information in the dendritic output.
            # These are useful for interaction-information identities such as
            # I(Vout;C) - I(Vout;C|E) = I(E;Vout) - I(E;Vout|C).
            metrics["I(Vout;C|E)"] = self._estimate_cmi("I(Vout;C|E)", Vout, C_col, E)
            metrics["I(Vout;C|I)"] = self._estimate_cmi(
                "I(Vout;C|I)", Vout, C_col, I_var
            )

        if self.compute_ablation_aligned_mi and Vb is not None:
            # How much class info is present in the layer output beyond upstream input
            metrics["I(Vout;C|Vb)"] = self._estimate_cmi(
                "I(Vout;C|Vb)", Vout, C_col, Vb
            )

            # Complement: how much upstream class information is *not* reflected in the layer output
            # (helps interpret upstream vs synapse contributions).
            metrics["I(Vb;C|Vout)"] = self._estimate_cmi(
                "I(Vb;C|Vout)", Vb, C_col, Vout
            )

            # How much class information in Vout remains after conditioning on synapse inputs
            # (proxy for upstream contribution to class information in Vout).
            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = self._estimate_cmi(
                "I(Vout;C|E,I)", Vout, C_col, EI
            )

            # Optional (can be expensive for high-dimensional synapse sets):
            # unique upstream class information beyond the full synapse set.
            if self.compute_upstream_unique_cmi:
                metrics["I(Vb;C|E,I)"] = self._estimate_cmi_or_zero(
                    "I(Vb;C|E,I)", Vb, C_col, EI
                )

            # ------------------------------------------------------------------
            # Information-processing conditionals (E/I/Vb -> Vout)
            #
            # These quantify *unique* statistical influence on the dendritic output,
            # independent of class labels. They complement ablation results because
            # they isolate each input's contribution to Vout beyond the others.
            # ------------------------------------------------------------------
            metrics["I(E;Vout|I,Vb)"] = self._estimate_cmi(
                "I(E;Vout|I,Vb)", E, Vout, np.hstack([I_var, Vb])
            )

            metrics["I(I;Vout|E,Vb)"] = self._estimate_cmi(
                "I(I;Vout|E,Vb)", I_var, Vout, np.hstack([E, Vb])
            )

            metrics["I(Vb;Vout|E,I)"] = self._estimate_cmi(
                "I(Vb;Vout|E,I)", Vb, Vout, EI
            )

            # Unique class information in E / I beyond upstream input
            metrics["I(E;C|Vb)"] = self._estimate_cmi("I(E;C|Vb)", E, C_col, Vb)

            metrics["I(I;C|Vb)"] = self._estimate_cmi("I(I;C|Vb)", I_var, C_col, Vb)

            # Unique class information in (E,I) beyond upstream input (proxy for "all synapses")
            metrics["I(E,I;C|Vb)"] = self._estimate_cmi("I(E,I;C|Vb)", EI, C_col, Vb)

            # Fine-grained: unique info in E (or I) beyond upstream + the other synapse type
            metrics["I(E;C|I,Vb)"] = self._estimate_cmi(
                "I(E;C|I,Vb)", E, C_col, np.hstack([I_var, Vb])
            )

            metrics["I(I;C|E,Vb)"] = self._estimate_cmi(
                "I(I;C|E,Vb)", I_var, C_col, np.hstack([E, Vb])
            )
        elif self.compute_ablation_aligned_mi:
            # No upstream branch input (outermost layer): still compute unique info between E and I
            metrics["I(E;C|I)"] = self._estimate_cmi("I(E;C|I)", E, C_col, I_var)

            metrics["I(I;C|E)"] = self._estimate_cmi("I(I;C|E)", I_var, C_col, E)

            # Information-processing conditionals without upstream input.
            metrics["I(E;Vout|I)"] = self._estimate_cmi("I(E;Vout|I)", E, Vout, I_var)

            metrics["I(I;Vout|E)"] = self._estimate_cmi("I(I;Vout|E)", I_var, Vout, E)

            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = self._estimate_cmi(
                "I(Vout;C|E,I)", Vout, C_col, EI
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            metrics["I(E;Vinf|C)"] = self._estimate_cmi("I(E;Vinf|C)", E, Vinf, C_col)

            # I(I;Vinf|C)
            metrics["I(I;Vinf|C)"] = self._estimate_cmi(
                "I(I;Vinf|C)", I_var, Vinf, C_col
            )

            # Somatic processing conditionals (low-dimensional and useful for "Vout -> Vinf -> C" chain).
            if self.compute_ablation_aligned_mi:
                metrics["I(Vinf;C|Vout)"] = self._estimate_cmi(
                    "I(Vinf;C|Vout)", Vinf, C_col, Vout
                )

                metrics["I(Vout;C|Vinf)"] = self._estimate_cmi(
                    "I(Vout;C|Vinf)", Vout, C_col, Vinf
                )

        self.logger.debug("  Conditional MI computation completed.")
        return metrics

    def _validate_inputs(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
    ):
        """Validate input arrays."""
        n_samples = len(E)

        if len(I_var) != n_samples:
            raise ValueError(
                f"I must have same number of samples as E: {len(I_var)} vs {n_samples}"
            )
        if len(Vout) != n_samples:
            raise ValueError(
                f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}"
            )
        if len(C) != n_samples:
            raise ValueError(
                f"C must have same number of samples as E: {len(C)} vs {n_samples}"
            )
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(
                f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}"
            )

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        return ensure_2d(arr)

    def get_summary(self, results: dict[str, dict[str, dict | float]]) -> str:
        """Generate a summary string of the results."""
        summary_lines = [
            "Information Analysis Summary",
            f"Method: {results['method']}",
            f"Samples: {results['n_samples']}",
            "",
        ]

        if "basic_mi" in results:
            summary_lines.append("Basic MI with C:")
            for key, value in results["basic_mi"].items():
                if not key.endswith("_normalized"):
                    summary_lines.append(f"  {key}: {value:.4f}")

        if "pairwise_mi" in results:
            summary_lines.append("\nPairwise MI:")
            for key, value in results["pairwise_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "conditional_mi" in results:
            summary_lines.append("\nConditional MI given C:")
            for key, value in results["conditional_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "gaussian_fisher" in results:
            summary_lines.append("\nGaussian/Fisher proxy:")
            for key, value in results["gaussian_fisher"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "pid" in results:
            summary_lines.append("\nPID Results:")
            for pid_key, pid_values in results["pid"].items():
                summary_lines.append(f"  {pid_key}:")
                for metric, value in pid_values.items():
                    summary_lines.append(f"    {metric}: {value:.4f}")

        return "\n".join(summary_lines)

    def compute_general_class_mi(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        dims: list[int] | None = None,
    ) -> dict:
        """Compute MI between activations (optionally specific dims) and class labels using all methods."""
        results = {}
        acts = activations if dims is None else activations[:, dims]
        for method, estimator in self.estimators.items():
            mi = estimator.mutual_information(acts, labels.reshape(-1, 1))
            results[f"{method}_mi_class"] = mi
            if self.normalize_mi:
                h_labels = estimator.entropy(labels.reshape(-1, 1))
                results[f"{method}_normalized_mi_class"] = (
                    mi / h_labels if h_labels > 0 else 0
                )
        return results

    def compute_metrics(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I_var : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        Vb : np.ndarray, optional
            Branch input (upstream dendritic input), shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I_var, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I_var)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D

        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)

        if Vb is not None:
            Vb = self._ensure_2d(Vb)

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I_var, Vout, C, Vinf, Vb)

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I_var, Vout, Vinf, Vb)

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. Gaussian/Fisher MI proxy
        if self.compute_gaussian_fisher:
            self.logger.info("Computing Gaussian/Fisher proxy metrics...")
            results["gaussian_fisher"] = self._compute_gaussian_fisher(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(E, I_var, Vout, C)

        return results

    def _average_mi_results(
        self, results_list: list[dict[str, dict[str, dict | float]]]
    ) -> dict[str, Any]:
        """Average MI results across multiple branches."""
        return average_mi_results(results_list, method=self.method)

    @staticmethod
    def _is_discrete_class_labels(labels: np.ndarray) -> bool:
        """Heuristic: decide whether labels are discrete class IDs."""
        return is_discrete_class_labels(labels)

    @staticmethod
    def _discrete_entropy_nats(labels: np.ndarray) -> float:
        """Discrete entropy H(C) in natural units (nats)."""
        return discrete_entropy_nats(labels)

    @staticmethod
    def _metric_targets_class(metric_key: str) -> bool:
        """Return True if the metric is of the form I(X;C) or I(X;C|...)."""
        return metric_targets_class(metric_key)

    def _compute_layer_statistics(
        self,
        results_list: list[dict[str, Any]],
        *,
        entropy_C: float | None = None,
        topk: int = 10,
    ) -> dict[str, float]:
        """Compute per-layer summary stats from per-branch MI results."""
        return compute_layer_statistics(
            results_list,
            entropy_C=entropy_C,
            topk=topk,
        )

    @staticmethod
    def _extract_metric_tokens(metric_key: str) -> list[str]:
        """Extract variable tokens from metric keys like 'I(E;Vb|C)'.

        Returns an empty list for non-metric keys.
        """
        return extract_metric_tokens(metric_key)

    @staticmethod
    def _rename_metric_key_tokens(metric_key: str, token_map: dict[str, str]) -> str:
        """Rename variable tokens inside MI keys.

        This is used to generate LDA/shuffled metric names (e.g. E -> E_lin) while
        preserving separators, and it works for tokens in both the main and
        conditioning parts (after '|').
        """
        return rename_metric_key_tokens(metric_key, token_map)

    def _keep_metric_key_given_availability(
        self,
        metric_key: str,
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> bool:
        """Return True if the metric is meaningful given available variables.

        We drop any metric that references:
        - E / E_* when there are no excitatory synapses
        - I / I_* when there are no inhibitory synapses
        - Vb / Vb_* when branch input does not exist (e.g., outermost distal layer)
        """
        return keep_metric_key_given_availability(
            metric_key,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    def _filter_result_metrics_in_place(
        self,
        result: dict[str, Any],
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> None:
        """Remove metrics that depend on non-existent variables (in-place)."""
        filter_result_metrics_in_place(
            result,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    @staticmethod
    def _should_convert_information_key_to_bits(key: Any) -> bool:
        """Heuristic: return True if `key` is an MI/CMI-like metric name in nats.

        We convert keys that contain an MI-style token (e.g. "I(E;C)") and are
        *not* clearly dimensionless or in non-MI units.
        """
        return should_convert_information_key_to_bits(key)

    def _convert_information_units_in_place(self, obj: Any) -> None:
        """Convert MI/CMI-like quantities in-place from nats -> bits.

        Estimators operate in nats internally. Many plots and summaries are in bits,
        so we convert any MI/CMI-like metrics (including *_std and *_null variants)
        before saving/plotting when `compute.output_units="bits"`.
        """
        convert_information_units_in_place(
            obj,
            output_units=getattr(self, "output_units", "bits"),
        )

    def _add_label_shuffle_null_baselines_in_place(
        self,
        result: dict[str, Any],
        *,
        E: np.ndarray | None,
        I_var: np.ndarray | None,
        Vout: np.ndarray | None,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
        has_exc_synapses: bool = True,
        has_inh_synapses: bool = True,
        has_branch_input: bool = True,
        seed_offset: int = 0,
    ) -> None:
        r"""Add permutation (*_null) baselines for selected MI/CMI metrics (in-place).

        This is a finite-sample bias / null diagnostic: it permutes *one of the
        variables* across samples and recomputes MI/CMI using the *same* estimator
        selected by ``estimator.method``.

        Notes
        -----
        - For ``I(X;Y)`` we use ``estimator.mutual_information_null(X, Y)`` which
          permutes ``Y`` across samples.
        - For ``I(X;Y|Z)`` we permute ``Y`` and recompute
          ``estimator.conditional_mutual_information(X, Y_perm, Z)``.
          If the conditioning variable is exactly ``Z=C`` (discrete labels),
          we permute ``Y`` \emph{within each class} to preserve the marginal
          distribution of ``Y|C`` while destroying within-class dependence.
        """
        if int(getattr(self, "mi_null_shuffles", 0) or 0) <= 0:
            return

        null_components = getattr(self, "mi_null_components", {"basic"})
        want_all = "all" in null_components
        want_basic = want_all or ("basic" in null_components)
        want_pairwise = want_all or ("pairwise" in null_components)
        want_conditional = want_all or bool(
            {"conditional", "ablation_aligned", "upstream_unique_cmi"}.intersection(
                null_components
            )
        )
        categories: list[str] = []
        if want_basic:
            categories.append("basic_mi")
        if want_pairwise:
            categories.append("pairwise_mi")
        if want_conditional:
            categories.append("conditional_mi")
        if not categories:
            return

        estimator = self.estimators[self.method]
        n_shuf = int(self.mi_null_shuffles)
        seed_base = int(getattr(self, "mi_null_seed", 0) or 0) + int(seed_offset)
        C_vec = np.asarray(C).ravel()
        C_col = C_vec.reshape(-1, 1)

        def _var_data(token: str) -> np.ndarray | None:
            return _metric_variable_for_null_baseline(
                token,
                E=E,
                I_var=I_var,
                Vout=Vout,
                C_col=C_col,
                Vinf=Vinf,
                Vb=Vb,
                has_exc_synapses=has_exc_synapses,
                has_inh_synapses=has_inh_synapses,
                has_branch_input=has_branch_input,
            )

        perms: list[np.ndarray] | None = None
        seed_off = 0
        for category in categories:
            if category not in result or not isinstance(result[category], dict):
                continue
            metrics_dict: dict[str, Any] = result[category]
            for metric_key in list(metrics_dict.keys()):
                if not isinstance(metric_key, str):
                    continue
                if metric_key.endswith(("_null", "_std", "_normalized")):
                    continue
                if f"{metric_key}_null" in metrics_dict:
                    continue

                parsed = _parse_mi_key_for_null_baseline(metric_key)
                if parsed is None:
                    continue
                x_vars, y_var, z_vars = parsed

                x_parts = [_var_data(t) for t in x_vars]
                if any(p is None for p in x_parts):
                    continue
                x = np.hstack([p for p in x_parts if p is not None])

                y_arr = _var_data(y_var)
                if y_arr is None:
                    continue

                # Unconditional: I(X;Y)
                if not z_vars:
                    null_mean, null_std = estimator.mutual_information_null(
                        x,
                        y_arr,
                        n_shuffles=n_shuf,
                        seed=seed_base + seed_off,
                    )
                    metrics_dict[f"{metric_key}_null"] = float(null_mean)
                    metrics_dict[f"{metric_key}_null_std"] = float(null_std)
                    seed_off += 1
                    continue

                # Conditional: I(X;Y|Z)
                z_parts = [_var_data(t) for t in z_vars]
                if any(p is None for p in z_parts):
                    continue
                z = np.hstack([p for p in z_parts if p is not None])

                stratify_by_c = len(z_vars) == 1 and z_vars[0].split("_", 1)[0] == "C"
                if perms is None:
                    rng = np.random.default_rng(seed_base + 10_000)
                    perms = [rng.permutation(C_vec.shape[0]) for _ in range(n_shuf)]

                vals: list[float] = []
                for perm in perms:
                    if stratify_by_c:
                        # Permute Y within each class label to preserve Y|C.
                        y_perm = np.asarray(y_arr).copy()
                        for c_val in np.unique(C_vec):
                            idx = np.flatnonzero(C_vec == c_val)
                            if idx.size <= 1:
                                continue
                            idx_perm = rng.permutation(idx)
                            y_perm[idx] = y_arr[idx_perm]
                    else:
                        y_perm = y_arr[perm]
                    vals.append(
                        float(estimator.conditional_mutual_information(x, y_perm, z))
                    )
                metrics_dict[f"{metric_key}_null"] = float(np.mean(vals))
                metrics_dict[f"{metric_key}_null_std"] = float(np.std(vals))

    def _compute_variance_analysis(self, results: dict) -> dict:
        """Compute variance bookkeeping from information-analysis summaries."""
        return compute_variance_analysis(results)


__all__ = ["InformationMetricsMixin"]
