"""Mutual-information metric helpers for information analysis."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from dendritic_modeling.analysis.core.information_validation_mixin import (
    InformationValidationMixin,
)


class InformationMutualInformationMixin(InformationValidationMixin):
    """Computes basic, pairwise, conditional, and class MI metrics."""

    def compute_class_mi(self, activations, labels):
        return self.estimators[self.method].mutual_information(activations, labels)

    def compute_conditional_mi(self, activations, labels, condition):
        return self.estimators[self.method].conditional_mutual_information(
            activations, labels, condition
        )

    def _estimate_timed_metric(
        self,
        metric_key: str,
        estimate_fn: Callable[[], float],
    ) -> float:
        """Compute one metric while preserving the debug timing format."""
        self.logger.debug(f"  Computing {metric_key}...")
        start_time = time.time()
        value = estimate_fn()
        self.logger.debug(
            f"  {metric_key} = {value:.4f} (took {time.time() - start_time:.1f}s)"
        )
        return value

    def _estimate_mi(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute one MI metric with the existing debug timing format."""
        return self._estimate_timed_metric(
            metric_key,
            lambda: self.estimators[self.method].mutual_information(x, y),
        )

    def _estimate_cmi(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
        condition: np.ndarray,
    ) -> float:
        """Compute one conditional MI metric with the existing debug timing format."""
        return self._estimate_timed_metric(
            metric_key,
            lambda: self.estimators[self.method].conditional_mutual_information(
                x, y, condition
            ),
        )

    def _estimate_cmi_or_zero(
        self,
        metric_key: str,
        x: np.ndarray,
        y: np.ndarray,
        condition: np.ndarray,
    ) -> float:
        """Compute one conditional MI metric, returning zero on estimator failure."""

        def estimate() -> float:
            try:
                return self.estimators[self.method].conditional_mutual_information(
                    x, y, condition
                )
            except Exception as e:
                self.logger.warning(f"  Failed to compute {metric_key}: {e}")
                return 0.0

        return self._estimate_timed_metric(
            metric_key,
            estimate,
        )

    def _add_mi_metrics(
        self,
        metrics: dict[str, float],
        metric_specs: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> None:
        """Estimate ordered MI metric specs into an existing metrics dict."""
        for metric_key, x, y in metric_specs:
            metrics[metric_key] = self._estimate_mi(metric_key, x, y)

    def _add_cmi_metrics(
        self,
        metrics: dict[str, float],
        metric_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Estimate ordered conditional MI specs into an existing metrics dict."""
        for metric_key, x, y, condition in metric_specs:
            metrics[metric_key] = self._estimate_cmi(metric_key, x, y, condition)

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

        self._add_mi_metrics(
            metrics,
            [
                ("I(E;C)", E, C),
                ("I(I;C)", I_var, C),
            ],
        )

        if Vb is not None:
            self._add_mi_metrics(metrics, [("I(Vb;C)", Vb, C)])

        self._add_mi_metrics(metrics, [("I(Vout;C)", Vout, C)])

        if Vinf is not None and self.include_vinf:
            self._add_mi_metrics(metrics, [("I(Vinf;C)", Vinf, C)])

        EI = np.hstack([E, I_var])
        self._add_mi_metrics(metrics, [("I(E,I;C)", EI, C)])

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

        self._add_mi_metrics(
            metrics,
            [
                ("I(E;I)", E, I_var),
                ("I(E;Vout)", E, Vout),
                ("I(I;Vout)", I_var, Vout),
            ],
        )

        if Vb is not None:
            self._add_mi_metrics(
                metrics,
                [
                    ("I(E;Vb)", E, Vb),
                    ("I(I;Vb)", I_var, Vb),
                    ("I(Vb;Vout)", Vb, Vout),
                ],
            )

        if Vinf is not None and self.include_vinf:
            self._add_mi_metrics(
                metrics,
                [
                    ("I(E;Vinf)", E, Vinf),
                    ("I(I;Vinf)", I_var, Vinf),
                    ("I(Vout;Vinf)", Vout, Vinf),
                ],
            )

        self.logger.debug("  Pairwise MI computation completed.")
        return metrics

    def _add_ablation_aligned_conditional_metrics(
        self,
        metrics: dict[str, float],
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C_col: np.ndarray,
        Vb: np.ndarray | None,
    ) -> None:
        """Append ablation-aligned conditional MI metrics in legacy order."""
        # One-sided conditional class information in the dendritic output.
        # These are useful for interaction-information identities such as
        # I(Vout;C) - I(Vout;C|E) = I(E;Vout) - I(E;Vout|C).
        self._add_cmi_metrics(
            metrics,
            [
                ("I(Vout;C|E)", Vout, C_col, E),
                ("I(Vout;C|I)", Vout, C_col, I_var),
            ],
        )

        if Vb is not None:
            # How much class info is present in the layer output beyond upstream input
            # Complement: how much upstream class information is *not* reflected in the layer output
            # (helps interpret upstream vs synapse contributions).
            self._add_cmi_metrics(
                metrics,
                [
                    ("I(Vout;C|Vb)", Vout, C_col, Vb),
                    ("I(Vb;C|Vout)", Vb, C_col, Vout),
                ],
            )

            # How much class information in Vout remains after conditioning on synapse inputs
            # (proxy for upstream contribution to class information in Vout).
            EI = np.hstack([E, I_var])
            self._add_cmi_metrics(metrics, [("I(Vout;C|E,I)", Vout, C_col, EI)])

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
            self._add_cmi_metrics(
                metrics,
                [("I(E;Vout|I,Vb)", E, Vout, np.hstack([I_var, Vb]))],
            )

            self._add_cmi_metrics(
                metrics,
                [("I(I;Vout|E,Vb)", I_var, Vout, np.hstack([E, Vb]))],
            )

            self._add_cmi_metrics(metrics, [("I(Vb;Vout|E,I)", Vb, Vout, EI)])

            # Unique class information in E / I beyond upstream input
            # Unique class information in (E,I) beyond upstream input (proxy for "all synapses")
            self._add_cmi_metrics(
                metrics,
                [
                    ("I(E;C|Vb)", E, C_col, Vb),
                    ("I(I;C|Vb)", I_var, C_col, Vb),
                    ("I(E,I;C|Vb)", EI, C_col, Vb),
                ],
            )

            # Fine-grained: unique info in E (or I) beyond upstream + the other synapse type
            self._add_cmi_metrics(
                metrics,
                [("I(E;C|I,Vb)", E, C_col, np.hstack([I_var, Vb]))],
            )

            self._add_cmi_metrics(
                metrics,
                [("I(I;C|E,Vb)", I_var, C_col, np.hstack([E, Vb]))],
            )
            return

        # No upstream branch input (outermost layer): still compute unique info between E and I
        # Information-processing conditionals without upstream input.
        self._add_cmi_metrics(
            metrics,
            [
                ("I(E;C|I)", E, C_col, I_var),
                ("I(I;C|E)", I_var, C_col, E),
                ("I(E;Vout|I)", E, Vout, I_var),
                ("I(I;Vout|E)", I_var, Vout, E),
            ],
        )

        EI = np.hstack([E, I_var])
        self._add_cmi_metrics(metrics, [("I(Vout;C|E,I)", Vout, C_col, EI)])

    def _add_vinf_conditional_metrics(
        self,
        metrics: dict[str, float],
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C_col: np.ndarray,
        Vinf: np.ndarray,
    ) -> None:
        """Append soma-activation conditional MI metrics in legacy order."""
        self._add_cmi_metrics(
            metrics,
            [
                ("I(E;Vinf|C)", E, Vinf, C_col),
                ("I(I;Vinf|C)", I_var, Vinf, C_col),
            ],
        )

        # Somatic processing conditionals (low-dimensional and useful for "Vout -> Vinf -> C" chain).
        if self.compute_ablation_aligned_mi:
            self._add_cmi_metrics(
                metrics,
                [
                    ("I(Vinf;C|Vout)", Vinf, C_col, Vout),
                    ("I(Vout;C|Vinf)", Vout, C_col, Vinf),
                ],
            )

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

        self._add_cmi_metrics(
            metrics,
            [
                ("I(E;I|C)", E, I_var, C_col),
                ("I(E;Vout|C)", E, Vout, C_col),
                ("I(I;Vout|C)", I_var, Vout, C_col),
            ],
        )

        if Vb is not None:
            self._add_cmi_metrics(
                metrics,
                [
                    ("I(E;Vb|C)", E, Vb, C_col),
                    ("I(I;Vb|C)", I_var, Vb, C_col),
                    ("I(Vb;Vout|C)", Vb, Vout, C_col),
                ],
            )

        # ------------------------------------------------------------------
        # Ablation-aligned conditional MI proxies
        #
        # These measure *unique* class information that each component provides
        # beyond other components, which is often closer to what layer ablations
        # probe than raw I(E;C) / I(I;C).
        # ------------------------------------------------------------------
        if self.compute_ablation_aligned_mi:
            self._add_ablation_aligned_conditional_metrics(
                metrics,
                E,
                I_var,
                Vout,
                C_col,
                Vb,
            )

        if Vinf is not None and self.include_vinf:
            self._add_vinf_conditional_metrics(
                metrics,
                E,
                I_var,
                Vout,
                C_col,
                Vinf,
            )

        self.logger.debug("  Conditional MI computation completed.")
        return metrics

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


__all__ = ["InformationMutualInformationMixin"]
