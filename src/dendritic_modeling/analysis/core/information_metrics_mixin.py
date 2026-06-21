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
    fisher_dprime2_binary,
    fisher_dprime2_pairwise_avg,
    fisher_mi_from_f,
    gaussian_fisher_proxy,
    is_discrete_class_labels,
    keep_metric_key_given_availability,
    metric_targets_class,
    rename_metric_key_tokens,
    should_convert_information_key_to_bits,
)
from dendritic_modeling.analysis.core.information_results import (
    average_mi_results,
    compute_layer_statistics,
    compute_variance_analysis,
    filter_result_metrics_in_place,
)
from dendritic_modeling.utils.information.pid import pid_synergy_redundancy


class InformationMetricsMixin:
    """Computes MI/CMI/PID/Fisher metrics and result summaries."""

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
        self.logger.debug("  Computing I(E;C)...")
        start_time = time.time()
        metrics["I(E;C)"] = self.estimators[self.method].mutual_information(E, C)
        self.logger.debug(
            f"  I(E;C) = {metrics['I(E;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;C)
        self.logger.debug("  Computing I(I;C)...")
        start_time = time.time()
        metrics["I(I;C)"] = self.estimators[self.method].mutual_information(I_var, C)
        self.logger.debug(
            f"  I(I;C) = {metrics['I(I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vb;C) if available
        if Vb is not None:
            self.logger.debug("  Computing I(Vb;C)...")
            start_time = time.time()
            metrics["I(Vb;C)"] = self.estimators[self.method].mutual_information(Vb, C)
            self.logger.debug(
                f"  I(Vb;C) = {metrics['I(Vb;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # I(Vout;C)
        self.logger.debug("  Computing I(Vout;C)...")
        start_time = time.time()
        metrics["I(Vout;C)"] = self.estimators[self.method].mutual_information(Vout, C)
        self.logger.debug(
            f"  I(Vout;C) = {metrics['I(Vout;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            self.logger.debug("  Computing I(Vinf;C)...")
            start_time = time.time()
            metrics["I(Vinf;C)"] = self.estimators[self.method].mutual_information(
                Vinf, C
            )
            self.logger.debug(
                f"  I(Vinf;C) = {metrics['I(Vinf;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # Joint MI: I(E,I;C)
        self.logger.debug("  Computing I(E,I;C)...")
        start_time = time.time()
        EI = np.hstack([E, I_var])
        metrics["I(E,I;C)"] = self.estimators[self.method].mutual_information(EI, C)
        self.logger.debug(
            f"  I(E,I;C) = {metrics['I(E,I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

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
        self.logger.debug("  Computing I(E;I)...")
        start_time = time.time()
        metrics["I(E;I)"] = self.estimators[self.method].mutual_information(E, I_var)
        self.logger.debug(
            f"  I(E;I) = {metrics['I(E;I)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout)
        self.logger.debug("  Computing I(E;Vout)...")
        start_time = time.time()
        metrics["I(E;Vout)"] = self.estimators[self.method].mutual_information(E, Vout)
        self.logger.debug(
            f"  I(E;Vout) = {metrics['I(E;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout)
        self.logger.debug("  Computing I(I;Vout)...")
        start_time = time.time()
        metrics["I(I;Vout)"] = self.estimators[self.method].mutual_information(
            I_var, Vout
        )
        self.logger.debug(
            f"  I(I;Vout) = {metrics['I(I;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Pairwise MI with Vb if available
        if Vb is not None:
            # I(E;Vb)
            self.logger.debug("  Computing I(E;Vb)...")
            start_time = time.time()
            metrics["I(E;Vb)"] = self.estimators[self.method].mutual_information(E, Vb)
            self.logger.debug(
                f"  I(E;Vb) = {metrics['I(E;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb)
            self.logger.debug("  Computing I(I;Vb)...")
            start_time = time.time()
            metrics["I(I;Vb)"] = self.estimators[self.method].mutual_information(
                I_var, Vb
            )
            self.logger.debug(
                f"  I(I;Vb) = {metrics['I(I;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout)
            self.logger.debug("  Computing I(Vb;Vout)...")
            start_time = time.time()
            metrics["I(Vb;Vout)"] = self.estimators[self.method].mutual_information(
                Vb, Vout
            )
            self.logger.debug(
                f"  I(Vb;Vout) = {metrics['I(Vb;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            self.logger.debug("  Computing I(E;Vinf)...")
            start_time = time.time()
            metrics["I(E;Vinf)"] = self.estimators[self.method].mutual_information(
                E, Vinf
            )
            self.logger.debug(
                f"  I(E;Vinf) = {metrics['I(E;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf)
            self.logger.debug("  Computing I(I;Vinf)...")
            start_time = time.time()
            metrics["I(I;Vinf)"] = self.estimators[self.method].mutual_information(
                I_var, Vinf
            )
            self.logger.debug(
                f"  I(I;Vinf) = {metrics['I(I;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vout;Vinf)
            self.logger.debug("  Computing I(Vout;Vinf)...")
            start_time = time.time()
            metrics["I(Vout;Vinf)"] = self.estimators[self.method].mutual_information(
                Vout, Vinf
            )
            self.logger.debug(
                f"  I(Vout;Vinf) = {metrics['I(Vout;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

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

        # I(E;I|C)
        self.logger.debug("  Computing I(E;I|C)...")
        start_time = time.time()
        metrics["I(E;I|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, I_var, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(E;I|C) = {metrics['I(E;I|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout|C)
        self.logger.debug("  Computing I(E;Vout|C)...")
        start_time = time.time()
        metrics["I(E;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, Vout, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(E;Vout|C) = {metrics['I(E;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout|C)
        self.logger.debug("  Computing I(I;Vout|C)...")
        start_time = time.time()
        metrics["I(I;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(I_var, Vout, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(I;Vout|C) = {metrics['I(I;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Conditional MI with Vb if available
        if Vb is not None:
            # I(E;Vb|C)
            self.logger.debug("  Computing I(E;Vb|C)...")
            start_time = time.time()
            metrics["I(E;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vb, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(E;Vb|C) = {metrics['I(E;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb|C)
            self.logger.debug("  Computing I(I;Vb|C)...")
            start_time = time.time()
            metrics["I(I;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vb, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(I;Vb|C) = {metrics['I(I;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout|C)
            self.logger.debug("  Computing I(Vb;Vout|C)...")
            start_time = time.time()
            metrics["I(Vb;Vout|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(Vb, Vout, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(Vb;Vout|C) = {metrics['I(Vb;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # ------------------------------------------------------------------
        # Ablation-aligned conditional MI proxies
        #
        # These measure *unique* class information that each component provides
        # beyond other components, which is often closer to what layer ablations
        # probe than raw I(E;C) / I(I;C).
        # ------------------------------------------------------------------
        estimator = self.estimators[self.method]
        C_col = C.reshape(-1, 1)

        if self.compute_ablation_aligned_mi:
            # One-sided conditional class information in the dendritic output.
            # These are useful for interaction-information identities such as
            # I(Vout;C) - I(Vout;C|E) = I(E;Vout) - I(E;Vout|C).
            self.logger.debug("  Computing I(Vout;C|E)...")
            start_time = time.time()
            metrics["I(Vout;C|E)"] = estimator.conditional_mutual_information(
                Vout, C_col, E
            )
            self.logger.debug(
                f"  I(Vout;C|E) = {metrics['I(Vout;C|E)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(Vout;C|I)...")
            start_time = time.time()
            metrics["I(Vout;C|I)"] = estimator.conditional_mutual_information(
                Vout, C_col, I_var
            )
            self.logger.debug(
                f"  I(Vout;C|I) = {metrics['I(Vout;C|I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if self.compute_ablation_aligned_mi and Vb is not None:
            # How much class info is present in the layer output beyond upstream input
            self.logger.debug("  Computing I(Vout;C|Vb)...")
            start_time = time.time()
            metrics["I(Vout;C|Vb)"] = estimator.conditional_mutual_information(
                Vout, C_col, Vb
            )
            self.logger.debug(
                f"  I(Vout;C|Vb) = {metrics['I(Vout;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Complement: how much upstream class information is *not* reflected in the layer output
            # (helps interpret upstream vs synapse contributions).
            self.logger.debug("  Computing I(Vb;C|Vout)...")
            start_time = time.time()
            metrics["I(Vb;C|Vout)"] = estimator.conditional_mutual_information(
                Vb, C_col, Vout
            )
            self.logger.debug(
                f"  I(Vb;C|Vout) = {metrics['I(Vb;C|Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # How much class information in Vout remains after conditioning on synapse inputs
            # (proxy for upstream contribution to class information in Vout).
            self.logger.debug("  Computing I(Vout;C|E,I)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = estimator.conditional_mutual_information(
                Vout, C_col, EI
            )
            self.logger.debug(
                f"  I(Vout;C|E,I) = {metrics['I(Vout;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Optional (can be expensive for high-dimensional synapse sets):
            # unique upstream class information beyond the full synapse set.
            if self.compute_upstream_unique_cmi:
                self.logger.debug("  Computing I(Vb;C|E,I)...")
                start_time = time.time()
                try:
                    metrics["I(Vb;C|E,I)"] = estimator.conditional_mutual_information(
                        Vb, C_col, EI
                    )
                except Exception as e:
                    self.logger.warning(f"  Failed to compute I(Vb;C|E,I): {e}")
                    metrics["I(Vb;C|E,I)"] = 0.0
                self.logger.debug(
                    f"  I(Vb;C|E,I) = {metrics['I(Vb;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

            # ------------------------------------------------------------------
            # Information-processing conditionals (E/I/Vb -> Vout)
            #
            # These quantify *unique* statistical influence on the dendritic output,
            # independent of class labels. They complement ablation results because
            # they isolate each input's contribution to Vout beyond the others.
            # ------------------------------------------------------------------
            self.logger.debug("  Computing I(E;Vout|I,Vb)...")
            start_time = time.time()
            metrics["I(E;Vout|I,Vb)"] = estimator.conditional_mutual_information(
                E, Vout, np.hstack([I_var, Vb])
            )
            self.logger.debug(
                f"  I(E;Vout|I,Vb) = {metrics['I(E;Vout|I,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;Vout|E,Vb)...")
            start_time = time.time()
            metrics["I(I;Vout|E,Vb)"] = estimator.conditional_mutual_information(
                I_var, Vout, np.hstack([E, Vb])
            )
            self.logger.debug(
                f"  I(I;Vout|E,Vb) = {metrics['I(I;Vout|E,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(Vb;Vout|E,I)...")
            start_time = time.time()
            metrics["I(Vb;Vout|E,I)"] = estimator.conditional_mutual_information(
                Vb, Vout, EI
            )
            self.logger.debug(
                f"  I(Vb;Vout|E,I) = {metrics['I(Vb;Vout|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Unique class information in E / I beyond upstream input
            self.logger.debug("  Computing I(E;C|Vb)...")
            start_time = time.time()
            metrics["I(E;C|Vb)"] = estimator.conditional_mutual_information(
                E, C_col, Vb
            )
            self.logger.debug(
                f"  I(E;C|Vb) = {metrics['I(E;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|Vb)...")
            start_time = time.time()
            metrics["I(I;C|Vb)"] = estimator.conditional_mutual_information(
                I_var, C_col, Vb
            )
            self.logger.debug(
                f"  I(I;C|Vb) = {metrics['I(I;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Unique class information in (E,I) beyond upstream input (proxy for "all synapses")
            self.logger.debug("  Computing I(E,I;C|Vb)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(E,I;C|Vb)"] = estimator.conditional_mutual_information(
                EI, C_col, Vb
            )
            self.logger.debug(
                f"  I(E,I;C|Vb) = {metrics['I(E,I;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Fine-grained: unique info in E (or I) beyond upstream + the other synapse type
            self.logger.debug("  Computing I(E;C|I,Vb)...")
            start_time = time.time()
            metrics["I(E;C|I,Vb)"] = estimator.conditional_mutual_information(
                E, C_col, np.hstack([I_var, Vb])
            )
            self.logger.debug(
                f"  I(E;C|I,Vb) = {metrics['I(E;C|I,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|E,Vb)...")
            start_time = time.time()
            metrics["I(I;C|E,Vb)"] = estimator.conditional_mutual_information(
                I_var, C_col, np.hstack([E, Vb])
            )
            self.logger.debug(
                f"  I(I;C|E,Vb) = {metrics['I(I;C|E,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )
        elif self.compute_ablation_aligned_mi:
            # No upstream branch input (outermost layer): still compute unique info between E and I
            self.logger.debug("  Computing I(E;C|I)...")
            start_time = time.time()
            metrics["I(E;C|I)"] = estimator.conditional_mutual_information(
                E, C_col, I_var
            )
            self.logger.debug(
                f"  I(E;C|I) = {metrics['I(E;C|I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|E)...")
            start_time = time.time()
            metrics["I(I;C|E)"] = estimator.conditional_mutual_information(
                I_var, C_col, E
            )
            self.logger.debug(
                f"  I(I;C|E) = {metrics['I(I;C|E)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Information-processing conditionals without upstream input.
            self.logger.debug("  Computing I(E;Vout|I)...")
            start_time = time.time()
            metrics["I(E;Vout|I)"] = estimator.conditional_mutual_information(
                E, Vout, I_var
            )
            self.logger.debug(
                f"  I(E;Vout|I) = {metrics['I(E;Vout|I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;Vout|E)...")
            start_time = time.time()
            metrics["I(I;Vout|E)"] = estimator.conditional_mutual_information(
                I_var, Vout, E
            )
            self.logger.debug(
                f"  I(I;Vout|E) = {metrics['I(I;Vout|E)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(Vout;C|E,I)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = estimator.conditional_mutual_information(
                Vout, C_col, EI
            )
            self.logger.debug(
                f"  I(Vout;C|E,I) = {metrics['I(Vout;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            self.logger.debug("  Computing I(E;Vinf|C)...")
            start_time = time.time()
            metrics["I(E;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vinf, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(E;Vinf|C) = {metrics['I(E;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf|C)
            self.logger.debug("  Computing I(I;Vinf|C)...")
            start_time = time.time()
            metrics["I(I;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vinf, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(I;Vinf|C) = {metrics['I(I;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Somatic processing conditionals (low-dimensional and useful for "Vout -> Vinf -> C" chain).
            if self.compute_ablation_aligned_mi:
                self.logger.debug("  Computing I(Vinf;C|Vout)...")
                start_time = time.time()
                metrics["I(Vinf;C|Vout)"] = estimator.conditional_mutual_information(
                    Vinf, C_col, Vout
                )
                self.logger.debug(
                    f"  I(Vinf;C|Vout) = {metrics['I(Vinf;C|Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

                self.logger.debug("  Computing I(Vout;C|Vinf)...")
                start_time = time.time()
                metrics["I(Vout;C|Vinf)"] = estimator.conditional_mutual_information(
                    Vout, C_col, Vinf
                )
                self.logger.debug(
                    f"  I(Vout;C|Vinf) = {metrics['I(Vout;C|Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

        self.logger.debug("  Conditional MI computation completed.")
        return metrics

    def _gaussian_fisher_proxy(
        self, x: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        """Return (F, I_proxy) for I(X;C) under a Gaussian/Fisher proxy."""
        return gaussian_fisher_proxy(
            x,
            labels,
            eps=float(getattr(self, "gaussian_fisher_eps", 1e-8)),
            aggregation=str(getattr(self, "gaussian_fisher_aggregation", "sum")),
            mi_transform=str(
                getattr(self, "gaussian_fisher_mi_transform", "half_log1p")
            ),
        )

    def _fisher_mi_from_f(self, f: float) -> float:
        """Map a Fisher/discriminability-like scalar to an MI-like proxy."""
        return fisher_mi_from_f(
            f,
            mi_transform=str(
                getattr(self, "gaussian_fisher_mi_transform", "half_log1p")
            ),
        )

    def _fisher_dprime2_binary(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Binary-class Fisher discriminability: d'^2 = Δμᵀ Σ^{-1} Δμ (ridge-regularized)."""
        return fisher_dprime2_binary(
            x, labels, eps=float(getattr(self, "gaussian_fisher_eps", 1e-8))
        )

    def _fisher_dprime2_pairwise_avg(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Multi-class surrogate: average binary d'^2 over all class pairs."""
        return fisher_dprime2_pairwise_avg(
            x, labels, eps=float(getattr(self, "gaussian_fisher_eps", 1e-8))
        )

    def _compute_gaussian_fisher(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute Gaussian/Fisher proxy metrics for available variables."""
        metrics: dict[str, float] = {}

        for name, X in [
            ("E", E),
            ("I", I_var),
            ("Vb", Vb),
            ("Vout", Vout),
            ("Vinf", Vinf if (Vinf is not None and self.include_vinf) else None),
        ]:
            if X is None:
                continue
            F, I_proxy = self._gaussian_fisher_proxy(X, C)
            # Use an I(.;.) prefix so LDA/shuffled variants get token-renamed.
            metrics[f"I({name};C)_fisher_F"] = float(F)
            metrics[f"I({name};C)_fisher"] = float(I_proxy)

        # Information-flow-style increment from upstream branches (Vb) beyond local synapses (E,I).
        # This mirrors the Schur-complement increment in the theory draft (Sec. 'Multi-branch Trees
        # and Information Flow') but uses a Fisher/geometry surrogate (d'^2) computed from moments.
        if Vb is not None:
            try:
                EI = np.hstack([E, I_var])
                EIVb = np.hstack([E, I_var, Vb])
                d2_full = self._fisher_dprime2_pairwise_avg(EIVb, C)
                d2_base = self._fisher_dprime2_pairwise_avg(EI, C)
                inc = float(max(0.0, d2_full - d2_base))
                metrics["I(Vb;C|E,I)_fisher_F"] = inc
                metrics["I(Vb;C|E,I)_fisher"] = float(self._fisher_mi_from_f(inc))
            except Exception as e:
                self.logger.warning(
                    f"Failed to compute Fisher flow increment I(Vb;C|E,I)_fisher: {e}"
                )
                metrics["I(Vb;C|E,I)_fisher_F"] = 0.0
                metrics["I(Vb;C|E,I)_fisher"] = 0.0

        return metrics

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        self.logger.info("Computing Partial Information Decomposition (PID)...")
        pid_results = {}

        # Prepare data for PID (may need binarization)
        self.logger.info(f"  Data preprocessing: binarize={self.pid_binarize}")
        if self.pid_binarize:
            self.logger.info(
                f"  Binarizing data using method: {self.pid_binarize_method}"
            )
            start_time = time.time()
            E_pid = self._binarize(E)
            I_pid = self._binarize(I_var)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
            self.logger.info(
                f"  Data binarization completed (took {time.time() - start_time:.1f}s)"
            )
        else:
            E_pid = E
            I_pid = I_var
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)

        def _pid_helper(
            source1_array: np.ndarray,
            source1_name: str,
            source2_array: np.ndarray,
            source2_name: str,
            target_array: np.ndarray,
            target_name: str,
        ) -> None:
            """Helper function to compute PID."""
            pid_key = f"PID({source1_name},{source2_name}->{target_name})"
            self.logger.info(f"  Computing {pid_key}...")
            start_time = time.time()
            try:
                synergy, redundancy = pid_synergy_redundancy(
                    [source1_array, source2_array], target_array
                )

                # Compute unique information
                unique_1 = (
                    self.estimators[self.method].mutual_information(
                        source1_array, target_array
                    )
                    - redundancy
                )
                unique_2 = (
                    self.estimators[self.method].mutual_information(
                        source2_array, target_array
                    )
                    - redundancy
                )

                pid_results[pid_key] = {
                    "synergy": synergy,
                    "redundancy": redundancy,
                    f"unique_{source1_name}": unique_1,
                    f"unique_{source2_name}": unique_2,
                }
                self.logger.info(
                    f"  {pid_key}: synergy={synergy:.4f}, redundancy={redundancy:.4f}, unique_{source1_name}={unique_1:.4f}, unique_{source2_name}={unique_2:.4f} (took {time.time() - start_time:.1f}s)"
                )
            except Exception as e:
                self.logger.warning(
                    f"  {pid_key} computation failed: {e} (took {time.time() - start_time:.1f}s)"
                )

        # PID: (E, I) -> C
        _pid_helper(E_pid, "E", I_pid, "I", C_pid, "C")
        # PID: (E, Vout) -> C
        _pid_helper(E_pid, "E", Vout_pid, "Vout", C_pid, "C")
        # PID: (I, Vout) -> C
        _pid_helper(I_pid, "I", Vout_pid, "Vout", C_pid, "C")
        # PID: (E, I) -> Vout
        _pid_helper(E_pid, "E", I_pid, "I", Vout_pid, "Vout")

        self.logger.info("  PID computation completed.")
        return pid_results

    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Binarize data for PID computation."""
        if self.pid_binarize_method == "median":
            threshold = np.median(data, axis=0)
        elif self.pid_binarize_method == "mean":
            threshold = np.mean(data, axis=0)
        elif self.pid_binarize_method == "quantile":
            threshold = np.quantile(data, self.pid_binarize_threshold, axis=0)
        else:
            raise ValueError(f"Unknown binarization method: {self.pid_binarize_method}")

        binarized: np.ndarray = data > threshold
        return binarized.astype(int)

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

        def _parse_mi_key(mi_key: Any) -> Any:
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

        def _var_data(token: str) -> np.ndarray | None:
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

                parsed = _parse_mi_key(metric_key)
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
