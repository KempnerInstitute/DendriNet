"""Node-perturbation estimates of compartment credit at a fixed checkpoint.

For each selected level of one dendritic tree, the error of every compartment
(the derivative of the loss with respect to the compartment's output, voltage
or injected current) is estimated from loss changes under antithetic
perturbations of that level
(:func:`dendritic_modeling.training.node_perturbation.node_perturbation_estimates`
with :func:`~dendritic_modeling.networks.perturb_compartments`) and compared
with the exact error, the gradient of the loss with respect to a zero
perturbation. Reported per level, dictionary, scale and draw count: cosine over
all compartments, mean cosine within each neuron, sign agreement, relative
error, norm ratio, for structured dictionaries the share of the exact error's
energy in their span, the forward evaluations per example and the wall time
per draw. Final-only; feedforward trees whose levels run
``DendriticBranchLayer.forward``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.tools.path_matched_intervention import (
    inventory_nested_tree,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    NodePerturbationCreditAnalysisParams,
)
from dendritic_modeling.networks import (
    CompartmentPerturbation,
    capture_analysis_currents,
    perturb_compartments,
)
from dendritic_modeling.training.node_perturbation import (
    group_basis,
    node_perturbation_estimates,
)
from dendritic_modeling.utils.general import save_dict
from dendritic_modeling.utils.stable_hash import stable_seed_offset


def _output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"the model returned {type(value).__name__}, not a tensor")
    return value


def per_example_loss(
    output: torch.Tensor, target: torch.Tensor, loss: str
) -> torch.Tensor:
    """Unreduced loss of each example: ``cross_entropy`` (class labels) or ``mse``."""
    if loss == "cross_entropy":
        return F.cross_entropy(output, target.long(), reduction="none")
    return (
        (output - target.reshape(output.shape).to(output.dtype))
        .square()
        .flatten(1)
        .mean(1)
    )


def agreement(
    estimate: torch.Tensor,
    exact: torch.Tensor,
    per_neuron: int,
    projection: torch.Tensor | None = None,
) -> dict[str, float]:
    """Agreement of an estimated error field ``[B, D]`` with the exact one.

    ``within_neuron_cosine`` averages the cosine over (example, neuron) pairs
    whose vectors of ``per_neuron`` compartments are nonzero.
    """

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        norm = a.norm() * b.norm()
        return float(a.flatten().dot(b.flatten()) / norm) if norm > 0 else math.nan

    batch = estimate.shape[0]
    a = estimate.reshape(batch, -1, per_neuron)
    b = exact.reshape(batch, -1, per_neuron)
    norms = a.norm(dim=-1) * b.norm(dim=-1)
    keep = norms > 0
    within = (a * b).sum(-1)[keep] / norms[keep]
    nonzero = exact.abs() > 1e-12 * exact.abs().max()
    result = {
        "cosine": cosine(estimate, exact),
        "within_neuron_cosine": float(within.mean()) if within.numel() else math.nan,
        "sign_agreement": float(
            (estimate.sign() == exact.sign())[nonzero].double().mean()
        ),
        "relative_error": float((estimate - exact).norm() / exact.norm()),
        "norm_ratio": float(estimate.norm() / exact.norm()),
    }
    if projection is not None:
        result["projection_energy"] = float(
            projection.square().sum() / exact.square().sum()
        )
        result["cosine_to_projection"] = cosine(estimate, projection)
    return result


class NodePerturbationCreditAnalyzer(AbstractAnalyzer):
    """Estimate compartment credit by node perturbation and compare it with exact errors."""

    def __init__(self, params: NodePerturbationCreditAnalysisParams):
        super().__init__("NodePerturbationCreditAnalyzer")
        self.params = params

    def analyze(
        self,
        model: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        device: str | torch.device = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Run every configured level, dictionary, scale and draw count."""
        if training:
            raise ValueError("node-perturbation credit is a final-only analysis")
        params = self.params
        is_recurrent = getattr(
            getattr(model, "core_network", None), "is_recurrent", False
        )
        if callable(is_recurrent):
            is_recurrent = is_recurrent()
        if bool(is_recurrent):
            raise ValueError("node-perturbation credit requires a feedforward model")
        modules, n_somas, branch_factors = inventory_nested_tree(
            model, module_name_prefix=params.module_name_prefix
        )
        missing = [depth for depth in params.depths if depth not in modules]
        if missing:
            raise ValueError(f"the tree has depths {sorted(modules)}; not {missing}")
        tensors = materialize_dataset(
            test_dataset, runtime, explicit_max_samples=params.max_examples
        )
        x, y = tensors[0], tensors[1]
        rows: list[dict[str, Any]] = []
        with analysis_device_context(model, device) as target:
            x, y = x.to(target), y.to(target)
            dtype = next(model.parameters()).dtype
            for depth in params.depths:
                name, layer = modules[depth]
                rows.extend(
                    self._level(
                        model,
                        layer,
                        name,
                        depth,
                        x,
                        y,
                        n_somas,
                        branch_factors,
                        dtype,
                        target,
                    )
                )
        result = {
            "coordinate": params.coordinate,
            "loss": params.loss,
            "n_examples": int(x.shape[0]),
            "n_somas": int(n_somas),
            "branch_factors": [int(f) for f in branch_factors],
            "modules": {str(depth): modules[depth][0] for depth in params.depths},
            "seed": int(params.seed),
            "rows": rows,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{Path(filename).stem}.json")
        return result

    def _level(
        self, model, layer, name, depth, x, y, n_somas, branch_factors, dtype, device
    ):
        params = self.params
        width = int(layer.branch_config.output_dim)
        per_neuron = width // n_somas
        batch = int(x.shape[0])
        exact = self._exact_errors(model, layer, x, y, width, dtype, device)
        reference = self._coordinate_rms(model, layer, x)

        def loss(offset: torch.Tensor) -> torch.Tensor:
            copies = offset.shape[0]
            repeated_x = x.repeat(copies, *([1] * (x.ndim - 1)))
            repeated_y = y.repeat(copies, *([1] * (y.ndim - 1)))
            perturbation = CompartmentPerturbation(
                offset.reshape(copies * batch, width), params.coordinate
            )
            with perturb_compartments({layer: perturbation}):
                output = _output_tensor(model(repeated_x))
            return per_example_loss(output, repeated_y, params.loss).reshape(
                copies, batch
            )

        rows = []
        for dictionary in params.dictionaries:
            basis = self._basis(
                dictionary, depth, width, n_somas, branch_factors, dtype, device
            )
            projection = None if basis is None else (exact @ basis) @ basis.T
            energies = [("scalar", 1.0)]
            if basis is not None and params.match_perturbation_energy:
                energies.append(("matched", math.sqrt(width / basis.shape[1])))
            for energy, factor in energies:
                for relative in params.relative_scales:
                    scale = relative * reference * factor
                    seed = (
                        int(params.seed)
                        + stable_seed_offset(depth, dictionary, energy, relative)
                    ) % 2**63
                    generator = torch.Generator(device=device).manual_seed(seed)
                    started = time.perf_counter()
                    found = node_perturbation_estimates(
                        loss,
                        (batch, width),
                        scale=scale,
                        draws=params.draws,
                        generator=generator,
                        basis=basis,
                        chunk=params.chunk,
                        dtype=dtype,
                        device=device,
                    )
                    seconds_per_draw = (time.perf_counter() - started) / max(
                        found.estimates
                    )
                    for draws, estimate in found.estimates.items():
                        rows.append(
                            {
                                "depth": int(depth),
                                "module": name,
                                "dictionary": dictionary,
                                "energy": energy,
                                "relative_scale": float(relative),
                                "scale": float(scale),
                                "rank": int(found.rank),
                                "draws": int(draws),
                                "evaluations_per_example": found.evaluations_per_example(
                                    draws
                                ),
                                "seconds_per_draw": float(seconds_per_draw),
                                **agreement(estimate, exact, per_neuron, projection),
                            }
                        )
        return rows

    def _exact_errors(self, model, layer, x, y, width, dtype, device):
        offset = torch.zeros(
            x.shape[0], width, dtype=dtype, device=device, requires_grad=True
        )
        with perturb_compartments(
            {layer: CompartmentPerturbation(offset, self.params.coordinate)}
        ):
            output = _output_tensor(model(x))
        loss = per_example_loss(output, y, self.params.loss).sum()
        return torch.autograd.grad(loss, offset)[0].detach()

    def _coordinate_rms(self, model, layer, x) -> float:
        """RMS of the perturbed quantity on the analysed examples (the scales' unit)."""
        with torch.no_grad(), capture_analysis_currents(layer):
            model(x)
            currents = layer.get_analysis_currents()
        if self.params.coordinate == "output":
            value = currents["post_gate_output"]
        elif self.params.coordinate == "voltage":
            value = currents["pre_gate_voltage"]
        else:
            # The shunting numerator that an injected current joins.
            parts = [
                currents[k]
                for k in ("pre_gate_excitation_current", "pre_gate_upstream_current")
                if currents[k] is not None
            ]
            if not parts:
                raise ValueError(
                    "a current perturbation needs a layer with excitatory or upstream current"
                )
            value = sum(parts)
        rms = float(value.double().square().mean().sqrt())
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError(
                f"the perturbed quantity has RMS {rms}; relative scales are undefined"
            )
        return rms

    @staticmethod
    def _basis(dictionary, depth, width, n_somas, branch_factors, dtype, device):
        if dictionary == "independent":
            return None
        if dictionary == "neuron":
            per_neuron = width // n_somas
            groups = [
                range(s * per_neuron, (s + 1) * per_neuron) for s in range(n_somas)
            ]
        else:  # siblings: the children of each parent, consecutive in the level's output
            if depth == 0:
                raise ValueError("the soma level has no sibling groups")
            factor = int(branch_factors[depth - 1])
            groups = [
                range(g * factor, (g + 1) * factor) for g in range(width // factor)
            ]
        return group_basis(
            [list(group) for group in groups], width, dtype=dtype, device=device
        )


__all__ = ["NodePerturbationCreditAnalyzer", "agreement", "per_example_loss"]
