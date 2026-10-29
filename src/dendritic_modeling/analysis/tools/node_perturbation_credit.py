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
per draw. Final-only. The perturbed levels must run
``DendriticBranchLayer.forward`` once per model forward. Recurrent cores, cores
whose autograd credit is not exact (``soma_broadcast``) and models with signal
rounding are refused: their gradients are not the derivatives of the forward
that the perturbations probe. Sample-rolled and permuted-denominator overrides
are also refused anywhere in the model: their example coupling violates the
estimator's independent-example assumption and changes when draws are batched.
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
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_rule_override import (
    active_branch_rule_override,
)
from dendritic_modeling.networks.utils.signal_rounding import signal_rounding_dtype
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
    """Unreduced loss of each example: ``cross_entropy`` (class labels) or ``mse``.

    ``mse`` averages the squared error over an example's outputs, which may be a
    single value per example (``[batch]``) or any trailing shape.
    """
    if loss == "cross_entropy":
        return F.cross_entropy(output, target.long(), reduction="none")
    error = output - target.reshape(output.shape).to(output.dtype)
    return error.square().reshape(output.shape[0], -1).mean(1)


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
        for name, module in model.named_modules():
            override = active_branch_rule_override(module)
            if override is not None and override.rule in (
                "sample_rolled_denominator",
                "sample_rolled_tangent",
                "permuted_current_denominator",
            ):
                raise ValueError(
                    "node-perturbation credit requires independent examples; "
                    f"batch-dependent override {override.rule!r} on {name!r} "
                    "changes the loss when perturbation draws are batched"
                )
        params = self.params
        is_recurrent = getattr(
            getattr(model, "core_network", None), "is_recurrent", False
        )
        if callable(is_recurrent):
            is_recurrent = is_recurrent()
        if bool(is_recurrent):
            raise ValueError("node-perturbation credit requires a feedforward model")
        surrogate = sorted(
            {
                str(module.autograd_credit_mode)
                for module in model.modules()
                if getattr(module, "autograd_credit_mode", "exact") != "exact"
            }
        )
        if surrogate:
            raise ValueError(
                "node-perturbation credit compares estimates with exact errors, but "
                f"the model's autograd credit is {surrogate}"
            )
        rounded = sorted(
            {
                module.signal_rounding
                for module in model.modules()
                if isinstance(getattr(module, "signal_rounding", None), str)
                and signal_rounding_dtype(module.signal_rounding) is not None
            }
        )
        if rounded:
            # Rounding makes the forward piecewise constant, while autograd passes
            # gradients through the casts: its errors are a surrogate there.
            raise ValueError(
                "node-perturbation credit compares estimates with exact errors, but "
                f"the model rounds its signals to {rounded}"
            )
        modules, n_somas, branch_factors = inventory_nested_tree(
            model, module_name_prefix=params.module_name_prefix
        )
        missing = [depth for depth in params.depths if depth not in modules]
        if missing:
            raise ValueError(f"the tree has depths {sorted(modules)}; not {missing}")
        runtime = runtime if runtime is not None else EvaluationRuntimeConfig()
        # A row is one example under one draw; every forward takes at most this many.
        rows_per_forward = int(runtime.batch_size)
        if rows_per_forward < 1:
            raise ValueError("analysis.runtime.batch_size must be positive")
        tensors = materialize_dataset(
            test_dataset, runtime, explicit_max_samples=params.max_samples
        )
        x, y = tensors[0], tensors[1]
        seed = 0 if params.seed is None else int(params.seed)
        rows: list[dict[str, Any]] = []
        with analysis_device_context(model, device) as target:
            # Tensors carry a concrete device (cuda:0, not cuda).
            target = torch.empty(0, device=target).device
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
                        rows_per_forward,
                        seed,
                    )
                )
        result = {
            "coordinate": params.coordinate,
            "loss": params.loss,
            "n_examples": int(x.shape[0]),
            "n_somas": int(n_somas),
            "branch_factors": [int(f) for f in branch_factors],
            "modules": {str(depth): modules[depth][0] for depth in params.depths},
            "seed": seed,
            "device": str(target),
            "dtype": str(dtype).removeprefix("torch."),
            "chunk": int(params.chunk),
            "rows_per_forward": rows_per_forward,
            "rows": rows,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{Path(filename).stem}.json")
        return result

    def _level(
        self,
        model,
        layer,
        name,
        depth,
        x,
        y,
        n_somas,
        branch_factors,
        dtype,
        device,
        rows_per_forward,
        base_seed,
    ):
        params = self.params
        width = int(layer.branch_config.output_dim)
        per_neuron = width // n_somas
        batch = int(x.shape[0])
        exact = self._exact_errors(
            model, layer, name, x, y, width, dtype, device, rows_per_forward
        )
        reference = self._coordinate_rms(model, layer, x, rows_per_forward)

        def loss(offset: torch.Tensor) -> torch.Tensor:
            # Copy-major rows: row r is draw r // batch of example r % batch.
            copies = offset.shape[0]
            total = copies * batch
            offsets = offset.reshape(total, width)
            losses = []
            for start in range(0, total, rows_per_forward):
                stop = min(total, start + rows_per_forward)
                index = torch.arange(start, stop, device=x.device) % batch
                perturbation = CompartmentPerturbation(
                    offsets[start:stop], params.coordinate
                )
                with perturb_compartments({layer: perturbation}):
                    output = _output_tensor(model(x[index]))
                losses.append(per_example_loss(output, y[index], params.loss))
            return torch.cat(losses).reshape(copies, batch)

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
                        base_seed
                        + stable_seed_offset(depth, dictionary, energy, relative)
                    ) % 2**63
                    generator = torch.Generator(device=device).manual_seed(seed)
                    _synchronize(device)
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
                    _synchronize(device)
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
                                "seed": int(seed),
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

    def _exact_errors(self, model, layer, name, x, y, width, dtype, device, examples):
        """Gradients of the loss with respect to zero offsets, a batch of examples at a time."""
        calls: list[int] = []
        handle = layer.register_forward_hook(lambda *_, calls=calls: calls.append(1))
        errors = []
        try:
            with torch.enable_grad():
                for start in range(0, x.shape[0], examples):
                    inputs, targets = (
                        x[start : start + examples],
                        y[start : start + examples],
                    )
                    offset = torch.zeros(
                        inputs.shape[0],
                        width,
                        dtype=dtype,
                        device=device,
                        requires_grad=True,
                    )
                    calls.clear()
                    with perturb_compartments(
                        {layer: CompartmentPerturbation(offset, self.params.coordinate)}
                    ):
                        output = _output_tensor(model(inputs))
                    if len(calls) != 1:
                        raise ValueError(
                            f"{name} ran {len(calls)} times in one model forward; "
                            "node-perturbation credit needs a level that runs once"
                        )
                    loss = per_example_loss(output, targets, self.params.loss).sum()
                    if loss.requires_grad:
                        gradient = torch.autograd.grad(
                            loss, offset, allow_unused=True, materialize_grads=True
                        )[0]
                    else:
                        gradient = torch.zeros_like(offset)
                    errors.append(gradient.detach())
        finally:
            handle.remove()
        return torch.cat(errors)

    def _coordinate_rms(self, model, layer, x, examples) -> float:
        """RMS of the perturbed quantity on the analysed examples (the scales' unit)."""
        total, count = 0.0, 0
        with torch.no_grad(), capture_analysis_currents(layer):
            for start in range(0, x.shape[0], examples):
                model(x[start : start + examples])
                currents = layer.get_analysis_currents()
                if self.params.coordinate == "output":
                    value = currents["post_gate_output"]
                elif self.params.coordinate == "voltage":
                    value = currents["pre_gate_voltage"]
                else:
                    # The shunting numerator that an injected current joins.
                    parts = [
                        currents[k]
                        for k in (
                            "pre_gate_excitation_current",
                            "pre_gate_upstream_current",
                        )
                        if currents[k] is not None
                    ]
                    if not parts:
                        raise ValueError(
                            "a current perturbation needs a layer with excitatory or "
                            "upstream current"
                        )
                    value = sum(parts)
                total += float(value.double().square().sum())
                count += value.numel()
        rms = math.sqrt(total / count)
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


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work, so wall times cover it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = ["NodePerturbationCreditAnalyzer", "agreement", "per_example_loss"]
