"""Matrix-free linearization of recurrent population-network state dynamics.

The recurrent state analyzed here is the state that is actually carried from
one :class:`PopulationNetwork` timestep to the next.  In rate mode that state
contains, for every population, all five dendritic trace banks and the
population output used by delayed projections.  Keeping the output tensors is
important: differentiating only the trace banks would omit recurrent routes
that consume the previous population outputs.

This module deliberately supports only the basic rate-mode state.  Optional
typed-conductance, soma-feedback, and spiking state is rejected rather than
being silently left out of the flattened vector.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)

_TRACE_FIELDS = (
    "trace_E",
    "trace_E_rec",
    "trace_I",
    "trace_I_rec",
    "trace_branch",
)
_OPTIONAL_STATE_FIELDS = (
    "branch_voltage",
    "typed_traces",
    "soma_output",
    "v_soma",
    "refractory_counter",
    "spike_readout",
    "dendritic_spike_plateau",
    "dendritic_spike_refractory",
    "dendritic_spike_events",
)


class UnsupportedPopulationStateError(ValueError):
    """Raised when flattening would omit an active recurrent state tensor."""


@dataclass(frozen=True)
class StateTensorSpec:
    """Location and shape of one tensor in the canonical state vector."""

    layer_index: int
    layer_name: str
    population_name: str
    field: str
    level_index: int | None
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def numel(self) -> int:
        return self.stop - self.start

    @property
    def path(self) -> str:
        prefix = f"layers[{self.layer_index}].{self.layer_name}.{self.population_name}"
        if self.level_index is None:
            return f"{prefix}.{self.field}"
        return f"{prefix}.{self.field}[{self.level_index}]"


def _unsupported_fields(state: DendriNetState) -> list[str]:
    return [
        field
        for field in _OPTIONAL_STATE_FIELDS
        if getattr(state, field, None) is not None
    ]


def _validate_rate_population(population: torch.nn.Module, path: str) -> None:
    unsupported_modes = []
    if getattr(population, "spiking_soma", None) is not None:
        unsupported_modes.append("spiking_soma")
    if bool(getattr(population, "synapse_types_enabled", False)):
        unsupported_modes.append("typed_synapses")
    if bool(getattr(population, "dendritic_spikes_enabled", False)):
        unsupported_modes.append("dendritic_spikes")
    if bool(getattr(population, "soma_feedback_enabled", False)):
        unsupported_modes.append("soma_feedback")
    if unsupported_modes:
        raise UnsupportedPopulationStateError(
            f"{path} is not a supported basic rate-mode population; enabled: "
            + ", ".join(unsupported_modes)
        )


def _validate_deterministic_network(network: PopulationNetwork) -> None:
    """Reject forward-time randomness that does not define one fixed Jacobian."""

    for name, module in network.named_modules():
        selection = str(getattr(module, "selection", "")).lower()
        class_name = type(module).__name__.lower()
        noise_level = float(getattr(module, "noise_level", 0.0) or 0.0)
        stochastic = (
            "stochastic" in class_name
            or selection in {"stochastic", "rank_probabilistic"}
            or noise_level > 0.0
        )
        if stochastic:
            raise ValueError(
                "State-transition linearization requires a deterministic forward; "
                f"module {name!r} ({type(module).__name__}) is stochastic"
            )
        if getattr(module, "_supports_recurrent_weight_cache", None) is False:
            raise ValueError(
                "State-transition linearization requires a fixed forward map; "
                f"module {name!r} ({type(module).__name__}) uses a dynamic "
                "selection/sparsification policy"
            )
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training:
            raise ValueError(
                "State-transition linearization requires a deterministic forward; "
                f"dropout module {name!r} is in training mode"
            )


class PopulationStateCodec:
    """Deterministically flatten and rebuild the supported recurrent state.

    The order is network layer order, population declaration order, the five
    trace-bank names in ``TRACE_FIELDS`` order, level order within a bank, and
    finally that population's delayed output.  Dictionary insertion order in a
    runtime state therefore cannot change the vector convention.
    """

    TRACE_FIELDS = _TRACE_FIELDS

    def __init__(
        self,
        network: PopulationNetwork,
        reference_state: Sequence[PopulationLayerState],
    ) -> None:
        if not isinstance(network, PopulationNetwork):
            raise TypeError(
                "PopulationStateCodec supports PopulationNetwork, got "
                f"{type(network).__name__}"
            )
        if not network.is_recurrent:
            raise ValueError("PopulationNetwork must have recurrent state")

        self.network = network
        self._population_names = tuple(
            tuple(layer.populations.keys()) for layer in network.layers
        )
        self._layer_names = tuple(layer.config.name for layer in network.layers)
        self._specs = self._build_specs(reference_state)
        self._dimension = self._specs[-1].stop if self._specs else 0
        if self._dimension == 0:
            raise ValueError("PopulationNetwork recurrent state is empty")

        reference_tensors = self._ordered_tensors(reference_state)
        self._dtype = reference_tensors[0].dtype
        self._device = reference_tensors[0].device
        self._batch_size = int(reference_tensors[0].shape[0])

    @property
    def specs(self) -> tuple[StateTensorSpec, ...]:
        return self._specs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _validate_structure(
        self, state: Sequence[PopulationLayerState]
    ) -> list[torch.Tensor]:
        if not isinstance(state, (list, tuple)):
            raise TypeError(
                "PopulationNetwork state must be a list or tuple of "
                "PopulationLayerState objects"
            )
        if len(state) != len(self.network.layers):
            raise ValueError(
                f"expected {len(self.network.layers)} layer states, got {len(state)}"
            )

        tensors: list[torch.Tensor] = []
        for layer_index, (layer_state, population_names) in enumerate(
            zip(state, self._population_names)
        ):
            if not isinstance(layer_state, PopulationLayerState):
                raise TypeError(
                    f"state layer {layer_index} must be PopulationLayerState, got "
                    f"{type(layer_state).__name__}"
                )
            expected_names = set(population_names)
            if set(layer_state.population_states) != expected_names:
                raise ValueError(
                    f"layer {layer_index} population-state keys differ: expected "
                    f"{sorted(expected_names)}, got "
                    f"{sorted(layer_state.population_states)}"
                )
            if set(layer_state.outputs) != expected_names:
                raise ValueError(
                    f"layer {layer_index} output keys differ: expected "
                    f"{sorted(expected_names)}, got {sorted(layer_state.outputs)}"
                )

            layer = self.network.layers[layer_index]
            for population_name in population_names:
                population = layer.populations[population_name]
                path = (
                    f"layers[{layer_index}].{self._layer_names[layer_index]}."
                    f"{population_name}"
                )
                _validate_rate_population(population, path)
                population_state = layer_state.population_states[population_name]
                if not isinstance(population_state, DendriNetState):
                    raise TypeError(
                        f"{path} state must be DendriNetState, got "
                        f"{type(population_state).__name__}"
                    )
                unsupported = _unsupported_fields(population_state)
                if unsupported:
                    raise UnsupportedPopulationStateError(
                        f"{path} has unsupported optional state fields: "
                        + ", ".join(unsupported)
                    )

                n_levels: int | None = None
                for field in _TRACE_FIELDS:
                    traces = getattr(population_state, field)
                    if not isinstance(traces, list):
                        raise TypeError(f"{path}.{field} must be a list of tensors")
                    if n_levels is None:
                        n_levels = len(traces)
                    elif len(traces) != n_levels:
                        raise ValueError(
                            f"{path} trace banks do not have equal level counts"
                        )
                    if not traces:
                        raise ValueError(f"{path}.{field} must not be empty")
                    expected_level_dims = tuple(
                        int(dim) for dim in getattr(population, "_level_dims", ())
                    )
                    if expected_level_dims and len(traces) != len(expected_level_dims):
                        raise ValueError(
                            f"{path}.{field} has {len(traces)} levels, expected "
                            f"{len(expected_level_dims)}"
                        )
                    for level_index, tensor in enumerate(traces):
                        if not isinstance(tensor, torch.Tensor):
                            raise TypeError(
                                f"{path}.{field}[{level_index}] must be a tensor"
                            )
                        if expected_level_dims and (
                            tensor.ndim != 2
                            or tensor.shape[1] != expected_level_dims[level_index]
                        ):
                            raise ValueError(
                                f"{path}.{field}[{level_index}] must have shape "
                                f"[batch, {expected_level_dims[level_index]}], got "
                                f"{tuple(tensor.shape)}"
                            )
                        tensors.append(tensor)

                output = layer_state.outputs[population_name]
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"{path}.outputs must be a tensor")
                expected_output_dim = int(getattr(population, "_n_soma", 0))
                if expected_output_dim and (
                    output.ndim != 2 or output.shape[1] != expected_output_dim
                ):
                    raise ValueError(
                        f"{path}.outputs must have shape "
                        f"[batch, {expected_output_dim}], got {tuple(output.shape)}"
                    )
                tensors.append(output)

        if not tensors:
            raise ValueError("PopulationNetwork recurrent state has no tensors")
        first = tensors[0]
        if not first.is_floating_point():
            raise TypeError("recurrent state tensors must be floating point")
        if first.ndim < 1:
            raise ValueError("recurrent state tensors must include a batch dimension")
        for tensor in tensors:
            if not tensor.is_floating_point():
                raise TypeError("recurrent state tensors must be floating point")
            if tensor.device != first.device or tensor.dtype != first.dtype:
                raise ValueError(
                    "all recurrent state tensors must share one device and dtype"
                )
            if tensor.ndim < 1 or tensor.shape[0] != first.shape[0]:
                raise ValueError(
                    "all recurrent state tensors must share a batch dimension"
                )
        return tensors

    def _build_specs(
        self, state: Sequence[PopulationLayerState]
    ) -> tuple[StateTensorSpec, ...]:
        tensors = self._validate_structure(state)
        specs: list[StateTensorSpec] = []
        offset = 0
        tensor_index = 0
        for layer_index, population_names in enumerate(self._population_names):
            layer_state = state[layer_index]
            for population_name in population_names:
                population_state = layer_state.population_states[population_name]
                for field in _TRACE_FIELDS:
                    traces = getattr(population_state, field)
                    for level_index, _ in enumerate(traces):
                        tensor = tensors[tensor_index]
                        tensor_index += 1
                        stop = offset + tensor.numel()
                        specs.append(
                            StateTensorSpec(
                                layer_index=layer_index,
                                layer_name=self._layer_names[layer_index],
                                population_name=population_name,
                                field=field,
                                level_index=level_index,
                                shape=tuple(tensor.shape),
                                start=offset,
                                stop=stop,
                            )
                        )
                        offset = stop
                tensor = tensors[tensor_index]
                tensor_index += 1
                stop = offset + tensor.numel()
                specs.append(
                    StateTensorSpec(
                        layer_index=layer_index,
                        layer_name=self._layer_names[layer_index],
                        population_name=population_name,
                        field="outputs",
                        level_index=None,
                        shape=tuple(tensor.shape),
                        start=offset,
                        stop=stop,
                    )
                )
                offset = stop
        return tuple(specs)

    def _ordered_tensors(
        self, state: Sequence[PopulationLayerState]
    ) -> list[torch.Tensor]:
        tensors = self._validate_structure(state)
        if hasattr(self, "_specs"):
            if len(tensors) != len(self._specs):
                raise ValueError("recurrent state tensor count changed")
            for tensor, spec in zip(tensors, self._specs):
                if tuple(tensor.shape) != spec.shape:
                    raise ValueError(
                        f"state tensor {spec.path} shape changed: expected "
                        f"{spec.shape}, got {tuple(tensor.shape)}"
                    )
        return tensors

    def flatten(self, state: Sequence[PopulationLayerState]) -> torch.Tensor:
        """Return the canonical one-dimensional differentiable state vector."""

        tensors = self._ordered_tensors(state)
        return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)

    def unflatten(self, vector: torch.Tensor) -> list[PopulationLayerState]:
        """Rebuild a state container from the canonical vector without detaching."""

        if not isinstance(vector, torch.Tensor):
            raise TypeError("state vector must be a tensor")
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError(
                f"state vector must have shape ({self.dimension},), got "
                f"{tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(
                f"state vector must use {self.device}/{self.dtype}, got "
                f"{vector.device}/{vector.dtype}"
            )

        values = {
            spec.path: vector[spec.start : spec.stop].view(spec.shape)
            for spec in self.specs
        }
        rebuilt: list[PopulationLayerState] = []
        for layer_index, population_names in enumerate(self._population_names):
            population_states: dict[str, DendriNetState] = {}
            outputs: dict[str, torch.Tensor] = {}
            for population_name in population_names:
                prefix = (
                    f"layers[{layer_index}].{self._layer_names[layer_index]}."
                    f"{population_name}"
                )
                traces_by_field: dict[str, list[torch.Tensor]] = {}
                for field in _TRACE_FIELDS:
                    matching_specs = [
                        spec
                        for spec in self.specs
                        if spec.layer_index == layer_index
                        and spec.population_name == population_name
                        and spec.field == field
                    ]
                    traces_by_field[field] = [
                        values[spec.path] for spec in matching_specs
                    ]
                population_states[population_name] = DendriNetState(
                    trace_E=traces_by_field["trace_E"],
                    trace_E_rec=traces_by_field["trace_E_rec"],
                    trace_I=traces_by_field["trace_I"],
                    trace_I_rec=traces_by_field["trace_I_rec"],
                    trace_branch=traces_by_field["trace_branch"],
                )
                outputs[population_name] = values[f"{prefix}.outputs"]
            rebuilt.append(
                PopulationLayerState(
                    population_states=population_states,
                    outputs=outputs,
                )
            )
        return rebuilt

    def metadata(self) -> list[dict[str, object]]:
        """Return a JSON-safe description of the canonical state convention."""

        return [
            {
                "path": spec.path,
                "shape": list(spec.shape),
                "start": spec.start,
                "stop": spec.stop,
            }
            for spec in self.specs
        ]


class StateTransitionLinearization:
    """Matrix-free Jacobian of ``q_t = F(q_{t-1}, x_t)`` at one point."""

    def __init__(
        self,
        network: PopulationNetwork,
        state: Sequence[PopulationLayerState],
        input_t: torch.Tensor,
    ) -> None:
        if not isinstance(network, PopulationNetwork):
            raise TypeError(
                "StateTransitionLinearization supports PopulationNetwork, got "
                f"{type(network).__name__}"
            )
        _validate_deterministic_network(network)
        self.network = network
        self.codec = PopulationStateCodec(network, state)
        if not isinstance(input_t, torch.Tensor):
            raise TypeError("input_t must be a tensor")
        if input_t.ndim != 2:
            raise ValueError(
                "input_t must have shape [batch, input_dim], got "
                f"{tuple(input_t.shape)}"
            )
        if input_t.shape[0] != self.codec.batch_size:
            raise ValueError(
                f"input batch {input_t.shape[0]} differs from state batch "
                f"{self.codec.batch_size}"
            )
        if input_t.shape[1] != network.input_dim:
            raise ValueError(
                f"input feature dimension must be {network.input_dim}, got "
                f"{input_t.shape[1]}"
            )
        if input_t.device != self.codec.device or input_t.dtype != self.codec.dtype:
            raise ValueError(
                "input_t and recurrent state must share device and dtype; got "
                f"{input_t.device}/{input_t.dtype} and "
                f"{self.codec.device}/{self.codec.dtype}"
            )
        self.input_t = input_t.detach().clone()
        self.state_vector = self.codec.flatten(state).detach().clone()

        # Validate that the actual transition returns exactly the same supported
        # state structure before any expensive iterative analysis starts.
        with torch.no_grad():
            next_vector = self.transition(self.state_vector)
        if next_vector.shape != self.state_vector.shape:
            raise RuntimeError(
                "state transition changed the flattened state dimension: "
                f"{self.state_vector.shape} -> {next_vector.shape}"
            )
        self.next_state_vector = next_vector.detach()

    @property
    def dimension(self) -> int:
        return self.codec.dimension

    @property
    def device(self) -> torch.device:
        return self.codec.device

    @property
    def dtype(self) -> torch.dtype:
        return self.codec.dtype

    def transition(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Evaluate ``F(q, x_t)`` and return the canonical next-state vector."""

        state = self.codec.unflatten(state_vector)
        _, next_state = self.network(
            self.input_t,
            hidden=state,
            return_hidden=True,
        )
        if next_state is None:
            raise RuntimeError("PopulationNetwork did not return recurrent state")
        return self.codec.flatten(next_state)

    def _validate_vector(self, vector: torch.Tensor, name: str) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if vector.shape != self.state_vector.shape:
            raise ValueError(
                f"{name} must have shape {tuple(self.state_vector.shape)}, got "
                f"{tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(
                f"{name} must use {self.device}/{self.dtype}, got "
                f"{vector.device}/{vector.dtype}"
            )

    def jvp(self, vector: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Apply ``(d q_t / d q_{t-1})`` to ``vector`` without building it."""

        self._validate_vector(vector, "vector")
        _, product = torch.autograd.functional.jvp(
            self.transition,
            self.state_vector,
            vector,
            create_graph=create_graph,
            strict=False,
        )
        return product

    def vjp(
        self, cotangent: torch.Tensor, *, create_graph: bool = False
    ) -> torch.Tensor:
        """Apply the transpose state Jacobian to ``cotangent`` matrix-free."""

        self._validate_vector(cotangent, "cotangent")
        _, product = torch.autograd.functional.vjp(
            self.transition,
            self.state_vector,
            cotangent,
            create_graph=create_graph,
            strict=False,
        )
        return product


@dataclass(frozen=True)
class SingularValueEstimate:
    """Result of two-sided power iteration on a matrix-free operator."""

    singular_value: float
    right_vector: torch.Tensor
    left_vector: torch.Tensor
    iterations: int
    converged: bool
    residual_norm: float

    def metadata(self) -> dict[str, object]:
        return {
            "singular_value": self.singular_value,
            "iterations": self.iterations,
            "converged": self.converged,
            "residual_norm": self.residual_norm,
        }


def _normalized(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(vector)
    eps = torch.finfo(vector.dtype).eps
    if bool(norm <= eps):
        return torch.zeros_like(vector), norm
    return vector / norm, norm


def _power_iteration(
    *,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    jvp: Callable[[torch.Tensor], torch.Tensor],
    vjp: Callable[[torch.Tensor], torch.Tensor],
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    seed: int,
    initial_vector: torch.Tensor | None,
) -> SingularValueEstimate:
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("power-iteration tolerances must be non-negative")

    if initial_vector is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        right = torch.randn(
            dimension,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if initial_vector.shape != (dimension,):
            raise ValueError(
                f"initial_vector must have shape ({dimension},), got "
                f"{tuple(initial_vector.shape)}"
            )
        if initial_vector.device != device or initial_vector.dtype != dtype:
            raise ValueError("initial_vector device/dtype does not match the operator")
        right = initial_vector.detach().clone()
    right, right_norm = _normalized(right)
    if bool(right_norm == 0):
        raise ValueError("initial_vector must be nonzero")

    previous_sigma: float | None = None
    converged = False
    left = torch.zeros_like(right)
    sigma_tensor = torch.zeros((), device=device, dtype=dtype)
    iterations = 0
    with torch.enable_grad():
        for iteration in range(1, max_iterations + 1):
            left, sigma_tensor = _normalized(jvp(right))
            iterations = iteration
            sigma = float(sigma_tensor.detach().cpu())
            if sigma == 0.0:
                return SingularValueEstimate(
                    singular_value=0.0,
                    right_vector=right.detach(),
                    left_vector=left.detach(),
                    iterations=iteration,
                    converged=True,
                    residual_norm=0.0,
                )
            next_right, transpose_norm = _normalized(vjp(left))
            if bool(transpose_norm == 0):
                break
            right = next_right
            if previous_sigma is not None:
                threshold = absolute_tolerance + relative_tolerance * abs(sigma)
                if abs(sigma - previous_sigma) <= threshold:
                    converged = True
                    break
            previous_sigma = sigma

        left, sigma_tensor = _normalized(jvp(right))
        sigma = float(sigma_tensor.detach().cpu())
        transpose_product = vjp(left) if sigma > 0.0 else torch.zeros_like(right)
        residual = torch.linalg.vector_norm(transpose_product - sigma_tensor * right)

    return SingularValueEstimate(
        singular_value=sigma,
        right_vector=right.detach(),
        left_vector=left.detach(),
        iterations=iterations,
        converged=converged,
        residual_norm=float(residual.detach().cpu()),
    )


def leading_state_transition_singular_value(
    linearization: StateTransitionLinearization,
    *,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-7,
    seed: int = 0,
    initial_vector: torch.Tensor | None = None,
) -> SingularValueEstimate:
    """Estimate ``sigma_max(d q_t / d q_{t-1})`` without forming the Jacobian."""

    return _power_iteration(
        dimension=linearization.dimension,
        device=linearization.device,
        dtype=linearization.dtype,
        jvp=linearization.jvp,
        vjp=linearization.vjp,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        seed=seed,
        initial_vector=initial_vector,
    )


class StateTransitionProduct:
    """Matrix-free product of one-step Jacobians along a fixed trajectory."""

    def __init__(self, linearizations: Sequence[StateTransitionLinearization]) -> None:
        if not linearizations:
            raise ValueError("StateTransitionProduct requires at least one step")
        self.linearizations = tuple(linearizations)
        first = self.linearizations[0]
        for linearization in self.linearizations[1:]:
            if (
                linearization.dimension != first.dimension
                or linearization.device != first.device
                or linearization.dtype != first.dtype
            ):
                raise ValueError(
                    "all state-transition linearizations must share shape/device/dtype"
                )

    @property
    def dimension(self) -> int:
        return self.linearizations[0].dimension

    @property
    def device(self) -> torch.device:
        return self.linearizations[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.linearizations[0].dtype

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        product = vector
        for linearization in self.linearizations:
            product = linearization.jvp(product)
        return product

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        product = cotangent
        for linearization in reversed(self.linearizations):
            product = linearization.vjp(product)
        return product

    def leading_singular_value(
        self,
        *,
        max_iterations: int = 50,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-7,
        seed: int = 0,
        initial_vector: torch.Tensor | None = None,
    ) -> SingularValueEstimate:
        return _power_iteration(
            dimension=self.dimension,
            device=self.device,
            dtype=self.dtype,
            jvp=self.jvp,
            vjp=self.vjp,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            seed=seed,
            initial_vector=initial_vector,
        )


__all__ = [
    "PopulationStateCodec",
    "SingularValueEstimate",
    "StateTensorSpec",
    "StateTransitionLinearization",
    "StateTransitionProduct",
    "UnsupportedPopulationStateError",
    "leading_state_transition_singular_value",
]
