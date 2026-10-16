"""The biological constraint set declared by ``model.core.biological_neuron``.

``biological_neuron: true`` declares that a core is a biologically constrained
E/I network. The check runs on the built model, on the modules the core
actually contains, so core-type aliases, defaults and per-population settings
are all taken into account. A core satisfies the set when:

1. Synaptic weights are nonnegative. Every synapse bank uses a nonnegative
   weight transform (``exp``, ``softplus`` or ``relu``), and the core has no
   layer with unconstrained weights, such as the ``nn.Linear`` layers of an
   input projection, an MLP inhibitory network or soma feedback.
2. Excitation and inhibition stay separate (Dale's law). No synapse bank merges
   excitatory and inhibitory inputs, and every source acts with one sign. An
   inhibitory pathway reads the network input only when the transfer layer
   splits it into separate excitatory and inhibitory channels
   (``independent_pathways: true``), or explicitly declares
   ``input_source_identity: paired_afferents``: distinct external E/I sources
   with identical tuning. Undeclared duplicated source identities cannot have
   both polarities. Each population-network connection uses the
   pathway of its source's polarity.
3. The input is nonnegative. The core checks or rectifies input before the first
   synapses: ``transfer.output_activation`` is ``relu``, ``softplus`` or
   ``sigmoid`` or ``nonnegative`` (checked identity), or a population network's
   ``input_transform`` is ``relu``, ``signed_split`` or ``nonnegative``.
4. Every population emits nonnegative activity: spikes, a spike rate, or the
   output of its soma gate. A gate that is nonnegative for every voltage
   (``relu``, ``softplus``, ``sigmoid``, ``param_relu``, ``param_tanh``,
   ``param_tanh_only_m``) qualifies. A gate that is nonnegative only for a
   nonnegative voltage (``none``, ``endpoint_tanh``, ``linear_sigmoid``,
   ``linear_tanh`` and their ``param_`` forms) qualifies when every level of the
   population integrates by shunting with such gates and no synapse type has a
   negative reversal potential, because shunting then keeps every voltage
   nonnegative.

The set covers structured EINet, unified E/I and population-network cores
(:data:`dendritic_modeling.config.biological.BIOLOGICAL_CORE_TYPES`).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.activations.parametric import (
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricReLU,
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.identity import (
    IdentityInputTransform,
)
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _FF_INH,
    _INPUT_E_SOURCE,
    _INPUT_I_SOURCE,
    _INPUT_SOURCE,
    _REC_INH,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.utils.input_domain import NonnegativeInputCheck
from dendritic_modeling.networks.utils.weight_transforms import (
    NONNEGATIVE_WEIGHT_TRANSFORMS,
)

# Gates whose output is nonnegative for every input voltage.
_NONNEGATIVE_GATES = (
    nn.ReLU,
    nn.Softplus,
    nn.Sigmoid,
    ParametricReLU,
    ParametricTanh,
    ParametricTanhOnlyM,
)
# Gates whose output is nonnegative when the voltage is nonnegative. nn.Identity
# is the soma output of a population without a gate (reactivate: false).
_NONNEGATIVE_ON_NONNEGATIVE_VOLTAGE = (
    nn.Identity,
    EndpointNormalizedTanh,
    ParametricLinearSigmoid,
    ParametricLinearTanh,
)
_RECTIFYING_ACTIVATIONS = (nn.ReLU, nn.Softplus, nn.Sigmoid, NonnegativeInputCheck)
_UNCONSTRAINED_WEIGHT_LAYERS = (
    nn.Linear,
    nn.Bilinear,
    nn.modules.conv._ConvNd,
    nn.Embedding,
    nn.RNNBase,
)
# Where an unconstrained layer comes from, by a name in its module path.
_UNCONSTRAINED_ORIGINS = (
    ("input_projection", "the input projection (input_projection_dims)"),
    ("soma_feedback", "soma feedback (soma_feedback_enabled)"),
    ("inhibitory_cells", "the MLP inhibitory network (inhibitory_network_type: mlp)"),
)
_POPULATION_MODULES = (DendriNet, StatefulDendriNet)


def biological_violations(core: nn.Module) -> list[str]:
    """Return a description of every biological constraint ``core`` breaks."""
    # The declaration concerns the neuronal core, not a task decoder or encoder.
    # Analysis callers may pass the complete classification/regression wrapper.
    core = getattr(core, "core_network", core)
    return [
        *_weight_violations(core),
        *_wiring_violations(core),
        *_input_violations(core),
        *_output_violations(core),
    ]


def enforce_biological_neuron(core_type: str, core: nn.Module) -> None:
    """Raise when a core declared ``biological_neuron: true`` breaks the set."""
    if not any(isinstance(module, _POPULATION_MODULES) for module in core.modules()):
        raise ValueError(
            f"model.core.biological_neuron is true, but the {core_type!r} core "
            "contains no dendritic populations."
        )
    violations = biological_violations(core)
    if violations:
        raise ValueError(
            f"model.core.biological_neuron is true, but the {core_type!r} core "
            "breaks the biological constraint set:\n- "
            + "\n- ".join(violations)
            + "\nChange these settings, or set model.core.biological_neuron: false."
        )


def _owns_parameters(module: nn.Module) -> bool:
    return next(module.parameters(recurse=False), None) is not None


def _examples(names: list[str]) -> str:
    shown = ", ".join(names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _weight_violations(core: nn.Module) -> list[str]:
    signed_banks: dict[str, list[str]] = {}
    unconstrained: dict[str, list[str]] = {}
    merged: list[str] = []
    for name, module in core.named_modules():
        if isinstance(module, _UNCONSTRAINED_WEIGHT_LAYERS):
            origin = next(
                (text for key, text in _UNCONSTRAINED_ORIGINS if key in name),
                f"a {type(module).__name__} layer",
            )
            unconstrained.setdefault(origin, []).append(name)
            continue
        if getattr(module, "synapse_mode", None) == "mlp":
            merged.append(name)
        transform = getattr(module, "weight_transform", None)
        if transform is None or not _owns_parameters(module):
            continue
        transform = str(transform).lower()
        if transform not in NONNEGATIVE_WEIGHT_TRANSFORMS:
            signed_banks.setdefault(transform, []).append(name)
    violations = [
        f"synapse banks use the signed weight transform {transform!r} "
        f"({_examples(names)}); use exp, softplus or relu"
        for transform, names in sorted(signed_banks.items())
    ]
    violations.extend(
        f"{origin} has unconstrained (signed) weights ({_examples(names)})"
        for origin, names in sorted(unconstrained.items())
    )
    if merged:
        violations.append(
            "the MLP synapse mode merges excitatory and inhibitory inputs into one "
            f"signed synapse bank ({_examples(merged)})"
        )
    return violations


def _has_inhibitory_pathway(cells: nn.Module | None) -> bool:
    return any(
        getattr(layer, "branch_inhibition", None) is not None
        for layer in getattr(cells, "branch_layers", ())
    )


def _duplicates_input(transfer: nn.Module | None) -> bool:
    """Whether the inhibitory input stream repeats the excitatory channels."""
    if getattr(transfer, "input_source_identity", "shared") == "paired_afferents":
        return False
    return transfer is None or isinstance(
        getattr(transfer, "pathway_split", None), IdentityInputTransform
    )


def _duplicated_stream_message(readers: Iterable[str]) -> str:
    return (
        f"{', '.join(readers)} read the transferred inhibitory stream, which "
        "repeats the excitatory input channels (transfer independent_pathways: "
        "false), so every input channel both excites and inhibits; set "
        "independent_pathways: true to split the input into excitatory and "
        "inhibitory channels, or explicitly declare input_source_identity: "
        "paired_afferents for distinct external E/I sources with shared tuning"
    )


def _wiring_violations(core: nn.Module) -> list[str]:
    if isinstance(core, ExcitationInhibitionNetwork):
        if not _duplicates_input(core.transfer_fn):
            return []
        first = core.layers[0]
        readers = []
        if first.inhibitory_cells is None:
            if _has_inhibitory_pathway(first.excitatory_cells):
                readers.append("first-layer excitatory cells")
        elif _has_inhibitory_pathway(first.inhibitory_cells):
            readers.append("first-layer inhibitory cells")
        return [_duplicated_stream_message(readers)] if readers else []
    if isinstance(core, EINetwork):
        if core.transfer_fn is None or not _duplicates_input(core.transfer_fn):
            return []
        readers = []
        for index, layer in enumerate(core.layers):
            direct = bool(layer.direct_ff_inhibitory_to_excitatory)
            if index > 0 and not direct:
                continue
            if layer.i_population is not None:
                if _has_inhibitory_pathway(layer.i_population):
                    readers.append(f"layer {index} inhibitory cells")
            elif direct and _has_inhibitory_pathway(layer.e_population):
                readers.append(f"layer {index} excitatory cells")
        return [_duplicated_stream_message(readers)] if readers else []
    if isinstance(core, PopulationNetwork):
        return _population_wiring_violations(core)
    return []


def _population_wiring_violations(core: PopulationNetwork) -> list[str]:
    violations = []
    first_sources = {
        connection.source
        for connection in core.layers[0].connections
        if connection.enabled
    }
    if (
        _duplicates_input(core.transfer_fn)
        and _INPUT_I_SOURCE in first_sources
        and first_sources & {_INPUT_SOURCE, _INPUT_E_SOURCE}
    ):
        violations.append(
            "the first layer reads input_i together with input or input_e, which "
            "carry the same channels because the input is not split, so every "
            "input channel both excites and inhibits; set use_transfer: true with "
            "transfer_params.independent_pathways: true, or explicitly declare "
            "transfer_params.input_source_identity: paired_afferents"
        )
    for layer in core.layers:
        for connection in layer.connections:
            if not connection.enabled:
                continue
            polarity = layer._source_polarity(connection.source)
            sign = (
                "inhibitory"
                if connection.pathway in {_FF_INH, _REC_INH}
                else "excitatory"
            )
            if sign != polarity:
                violations.append(
                    f"connection {connection.source} -> {connection.target} in "
                    f"layer {layer.config.name!r} uses pathway "
                    f"{connection.pathway!r}, but its source is {polarity}"
                )
    return violations


def _rectifies(transfer: nn.Module | None) -> bool:
    return transfer is not None and isinstance(
        getattr(transfer, "activation_fn", None), _RECTIFYING_ACTIVATIONS
    )


def _projection_rectifies(core: nn.Module) -> bool:
    projection = getattr(core, "input_projection", None)
    return (
        isinstance(projection, nn.Sequential)
        and len(projection) > 0
        and isinstance(projection[-1], _RECTIFYING_ACTIVATIONS)
    )


def _input_violations(core: nn.Module) -> list[str]:
    unrectified = "use transfer output_activation nonnegative (checked identity), relu, softplus or sigmoid"
    if isinstance(core, ExcitationInhibitionNetwork):
        if _rectifies(core.transfer_fn):
            return []
        activation = core.transfer_fn.output_activation
        return [
            f"the input reaches the synapses unrectified (transfer "
            f"output_activation {activation!r}); {unrectified}"
        ]
    if isinstance(core, EINetwork):
        # The input adapter acts before the transfer layer, whose sign-preserving
        # normalization and stream split keep a nonnegative input nonnegative.
        if getattr(core, "input_transform", "identity") in {
            "nonnegative",
            "relu",
            "signed_split",
        }:
            return []
        if core.transfer_fn is None:
            if _projection_rectifies(core):
                return []
            return [
                "the core has no transfer layer, so nothing rectifies its input; "
                f"enable the transfer layer and {unrectified}"
            ]
        if _rectifies(core.transfer_fn):
            return []
        return [
            "the input reaches the synapses unrectified (transfer "
            f"output_activation {core.transfer_fn.output_activation!r}); "
            f"{unrectified}"
        ]
    if isinstance(core, PopulationNetwork):
        if core.input_transform in {
            "nonnegative",
            "relu",
            "signed_split",
        } or _rectifies(core.transfer_fn):
            return []
        # The projection rectifies input/input_e, but a transfer layer's
        # input_i bypasses it. Only accept this guarantee if that bypass is
        # unused. Signed projection weights remain a separate violation.
        reads_unprojected_i = core.transfer_fn is not None and any(
            connection.enabled and connection.source == _INPUT_I_SOURCE
            for connection in core.layers[0].connections
        )
        if _projection_rectifies(core) and not reads_unprojected_i:
            return []
        return [
            "the input reaches the synapses unrectified (population_network "
            f"input_transform {core.input_transform!r}); set input_transform "
            "nonnegative, relu or signed_split, or a rectifying transfer_params "
            "output_activation"
        ]
    return []


def _gate_name(gate: nn.Module) -> str:
    return "none" if isinstance(gate, nn.Identity) else type(gate).__name__


def _has_negative_reversal(population: nn.Module) -> bool:
    if not getattr(population, "synapse_types_enabled", False):
        return False
    types = population.synapse_types
    return any(
        float(synapse_type.reversal_potential) < 0.0
        for group in (types.excitatory, types.inhibitory)
        for synapse_type in group.types
    )


def _population_output_problem(population: nn.Module) -> str | None:
    """Why ``population`` can emit negative activity, or None when it cannot."""
    spiking_soma = getattr(population, "spiking_soma", None)
    if spiking_soma is not None:
        if spiking_soma.output_mode in {"spikes", "rate"}:
            return None
        return (
            "its spiking soma reads out the membrane potential "
            "(spike_readout: membrane), which can be negative"
        )
    layers = list(population.branch_layers)
    gates = [layer.reactivation for layer in layers]
    soma_gate = gates[-1]
    if isinstance(soma_gate, _NONNEGATIVE_GATES):
        return None
    if not isinstance(soma_gate, _NONNEGATIVE_ON_NONNEGATIVE_VOLTAGE):
        return f"its soma gate {_gate_name(soma_gate)} can emit negative values"
    reason = None
    if not all(layer.use_shunting for layer in layers):
        reason = "additive integration can make the voltage negative"
    elif not all(
        isinstance(gate, _NONNEGATIVE_GATES + _NONNEGATIVE_ON_NONNEGATIVE_VOLTAGE)
        for gate in gates
    ):
        reason = "a dendritic gate can emit negative values"
    elif _has_negative_reversal(population):
        reason = "a synapse type has a negative reversal potential"
    if reason is None:
        return None
    return (
        f"its soma gate {_gate_name(soma_gate)} is nonnegative only for a "
        f"nonnegative voltage, and {reason}; use a nonnegative gate such as "
        "param_tanh, param_relu or relu"
    )


def _output_violations(core: nn.Module) -> list[str]:
    violations = []
    for name, module in core.named_modules():
        if not isinstance(module, _POPULATION_MODULES):
            continue
        problem = _population_output_problem(module)
        if problem is not None:
            violations.append(f"population {name or 'core'}: {problem}")
    return violations


__all__ = ["biological_violations", "enforce_biological_neuron"]
