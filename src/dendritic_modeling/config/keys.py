"""Key checking for configuration sections.

Every configuration section is checked against the keys its builder reads.
A key outside that set raises :class:`UnknownConfigKeyError` with a
did-you-mean hint, so a misspelled option can no longer fall back to its
default without notice.

Two kinds of keys are accepted without a schema field:

* The top-level ``metadata`` mapping and top-level keys that start with an
  underscore hold annotations written by people and tools (study contracts,
  sweep identifiers). The loader keeps them in ``Config.metadata`` and never
  reads them.
* Keys listed in :data:`RETIRED_KEYS` were written into run configs by earlier
  library versions or by library tools. They are accepted with a warning so
  archived as-run configs still load. A retired key whose value the current
  library cannot reproduce raises instead; such a run must be replayed at the
  library revision that produced it.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

METADATA_SECTION = "metadata"


class UnknownConfigKeyError(ValueError):
    """Raised when a configuration section contains keys no builder reads."""

    def __init__(self, section: str, keys: Iterable[str], allowed: Iterable[str]):
        self.section = section
        self.keys = tuple(sorted(str(key) for key in keys))
        allowed = sorted(str(key) for key in allowed)
        details = []
        unmatched = False
        for key in self.keys:
            matches = difflib.get_close_matches(key, allowed, n=1)
            if matches:
                details.append(f"{key} (did you mean {matches[0]!r}?)")
            else:
                details.append(key)
                unmatched = True
        where = f"{section} option(s)" if section else "top-level section(s)"
        message = (
            f"Unknown {where}: {', '.join(details)}. "
            "These keys would be silently ignored."
        )
        if unmatched and allowed:
            message += " Valid keys: " + ", ".join(allowed) + "."
        if not section:
            message += (
                f" Put annotations under '{METADATA_SECTION}:' or in keys that "
                "start with an underscore."
            )
        super().__init__(message)


@dataclass(frozen=True)
class RetiredKey:
    """A key that earlier library versions accepted and the current one does not.

    ``accepts`` decides which values are still loadable: ``None`` accepts any
    value, a tuple accepts only the listed values (normally the former default,
    which is what serialized configs contain), and a callable decides per value.
    """

    reason: str
    accepts: tuple[Any, ...] | Callable[[Any], bool] | None = None

    def accepted(self, value: Any) -> bool:
        if self.accepts is None:
            return True
        if callable(self.accepts):
            return bool(self.accepts(value))
        normalized = list(value) if isinstance(value, tuple) else value
        return any(normalized == candidate for candidate in self.accepts)


def _any(reason: str) -> RetiredKey:
    return RetiredKey(reason)


def _only(value: Any, reason: str) -> RetiredKey:
    return RetiredKey(reason, accepts=(value,))


def _never(reason: str) -> RetiredKey:
    return RetiredKey(reason, accepts=())


def _inactive(value: Any) -> bool:
    """Return True for an empty or explicitly disabled section."""
    if value is None:
        return True
    if isinstance(value, Mapping):
        return not value or value.get("enabled", False) is False
    return False


def _disabled(reason: str) -> RetiredKey:
    return RetiredKey(reason, accepts=_inactive)


def _wandb_off(value: Any) -> bool:
    """Return True for a wandb section that did not request logging."""
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    return not value.get("use_wandb", False) and not value.get("enabled", False)


_OLD_LAYOUT = "belongs to the configuration layout used before September 2025"
_NO_WANDB = "was removed; the library does not log to Weights & Biases"
_NEVER_CHECKPOINTS = (
    "never configured the trainer; periodic checkpoints are configured by "
    "experiment.checkpointing"
)
_FLAT_ABLATION = "was replaced by params.selection and params.methods on 2026-03-12"
_ROUTING = (
    "was removed on 2026-10-08; which cells read the transferred inhibitory "
    "stream follows from architecture.inhibitory_layer_sizes and the "
    "connectivity synapse counts"
)
_NEVER_READ_RLS = "was never read; no RLS estimator was implemented"
_NEVER_READ_BATCH_PROCESS = (
    "was never read (the correlation analysis computes each correlation matrix "
    "in one pass) and was retired on 2026-10-13"
)
_NEVER_READ_CENTRALITY = (
    "was never read; the centrality modulator uses inverse branch depth"
)
_DERIVED_CONV = (
    "was never read; convolutional autoencoders derive kernel sizes and strides "
    "from input_shape and latent_shape"
)


def _common_trainer_keys() -> dict[str, RetiredKey]:
    return {
        "save_path": _any(
            "was removed on 2026-03-17; run outputs are written under "
            "outputs.results_dir"
        ),
        "checkpointing": _any(_NEVER_CHECKPOINTS),
        "checkpoint_interval": _any(_NEVER_CHECKPOINTS),
    }


def _local_rule_keys() -> dict[str, RetiredKey]:
    return {
        "rho_value": _only(
            None,
            "is not an option of the four-factor rule, which estimates rho "
            "from activity",
        ),
        "weight_lr_multiplier": _only(
            None, "is not an option; set learning rates under common.param_groups"
        ),
        "compute_gradient_stats": _any("was a diagnostic switch with no effect"),
        "rls_forgetting": _any(_NEVER_READ_RLS),
        "morphology_centrality_metric": _any(_NEVER_READ_CENTRALITY),
    }


RETIRED_KEYS: dict[str, dict[str, RetiredKey]] = {
    "": {
        "task": _never(_OLD_LAYOUT),
        "train": _never(_OLD_LAYOUT),
        "visualization": _never(_OLD_LAYOUT),
        "wandb": RetiredKey(_NO_WANDB, accepts=_wandb_off),
    },
    "experiment": {
        "enable_amp": _any(
            "has had no effect since ExperimentConfig dropped it (2026-03-17); "
            "mixed precision is controlled by training.<phase>.common.use_amp"
        ),
        "enable_nvtx": _any("was removed on 2026-05-08; it only added profiler ranges"),
        "use_amp": _any(
            "has never been an option; mixed precision is controlled by "
            "training.<phase>.common.use_amp"
        ),
        "enable_profiling": _only(
            False,
            "was never implemented; wrap the training call in torch.profiler "
            "instead",
        ),
        "profiling_output_dir": _any("was never read"),
        "enable_hooks": _any(
            "never disabled anything: gradient-scaling hooks run whenever a "
            "model.core gradient_scaling option is not 'none'"
        ),
    },
    "experiment.checkpointing": {
        "checkpoint_dir": _any(
            "has never chosen where checkpoints are written; they go under the run "
            "directory"
        ),
    },
    "model": {
        key: _never(_OLD_LAYOUT)
        for key in (
            "base",
            "network",
            "input_network",
            "core_network",
            "encoder_network",
            "decoder_network",
            "probabilistic",
        )
    },
    "model.encoder": {
        "load_save_root": _any(
            "was read only by the removed encoder pretraining script"
        ),
    },
    "model.encoder.params": {
        "backbone": _only(None, "was removed on 2026-03-13"),
        "drop_layers": _only([], "was removed on 2026-03-13"),
        "freeze": _only(False, "was removed on 2026-03-13"),
        "input_size": _only(None, "was removed on 2026-03-13"),
        "split_after": _only(None, "was removed on 2026-03-13"),
        "weights": _only(None, "was removed on 2026-03-13"),
        "encoder_kernel_sizes": _any(_DERIVED_CONV),
        "encoder_strides": _any(_DERIVED_CONV),
    },
    "model.decoder.params": {
        "backbone": _only(None, "was removed on 2026-03-13"),
        "drop_layers": _only([], "was removed on 2026-03-13"),
        "freeze": _only(False, "was removed on 2026-03-13"),
        "split_from": _only(None, "was removed on 2026-03-13"),
        "weights": _only(None, "was removed on 2026-03-13"),
    },
    "model.core": {
        "lda_init": _disabled("was removed on 2025-12-12"),
        "multi_population": _disabled(
            "was replaced by model.core.population_network on 2026-06-08"
        ),
        "population_graph": _disabled(
            "was replaced by model.core.population_network on 2026-06-10"
        ),
        "recurrent": _disabled(
            "selected the legacy recurrent cell stack, removed on 2026-02-27; use "
            "connectivity.rec_* synapse counts (EINet types), model.core.unified_ei "
            "or model.core.population_network with recurrent layers"
        ),
    },
    "model.core.transfer": {
        "e_stream_dims": _only(None, "was removed on 2026-08-05"),
        "split_input_stream": _only(False, "was removed on 2026-08-05"),
        # Checked against the rest of the core by
        # inhibitory_wiring.check_retired_inhibitory_routing.
        "input_mode": RetiredKey(_ROUTING, accepts=(0, 1)),
        "inhibitory_mode": RetiredKey(_ROUTING, accepts=("first", "none", "all")),
        **{
            key: _any(_ROUTING)
            for key in (
                "allow_direct_inhibitory_stream",
                "allow_direct_inhibitory_input",
                "require_explicit_direct_inhibitory_stream",
                "input_mode1_build_inhibitory_population",
                "build_inhibitory_population_for_input_mode1",
            )
        },
    },
    "model.core.sparsity": {
        "strategy": _only("none", "was renamed gradient_scaling on 2025-09-23"),
    },
    "model.core.blocklinear": {
        "strategy": _only("none", "was renamed gradient_scaling on 2025-09-23"),
    },
    "model.core.reactivation": {
        "strategy": _only("none", "was renamed gradient_scaling on 2025-09-23"),
        "match_additive_init_to_shunting": _only(
            False,
            "was replaced by model.core.reactivation.init_policy on 2026-04-18; "
            "additive cores now initialize their gates by that policy",
        ),
    },
    "training": {
        "encoder": _any(
            "was retired on 2026-10-07: the main phase trains an encoder jointly "
            "(training.main.common.param_groups.encoder_lr), and "
            "training.encoder_pretraining pretrains autoencoder encoders"
        ),
    },
    "training.main.common": _common_trainer_keys(),
    "training.main.optimizer": {
        "state_dtype": _only(None, "was removed on 2026-08-10"),
    },
    "training.main.learning_strategy_config": _local_rule_keys(),
    "training.main.learning_strategy_config.five_factor": {
        "rls_forgetting": _any(_NEVER_READ_RLS)
    },
    "training.main.learning_strategy_config.morphology_aware": {
        "morphology_centrality_metric": _any(_NEVER_READ_CENTRALITY)
    },
    "training.main.pruning": {
        "dynamic_pruning": _only(
            False, "was never implemented; pruning runs once after training"
        ),
        "pruning_frequency": _any("was never read; pruning runs once after training"),
    },
    "distributed.fsdp": {
        "use_fsdp": _only(
            False, "was removed on 2026-03-14; select FSDP with distributed.mode"
        ),
        "two_tier_min_numel": _only(25000000, "was removed on 2026-08-10"),
        "cpu_init": _any(
            "was retired on 2026-10-09: FSDP always receives the model on the "
            "CPU and moves each wrapped unit to its GPU while sharding it"
        ),
        "reduce_communication_overhead": _only(
            True,
            "was retired on 2026-10-09; set limit_all_gathers and "
            "forward_prefetch, which it switched on together",
        ),
    },
    "analysis": {
        "branch_information": _disabled(_OLD_LAYOUT),
        "ei_info_analysis": _disabled(_OLD_LAYOUT),
        "synaptic_ablation": _disabled(_OLD_LAYOUT),
        "synaptic_weight_expectation": _disabled("was removed on 2025-09-12"),
        "synaptic_attribution_analysis": _disabled("was removed on 2026-07-19"),
        "input_regime_operating_point_analysis": _disabled("was removed on 2026-07-19"),
        "compositional_ood_analysis": _disabled("was removed on 2026-08-29"),
    },
    "analysis.runtime.training": {
        "splits": _any(
            "was removed on 2026-03-16; choose evaluated splits with "
            "analysis.performance.training_splits"
        ),
    },
    "analysis.runtime.final": {
        "splits": _any(
            "was removed on 2026-03-16; choose evaluated splits with "
            "analysis.performance.final_splits"
        ),
    },
    "analysis.ablation.params": {
        **{
            key: _any(_FLAT_ABLATION)
            for key in (
                "ablate_upstream",
                "ablation_methods",
                "accuracy",
                "auc",
                "categorical_loglikelihood",
                "clamp_statistic",
                "compartment_ablation",
                "compute_pred_label_mi",
                "cosine_similarity",
                "layer_ablation",
                "levels",
                "metrics",
                "mse",
                "shuffle_seed",
                "targets",
            )
        },
        **{
            key: _any("has never been an option of the ablation analysis")
            for key in (
                "ablate_all_synapses",
                "ablate_excitation",
                "ablate_inhibition",
                "ablate_components",
                "ablation_method",
                "ablation_types",
            )
        },
    },
    "analysis.compartment_snr.params": {
        "global_analysis": _any(
            "was never read; global_aggregation controls the global summary"
        ),
    },
    "analysis.correlation.params": {
        **{
            key: _any("was removed in August 2025")
            for key in (
                "compute_branch_correlations",
                "compute_conditional_correlations",
                "compute_synaptic_correlations",
                "per_branch_analysis",
                "save_plots",
            )
        },
        "batch_process": _any(_NEVER_READ_BATCH_PROCESS),
    },
    "analysis.correlation.params.compute": {
        "batch_process": _any(_NEVER_READ_BATCH_PROCESS),
    },
    "analysis.information.params": {
        "compute_pid": _only(
            False, "was replaced by selection.components: [pid], which runs PID"
        ),
        "include_branch_input": _any("was removed on 2025-12-12"),
        "compute_per_branch": _any("was removed in July 2025"),
        "ei_specific_analysis": _any("was removed in August 2025"),
        "neuron_selection": _any("was removed in August 2025"),
    },
    "analysis.information.params.baselines.label_shuffle_null": {
        "conditional_bins": _only(10, "was removed on 2026-09-30"),
        "conditional_shuffle": _only("auto", "was removed on 2026-09-30"),
    },
    "analysis.multiplicative_gain.params": {
        "gain_sampling": _only("log_uniform", "was removed on 2026-08-29"),
    },
    "analysis.path_matched_information.params": {
        "add_probe_seed_offset": _any(
            "was retired on 2026-10-08: an unset information_seed is derived "
            "from experiment.probe_seed, and a written one is used unchanged"
        ),
    },
    "analysis.synapse_turnover.params": {
        key: _any("was never read and was retired on 2026-10-08")
        for key in ("synaptic_analysis", "save_synaptic_report")
    },
    "analysis.single_layer_contribution.params": {
        key: _any("was removed in July 2025")
        for key in ("batch_size", "save_csv", "save_plots")
    },
    "analysis.weight.params": {
        key: _any("has never been an option of the weight analysis")
        for key in ("sparsity", "weight_norm", "weight_distribution")
    },
}


@dataclass(frozen=True)
class RetiredValue:
    """List entries that earlier library versions accepted but never read.

    A retired entry of a list option loads with a warning and is dropped from
    the list, which the library reproduces exactly since the entry had no
    effect; ``dendritic-upgrade-config`` removes it from files.
    """

    values: tuple[Any, ...]
    reason: str


_NEVER_IMPLEMENTED_ACROSS_LAYER = RetiredValue(
    ("across_layer",),
    "was never implemented (stratified sampling draws within-branch and "
    "within-layer pairs only) and was retired on 2026-10-13",
)

#: Retired entries of list options, by section and key.
RETIRED_VALUES: dict[str, dict[str, RetiredValue]] = {
    "analysis.correlation.params": {
        "stratified_categories": _NEVER_IMPLEMENTED_ACROSS_LAYER,
    },
    "analysis.correlation.params.sampling": {
        "stratified_categories": _NEVER_IMPLEMENTED_ACROSS_LAYER,
    },
}

_WARNED_RETIRED_VALUES: set[str] = set()


def drop_retired_values(section: str, key: str, values: Any) -> Any:
    """Return the list ``values`` of ``section.key`` without its retired entries.

    Each retired entry warns once per process. Values that are not lists, and
    options without retired entries, are returned unchanged.
    """
    entry = RETIRED_VALUES.get(section, {}).get(key)
    if entry is None or not isinstance(values, (list, tuple)):
        return values
    retired = [value for value in values if value in entry.values]
    if not retired:
        return values
    for value in retired:
        message = f"{section}.{key} entry {value!r}"
        if message not in _WARNED_RETIRED_VALUES:
            _WARNED_RETIRED_VALUES.add(message)
            logger.warning("Ignoring %s: it %s.", message, entry.reason)
    kept = [value for value in values if value not in entry.values]
    return tuple(kept) if isinstance(values, tuple) else kept


def drop_retired_values_in(data: dict[str, Any], *, section: str) -> dict[str, Any]:
    """Return ``data`` (a section mapping) without the retired list entries.

    Nested sections of ``section`` are cleaned too. Mappings are copied where
    they change, so the caller's mapping is left as it was.
    """
    for path, retired in RETIRED_VALUES.items():
        if path != section and not path.startswith(f"{section}."):
            continue
        parts = [part for part in path[len(section) :].split(".") if part]
        data = _drop_retired_values_at(data, parts, path, retired)
    return data


def _drop_retired_values_at(
    node: Any, parts: list[str], path: str, retired: dict[str, RetiredValue]
) -> Any:
    if not isinstance(node, Mapping):
        return node
    if parts:
        head, rest = parts[0], parts[1:]
        if head not in node:
            return node
        child = _drop_retired_values_at(node[head], rest, path, retired)
        if child is node[head]:
            return node
        return {**node, head: child}
    changed = {
        key: drop_retired_values(path, key, node[key]) for key in retired if key in node
    }
    if all(changed[key] is node[key] for key in changed):
        return node
    return {**node, **changed}


def to_mapping(raw: Any, *, section: str) -> dict[str, Any]:
    """Return a section payload as a plain dict, or raise if it is not a mapping."""
    if raw is None:
        return {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, Mapping):
        label = section or "the configuration"
        raise ValueError(f"{label} must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def check_section_keys(
    data: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    section: str,
) -> dict[str, Any]:
    """Return the entries of ``data`` whose keys are in ``allowed``.

    Retired keys of the section warn and are dropped, or raise when their value
    cannot be reproduced. Any other unknown key raises
    :class:`UnknownConfigKeyError`.
    """
    allowed = set(allowed)
    retired = RETIRED_KEYS.get(section, {})
    kept: dict[str, Any] = {}
    unknown = []
    for key, value in data.items():
        if key in allowed:
            kept[key] = value
            continue
        entry = retired.get(key)
        if entry is None:
            unknown.append(key)
            continue
        _check_retired_value(section, key, value, entry)
        logger.warning(
            "Ignoring %s: it %s.", f"{section}.{key}" if section else key, entry.reason
        )
    if unknown:
        raise UnknownConfigKeyError(section, unknown, allowed)
    return kept


def config_from_payload(config_cls: type, payload: Mapping[str, Any], *, section: str):
    """Build the dataclass ``config_cls`` from a payload mapping.

    Unknown keys raise :class:`UnknownConfigKeyError` naming ``section`` and the
    closest valid key, instead of the constructor's bare ``TypeError``.
    """
    allowed = {item.name for item in dataclasses.fields(config_cls)}
    return config_cls(**check_section_keys(dict(payload), allowed, section=section))


def _check_retired_value(section: str, key: str, value: Any, entry: RetiredKey):
    path = f"{section}.{key}" if section else str(key)
    if not entry.accepted(value):
        raise ValueError(
            f"{path}={value!r} cannot be loaded: {path} {entry.reason}. "
            "Replay this config at the library revision that produced it, "
            "or remove the key."
        )


_WARNED_RETIRED: set[str] = set()


def check_retired_keys(data: Mapping[str, Any], *, section: str) -> None:
    """Check retired keys that reach a builder without passing the loader.

    Every key of ``data`` must be a retired key of ``section``. A value that
    cannot be reproduced raises; otherwise each key warns once per process,
    since builders can see the same mapping several times.
    """
    retired = RETIRED_KEYS[section]
    for key, value in data.items():
        entry = retired[key]
        _check_retired_value(section, key, value, entry)
        path = f"{section}.{key}"
        if path not in _WARNED_RETIRED:
            _WARNED_RETIRED.add(path)
            logger.warning("Ignoring %s: it %s.", path, entry.reason)


_HINTS: dict[type, dict[str, Any]] = {}


def _field_types(config_cls: type) -> dict[str, Any]:
    if config_cls not in _HINTS:
        try:
            _HINTS[config_cls] = typing.get_type_hints(config_cls)
        except Exception:  # unresolved forward references: use raw annotations
            _HINTS[config_cls] = {
                field.name: field.type for field in dataclasses.fields(config_cls)
            }
    return _HINTS[config_cls]


def _nested_config_class(annotation: Any) -> type | None:
    """Return the dataclass a field holds, unwrapping ``Optional``."""
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    candidates = [
        arg
        for arg in typing.get_args(annotation)
        if isinstance(arg, type) and dataclasses.is_dataclass(arg)
    ]
    if len(candidates) == 1 and typing.get_origin(annotation) is not list:
        return candidates[0]
    return None


def _list_item_config_class(annotation: Any) -> type | None:
    """Return the dataclass held by a ``list[...]`` field."""
    if typing.get_origin(annotation) is not list:
        return None
    args = typing.get_args(annotation)
    if len(args) == 1 and isinstance(args[0], type):
        if dataclasses.is_dataclass(args[0]):
            return args[0]
    return None


_WARNED_RENAMED: set[str] = set()


def apply_renamed_fields(
    config_cls: type, data: Mapping[str, Any], *, section: str
) -> dict[str, Any]:
    """Return ``data`` with renamed fields of ``config_cls`` under their new names.

    A config class lists its renamed fields in a ``RENAMED_FIELDS`` class
    attribute, ``{old: (new, date)}``. An old name loads under the new one and
    warns once per process. Writing both names with different values raises.
    """
    renamed = getattr(config_cls, "RENAMED_FIELDS", None) or {}
    if not set(data) & set(renamed):
        # Nothing to rename: keep the caller's mapping.
        return data if isinstance(data, dict) else dict(data)
    data = dict(data)
    for old, (new, date) in renamed.items():
        if old not in data:
            continue
        value = data.pop(old)
        if new in data and data[new] != value:
            raise ValueError(
                f"{section}.{old} was renamed {section}.{new} on {date}, and the "
                f"config sets both ({value!r} and {data[new]!r}); keep only "
                f"{section}.{new}"
            )
        data.setdefault(new, value)
        path = f"{section}.{old}"
        if path not in _WARNED_RENAMED:
            _WARNED_RENAMED.add(path)
            logger.warning(
                "%s was renamed %s.%s on %s; the old name still loads.",
                path,
                section,
                new,
                date,
            )
    return data


def build_section(config_cls: type, raw: Any, *, section: str) -> Any:
    """Build ``config_cls`` from a mapping, checking keys at every level.

    Nested dataclass fields (including ``Optional`` and ``list`` of
    dataclasses) are built recursively with their own section paths. Values
    that are already instances of the target class are used as they are, so
    callers can pre-build sections that need aliases or migrations. Renamed
    fields (``RENAMED_FIELDS``) load under their new names.
    """
    if isinstance(raw, config_cls):
        return raw
    data = apply_renamed_fields(
        config_cls, to_mapping(raw, section=section), section=section
    )
    allowed = {field.name for field in dataclasses.fields(config_cls)}
    kwargs = check_section_keys(data, allowed, section=section)
    types = _field_types(config_cls)
    for name, value in list(kwargs.items()):
        annotation = types.get(name)
        path = f"{section}.{name}" if section else name
        nested = _nested_config_class(annotation)
        if nested is not None and (
            isinstance(value, Mapping) or OmegaConf.is_config(value)
        ):
            kwargs[name] = build_section(nested, value, section=path)
            continue
        item_cls = _list_item_config_class(annotation)
        if item_cls is not None and isinstance(value, (list, tuple)):
            kwargs[name] = [
                (
                    build_section(item_cls, item, section=f"{path}[{index}]")
                    if isinstance(item, Mapping)
                    else item
                )
                for index, item in enumerate(value)
            ]
            continue
        if OmegaConf.is_config(value):
            kwargs[name] = OmegaConf.to_container(value, resolve=True)
    return config_cls(**kwargs)


def field_default(config_cls: type, name: str) -> Any:
    """Return the default value of a dataclass field (calling its factory)."""
    for item in dataclasses.fields(config_cls):
        if item.name == name:
            if item.default is not dataclasses.MISSING:
                return item.default
            if item.default_factory is not dataclasses.MISSING:
                return item.default_factory()
            raise ValueError(f"{config_cls.__name__}.{name} has no default")
    raise AttributeError(f"{config_cls.__name__} has no field {name!r}")


def build_nested_sections(obj: Any, *, section: str) -> None:
    """Type the mapping-valued nested sections of a dataclass instance in place.

    Used by ``__post_init__`` methods so that programmatic construction with
    plain dicts gets the same typed, key-checked sections as ``load_config``.
    """
    types = _field_types(type(obj))
    for item in dataclasses.fields(obj):
        value = getattr(obj, item.name)
        annotation = types.get(item.name)
        path = f"{section}.{item.name}"
        nested = _nested_config_class(annotation)
        if nested is not None and (
            isinstance(value, Mapping) or OmegaConf.is_config(value)
        ):
            setattr(obj, item.name, build_section(nested, value, section=path))
            continue
        item_cls = _list_item_config_class(annotation)
        if item_cls is not None and isinstance(value, (list, tuple)):
            setattr(
                obj,
                item.name,
                [
                    (
                        build_section(item_cls, entry, section=f"{path}[{index}]")
                        if isinstance(entry, Mapping)
                        else entry
                    )
                    for index, entry in enumerate(value)
                ],
            )


__all__ = [
    "METADATA_SECTION",
    "RETIRED_KEYS",
    "RETIRED_VALUES",
    "RetiredKey",
    "RetiredValue",
    "UnknownConfigKeyError",
    "apply_renamed_fields",
    "build_nested_sections",
    "build_section",
    "check_retired_keys",
    "check_section_keys",
    "drop_retired_values",
    "drop_retired_values_in",
    "field_default",
    "to_mapping",
]
