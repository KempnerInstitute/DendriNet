"""Checks on the parameters the training pipeline passes to dataset loaders.

The pipeline merges ``data.processing``, the selected
``data.dataset_params.<dataset_name>`` block and the resolved seeds into one
flat mapping. Each loader reads only some of these keys. The helpers here
reject dataset-specific keys a loader never reads and label noise on datasets
that do not apply it, so a misspelled or unsupported option fails instead of
being ignored.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from typing import Any

# Seeds the pipeline always adds; loaders use the ones they need.
PIPELINE_SEED_KEYS = frozenset({"label_noise_seed", "split_seed", "seed"})
# The fields of ``data.processing``, merged into every dataset's parameters.
PROCESSING_KEYS = frozenset({"flatten", "normalize", "label_noise_rate"})
PIPELINE_PARAMETER_KEYS = PIPELINE_SEED_KEYS | PROCESSING_KEYS

# Datasets whose loader flips training labels for ``label_noise_rate > 0``.
LABEL_NOISE_DATASETS = frozenset({"mnist", "fashion_mnist", "cifar10", "cifar100"})


class ReadRecorder(dict):
    """Parameter mapping that records every key a loader looks up.

    ``get``, ``[]`` and ``in`` count as reads, including lookups of keys that
    are absent, so the recorded set also lists the keys a loader accepts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def __getitem__(self, key: Any) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self.read.add(key)  # type: ignore[arg-type]
        return super().__contains__(key)


def reject_unread_dataset_params(
    dataset_name: str,
    parameters: Mapping[str, Any],
    read_keys: Iterable[str],
) -> None:
    """Raise when ``parameters`` hold dataset-specific keys the loader ignores.

    Keys the pipeline adds for every dataset (``PIPELINE_PARAMETER_KEYS``) are
    not checked here.
    """
    read = set(read_keys)
    unread = sorted(set(parameters) - read - PIPELINE_PARAMETER_KEYS)
    if not unread:
        return
    hints = []
    for key in unread:
        match = difflib.get_close_matches(str(key), sorted(map(str, read)), n=1)
        hints.append(f"{key} (did you mean {match[0]}?)" if match else str(key))
    raise ValueError(
        f"data.dataset_params.{dataset_name} sets "
        + ", ".join(hints)
        + f", which the {dataset_name} loader does not read. Remove "
        + ("it." if len(unread) == 1 else "them.")
    )


# What the loaders use for ``data.processing`` fields left unset (None).
_PROCESSING_DEFAULTS = {"flatten": True, "normalize": False}
# Datasets whose own ``normalize`` default is true.
_NORMALIZED_BY_DEFAULT = frozenset({"sequential_mnist", "stringer_v1"})


def set_processing_parameters(processing: Any) -> dict[str, Any]:
    """Return the ``data.processing`` fields that are set.

    Unset fields (None) are not passed, so each loader applies its own default.
    """
    from dendritic_modeling.config.conversion import to_plain_dict

    return {
        key: value
        for key, value in to_plain_dict(processing).items()
        if value is not None
    }


def effective_processing(
    dataset_name: str,
    processing: Any,
    dataset_params: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Return the ``flatten`` and ``normalize`` values a dataset is loaded with.

    A value in the dataset's ``data.dataset_params`` block wins over
    ``data.processing``; an unset field takes the loader default.
    """
    written = {**set_processing_parameters(processing), **dict(dataset_params or {})}
    resolved = {}
    for name, default in _PROCESSING_DEFAULTS.items():
        if written.get(name) is not None:
            resolved[name] = bool(written[name])
        elif name == "normalize" and dataset_name in _NORMALIZED_BY_DEFAULT:
            resolved[name] = True
        else:
            resolved[name] = default
    return resolved


def reject_unapplied_processing(
    dataset_name: str,
    parameters: Mapping[str, Any],
    applied: Iterable[str],
) -> None:
    """Raise when ``normalize``/``flatten`` request an effect the loader lacks.

    ``applied`` names the processing options the loader implements. A written
    ``true`` for another option raises; ``false`` or unset asks for no effect
    and passes.
    """
    unapplied = sorted(
        key
        for key in ("flatten", "normalize")
        if parameters.get(key) is True and key not in set(applied)
    )
    if unapplied:
        raise ValueError(
            f"data.processing {', '.join(f'{key}: true' for key in unapplied)} "
            f"would be ignored: the {dataset_name} loader does not apply "
            f"{' or '.join(unapplied)}. Remove it or set it to false."
        )


def reject_unapplied_label_noise(dataset_name: str, label_noise_rate: Any) -> None:
    """Raise when label noise is requested for a dataset that does not apply it."""
    if label_noise_rate is None or float(label_noise_rate) == 0.0:
        return
    if dataset_name in LABEL_NOISE_DATASETS:
        return
    raise ValueError(
        f"data.processing.label_noise_rate={label_noise_rate} would be ignored: "
        f"label noise is applied only to {', '.join(sorted(LABEL_NOISE_DATASETS))}, "
        f"and dataset {dataset_name!r} would train on clean labels. Set "
        "label_noise_rate: 0."
    )


__all__ = [
    "LABEL_NOISE_DATASETS",
    "PIPELINE_PARAMETER_KEYS",
    "PIPELINE_SEED_KEYS",
    "PROCESSING_KEYS",
    "ReadRecorder",
    "effective_processing",
    "reject_unapplied_label_noise",
    "reject_unapplied_processing",
    "reject_unread_dataset_params",
    "set_processing_parameters",
]
