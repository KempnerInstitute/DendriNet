"""Pure helper functions for information analysis."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""

    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def is_discrete_class_labels(labels: np.ndarray) -> bool:
    """Heuristic: decide whether labels are discrete class IDs."""

    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return False
    unique = np.unique(labels)
    # Typical classification: small number of unique values
    if unique.size <= 50:
        return True
    # If labels are integer-like and the unique-label count is still smallish.
    if unique.size <= 200 and np.allclose(unique, np.round(unique), atol=1e-6):
        return True
    return False


def discrete_entropy_nats(labels: np.ndarray) -> float:
    """Discrete entropy H(C) in natural units (nats)."""

    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    # Natural log: matches the Kraskov MI units (nats)
    return float(-np.sum(p * np.log(p + 1e-12)))


def metric_targets_class(metric_key: str) -> bool:
    """Return True if the metric is of the form I(X;C) or I(X;C|...)."""

    if (
        not isinstance(metric_key, str)
        or "I(" not in metric_key
        or ")" not in metric_key
    ):
        return False
    lpar = metric_key.find("I(")
    lpar = metric_key.find("(", lpar)
    rpar = metric_key.find(")", lpar)
    if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
        return False
    inside = metric_key[lpar + 1 : rpar]
    main = inside.split("|", 1)[0]
    parts = [p.strip() for p in main.split(";")]
    if len(parts) < 2:
        return False
    return parts[1] == "C"


def extract_metric_tokens(metric_key: str) -> list[str]:
    """Extract variable tokens from metric keys like 'I(E;Vb|C)'."""

    if not isinstance(metric_key, str):
        return []
    if "I(" not in metric_key or ")" not in metric_key:
        return []

    lpar = metric_key.find("I(")
    lpar = metric_key.find("(", lpar)
    rpar = metric_key.find(")", lpar)
    if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
        return []

    inside = metric_key[lpar + 1 : rpar]
    inside = inside.replace("|", ";").replace(",", ";")
    return [t.strip() for t in inside.split(";") if t.strip()]


def rename_metric_key_tokens(metric_key: str, token_map: dict[str, str]) -> str:
    """Rename variable tokens inside MI keys while preserving separators."""

    if not isinstance(metric_key, str):
        return metric_key
    if "I(" not in metric_key:
        return metric_key

    renamed = metric_key
    for src, dst in token_map.items():
        # Replace full-token occurrences inside the I( ... ) block. Tokens are
        # delimited by one of: (, ;, ,, |, )
        renamed = re.sub(
            rf"(?<=[(;,|]){re.escape(src)}(?=[,;|)])",
            dst,
            renamed,
        )
    return renamed


def keep_metric_key_given_availability(
    metric_key: str,
    *,
    has_exc_synapses: bool,
    has_inh_synapses: bool,
    has_branch_input: bool,
) -> bool:
    """Return True if the metric is meaningful given available variables."""

    tokens = extract_metric_tokens(metric_key)
    if not tokens:
        return True

    if (not has_exc_synapses) and any(t.startswith("E") for t in tokens):
        return False
    if (not has_inh_synapses) and any(t.startswith("I") for t in tokens):
        return False
    if (not has_branch_input) and any(t.startswith("Vb") for t in tokens):
        return False

    return True


def should_convert_information_key_to_bits(key: Any) -> bool:
    """Heuristic: return True if `key` is an MI/CMI-like metric name in nats."""

    if not isinstance(key, str):
        return False
    if "I(" not in key:
        return False
    # Dimensionless ratios; do not convert.
    if "_normalized" in key:
        return False
    # Fisher discriminability proxies (d'^2-like); not information units.
    if "_fisher" in key:
        return False
    # Counts used for proxy bookkeeping.
    if key.endswith("_n"):
        return False
    return True


def convert_information_units_in_place(obj: Any, *, output_units: str) -> None:
    """Convert MI/CMI-like quantities in-place from nats to bits."""

    if output_units != "bits":
        return

    nat_to_bit = 1.0 / float(np.log(2.0))
    nat_to_bit_sq = nat_to_bit * nat_to_bit

    def _convert_any(x: Any) -> None:
        if isinstance(x, dict):
            for k in list(x.keys()):
                v = x.get(k)
                if isinstance(v, (dict, list)):
                    _convert_any(v)
                elif isinstance(v, (int, float, np.floating)):
                    if not should_convert_information_key_to_bits(k):
                        continue
                    scale = (
                        nat_to_bit_sq
                        if isinstance(k, str) and k.endswith(("_var", "_variance"))
                        else nat_to_bit
                    )
                    x[k] = float(v) * scale
        elif isinstance(x, list):
            for item in x:
                _convert_any(item)

    _convert_any(obj)


__all__ = [
    "convert_information_units_in_place",
    "discrete_entropy_nats",
    "ensure_2d",
    "extract_metric_tokens",
    "is_discrete_class_labels",
    "keep_metric_key_given_availability",
    "metric_targets_class",
    "rename_metric_key_tokens",
    "should_convert_information_key_to_bits",
]
