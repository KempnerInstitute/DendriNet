"""Partial information decomposition (PID) of two sources about a target.

The minimum-mutual-information (MMI) measure takes the redundancy to be the
smaller single-source information (Barrett, Phys. Rev. E 91, 052802, 2015)::

    R   = min(I(S1;T), I(S2;T))
    U1  = I(S1;T) - R
    U2  = I(S2;T) - R
    Syn = I(S1,S2;T) - I(S1;T) - I(S2;T) + R

For jointly Gaussian sources and a univariate target the common PID measures
reduce to MMI. The three mutual informations come from any estimator, so the
terms inherit its bias: with estimated values the synergy can come out slightly
negative, and it is reported unclipped. A non-finite mutual information makes
every term NaN.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

PID_MEASURES = ("mmi",)


@dataclass(frozen=True)
class PIDTerms:
    """Redundant, unique and synergistic information, in the estimator's units."""

    redundancy: float
    unique_1: float
    unique_2: float
    synergy: float
    joint: float


def mmi_pid(i_1: float, i_2: float, i_joint: float) -> PIDTerms:
    """Decompose ``I(S1,S2;T)`` with the minimum-mutual-information redundancy."""
    i_1, i_2, i_joint = float(i_1), float(i_2), float(i_joint)
    if not np.all(np.isfinite([i_1, i_2, i_joint])):
        nan = float("nan")
        return PIDTerms(nan, nan, nan, nan, i_joint)
    redundancy = min(i_1, i_2)
    return PIDTerms(
        redundancy=redundancy,
        unique_1=i_1 - redundancy,
        unique_2=i_2 - redundancy,
        synergy=i_joint - i_1 - i_2 + redundancy,
        joint=i_joint,
    )


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.reshape(-1, 1) if array.ndim == 1 else array


_RETIRED_HELPER = (
    "{name} was a placeholder for an external PID package that DendriNet never "
    "provided, so it always raised ImportError; it is deprecated and will be "
    "removed. The decomposition is built in: use "
    "partial_information_decomposition(source_1, source_2, target, "
    "mutual_information) or mmi_pid(i_1, i_2, i_joint) from "
    "dendritic_modeling.utils.information."
)


def check_pid_available() -> None:
    """Deprecated: raises :class:`ImportError`, as it always did.

    It checked for an external ``PID`` package that DendriNet never provided.
    The decomposition is built in (:func:`partial_information_decomposition`).
    """
    raise ImportError(_RETIRED_HELPER.format(name="check_pid_available"))


def pid_synergy_redundancy(
    sources: list[np.ndarray], target: np.ndarray
) -> tuple[float, float]:
    """Deprecated: raises :class:`ImportError`, as it always did.

    It wrapped an external ``PID`` package that DendriNet never provided. The
    ``synergy`` and ``redundancy`` of :func:`partial_information_decomposition`,
    with a mutual-information estimator of your choice, replace its result.
    """
    raise ImportError(_RETIRED_HELPER.format(name="pid_synergy_redundancy"))


def partial_information_decomposition(
    source_1: np.ndarray,
    source_2: np.ndarray,
    target: np.ndarray,
    mutual_information: Callable[[np.ndarray, np.ndarray], float],
) -> PIDTerms:
    """MMI decomposition of the information two sources carry about ``target``.

    ``mutual_information(x, y)`` estimates ``I(X;Y)`` from samples in rows; the
    joint term passes both sources side by side.
    """
    s1, s2 = _as_2d(source_1), _as_2d(source_2)
    if len(s1) != len(s2) or len(s1) != len(target):
        raise ValueError("sources and target need the same number of samples")
    return mmi_pid(
        mutual_information(s1, target),
        mutual_information(s2, target),
        mutual_information(np.hstack([s1, s2]), target),
    )


__all__ = [
    "PID_MEASURES",
    "PIDTerms",
    "check_pid_available",
    "mmi_pid",
    "partial_information_decomposition",
    "pid_synergy_redundancy",
]
