"""Utility functions for Partial Information Decomposition (PID).

These helpers attempt to import the PID package located in the project
``external`` directory.  If the package cannot be imported, the
functions will raise ``ImportError``.  The intention is to keep the code
self-contained while still allowing PID-based analyses when the
external dependency is available.
"""

from __future__ import annotations

import logging

# typing imports removed - not needed
import numpy as np

try:
    # The external PID library should be placed inside ``external``
    # so that ``external.PID`` becomes importable.  Adjust the import as
    # necessary for your local setup.
    from external import PID  # type: ignore
except Exception as e:  # pragma: no cover - best effort
    PID = None
    _import_error = e
else:
    _import_error = None


def check_pid_available() -> None:
    """Raise an ImportError if the PID package is not available."""

    if PID is None:
        raise ImportError(
            "PID package not found. Ensure the external PID library is "
            "available in the 'external' directory."
        ) from _import_error


def pid_synergy_redundancy(
    sources: list[np.ndarray], target: np.ndarray
) -> tuple[float, float]:
    """Compute synergy and redundancy using the external PID library.

    Parameters
    ----------
    sources : list of ndarrays
        List of source variables with shape ``(n_samples, n_features)``.
    target : ndarray
        Target variable with shape ``(n_samples, n_features)``.

    Returns
    -------
    tuple of floats
        ``(synergy, redundancy)`` from the chosen PID method.
    """

    check_pid_available()

    # The exact API will depend on the external library.  Here we
    # demonstrate a generic usage pattern expected from typical PID
    # implementations.
    data = np.hstack([*sources, target])
    pid_obj = PID(data)

    try:
        synergy = pid_obj.synergy()
        redundancy = pid_obj.redundancy()
    except AttributeError as exc:  # pragma: no cover - depends on PID API
        logging.error("Unexpected PID API: %s", exc)
        raise

    return float(synergy), float(redundancy)
