"""Opt-in reduced-precision rounding of dendritic signals and conductances.

A dendritic layer with ``signal_rounding`` set to ``float16`` or ``bfloat16``
rounds its presynaptic signals, synaptic and coupling conductances, leak,
summed synaptic drives, voltages and outputs to that format, while the sums
inside each reduction, the parameters and the optimizer state keep the
parameter dtype. Rounding is a cast to the format and back, so gradients that
flow through a rounded value are rounded to the same format, as in
half-precision autocast without loss scaling. Values beyond the format's range
become infinite.
"""

import torch

SIGNAL_ROUNDING_FORMATS: dict[str, torch.dtype | None] = {
    "none": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def signal_rounding_dtype(name: str | None) -> torch.dtype | None:
    """Return the rounding dtype named by a ``signal_rounding`` option."""
    key = "none" if name is None else str(name).lower()
    if key not in SIGNAL_ROUNDING_FORMATS:
        raise ValueError(
            f"signal_rounding must be one of {sorted(SIGNAL_ROUNDING_FORMATS)}, "
            f"got {name!r}"
        )
    return SIGNAL_ROUNDING_FORMATS[key]


def round_signal(value, dtype: torch.dtype | None):
    """Round ``value`` to ``dtype`` and back; ``None`` returns it unchanged."""
    if dtype is None:
        return value
    if not torch.is_tensor(value):
        return float(torch.tensor(value, dtype=torch.float64).to(dtype))
    return value.to(dtype).to(value.dtype)


__all__ = ["SIGNAL_ROUNDING_FORMATS", "round_signal", "signal_rounding_dtype"]
