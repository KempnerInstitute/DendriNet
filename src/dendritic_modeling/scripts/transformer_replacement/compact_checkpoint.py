"""Rewrite a transformer replacement checkpoint with packed sparse indices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _atomic_torch_save,
    _compact_sparse_bitmask_state_dict,
    _compact_sparse_index_state_dict,
)


def compact_replacement_checkpoint(
    source: Path,
    output: Path,
    *,
    encoding: str = "uint",
) -> dict[str, Any]:
    """Write a load-compatible checkpoint with compact topology storage."""
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Output must differ from the source checkpoint")
    if encoding not in {"uint", "bitmask"}:
        raise ValueError("encoding must be 'uint' or 'bitmask'")
    compact_state = (
        _compact_sparse_bitmask_state_dict
        if encoding == "bitmask"
        else _compact_sparse_index_state_dict
    )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    encodings: dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict, state_encoding = compact_state(payload["state_dict"])
        payload["state_dict"] = state_dict
        payload["sparse_index_encoding"] = state_encoding
        encodings["state_dict"] = state_encoding
    elif isinstance(payload, dict) and isinstance(
        payload.get("replacement_state_dicts"), list
    ):
        compact_states = []
        state_encodings = []
        for state_dict in payload["replacement_state_dicts"]:
            compact, state_encoding = compact_state(state_dict)
            compact_states.append(compact)
            state_encodings.append(state_encoding)
        payload["replacement_state_dicts"] = compact_states
        payload["sparse_index_encodings"] = state_encodings
        encodings["replacement_state_dicts"] = state_encodings
    else:
        raise ValueError(
            "Checkpoint must contain state_dict or replacement_state_dicts"
        )

    _atomic_torch_save(payload, str(output))
    source_bytes = os.path.getsize(source)
    output_bytes = os.path.getsize(output)
    return {
        "source": str(source),
        "output": str(output),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "size_reduction_factor": source_bytes / max(output_bytes, 1),
        "encoding": encoding,
        "index_encodings": encodings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--encoding",
        choices=("uint", "bitmask"),
        default="uint",
        help="Store indices as compact integers or row-wise topology bitmasks.",
    )
    args = parser.parse_args()
    report = compact_replacement_checkpoint(
        args.source,
        args.output,
        encoding=args.encoding,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
