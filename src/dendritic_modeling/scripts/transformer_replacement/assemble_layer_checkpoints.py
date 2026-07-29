"""Assemble independently distilled layer checkpoints for all-layer warm starts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("OLMo generation manifest must contain a JSON object")
    return payload


def _checkpoint_sources(manifest: dict[str, Any]) -> list[tuple[int, Path]]:
    runs = manifest.get("layer_runs")
    if isinstance(runs, list) and runs:
        return [(int(run["layer"]), Path(str(run["checkpoint"]))) for run in runs]

    num_layers = int(manifest.get("num_layers", 0))
    results_root_value = str(manifest.get("results_root", "")).strip()
    if num_layers < 1 or not results_root_value:
        raise ValueError("Manifest lacks layer_runs and legacy result metadata")
    results_root = Path(results_root_value)
    return [
        (
            layer,
            results_root / f"layer_{layer:02d}" / f"layer_{layer}_replacement.pt",
        )
        for layer in range(num_layers)
    ]


def _place_checkpoint(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if mode == "symlink":
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError("mode must be 'symlink', 'hardlink', or 'copy'")


def assemble_layer_checkpoints(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    mode: str = "symlink",
) -> dict[str, Any]:
    """Create the flat checkpoint directory consumed by joint warm starts."""
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    sources = _checkpoint_sources(manifest)
    missing = [str(source) for _, source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing independently distilled checkpoints: " + ", ".join(missing)
        )

    destination_value = (
        str(output_dir)
        if output_dir is not None
        else str(manifest.get("assembled_checkpoint_dir", ""))
    ).strip()
    if not destination_value:
        raise ValueError("No output directory was provided or stored in the manifest")
    destination = Path(destination_value)
    destination.mkdir(parents=True, exist_ok=True)

    targets = [destination / f"layer_{layer}_replacement.pt" for layer, _ in sources]
    conflicts = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if conflicts:
        raise FileExistsError(
            "Refusing a partial assembly because targets already exist: "
            + ", ".join(conflicts)
        )

    assembled = []
    for (layer, source), target in zip(sources, targets):
        _place_checkpoint(source, target, mode)
        assembled.append(
            {
                "layer": layer,
                "source": str(source),
                "target": str(target),
                "size_bytes": source.stat().st_size,
            }
        )

    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "mode": mode,
        "output_dir": str(destination),
        "num_layers": len(assembled),
        "checkpoints": assembled,
    }
    (destination / "assembly_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="manifest.json from the all-layer generator")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = assemble_layer_checkpoints(
        args.manifest,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
