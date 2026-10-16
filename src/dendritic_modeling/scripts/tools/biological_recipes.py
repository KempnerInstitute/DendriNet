"""Validate or materialize hash-bound prospective biological study recipes.

Run as python -m dendritic_modeling.scripts.tools.biological_recipes. A study
manifest supplies the selected experiment/sweep sources, their scientific
bindings and explicit adoption options. Existing results are never relabelled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


def check_source(root, expected):
    import dendritic_modeling

    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError("A real pinned library Git checkout is required")
    if (
        Path(dendritic_modeling.__file__).resolve()
        != root / "src/dendritic_modeling/__init__.py"
    ):
        raise ValueError("Imported library differs from the selected checkout")
    if git(root, "rev-parse", "HEAD") != expected or git(root, "status", "--porcelain"):
        raise ValueError("Use a clean library checkout at the manifest's exact commit")


def read_experiments(entry, root):
    import yaml
    from omegaconf import OmegaConf

    from dendritic_modeling.config.composition import load_composed_yaml
    from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
        UnifiedSweepGenerator,
    )

    if entry.get("format") == "collection":
        for source in entry["sources"]:
            yield from read_experiments(
                {"source": source["path"], "sha256": source["sha256"]}, root
            )
        return
    path = root / entry["source"]
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Source hash changed: {path}")
    if entry.get("format", "experiment") == "experiment":
        yield yaml.safe_load(path.read_text())
    elif entry["format"] == "sweep":
        composed, sources = load_composed_yaml(path)
        # Composition can read other files. All must be bound by the inventory.
        expected = entry["composition_sha256"]
        for source in sources:
            source = Path(source).resolve()
            key = os.path.relpath(source, root)
            if expected.get(key) != sha(source):
                raise ValueError(f"Unbound or changed composition input: {source}")
        generator = UnifiedSweepGenerator()
        if not generator.validate_config(composed):
            raise ValueError(f"Invalid sweep: {path}")
        for generated in generator.generate_configs(composed):
            yield OmegaConf.to_container(generated, resolve=True)
    else:
        raise ValueError(f"Unknown source format: {entry['format']}")


def compare_models(original, prepared, input_dim, *, recurrent_steps=3):
    """Same-core adoption check, before data-dependent calibration or training.

    Exact equality is required for recipes declared computation-preserving.
    Includes zero-valued inputs, parameter gradients and one optimizer step.
    This is deliberately not a historical-version or full-training comparison.
    """
    import torch

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.networks.architectures.biological_constraints import (
        biological_violations,
    )
    from dendritic_modeling.scripts.script_utils import setup_utils
    from dendritic_modeling.utils import resolve_experiment_seeds, set_seed

    def capture(mapping):
        payload = copy.deepcopy(mapping)
        payload["model"].setdefault("encoder", {}).setdefault("params", {})[
            "input_dim"
        ] = input_dim
        config = config_from_mapping(payload)
        seeds = resolve_experiment_seeds(config.experiment)
        setup_utils._apply_topology_seed_defaults(config.model, seeds.topology_seed)
        setup_utils._apply_initialization_seed_defaults(config.model, seeds.model_seed)
        set_seed(int(seeds.model_seed), deterministic=True)
        model, _ = setup_utils.initialize_model(config.model)
        model.train()
        core = model.core_network
        generator = torch.Generator().manual_seed(913)
        shape = (
            (2, recurrent_steps, input_dim)
            if getattr(core, "is_recurrent", False)
            else (2, input_dim)
        )
        x = torch.rand(*shape, generator=generator)
        x[..., ::7] = 0
        x.requires_grad_()
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        y.square().mean().backward()
        if (
            x.grad is None
            or not torch.isfinite(y).all()
            or not torch.isfinite(x.grad).all()
        ):
            raise ValueError("Nonfinite output/gradient or missing input gradient")
        tensors = {
            "output": y.detach().clone(),
            "input_gradient": x.grad.detach().clone(),
        }
        tensors.update(
            {f"state/{k}": v.detach().clone() for k, v in model.state_dict().items()}
        )
        for k, p in model.named_parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise ValueError(f"Nonfinite parameter gradient: {k}")
                tensors[f"gradient/{k}"] = p.grad.detach().clone()
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        tensors.update(
            {f"updated/{k}": v.detach().clone() for k, v in model.state_dict().items()}
        )
        return tensors, (
            biological_violations(core) if config.model.core.biological_neuron else None
        )

    old, _ = capture(original)
    new, violations = capture(prepared)
    different = sorted(
        k
        for k in old.keys() | new.keys()
        if k not in old or k not in new or not torch.equal(old[k], new[k])
    )
    if violations:
        raise ValueError(violations)
    return {
        "exact": not different,
        "tensors": len(new),
        "differences": different,
        "biological_violations": violations,
        "rtol": 0,
        "atol": 0,
        "scope": "same-core initialization, nonnegative diagnostic inputs, gradients and one SGD step",
    }


def _set_dotted(mapping, dotted, value):
    """Set a dotted configuration path; numeric parts index lists."""
    node = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--check-models",
        choices=("none", "representative", "all"),
        default="representative",
    )
    parser.add_argument(
        "--entry",
        action="append",
        help="Restrict to named entries; receipt records partial scope",
    )
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    check_source(args.library_root, manifest["library_commit"])
    output = args.output_dir.resolve()
    if output.is_relative_to(args.library_root.resolve()) or output.is_relative_to(
        root
    ):
        raise ValueError("Outputs must be outside library and recipe roots")
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import yaml

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.config.biological_adoption import prepare_biological_config
    from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

    torch.set_num_threads(1)
    record = {
        "status": "running",
        "library_commit": manifest["library_commit"],
        "manifest_sha256": sha(args.manifest),
        "paper_commit": git(root, "rev-parse", "HEAD"),
        "paper_dirty": bool(git(root, "status", "--porcelain")),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": 1,
        "model_check_mode": args.check_models,
        "entries": [],
        "qualification": "prospective recipe validation; not historical retraining or figure reproduction",
    }
    wanted = set(args.entry or ())
    known = {entry["id"] for entry in manifest["entries"]}
    if not wanted <= known:
        raise ValueError(f"Unknown entries: {wanted - known}")
    try:
        for entry in manifest["entries"]:
            if Path(entry["id"]).name != entry["id"] or entry["id"] in {".", ".."}:
                raise ValueError("Entry IDs must be single safe directory names")
            if type(entry["runs"]) is not int or entry["runs"] < 1:
                raise ValueError("runs must be a positive integer")
            if not entry.get("check_indices") or any(
                type(i) is not int or i < 0 or i >= entry["runs"]
                for i in entry["check_indices"]
            ):
                raise ValueError("Declare valid representative check_indices")
            if entry["expected_computation"] not in {
                "identical",
                "retraining_required",
            }:
                raise ValueError("Declare expected_computation explicitly")
            if wanted and entry["id"] not in wanted:
                continue
            item = {"id": entry["id"], "count": 0, "model_checks": [], "errors": []}
            for index, raw in enumerate(read_experiments(entry, root)):
                try:
                    raw = {k: v for k, v in raw.items() if not k.startswith("_sweep_")}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        old, _, unresolved = upgrade_mapping(raw)
                        if unresolved:
                            raise ValueError(unresolved)
                        new = prepare_biological_config(raw, **entry["adoption"])
                        # Study-specific explicit circuit revisions, if required.
                        for dotted, value in entry.get("overrides", {}).items():
                            _set_dotted(new, dotted, value)
                        config_from_mapping(new)
                    check = args.check_models == "all" or (
                        args.check_models == "representative"
                        and index in entry["check_indices"]
                    )
                    if check:
                        result = compare_models(old, new, entry["input_dim"])
                        result["index"] = index
                        item["model_checks"].append(result)
                        if (
                            entry["expected_computation"] == "identical"
                            and not result["exact"]
                        ):
                            raise ValueError(
                                "Computation-preserving adoption changed diagnostic tensors"
                            )
                    if args.materialize:
                        run = output / "runs" / entry["id"] / str(index)
                        new.setdefault("outputs", {})["results_dir"] = str(run)
                        if args.data_root is not None:
                            new.setdefault("data", {})["base_dir"] = str(
                                args.data_root.resolve()
                            )
                        checkpointing = new.get("experiment", {}).get("checkpointing")
                        if isinstance(checkpointing, dict):
                            checkpointing["checkpoint_dir"] = str(run / "checkpoints")
                        destination = (
                            output / "configs" / entry["id"] / f"{index:05d}.yaml"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_text(yaml.safe_dump(new, sort_keys=False))
                    item["count"] += 1
                except Exception as exc:
                    item["errors"].append(
                        {"index": index, "error": f"{type(exc).__name__}: {exc}"}
                    )
            if item["count"] != entry["runs"]:
                item["errors"].append(
                    {"expected_runs": entry["runs"], "validated_runs": item["count"]}
                )
            record["entries"].append(item)
            print(
                json.dumps(
                    {
                        "id": item["id"],
                        "validated": item["count"],
                        "errors": len(item["errors"]),
                        "checks": len(item["model_checks"]),
                    }
                ),
                flush=True,
            )
        check_source(args.library_root, manifest["library_commit"])
        record["status"] = (
            "failed" if any(i["errors"] for i in record["entries"]) else "passed"
        )
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
