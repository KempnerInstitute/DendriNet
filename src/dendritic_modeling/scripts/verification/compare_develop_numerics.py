"""Compare deterministic numerical probes between two repository checkouts.

The script runs the same probe program in two separate Python subprocesses with
different ``PYTHONPATH`` roots. This avoids importing both versions into one
interpreter and keeps branch-to-branch comparisons honest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dendritic_modeling.utils.logging_config import setup_logging

PROBE_SOURCE = r"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
from dendritic_modeling.analysis.core.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import SynapticActivationAnalyzer
from dendritic_modeling.config import (
    AblationAnalysisParams,
    BranchActivationAnalysisParams,
    InformationAnalysisParams,
    PerformanceAnalysisParams,
    SynapticActivationAnalysisParams,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity
from dendritic_modeling.training.strategies.standard import Trainer


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _normalize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return _normalize(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return _normalize(obj.tolist())
    if isinstance(obj, np.generic):
        return _normalize(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    return obj


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().cpu().float().reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(tensor.shape), "numel": 0}
    return {
        "shape": list(tensor.shape),
        "numel": int(flat.numel()),
        "sum": float(flat.sum().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "first8": [float(v) for v in flat[:8].tolist()],
    }


def _state_digest(model: torch.nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    metadata = getattr(state, "_metadata", {})
    try:
        from dendritic_modeling.networks.state_versions import state_keys_added
    except ImportError:  # checkouts before the version declarations
        state_keys_added = None
    version_added = {}
    if state_keys_added is not None:
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for version, names in state_keys_added(module).items():
                for name in names:
                    if prefix + name in state:
                        version_added[prefix + name] = int(version)
    return {
        "tensors": {name: _tensor_digest(value) for name, value in state.items()},
        "module_versions": {
            name: meta.get("version") for name, meta in metadata.items()
        },
        "version_added": version_added,
    }


def _grad_digest(model: torch.nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            out[name] = _tensor_digest(param.grad)
    return out


def _make_dataset(n_samples: int = 64, input_dim: int = 8) -> TensorDataset:
    base = torch.linspace(-1.5, 1.5, steps=n_samples * input_dim, dtype=torch.float32)
    x = base.reshape(n_samples, input_dim)
    x = torch.stack(
        [
            torch.sin(x[:, 0]),
            torch.cos(x[:, 1]),
            x[:, 2],
            x[:, 3] ** 2,
            torch.tanh(x[:, 4]),
            x[:, 5] * x[:, 6],
            x[:, 6] - x[:, 7],
            x[:, 7],
        ],
        dim=1,
    )
    labels = ((x[:, 0] > 0).long() + 2 * (x[:, 2] > 0).long()) % 4
    return TensorDataset(x.contiguous(), labels.contiguous())


MODEL_VARIANTS = [
    {
        "name": "standard_shunting_softplus",
        "topk_type": "standard",
        "use_shunting": True,
        "weight_transform": "softplus",
        "efficient_blocklinear": False,
    },
    {
        "name": "standard_additive_identity",
        "topk_type": "standard",
        "use_shunting": False,
        "weight_transform": "identity",
        "efficient_blocklinear": False,
    },
    {
        "name": "indexed_dynamic_shunting",
        "topk_type": "indexed_dynamic",
        "use_shunting": True,
        "weight_transform": "softplus",
        "efficient_blocklinear": False,
        "indexed_candidate_size": 3,
        "indexed_seed": 31,
    },
    {
        "name": "standard_efficient_blocklinear",
        "topk_type": "standard",
        "use_shunting": True,
        "weight_transform": "softplus",
        "efficient_blocklinear": True,
    },
]


def _build_model(seed: int = 123, variant: Mapping[str, Any] | None = None) -> Classifier:
    _seed(seed)
    variant = dict(variant or MODEL_VARIANTS[0])
    transfer_params = {
        "independent_pathways": False,
        "output_activation": "softplus",
    }
    extra_kwargs = {
        key: variant[key]
        for key in ("indexed_candidate_size", "indexed_seed")
        if key in variant
    }
    core = ExcitationInhibitionNetwork(
        input_dim=8,
        transfer_params=transfer_params,
        excitatory_layer_sizes=[5, 4],
        inhibitory_layer_sizes=[4, 3],
        excitatory_branch_factors=[2, 1],
        inhibitory_branch_factors=[1, 1],
        ee_synapses_per_branch_per_layer=[3, 2],
        ei_synapses_per_branch_per_layer=[3, 2],
        ie_synapses_per_branch_per_layer=[2, 2],
        ii_synapses_per_branch_per_layer=[2, 1],
        reactivate=True,
        reactivation_init_m=1.2,
        reactivation_init_b=0.1,
        reactivation_init_policy="fixed",
        somatic_synapses=True,
        inhibitory_network_type="dendritic",
        use_shunting=bool(variant["use_shunting"]),
        topk_type=variant["topk_type"],
        topk_init_method="xavier_normal",
        weight_transform=variant["weight_transform"],
        use_noise=False,
        efficient_blocklinear=bool(variant["efficient_blocklinear"]),
        **extra_kwargs,
    )
    return Classifier(
        encoder_network=Identity(input_dim=8),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )


def _configure_information_params() -> InformationAnalysisParams:
    params = InformationAnalysisParams()
    params.method = "binned"
    params.n_bins = 4
    params.max_samples = 32
    params.components = ["basic", "pairwise", "conditional", "ablation_aligned"]
    params.scopes = []
    params.compute_basic_mi = True
    params.compute_pairwise_mi = True
    params.compute_conditional_mi = True
    params.compute_ablation_aligned_mi = True
    params.compute_layer_total_proxies = False
    params.compute_upstream_unique_cmi = False
    params.include_vinf = True
    params.normalize_mi = False
    params.computation_level = "single_branch"
    params.branch_aggregation = "mean"
    params.per_layer_analysis = False
    params.per_neuron_analysis = False
    params.per_einet_analysis = False
    if hasattr(params, "selection"):
        params.selection.components = list(params.components)
        params.selection.scopes = []
        params.selection.signal_variants = ["network"]
    if hasattr(params, "compute"):
        params.compute.max_samples = params.max_samples
        params.compute.include_vinf = params.include_vinf
        params.compute.normalize_mi = params.normalize_mi
        params.compute.computation_level = params.computation_level
        params.compute.branch_aggregation = params.branch_aggregation
        params.compute.per_layer_analysis = False
        params.compute.per_neuron_analysis = False
        params.compute.per_einet_analysis = False
        params.compute.compute_layer_total_proxies = False
        params.compute.compute_upstream_unique_cmi = False
    if hasattr(params, "estimator"):
        params.estimator.method = params.method
        params.estimator.n_bins = params.n_bins
    return params


def _configure_ablation_params() -> AblationAnalysisParams:
    params = AblationAnalysisParams()
    if hasattr(params, "selection"):
        params.selection.levels = ["layer"]
        params.selection.targets = ["all_synapses", "excitation", "inhibition"]
        params.selection.metrics = ["accuracy", "categorical_loglikelihood"]
    if hasattr(params, "methods"):
        params.methods.ablation_methods = ["lesion", "shuffle", "mean_clamp"]
        params.methods.shuffle_seed = 17
    # Legacy fallback attributes, harmless for grouped configs.
    params.layer_ablation = True
    params.compartment_ablation = False
    params.ablate_all_synapses = True
    params.ablate_excitation = True
    params.ablate_inhibition = True
    params.ablate_upstream = False
    params.accuracy = True
    params.categorical_ll = True
    return params


def _probe_forward_and_gradients(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset()
    x, y = dataset.tensors
    model = _build_model(variant=variant)
    model.train()
    logits = model(x[:16])
    loss = torch.nn.functional.cross_entropy(logits, y[:16])
    loss.backward()
    return {
        "logits": _normalize(logits),
        "loss": float(loss.item()),
        "state": _state_digest(model),
        "grads": _grad_digest(model),
    }


def _probe_training(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset()
    train_ds = TensorDataset(dataset.tensors[0][:48], dataset.tensors[1][:48])
    valid_ds = TensorDataset(dataset.tensors[0][48:], dataset.tensors[1][48:])
    model = _build_model(variant=variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(
        optimizer=optimizer,
        loss_function="cross_entropy",
        epochs=2,
        batch_size=16,
        suppress_prints=True,
        num_workers=0,
        use_amp=False,
    )
    _seed(456)
    results = trainer.train(model=model, train_data=train_ds, valid_data=valid_ds)
    return {
        "train_losses": _normalize(results.get("train_losses")),
        "valid_losses": _normalize(results.get("valid_losses")),
        "final_state": _state_digest(model),
    }


def _probe_performance_analysis(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset()
    model = _build_model(variant=variant)
    params = PerformanceAnalysisParams()
    params.accuracy = True
    params.categorical_loglikelihood = True
    analyzer = PerformanceAnalyzer(params)
    return analyzer.analyze(model=model, test_ds=dataset, device="cpu", save_path=None)


def _probe_ablation_analysis(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset(n_samples=32)
    model = _build_model(variant=variant)
    analyzer = AblationAnalyzer(_configure_ablation_params())
    return analyzer.analyze(model=model, test_dataset=dataset, device="cpu", save_path=None)


def _probe_activation_analysis(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset(n_samples=24)
    model = _build_model(variant=variant)
    branch_params = BranchActivationAnalysisParams()
    branch_params.downsample = True
    branch_params.samples = 12
    branch_params.raw_summary = True
    syn_params = SynapticActivationAnalysisParams()
    syn_params.downsample = True
    syn_params.samples = 12
    syn_params.raw_summary = True
    return {
        "branch": BranchActivationAnalyzer(branch_params).analyze(
            model=model,
            test_dataset=dataset,
            device="cpu",
            save_path=None,
        ),
        "synaptic": SynapticActivationAnalyzer(syn_params).analyze(
            model=model,
            test_dataset=dataset,
            device="cpu",
            save_path=None,
        ),
    }


def _probe_information_metrics() -> dict[str, Any]:
    analyzer = InformationAnalyzer(_configure_information_params())
    E = np.array([[0.0, 1.0], [0.1, 1.1], [1.0, 0.0], [1.1, 0.1], [0.2, 1.2], [1.2, 0.2]])
    I = np.array([[1.0], [1.1], [0.0], [0.1], [1.2], [0.2]])
    Vout = np.array([[0.0], [0.2], [1.0], [1.2], [0.1], [1.1]])
    C = np.array([0, 0, 1, 1, 0, 1])
    Vinf = np.array([[0.3], [0.4], [0.9], [1.0], [0.5], [1.1]])
    Vb = np.array([[0.8], [0.7], [0.2], [0.1], [0.6], [0.3]])
    return analyzer.compute_metrics(E, I, Vout, C, Vinf=Vinf, Vb=Vb)


def _probe_information_analysis(variant: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _make_dataset(n_samples=32)
    model = _build_model(variant=variant)
    analyzer = InformationAnalyzer(_configure_information_params())
    return analyzer.analyze(model=model, test_dataset=dataset, device="cpu", save_path=None)


def _probe_variant(variant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "forward_and_gradients": _probe_forward_and_gradients(variant),
        "training": _probe_training(variant),
        "performance_analysis": _probe_performance_analysis(variant),
        "ablation_analysis": _probe_ablation_analysis(variant),
        "activation_analysis": _probe_activation_analysis(variant),
        "information_analysis": _probe_information_analysis(variant),
    }


def main() -> None:
    _seed(0)
    results = {
        "feedforward_variants": {
            variant["name"]: _probe_variant(variant) for variant in MODEL_VARIANTS
        },
        "information_metrics": _probe_information_metrics(),
    }
    print(json.dumps(_normalize(results), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
"""


def _repo_src(repo: Path) -> Path:
    src = repo / "src"
    if not src.exists():
        raise FileNotFoundError(f"repository src directory not found: {src}")
    return src


def run_probe(repo: Path, python: str) -> dict[str, Any]:
    env = os.environ.copy()
    src = str(_repo_src(repo))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [python, "-c", PROBE_SOURCE],
        cwd=repo,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"probe failed for {repo} with exit {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"probe produced no stdout for {repo}")
    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse probe JSON for {repo}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        ) from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_state_digest(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {
        "tensors",
        "module_versions",
        "version_added",
    }


def _layout_only(key: str, side: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Whether ``side`` has ``key`` only because its module version added it."""
    added = side["version_added"].get(key)
    if added is None:
        return False
    module = key.rsplit(".", 1)[0] if "." in key else ""
    other_version = other["module_versions"].get(module)
    return other_version is None or int(other_version) < added


def _compare_state_digests(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    path: str,
    atol: float,
    rtol: float,
    failures: list[str],
    max_failures: int,
    layout_notes: list[str],
) -> None:
    """Compare shared tensors; keys a newer module version added are notes."""
    left_tensors, right_tensors = left["tensors"], right["tensors"]
    for key in sorted(set(left_tensors) ^ set(right_tensors)):
        side, other = (left, right) if key in left_tensors else (right, left)
        if _layout_only(key, side, other):
            layout_notes.append(f"{path}.{key}: added by a newer module version")
        else:
            failures.append(f"{path}: key {key!r} present on one side only")
    for key in sorted(set(left_tensors) & set(right_tensors)):
        _compare_values(
            left_tensors[key],
            right_tensors[key],
            path=f"{path}.{key}",
            atol=atol,
            rtol=rtol,
            failures=failures,
            max_failures=max_failures,
            layout_notes=layout_notes,
        )


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    failures: list[str],
    max_failures: int,
    layout_notes: list[str],
) -> None:
    if len(failures) >= max_failures:
        return
    if _is_state_digest(left) and _is_state_digest(right):
        _compare_state_digests(
            left,
            right,
            path=path,
            atol=atol,
            rtol=rtol,
            failures=failures,
            max_failures=max_failures,
            layout_notes=layout_notes,
        )
        return
    if _is_number(left) and _is_number(right):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return
        if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
            failures.append(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            failures.append(
                f"{path}: key mismatch left_only={sorted(left_keys - right_keys)!r} "
                f"right_only={sorted(right_keys - left_keys)!r}"
            )
            return
        for key in sorted(left_keys):
            _compare_values(
                left[key],
                right[key],
                path=f"{path}.{key}",
                atol=atol,
                rtol=rtol,
                failures=failures,
                max_failures=max_failures,
                layout_notes=layout_notes,
            )
        return
    if (
        isinstance(left, Sequence)
        and not isinstance(left, str | bytes)
        and isinstance(right, Sequence)
        and not isinstance(right, str | bytes)
    ):
        if len(left) != len(right):
            failures.append(f"{path}: length mismatch {len(left)} != {len(right)}")
            return
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            _compare_values(
                left_item,
                right_item,
                path=f"{path}[{idx}]",
                atol=atol,
                rtol=rtol,
                failures=failures,
                max_failures=max_failures,
                layout_notes=layout_notes,
            )
        return
    if left != right:
        failures.append(f"{path}: {left!r} != {right!r}")


def compare_payloads(
    current: dict[str, Any],
    develop: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    max_failures: int,
    layout_notes: list[str] | None = None,
) -> list[str]:
    """Return the numerical differences; ``layout_notes`` collects state keys
    that only a newer module version writes (not differences)."""
    failures: list[str] = []
    _compare_values(
        current,
        develop,
        path="$",
        atol=atol,
        rtol=rtol,
        failures=failures,
        max_failures=max_failures,
        layout_notes=layout_notes if layout_notes is not None else [],
    )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-path",
        type=Path,
        default=Path.cwd(),
        help="Path to the current branch checkout.",
    )
    parser.add_argument(
        "--develop-path",
        type=Path,
        required=True,
        help="Path to an origin/develop checkout or worktree.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable with project dependencies installed.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument(
        "--write-current-json",
        type=Path,
        default=None,
        help="Optional path to save the current-branch probe payload.",
    )
    parser.add_argument(
        "--write-develop-json",
        type=Path,
        default=None,
        help="Optional path to save the develop probe payload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_path = args.current_path.resolve()
    develop_path = args.develop_path.resolve()

    current = run_probe(current_path, args.python)
    develop = run_probe(develop_path, args.python)

    if args.write_current_json is not None:
        args.write_current_json.write_text(
            json.dumps(current, indent=2, sort_keys=True)
        )
    if args.write_develop_json is not None:
        args.write_develop_json.write_text(
            json.dumps(develop, indent=2, sort_keys=True)
        )

    layout_notes: list[str] = []
    failures = compare_payloads(
        current,
        develop,
        atol=args.atol,
        rtol=args.rtol,
        max_failures=args.max_failures,
        layout_notes=layout_notes,
    )
    if layout_notes:
        print(f"State keys written only by newer module versions: {len(layout_notes)}")
        for note in layout_notes[:20]:
            print(f"- {note}")
    if failures:
        print("Numerical parity FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Numerical parity passed")
    print(f"Compared current={current_path}")
    print(f"Against develop={develop_path}")
    print("Variants: " + ", ".join(current["feedforward_variants"].keys()))
    print(
        "Per-variant probes: "
        + ", ".join(
            [
                "forward_and_gradients",
                "training",
                "performance_analysis",
                "ablation_analysis",
                "activation_analysis",
                "information_analysis",
            ]
        )
    )
    print("Shared probes: information_metrics")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
