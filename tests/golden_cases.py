"""Small CPU models whose state and outputs are recorded as cross-commit goldens.

``tests/test_default_path_goldens.py`` rebuilds every case and compares it with
``tests/fixtures/default_path_goldens.json``. Regenerate that file only with::

    PYTHONPATH=src python tests/golden_cases.py --write --reason "<why>"

which records the generating commit, the numerical stack and the reason in the
file's history. A case is built at library defaults apart from its sizes, in
float32 on one CPU thread with fixed seeds, in evaluation mode. Its record holds
a SHA-256 of every state tensor (names, dtypes, shapes and bytes), the outputs
for a fixed input and the gradient of their sum with respect to that input.

The cases run in a child process whose numerical libraries take fixed code
paths (:data:`PINNED_NUMERICS`), so the recorded bytes hold on every x86 CPU.
Otherwise MKL's AVX-512 kernels round a few initial weights differently from
the AVX2 kernels of CPUs without AVX-512, such as the pull-request runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "default_path_goldens.json"
#: Fixed code paths of the child process that runs the cases: Intel MKL in its
#: conditional numerical reproducibility mode, oneDNN and PyTorch's own CPU
#: kernels limited to AVX2, one thread.
PINNED_NUMERICS = {
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
}
SEED = 9461
INPUT_SEED = 17
INPUT_DIM = 8


def _einet_core(
    core_type: str, *, branch_factors=(2, 2), synapses: int = 3, **sections
) -> dict:
    core = {
        "type": core_type,
        "architecture": {
            "excitatory_layer_sizes": [4],
            "excitatory_branch_factors": list(branch_factors),
        },
        "connectivity": {"ee_synapses_per_branch_per_layer": [synapses]},
    }
    core.update(sections)
    return core


def _einet_two_layer_inhibitory() -> dict:
    """Two E layers whose branches receive inhibitory-population synapses."""
    return {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [4, 3],
            "inhibitory_layer_sizes": [2, 2],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3, 2],
            "ie_synapses_per_branch_per_layer": [2, 2],
            "ei_synapses_per_branch_per_layer": [2, 2],
            "ii_synapses_per_branch_per_layer": [0, 0],
        },
    }


def _population_core(
    use_shunting: bool, *, e_branch_factors=(2, 2), ff_excitatory_synapses: int = 3
) -> dict:
    return {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": ff_excitatory_synapses,
                        "ff_inhibitory_synapses": 2,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "use_shunting": use_shunting,
                    },
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": list(e_branch_factors),
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "i"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ]
        },
    }


def _recurrent_population_core() -> dict:
    """A recurrent rate-mode population of dendritic cells.

    Shunting E cells with per-level time constants, per-level feedforward and
    recurrent synapse counts and the mechanism-neutral initializer, recurrently
    coupled with an inhibitory population.
    """
    return {
        "type": "population_network",
        "population_network": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "recurrent": True,
                    "readout_population": "pyr",
                    "dt": 1.0,
                    "connection_seed": 31,
                    "population_defaults": {
                        "ff_excitatory_synapses": 0,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "use_shunting": True,
                        "weight_transform": "exp",
                        "dbl_init_method": "mechanism_neutral",
                        "adaptive_initialization": False,
                        "dynamics_mode": "rate",
                        "efficient_blocklinear": True,
                        "indexed_seed": 37,
                        "initialization_seed": 41,
                    },
                    "populations": [
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [2, 2],
                            "population": {
                                "ff_excitatory_synapses_by_level": [2, 2, 1],
                                "ff_inhibitory_synapses_by_level": [0, 0, 0],
                                "rec_excitatory_synapses_by_level": [2, 2, 2],
                                "rec_inhibitory_synapses_by_level": [1, 1, 1],
                                "level_taus": [20.0, 10.0, 4.0],
                            },
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [2],
                            "population": {
                                "ff_excitatory_synapses_by_level": [2, 1],
                                "ff_inhibitory_synapses_by_level": [0, 0],
                                "rec_excitatory_synapses_by_level": [2, 2],
                                "rec_inhibitory_synapses_by_level": [0, 0],
                                "level_taus": [8.0, 4.0],
                            },
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "pv"},
                        {"source": "pyr", "target": "pyr", "timing": "delayed"},
                        {"source": "pyr", "target": "pv", "timing": "delayed"},
                        {"source": "pv", "target": "pyr", "timing": "delayed"},
                    ],
                }
            ],
        },
    }


def _spatial_einet_config() -> dict:
    return {
        "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [0],
        },
    }


def _spatial_conv() -> torch.nn.Module:
    from dendritic_modeling.networks.architectures.factory import get_architecture

    return get_architecture(
        "hierarchical_dendritic_conv", _spatial_einet_config(), input_dim=3
    )


def _tensor_map() -> torch.nn.Module:
    from dendritic_modeling.networks.architectures.factory import get_architecture

    return get_architecture(
        "hierarchical_dendritic_tensor_map",
        _spatial_einet_config(),
        input_dim=12,
        suffix_input_dim=5,
    )


class _TwoStream(torch.nn.Module):
    """Feed one input to both the excitatory and inhibitory synapses."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, x)


def _typed_dendrinet() -> torch.nn.Module:
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
        DendriNet,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
        DendriNetConfig,
    )

    return _TwoStream(
        DendriNet(
            config=DendriNetConfig(
                n_soma=3,
                branch_factors=(2, 2),
                excitatory_input_dim=INPUT_DIM,
                excitatory_synapses_per_branch=3,
                inhibitory_input_dim=INPUT_DIM,
                inhibitory_synapses_per_branch=2,
            )
        )
    )


def _full_model(core: dict) -> torch.nn.Module:
    # A YAML file through load_config, the loader every library version has,
    # so the same case can be built at an older revision for comparison.
    from dendritic_modeling.config import load_config
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    mapping = {
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": INPUT_DIM}},
            "core": core,
            "decoder": {"params": {"output_dim": 3, "hidden_dims": [5]}},
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "case.yaml"
        path.write_text(yaml.safe_dump(mapping, sort_keys=False))
        config = load_config(str(path))
    model, _ = initialize_model(config.model)
    return model


def _unified_rnn() -> torch.nn.Module:
    from dendritic_modeling.networks.architectures.factory import get_architecture

    return get_architecture(
        "rnn_dendritic_shunting",
        {
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 6,
                            "branch_factors": [2],
                            "ff_excitatory_synapses": 3,
                            "rec_excitatory_synapses": 2,
                        },
                        "inhibitory": None,
                        "recurrent": True,
                        "dt": 1.0,
                    }
                ],
            }
        },
        input_dim=INPUT_DIM,
    )


def _affine_mlp() -> torch.nn.Module:
    from dendritic_modeling.config.affine_mlp import AffineMLPConfig
    from dendritic_modeling.networks.architectures.classical.mlp.affine_mlp import (
        build_affine_mlp,
    )

    return build_affine_mlp(
        AffineMLPConfig(dimensions=[INPUT_DIM, 7, 3], activate_output=False)
    )


def _indexed(backend: str) -> dict:
    return {
        "type": "indexed",
        "indexed": {"seed": 81, "projection_backend": backend},
    }


# name -> (builder, input shape)
CASES: dict[str, tuple[Callable[[], torch.nn.Module], tuple[int, ...]]] = {
    "einet_dendritic_shunting": (
        lambda: _full_model(_einet_core("dendritic_shunting")),
        (4, INPUT_DIM),
    ),
    "einet_dendritic_additive": (
        lambda: _full_model(_einet_core("dendritic_additive")),
        (4, INPUT_DIM),
    ),
    "einet_dendritic_normalized_additive": (
        lambda: _full_model(_einet_core("dendritic_normalized_additive")),
        (4, INPUT_DIM),
    ),
    "einet_dendritic_mlp": (
        lambda: _full_model(_einet_core("dendritic_mlp")),
        (4, INPUT_DIM),
    ),
    "einet_point_mlp": (
        lambda: _full_model(_einet_core("point_mlp")),
        (4, INPUT_DIM),
    ),
    "einet_flat_shunting": (
        lambda: _full_model(_einet_core("flat_shunting")),
        (4, INPUT_DIM),
    ),
    "einet_indexed_eager": (
        lambda: _full_model(
            _einet_core("dendritic_shunting", sparsity=_indexed("eager"))
        ),
        (4, INPUT_DIM),
    ),
    "einet_indexed_recompute": (
        lambda: _full_model(
            _einet_core("dendritic_shunting", sparsity=_indexed("recompute"))
        ),
        (4, INPUT_DIM),
    ),
    "population_network_shunting": (
        lambda: _full_model(_population_core(True)),
        (4, INPUT_DIM),
    ),
    "population_network_additive": (
        lambda: _full_model(_population_core(False)),
        (4, INPUT_DIM),
    ),
    "unified_rnn_dendritic_shunting": (_unified_rnn, (2, 4, INPUT_DIM)),
    "affine_mlp_control": (_affine_mlp, (4, INPUT_DIM)),
    # Morphologies: one wide level, three-way tree, and a branched tree at a
    # reduced synapse density.
    "einet_branched_4": (
        lambda: _full_model(_einet_core("dendritic_shunting", branch_factors=(4,))),
        (4, INPUT_DIM),
    ),
    "einet_branched_3_3": (
        lambda: _full_model(
            _einet_core("dendritic_shunting", branch_factors=(3, 3), synapses=2)
        ),
        (4, INPUT_DIM),
    ),
    "einet_branched_reduced_density": (
        lambda: _full_model(_einet_core("dendritic_shunting", synapses=1)),
        (4, INPUT_DIM),
    ),
    "einet_two_layer_inhibitory": (
        lambda: _full_model(_einet_two_layer_inhibitory()),
        (4, INPUT_DIM),
    ),
    # Every sparse-synapse family on the default shunting path (evaluation-mode
    # forward, so stochastic selections are deterministic).
    **{
        f"einet_sparsity_{kind}": (
            lambda kind=kind, options=options: _full_model(
                _einet_core("dendritic_shunting", sparsity={"type": kind, **options})
            ),
            (4, INPUT_DIM),
        )
        for kind, options in (
            ("stochastic", {}),
            ("variance", {}),
            ("dense_to_sparse", {}),
            ("indexed_rewire", {"indexed": {"seed": 83}}),
            ("indexed_release", {"indexed": {"seed": 89}}),
            ("indexed_dynamic", {"indexed": {"seed": 97}}),
        )
    },
    # DeepST draws its contacts from densities, so it needs the inhibitory
    # population for the analytical E/I initializer.
    "einet_sparsity_deepst": (
        lambda: _full_model(
            {
                **_einet_two_layer_inhibitory(),
                "sparsity": {
                    "type": "deepst",
                    "deepst": {
                        "excitatory_target_density": 0.5,
                        "inhibitory_target_density": 0.5,
                    },
                },
            }
        ),
        (4, INPUT_DIM),
    ),
    "population_network_branched_reduced_density": (
        lambda: _full_model(
            _population_core(True, e_branch_factors=(4,), ff_excitatory_synapses=2)
        ),
        (4, INPUT_DIM),
    ),
    "population_network_recurrent_rate": (
        lambda: _full_model(_recurrent_population_core()),
        (2, 5, INPUT_DIM),
    ),
    "spatial_dendritic_conv": (_spatial_conv, (2, 3, 5, 5)),
    "tensor_map": (_tensor_map, (2, 12)),
    "typed_dendrinet": (_typed_dendrinet, (4, INPUT_DIM)),
}


def state_sha256(module: torch.nn.Module) -> str:
    """Digest of every state tensor's name, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_case(name: str) -> dict:
    """Build case ``name`` and return its state digest, outputs and gradient."""
    builder, input_shape = CASES[name]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(SEED)
        model = builder().eval()
        generator = torch.Generator().manual_seed(INPUT_SEED)
        x = torch.rand(input_shape, generator=generator).requires_grad_(True)
        output = model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output.sum().backward()
        return {
            "state_sha256": state_sha256(model),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "output": output.detach().double().tolist(),
            "input_grad": x.grad.detach().double().tolist(),
        }
    finally:
        torch.set_num_threads(threads)


def pinned_records(names: list[str] | None = None) -> dict[str, dict]:
    """Records of ``names`` (all cases by default), run with :data:`PINNED_NUMERICS`."""
    command = [sys.executable, str(Path(__file__).resolve()), "--records"]
    result = subprocess.run(
        [*command, *(names or [])],
        capture_output=True,
        text=True,
        env={**os.environ, **PINNED_NUMERICS},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"golden cases failed in the child process:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _environment() -> dict:
    import dendritic_modeling

    root = Path(dendritic_modeling.__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "library_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pinned_numerics": PINNED_NUMERICS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="Rewrite the fixture")
    parser.add_argument("--reason", help="Why the goldens change (required to write)")
    parser.add_argument("--output", type=Path, help="Write the records here instead")
    parser.add_argument(
        "--records",
        nargs="*",
        metavar="CASE",
        help="Print the records of these cases (all by default) from this process; "
        "the other modes run them in a child process with PINNED_NUMERICS",
    )
    args = parser.parse_args(argv)
    if args.records is not None:
        print(json.dumps({name: run_case(name) for name in args.records or CASES}))
        return 0
    records = pinned_records()
    environment = _environment()
    if args.output is not None:
        args.output.write_text(
            json.dumps({"environment": environment, "cases": records}, indent=1)
        )
        return 0
    if not args.write:
        print(json.dumps({"environment": environment, "cases": records}, indent=1))
        return 0
    if not args.reason:
        parser.error("--write needs --reason")
    history = []
    if FIXTURE.is_file():
        previous = json.loads(FIXTURE.read_text())
        history = previous.get("history", [])
        changed = sorted(
            name
            for name in set(records) | set(previous.get("cases", {}))
            if records.get(name) != previous.get("cases", {}).get(name)
        )
    else:
        changed = sorted(records)
    history.append(
        {
            "date": date.today().isoformat(),
            **environment,
            "changed_cases": changed,
            "reason": args.reason,
        }
    )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps({"history": history, "cases": records}, indent=1) + "\n"
    )
    print(f"Wrote {FIXTURE} ({len(changed)} changed cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
