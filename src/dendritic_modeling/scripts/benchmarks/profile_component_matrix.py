"""Per-family performance matrix for dendritic components.

Different model families stress entirely different code paths — feedforward
einet (flat vs tree morphology), spatial convolution cores, recurrent
unified-EI cells (sequential step loops), population networks, and the
sparsity backends (indexed vs stochastic top-k) and weight transforms
(softplus/exp/identity) multiply the matrix. Optimization decisions (fused
kernels, CUDA graphs, compile modes) must therefore be ranked per family,
never generalized from one benchmark.

Each case builds a small representative model, times forward and
forward+backward passes (CUDA-synchronized medians), records peak memory,
and wraps every phase in an NVTX range (``case:<name>/fwd`` etc.) so an
``nsys`` session over this script yields a per-family timeline. Failing
cases are reported, not fatal.

Example::

    DENDRITIC_NVTX=1 nsys profile -o matrix python -m \
        dendritic_modeling.scripts.benchmarks.profile_component_matrix \
        --output matrix.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback

import torch

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils.logging_config import setup_logging
from dendritic_modeling.utils.nvtx import nvtx_range

BATCH = 64
DIM_IN = 512
DIM_OUT = 256
SYNAPSES = 64
SEQ_LEN = 32


def _einet_config(
    *,
    branch_factors: list[int],
    use_shunting: bool,
    weight_transform: str,
    sparsity_type: str = "indexed",
    reactivate: bool = True,
) -> dict:
    return {
        "type": "einet",
        "architecture": {
            "excitatory_layer_sizes": [DIM_OUT],
            "inhibitory_layer_sizes": [16] if use_shunting else [],
            "excitatory_branch_factors": branch_factors,
            "inhibitory_branch_factors": [1],
            "inhibitory_network_type": "dendritic",
            "mlp_inhibitory_network_params": {"hidden_dims": [], "activation": "relu"},
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [SYNAPSES],
            "ei_synapses_per_branch_per_layer": [8] if use_shunting else [0],
            "ie_synapses_per_branch_per_layer": [16 if use_shunting else SYNAPSES],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "independent_pathways": False,
            "excitatory_dim": None,
            "inhibitory_dim": None,
            "output_activation": None,
        },
        "morphology": {
            "somatic_synapses": False,
            "use_shunting": use_shunting,
            "weight_transform": weight_transform,
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "type": sparsity_type,
            "init_method": "xavier_normal",
            "indexed": {"seed": 7},
        },
        "reactivation": {
            "enabled": reactivate,
            "type": "param_tanh",
            "init_policy": "analytical",
            "init_m": 1.5,
            "init_b": 0.5,
        },
        "blocklinear": {"efficient": True},
        "implementation": {"adaptive_initialization": False, "print_hooks": False},
    }


def _conv_config() -> dict:
    config = _einet_config(
        branch_factors=[1], use_shunting=True, weight_transform="softplus"
    )
    config["type"] = "hierarchical_dendritic_conv"
    config["architecture"]["excitatory_layer_sizes"] = [128]
    config["architecture"]["inhibitory_layer_sizes"] = []
    config["connectivity"]["ei_synapses_per_branch_per_layer"] = [0]
    config["spatial"] = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_transform": "identity",
        "output_adapter_mode": "identity",
    }
    return config


def _unified_ei_config() -> dict:
    return {
        "unified_ei": {
            "n_excitatory": DIM_OUT,
            "n_inhibitory": 32,
            "output_mode": "last",
            "dt": 1.0,
            "learn_taus": True,
            "i_tau": 7.0,
            "layers": [
                {
                    "tau": 15.0,
                    "ff_synapses": SYNAPSES,
                    "rec_synapses": 16,
                    "ie_synapses": 8,
                    "compartment": "topk",
                    "use_shunting": True,
                    "reactivate": True,
                    "reactivation_type": "param_tanh",
                    "weight_transform": "softplus",
                }
            ],
        }
    }


def _population_config() -> dict:
    defaults = {
        "ff_excitatory_synapses": SYNAPSES,
        "ff_inhibitory_synapses": 8,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "reactivate": True,
        "dbl_init_method": "default",
        "weight_transform": "exp",
    }
    return {
        "population_network": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": defaults,
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 32,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": DIM_OUT,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ],
        }
    }


def _cases() -> dict:
    """name -> (architecture type, config, example-input factory, input_dim)."""
    flat = lambda: torch.rand(BATCH, DIM_IN)  # noqa: E731
    seq = lambda: torch.rand(BATCH, SEQ_LEN, DIM_IN)  # noqa: E731
    image = lambda: torch.rand(BATCH, 64, 14, 14)  # noqa: E731
    return {
        "ff_flat_shunting_softplus_indexed": (
            "einet",
            _einet_config(
                branch_factors=[1], use_shunting=True, weight_transform="softplus"
            ),
            flat,
            DIM_IN,
        ),
        "ff_flat_shunting_exp_indexed": (
            "einet",
            _einet_config(
                branch_factors=[1], use_shunting=True, weight_transform="exp"
            ),
            flat,
            DIM_IN,
        ),
        "ff_flat_additive_signed_indexed": (
            "einet",
            _einet_config(
                branch_factors=[1],
                use_shunting=False,
                weight_transform="identity",
                reactivate=False,
            ),
            flat,
            DIM_IN,
        ),
        "ff_tree42_shunting_softplus_indexed": (
            "einet",
            _einet_config(
                branch_factors=[4, 2], use_shunting=True, weight_transform="softplus"
            ),
            flat,
            DIM_IN,
        ),
        "ff_flat_shunting_softplus_topk": (
            "einet",
            _einet_config(
                branch_factors=[1],
                use_shunting=True,
                weight_transform="softplus",
                sparsity_type="standard",
            ),
            flat,
            DIM_IN,
        ),
        "conv_shunting_softplus_indexed": (
            "hierarchical_dendritic_conv",
            _conv_config(),
            image,
            64,
        ),
        "rnn_unified_ei_shunting_softplus": (
            "unified_ei",
            _unified_ei_config(),
            seq,
            DIM_IN,
        ),
        "population_ei_exp": (
            "population_network",
            _population_config(),
            flat,
            DIM_IN,
        ),
    }


def _time_case(model, example, device, warmup, iterations) -> dict:
    model = model.to(device)
    example = example.to(device)

    def _sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    timings: dict[str, float] = {}
    for phase in ("fwd", "fwd_bwd"):
        durations = []
        for i in range(warmup + iterations):
            _sync()
            start = time.perf_counter()
            with nvtx_range(f"{phase}:{i}"):
                if phase == "fwd":
                    with torch.no_grad():
                        model(example)
                else:
                    model.zero_grad(set_to_none=True)
                    model(example).sum().backward()
            _sync()
            if i >= warmup:
                durations.append(time.perf_counter() - start)
        timings[f"{phase}_ms_median"] = 1e3 * sorted(durations)[len(durations) // 2]
    if device.startswith("cuda"):
        timings["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
    timings["params"] = sum(p.numel() for p in model.parameters())
    return timings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--only", default=None, help="Comma-separated case filter")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    selected = set(args.only.split(",")) if args.only else None
    record: dict[str, dict] = {}
    for name, (core_type, config, example_fn, input_dim) in _cases().items():
        if selected is not None and name not in selected:
            continue
        torch.manual_seed(0)
        try:
            with nvtx_range(f"case:{name}"):
                model = get_architecture(
                    core_type, copy.deepcopy(config), input_dim=input_dim
                )
                result = _time_case(
                    model, example_fn(), args.device, args.warmup, args.iterations
                )
            record[name] = {"status": "ok", **result}
        except Exception as error:
            record[name] = {
                "status": f"failed: {type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=4),
            }
        summary = record[name]
        print(
            f"{name:<42} "
            + (
                f"fwd {summary['fwd_ms_median']:.2f}ms  "
                f"fwd+bwd {summary['fwd_bwd_ms_median']:.2f}ms  "
                f"params {summary['params'] / 1e3:.0f}k"
                if summary["status"] == "ok"
                else summary["status"]
            ),
            flush=True,
        )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(
                {"device": args.device, "batch": BATCH, "cases": record},
                handle,
                indent=1,
            )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    setup_logging()
    main()
