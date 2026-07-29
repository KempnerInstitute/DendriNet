"""Verify that the dendritic-RNN canary dispatches through CUDA gather.

Run this on a GPU node after sourcing ``setup_cuda_env.sh``.  It builds the
actual canary configuration, invokes every ``IndexedSparseLinear`` layer in
both fp32 and bf16, and counts calls to the production CUDA custom-autograd
wrapper.  A success verdict therefore proves that no Triton, fused-PyTorch,
or loop fallback was taken for any canary synapse shape.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cuda_gather_module,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    indexed_sparse as indexed_sparse_module,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dendritic_rnn/large_scale_500k_5k/canary.yaml",
        help="Dendritic RNN canary configuration to exercise.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for each direct indexed-synapse invocation.",
    )
    return parser.parse_args()


def _run_layer(
    layer: IndexedSparseLinear,
    *,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[tuple[int, int, int], bool, bool, bool]:
    layer.to(dtype=dtype)
    layer.zero_grad(set_to_none=True)
    x = torch.randn(
        batch_size,
        layer.in_features,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    out = layer(x)
    out.float().square().mean().backward()
    return (
        (layer.in_features, layer.out_features, layer.K),
        bool(torch.isfinite(out).all()),
        x.grad is not None and bool(torch.isfinite(x.grad).all()),
        layer.pre_w.grad is not None and bool(torch.isfinite(layer.pre_w.grad).all()),
    )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("verify_canary_cuda_gather.py requires a CUDA device")
    if not cuda_gather_module.cuda_gather_available():
        raise RuntimeError(
            "The CUDA gather extension did not compile. Source "
            "profilings/weight_transform_opt/setup_cuda_env.sh first."
        )

    config = load_config(args.config)
    model, _ = initialize_model(config.model)
    model = model.to("cuda")
    synapses = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, IndexedSparseLinear)
    ]
    if not synapses:
        raise RuntimeError("The canary did not construct any IndexedSparseLinear layers")

    disabled = [
        name
        for name, layer in synapses
        if not (
            layer.use_fused_gather
            and layer.use_triton_gather
            and layer.use_cuda_gather
        )
    ]
    if disabled:
        raise RuntimeError(f"Canary gather flags are not all enabled: {disabled}")

    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    original: Callable[..., torch.Tensor] = indexed_sparse_module.cuda_sparse_gather

    def traced_cuda_sparse_gather(
        x: torch.Tensor,
        connection_indices: torch.Tensor,
        weight: torch.Tensor,
        *call_args,
        **call_kwargs,
    ) -> torch.Tensor:
        calls.append(
            (tuple(x.shape), tuple(connection_indices.shape), tuple(weight.shape))
        )
        return original(
            x, connection_indices, weight, *call_args, **call_kwargs
        )

    indexed_sparse_module.cuda_sparse_gather = traced_cuda_sparse_gather
    try:
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"canary indexed synapses: {len(synapses)}"
        )
        for dtype in (torch.float32, torch.bfloat16):
            for name, layer in synapses:
                shape, finite_out, finite_x_grad, finite_weight_grad = _run_layer(
                    layer, dtype=dtype, batch_size=args.batch_size
                )
                if not (finite_out and finite_x_grad and finite_weight_grad):
                    raise RuntimeError(
                        f"Non-finite result from {name} at dtype={dtype}: "
                        f"out={finite_out}, x_grad={finite_x_grad}, "
                        f"weight_grad={finite_weight_grad}"
                    )
                print(
                    f"{dtype}: {name} "
                    f"in={shape[0]} out={shape[1]} K={shape[2]} OK"
                )
    finally:
        indexed_sparse_module.cuda_sparse_gather = original

    expected_calls = 2 * len(synapses)
    if len(calls) != expected_calls:
        raise RuntimeError(
            f"CUDA wrapper received {len(calls)} calls; expected {expected_calls}. "
            "A fallback path was selected."
        )
    if any(not cuda_gather_module._wpo_ok(shape[0][1], shape[1][1]) for shape in calls):
        raise RuntimeError("At least one canary shape did not select the WPO CUDA kernel")

    print(
        "VERDICT: CUDA WPO forward + split custom-autograd backward selected "
        f"for all {len(synapses)} canary synapses in fp32 and bf16 "
        f"({len(calls)} CUDA wrapper calls)."
    )


if __name__ == "__main__":
    main()
