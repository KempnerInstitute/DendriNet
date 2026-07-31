"""Is the staged forward's 1 block/SM a hardware limit, or a near-miss?

The original wpo experiment (profilings/weight_transform_opt/
cuda_gather_fwd_stagex_wpo.py) was designed and validated at IN=10,000:

    "Shared = just the x-row = IN*4 = 40KB (< 48KB default -> no opt-in, and
     ~2 blocks/SM). Fat blocks (T=512) -> ~66% occupancy target"

Two blocks x 16 warps = 32 warps = 66% -- exactly what it claimed. At the stress
config's IN=16,000 the row is 64 KB, only one block fits, and occupancy halves to
33%. ncu confirms it. So the kernel is not broken; its design point moved.

That raises a cheaper question than writing a new kernel: is 2 blocks/SM at 64 KB
actually unreachable, or is it a near-miss against the per-SM shared budget that a
carveout setting could recover? This queries the real limits and computes, per
in_features, how many blocks the hardware can host.

    python probe_shared_budget.py
"""

from __future__ import annotations

import sys

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)


def main():
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    p = torch.cuda.get_device_properties(0)
    optin = cg._optin_shared() if cg.cuda_gather_available() else 0

    per_block_default = getattr(p, "shared_memory_per_block", None)
    per_sm = getattr(p, "shared_memory_per_multiprocessor", None)
    max_threads_sm = getattr(p, "max_threads_per_multi_processor", None)

    print(f"GPU {p.name}  ({p.major}.{p.minor})  SMs={p.multi_processor_count}")
    print(f"  shared / block (default)      : {per_block_default:,} B"
          if per_block_default else "  shared / block: ?")
    print(f"  shared / block (opt-in max)   : {optin:,} B")
    print(f"  shared / SM (total)           : {per_sm:,} B" if per_sm
          else "  shared / SM: ?")
    print(f"  max threads / SM              : {max_threads_sm:,}"
          if max_threads_sm else "")
    warps_sm = (max_threads_sm // 32) if max_threads_sm else None
    if warps_sm:
        print(f"  max warps / SM                : {warps_sm}")

    print(f"\n  T=512 -> 16 warps per block. Occupancy = 16*blocks/{warps_sm}.\n")
    print(f"{'in_features':>12}{'row bytes':>12}{'blocks by shared':>18}"
          f"{'occupancy':>12}")
    for IN in (10000, 12000, 16000, 16384, 20000, 24000, 25344):
        row = IN * 4
        if not per_sm:
            continue
        blocks = per_sm // row if row else 0
        # a block also cannot exceed the per-block opt-in cap
        runnable = row <= optin
        occ = (16 * blocks / warps_sm * 100) if (warps_sm and blocks) else 0
        note = "" if runnable else "   (exceeds per-block cap)"
        print(f"{IN:>12,}{row:>12,}{blocks:>18}{occ:>11.0f}%{note}")

    if per_sm:
        print(f"\n  The design point IN=10,000 gives {per_sm // 40000} block(s)/SM; "
              f"IN=16,000 gives {per_sm // 64000}.")
        half = per_sm // 2
        print(f"  For 2 blocks/SM a block must stay under {half:,} B "
              f"= in_features {half // 4:,}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
