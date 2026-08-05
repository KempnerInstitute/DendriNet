"""The chunked resource-record fast path must equal the dense snapshot exactly.

`synapse_resource_records` used to call `effective_synapse_snapshot` per pathway,
which materializes ~5 dense [out, in] matrices per indexed layer (~45 GB at the
stress config's 144,000 x 16,000) to produce five integers. The fast path
computes the same scatters in row chunks.

"Same scatters" is a claim about semantics, so it is tested on the cases where a
clever analytic shortcut would silently diverge from the dense path:

  * duplicate indices in a row (scatter_add collapses them; a naive O*K count
    does not)
  * forbidden entries (candidate-slot accounting)
  * a relu weight transform (zero-valued weights -- must NOT affect mask counts,
    which are index-arithmetic only)

Every numeric field must be exactly equal, not close.

    python check_resource_records_equiv.py
"""

from __future__ import annotations

import sys

import torch

from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.model_resources import _indexed_record_counts
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)


def legacy_counts(synapse):
    s = effective_synapse_snapshot(synapse)
    rk = s.realized_k.to(dtype=torch.float32)
    return {
        "candidate_slots": int(s.candidate_mask.ne(0).sum().item()),
        "active_synapses": int(s.active_mask.ne(0).sum().item()),
        "realized_k_min": int(rk.min().item()),
        "realized_k_max": int(rk.max().item()),
        "realized_k_mean": float(rk.mean().item()),
        "selection_policy": s.selection_policy,
        "mask_source": s.mask_source,
    }


def build(out_f, in_f, k, transform="softplus", seed=0):
    torch.manual_seed(seed)
    return IndexedSparseLinear(
        in_f, out_f, k, weight_transform=transform, seed=seed,
    )


def main():
    cases = []

    m1 = build(2000, 16000, 40)
    cases.append(("plain softplus 2000x16000 K=40", m1))

    m2 = build(300, 5000, 7, transform="relu", seed=3)
    cases.append(("relu transform (zero weights possible)", m2))

    # duplicates: overwrite the index buffer so several rows repeat an index --
    # scatter_add collapses these, so realized_k < K there.
    m3 = build(400, 3000, 8, seed=5)
    with torch.no_grad():
        idx = m3.connection_indices.clone()
        idx[:100, 1] = idx[:100, 0]
        idx[:50, 2] = idx[:50, 0]
        m3.connection_indices.copy_(idx)
    cases.append(("duplicate indices in 100 rows", m3))

    # forbidden entries: half the rows exclude one input column.
    m4 = build(500, 4000, 10, seed=7)
    forb = torch.full((500,), -1, dtype=torch.long)
    forb[::2] = torch.arange(0, 500, 2) % 4000
    m4._forbidden_input_index_per_output = forb
    cases.append(("forbidden entries on half the rows", m4))

    ok = True
    for name, mod in cases:
        fast = _indexed_record_counts(mod)
        if fast is None:
            print(f"  {name:<44} FAST PATH DECLINED (unexpected)")
            ok = False
            continue
        legacy = legacy_counts(mod)
        bad = {k: (legacy[k], fast[k]) for k in legacy if legacy[k] != fast[k]}
        ok &= not bad
        print(f"  {name:<44} {'EXACT MATCH' if not bad else 'MISMATCH ' + str(bad)}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
