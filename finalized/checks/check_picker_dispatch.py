"""Assert the forward-gather picker dispatches the MEASURED winner set.

The picker (_pick_fwd_variant) is a lookup table fitted from the crossover /
hybrid sweeps; this check replays the fitted grid and fails if the code's
decisions drift from the measured winners recorded here. One expected set per
architecture, added as each arch's edge sweep lands (logs/edges_*).

    python check_picker_dispatch.py                       # on the live GPU
    python check_picker_dispatch.py --cc 12,0 --optin 101376   # logic-only

Simulate mode patches the capability/optin queries so the dispatch logic can
be regression-checked with no GPU; a hardware run on the real arch is still
the stronger statement (H100 branch was verified live on holygpu8a11301).
"""

from __future__ import annotations

import argparse
import sys

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

INS = (3072, 8000, 16000, 24000)
OUTS = (2000, 48000, 144000)
KS = (10, 15, 20, 25, 30, 40, 50, 60, 70, 80)


def expected_sm90():
    """From logs/edges_h100 (optin 232,448 B): no-stage wins ONLY at blk/SM=2
    (the IN=24k row): the whole row at K<=15, the 144k cell alone at K=20..60.
    Hybrid loses everywhere (0.54-0.94x): no band."""
    exp = {}
    for K in (10, 15):
        for O in OUTS:
            exp[(24000, O, K)] = "nostage"
    for K in (20, 25, 30, 40, 50, 60):
        exp[(24000, 144000, K)] = "nostage"
    return exp


def expected_sm120():
    """From logs/edges_blackwell (optin 101,376 B), 1-blk/SM rows IN=16k/24k:
    hybrid at K<=20 everywhere and at (IN<=20k, O<=96k) through K=60;
    no-stage at (IN<=20k, O>=96k) for ALL K>20, and at (IN>20k, O>=96k)
    only through K=40 (dies at 50: 0.89x)."""
    exp = {}
    for IN in (16000, 24000):
        for K in (10, 15, 20):
            for O in OUTS:
                exp[(IN, O, K)] = "hybrid"
    for K in (25, 30, 40, 50, 60):
        exp[(16000, 2000, K)] = "hybrid"
        if K <= 50:  # (48k, K60) trimmed: 0.94x at shipped config
            exp[(16000, 48000, K)] = "hybrid"
    for K in (25, 30, 40, 50, 60, 70, 80):
        exp[(16000, 144000, K)] = "nostage"
    for K in (25, 30, 40):
        exp[(24000, 144000, K)] = "nostage"
    return exp


def expected_sm80():
    """From logs/edges_a100 (80GB) + the 40GB variant verify. 1 blk/SM (IN=24k):
    every O for K<=40; only O>=40k for K>40 (small-O flips, 0.58-0.94x).
    2 blk/SM (IN=16k): trimmed to cells that win on BOTH sm80 variants --
    whole row K<=15; (48k, K20); (144k, K<=40)."""
    exp = {}
    for K in (10, 15, 20, 25, 30, 40):
        for O in OUTS:
            exp[(24000, O, K)] = "nostage"
    for K in (50, 60, 70, 80):
        for O in (48000, 144000):
            exp[(24000, O, K)] = "nostage"
    for K in (10, 15):
        for O in OUTS:
            exp[(16000, O, K)] = "nostage"
    exp[(16000, 48000, 20)] = "nostage"
    for K in (20, 25, 30, 40):  # K>=50 at 144k + (48k,K25) trimmed: lose on 40GB variant
        exp[(16000, 144000, K)] = "nostage"
    return exp


EXPECTED = {(8, 0): expected_sm80, (9, 0): expected_sm90, (12, 0): expected_sm120}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", type=str, default="",
                    help="simulate this capability, e.g. 12,0 (no GPU needed)")
    ap.add_argument("--optin", type=int, default=0,
                    help="simulate this opt-in shared budget in bytes")
    a = ap.parse_args()

    if a.cc:
        cc = tuple(int(x) for x in a.cc.split(","))
        optin = a.optin or {8: 167_936, 9: 232_448, 12: 101_376}.get(cc[0], 101_376)
        torch.cuda.get_device_capability = lambda *args, **kw: cc
        cg._optin_shared = lambda: optin
        cg._pick_wpo_params = lambda *args, **kw: (2048, 512)
        print(f"SIMULATED cc={cc} optin={optin:,}")
    else:
        if not torch.cuda.is_available():
            print("needs CUDA (or use --cc/--optin to simulate)")
            return 1
        cc = torch.cuda.get_device_capability()
        print(torch.cuda.get_device_name(0), cc)

    got = {}
    for IN in INS:
        for K in KS:
            for O in OUTS:
                v, P, T = cg._pick_fwd_variant(IN, O, K)
                if v != "staged":
                    got[(IN, O, K)] = v

    if cc not in EXPECTED:
        print(f"no expected set recorded for cc={cc}; dispatch table only:")
        for cell, v in sorted(got.items()):
            print("  ", cell, "->", v)
        return 0

    exp = EXPECTED[cc]()
    extra = {c: v for c, v in got.items() if c not in exp}
    missing = {c: v for c, v in exp.items() if c not in got}
    wrong = {c: (got[c], exp[c]) for c in got if c in exp and got[c] != exp[c]}
    print("extra (dispatched non-staged, not a measured winner):", sorted(extra.items()))
    print("missing (measured winner, still staged):", sorted(missing.items()))
    print("wrong variant:", sorted(wrong.items()))
    if extra or missing or wrong:
        print("MISMATCH")
        return 1
    print(f"MATCH: all {len(exp)} measured winner cells dispatch correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
