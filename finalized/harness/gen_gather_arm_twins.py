"""Generate gather-arm twins of a LEARNING config for the level-overlay figure.

Takes one real training config and emits four twins -- loop / fused / triton /
cuda -- that are byte-identical except for the three ``indexed_use_*_gather``
flags (set in every ``population_defaults`` block). Optional overrides keep the
run honest and bounded:

  --epochs N        cap training epochs (the figure needs a visible trajectory,
                    not convergence)
  --widen E,I       widen every excitatory/inhibitory population to E/I neurons
                    (the example configs are tiny; kernels matter at width)
  Rewiring is frozen (``indexed_rewire_until_step: 0``) so the topology is
  identical across arms for the whole run -- the same pin the profiling
  harness applies. Declared in the twin header.

    python gen_gather_arm_twins.py <base.yaml> <outdir> [--epochs N] [--widen E,I]
"""

from __future__ import annotations

import argparse
import copy
import os

import yaml

ALL_FLAGS = (
    "indexed_use_fused_gather",
    "indexed_use_triton_gather",
    "indexed_use_cuda_gather",
)
ARM_FLAG = {
    "loop": None,
    "fused": "indexed_use_fused_gather",
    "triton": "indexed_use_triton_gather",
    "cuda": "indexed_use_cuda_gather",
}


def walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            walk(v, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("outdir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--widen", type=str, default=None, help="E,I neuron counts")
    ap.add_argument("--data-base-dir", default=None,
                    help="pin data.base_dir (else each run's output_dir becomes "
                         "the dataset root and re-downloads)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    base = yaml.safe_load(open(a.base))
    widen = tuple(int(x) for x in a.widen.split(",")) if a.widen else None

    for arm, flag in ARM_FLAG.items():
        cfg = copy.deepcopy(base)
        if a.data_base_dir:
            cfg.setdefault("data", {})["base_dir"] = a.data_base_dir
        # Post-training analysis materializes the full dataset for scoring
        # (~83 GB spike): it would dominate the peak-memory measurement and pad
        # wall-clock, so the overlay twins skip it. Loss trajectories come from
        # training_summary.json regardless.
        perf = cfg.get("analysis", {}).get("performance")
        if isinstance(perf, dict):
            perf["enabled"] = False

        def set_arm(d):
            if "population_defaults" in d and isinstance(d["population_defaults"], dict):
                pd = d["population_defaults"]
                for k in ALL_FLAGS:
                    pd[k] = False
                if flag:
                    pd[flag] = True
                pd["indexed_rewire_until_step"] = 0  # frozen topology, all arms

        def set_widen(d):
            if widen and "n_neurons" in d and d.get("polarity") in (
                    "excitatory", "inhibitory"):
                d["n_neurons"] = widen[0] if d["polarity"] == "excitatory" else widen[1]

        def set_epochs(d):
            if a.epochs is not None and "batch_size" in d and "epochs" in d:
                d["epochs"] = a.epochs

        def set_names(d):
            if "run_name" in d:
                d["run_name"] = f"gather_{arm}"
            if "results_dir" in d:
                d["results_dir"] = os.path.join(a.outdir, "results", arm)

        for fn in (set_arm, set_widen, set_epochs, set_names):
            walk(cfg, fn)

        out = os.path.join(a.outdir, f"{arm}.yaml")
        with open(out, "w") as f:
            f.write(f"# gather-arm twin [{arm}] of {os.path.basename(a.base)}"
                    f" -- identical except gather flags"
                    f"{'; widened ' + a.widen if a.widen else ''}"
                    f"{'; epochs ' + str(a.epochs) if a.epochs else ''};"
                    f" rewiring frozen (until_step 0) in every arm\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        print("wrote", out)


if __name__ == "__main__":
    main()
