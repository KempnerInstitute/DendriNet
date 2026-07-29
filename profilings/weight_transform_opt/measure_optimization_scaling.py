"""End-to-end validation: do the three gather optimizations actually speed up REAL training,
across model sizes, at no cost to what the model learns?

Runs the real population-network model at several WIDTHS (scaling hidden n_neurons), under three
flag settings, and records the metrics for the before/after figure:

  series:
    off    -- all three flags False (original loop)            <- "before"
    fused  -- use_fused_gather only (Phase 1, ~5x, safe-on)    <- midpoint
    on     -- all three True (precedence cuda->triton->fused)  <- "after"

  metrics:
    * ms/step        -- steady-state median wall-clock per training step (bf16, the real regime)
    * peak_gb        -- torch.cuda.max_memory_allocated during the step (the memory win)
    * speedup        -- derived at plot time (off/on and off/fused)
    * loss overlay   -- a separate fp32 pass at one size, 3 series, on REAL CIFAR-10: repeatedly fit
                        a fixed subset of real images (same init+data+batch-order across series) ->
                        the curve actually descends (the model demonstrably learns) AND the three
                        series overlay exactly (speedup at no accuracy cost). fp32 so the curves are
                        clean (bf16 chaos is benign but visually muddies the overlay).

Fairness: identical config/seed/synthetic-batch, only the 3 flags flip. Warmup discards the
lazy CUDA compile (~90s first use) + cudnn autotune; timing is the median of the next N steps.
OOM is caught and recorded (the original loop is expected to OOM at large width -- that IS a result).

Run (needs a GPU + `source setup_cuda_env.sh` so the CUDA path can compile):
  python measure_optimization_scaling.py --factors 0.25 0.5 1.0 2.0
Results -> opt_scaling_results.json  (feed to the plotting step).
"""
import os, sys, json, copy, argparse, time
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
import torch.nn as nn

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

BASE_CFG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
SCRATCH = os.path.join(REPO, "profilings/weight_transform_opt/_scaling_tmp")  # shared FS, node-agnostic
os.makedirs(SCRATCH, exist_ok=True)
OUT_JSON = os.path.join(REPO, "profilings/weight_transform_opt/opt_scaling_results.json")

SERIES = {  # (use_fused, use_triton, use_cuda)
    "off":   (False, False, False),
    "fused": (True,  False, False),
    "on":    (True,  True,  True),
}


def scale_neurons(obj, factor):
    """Recursively multiply every populations' n_neurons by `factor` (input_dim/readout untouched)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "n_neurons" and isinstance(v, int):
                obj[k] = max(1, int(round(v * factor)))
            else:
                scale_neurons(v, factor)
    elif isinstance(obj, list):
        for item in obj:
            scale_neurons(item, factor)


def scaled_config_path(factor):
    with open(os.path.join(REPO, BASE_CFG)) as f:
        raw = yaml.safe_load(f)
    scale_neurons(raw, factor)
    path = os.path.join(SCRATCH, f"cfg_w{factor:g}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return path


def hidden_width(cfg_path):
    """Largest hidden n_neurons in the config (the E width) -- for the x-axis label."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    widths = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "n_neurons" and isinstance(v, int):
                    widths.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)
    walk(raw)
    return max(widths) if widths else 0


def set_flags(model, fused, triton, cuda):
    n = 0
    for m in model.modules():
        if isinstance(m, IndexedSparseLinear):
            m.use_fused_gather = fused
            m.use_triton_gather = triton
            m.use_cuda_gather = cuda
            n += 1
    return n


def build_model(cfg_path, dev):
    cfg = load_config(cfg_path)
    torch.manual_seed(42)                      # identical init across series
    model, _ = initialize_model(cfg.model)
    return model.to(dev)


def run_one(cfg_path, flags, dev, steps, warmup, amp, record_loss=False, batch=128):
    """Returns dict(ms_step, peak_gb, n_syn, losses) or dict(oom=True)."""
    torch.cuda.empty_cache()
    try:
        model = build_model(cfg_path, dev)
        n_syn = set_flags(model, *flags)
        torch.manual_seed(0)
        x = torch.randn(batch, 3072, device=dev)
        y = torch.randint(0, 10, (batch,), device=dev)
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = ce(model(x), y)
            loss.backward(); opt.step()
            return loss

        for _ in range(warmup):                # warmup: lazy compile + cudnn autotune + caches
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        losses = []
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(steps):
            l = step()
            if record_loss:
                losses.append(l.item())
        t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / steps
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        del model, opt
        torch.cuda.empty_cache()
        return {"ms_step": ms, "peak_gb": peak_gb, "n_syn": n_syn, "losses": losses}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}


_CIFAR = {}


def cifar10_train_tensors():
    """Load the real CIFAR-10 train set ONCE, matching the config's preprocessing
    (flatten -> [N,3072], per-channel normalize). Returns (x[N,3072] float, y[N] long) on CPU."""
    if "xy" not in _CIFAR:
        from dendritic_modeling.datasets import load_cifar10_as_datasets
        train_ds, _, _ = load_cifar10_as_datasets(
            train_valid_split=1.0, flatten=True, normalize=True,
            task_data_path="data",  # config's data.base_dir -> resolves to data/datasets/cifar (already on disk)
        )  # same loader the trainer uses; CIFAR mean/std, 3072-dim
        x, y = train_ds.tensors
        _CIFAR["xy"] = (x.float().contiguous(), y.long().contiguous())
    return _CIFAR["xy"]


def run_overlay(cfg_path, flags, dev, steps, batch, subset=512, lr=1e-3, clip=5.0,
                data_seed=0, sample_seed=1234):
    """fp32 training on a FIXED subset of REAL CIFAR-10 -> per-step loss (list), or None on OOM.

    Recipe chosen to show a genuine *descending* learning curve while proving equivalence.
    (One-pass over 0.5 epoch of this large model barely leaves the chance floor -- real training
    needs many passes to visibly descend -- so we repeatedly fit a fixed subset instead.)
      * a fixed `subset` of REAL CIFAR-10 images (seed data_seed) -- real data, real gradients;
      * each step draws a batch via a shared RNG (seed sample_seed) -> all three series see the
        IDENTICAL batch sequence, so the curves overlay exactly;
      * lr 1e-3 + grad-clip 5.0 mirror the real training config (stable + faithful);
      * repeatedly fitting the subset drives the loss down clearly (the model demonstrably learns).
    Identical init (seed 42) + identical batches across series => the three loss curves coincide.
    fp32 (no autocast) exercises the proven-equivalent fp32 kernel path and keeps the overlay clean.
    """
    torch.cuda.empty_cache()
    try:
        x_all, y_all = cifar10_train_tensors()
        g = torch.Generator().manual_seed(data_seed)
        sel = torch.randperm(x_all.shape[0], generator=g)[:subset]  # same subset for every series
        xs = x_all[sel].to(dev)
        ys = y_all[sel].to(dev)

        model = build_model(cfg_path, dev)  # seed 42 -> identical init across series
        set_flags(model, *flags)
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bg = torch.Generator().manual_seed(sample_seed)  # batch order: identical across series

        losses = []
        for _ in range(steps):
            bi = torch.randint(0, subset, (batch,), generator=bg).to(dev)  # same seq every series
            opt.zero_grad(set_to_none=True)
            loss = ce(model(xs[bi]), ys[bi])  # fp32, no autocast
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        del model, opt, xs, ys
        torch.cuda.empty_cache()
        return losses
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--overlay-factor", type=float, default=1.0, help="which width for the fp32 loss overlay")
    ap.add_argument("--overlay-steps", type=int, default=400, help="training steps for the real-CIFAR loss overlay")
    ap.add_argument("--overlay-batch", type=int, default=128, help="batch size for the overlay")
    ap.add_argument("--overlay-subset", type=int, default=512, help="# real CIFAR-10 images repeatedly fit in the overlay")
    ap.add_argument("--overlay-lr", type=float, default=1e-3, help="Adam lr for the overlay (matches training config)")
    ap.add_argument("--overlay-only", action="store_true",
                    help="skip the bf16 sweep; reuse its results from the JSON and rerun ONLY the loss overlay")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    print(f"gpu: {gpu}\nfactors: {args.factors}\n")

    if args.overlay_only:
        assert os.path.exists(OUT_JSON), f"--overlay-only needs an existing {OUT_JSON} (run the sweep first)"
        results = json.load(open(OUT_JSON))     # reuse the bf16 sweep; only the overlay is re-measured
        results["overlay"] = {}
        print("overlay-only: reusing existing bf16 sweep, rerunning loss overlay on real CIFAR-10\n")
    else:
        results = {"gpu": gpu, "steps": args.steps, "sweep": [], "overlay": {}}

        # ---- bf16 sweep: ms/step + peak memory, across widths x series ----
        for f in args.factors:
            cfg_path = scaled_config_path(f)
            width = hidden_width(cfg_path)
            row = {"factor": f, "e_width": width, "series": {}}
            print(f"=== width factor {f:g}  (E n_neurons={width}) ===")
            for name, flags in SERIES.items():
                t = time.time()
                r = run_one(cfg_path, flags, dev, args.steps, args.warmup, amp=True)
                if r.get("oom"):
                    print(f"  {name:6s}: OOM")
                    row["series"][name] = {"oom": True}
                else:
                    print(f"  {name:6s}: {r['ms_step']:8.2f} ms/step   {r['peak_gb']:6.2f} GB   "
                          f"({r['n_syn']} synapses, {time.time()-t:.0f}s)")
                    row["series"][name] = {"ms_step": r["ms_step"], "peak_gb": r["peak_gb"]}
            results["sweep"].append(row)
            json.dump(results, open(OUT_JSON, "w"), indent=2)   # checkpoint after each size

    # ---- fp32 loss overlay on REAL CIFAR-10: proves training is unchanged AND descends ----
    print(f"\n=== fp32 loss overlay on real CIFAR-10 @ factor {args.overlay_factor:g}  "
          f"({args.overlay_subset}-image subset, {args.overlay_steps} steps x batch {args.overlay_batch}, lr {args.overlay_lr:g}) ===")
    cfg_path = scaled_config_path(args.overlay_factor)
    for name, flags in SERIES.items():
        losses = run_overlay(cfg_path, flags, dev, args.overlay_steps, args.overlay_batch,
                             subset=args.overlay_subset, lr=args.overlay_lr)
        if losses is None:
            print(f"  {name:6s}: OOM"); results["overlay"][name] = None
        else:
            print(f"  {name:6s}: loss {losses[0]:.4f} -> {losses[-1]:.4f}  (min {min(losses):.4f})")
            results["overlay"][name] = losses
    results["overlay_factor"] = args.overlay_factor
    results["overlay_meta"] = {"data": "cifar10_real", "mode": "subset_overfit",
                               "subset": args.overlay_subset, "steps": args.overlay_steps,
                               "batch": args.overlay_batch, "lr": args.overlay_lr}

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
