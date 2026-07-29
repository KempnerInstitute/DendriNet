"""Real-trainer loss overlay: drive the EXACT 10k-E profiling model through its actual training
pipeline (CustomWeightDecayOptimizer -> weight_boosting/decay_weights + indexed_rewire + per-group
LRs, exactly what a plain Adam loop lacks) on REAL CIFAR-10, fp32, for the three gather-flag
settings, recording per-batch training loss. Identical init (seed) + fixed batch order (shuffle
off) across series -> the three curves should coincide while the model genuinely learns.

  off   = (fused, triton, cuda) all False  -- original loop
  fused = fused only
  on    = all three (precedence cuda->triton->fused)

Run on a GPU node after `source setup_cuda_env.sh`:
  # quick: does it descend on the fast path?
  python train_overlay_real.py --series on --epochs 2 --batch 256
  # full overlay:
  python train_overlay_real.py --series off,fused,on --epochs 3 --batch 256 --out overlay_real.json
"""
import os, sys, json, argparse
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
from torch.utils.data import Subset
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.datasets import load_cifar10_as_datasets, load_mnist_as_datasets
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config, prepare_trainer_config)
from dendritic_modeling.scripts.training.train_experiments import _resolve_main_training_settings
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear)

CFG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"  # default; --config overrides
SERIES = {"off": (False, False, False), "fused": (True, False, False), "on": (True, True, True)}
DEV = "cuda"

_DATA = {}
def datasets():
    """Load the config's OWN dataset (mnist or cifar10) with its declared preprocessing."""
    if "d" not in _DATA:
        raw = yaml.safe_load(open(CFG))
        dd = raw.get("data", {}); proc = dd.get("processing", {})
        name = dd.get("dataset_name")
        flat = bool(proc.get("flatten", True)); norm = bool(proc.get("normalize", True))
        tvs = float(raw.get("experiment", {}).get("train_valid_split", 0.8))
        loader = {"mnist": load_mnist_as_datasets, "cifar10": load_cifar10_as_datasets}.get(name)
        if loader is None:
            raise ValueError(f"driver supports mnist/cifar10, config asks for '{name}'")
        # task_data_path="data" -> data/datasets/<name> (downloads if absent; MNIST is tiny)
        _DATA["d"] = loader(train_valid_split=tvs, flatten=flat, normalize=norm, task_data_path="data")
        print(f"  data: {name} flatten={flat} normalize={norm} split={tvs}", flush=True)
    return _DATA["d"]


def set_flags(model, fused, triton, cuda):
    n = 0
    for m in model.modules():
        if isinstance(m, IndexedSparseLinear):
            m.use_fused_gather = fused; m.use_triton_gather = triton; m.use_cuda_gather = cuda; n += 1
    return n


def run_series(name, flags, args):
    train_ds, valid_ds, _ = datasets()
    if args.subset > 0:
        train_ds = Subset(train_ds, range(min(args.subset, len(train_ds))))
    valid_small = Subset(valid_ds, range(min(args.batch, len(valid_ds))))  # cheap validation

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)  # identical init across series
    cfg = load_config(CFG)
    c = cfg.training.main.common
    c.epochs = args.epochs; c.batch_size = args.batch; c.shuffle = False   # deterministic batch order
    c.use_amp = False; c.early_stopping = False; c.load_best_state_dict = False
    c.plot_losses = False; c.suppress_prints = True
    c.num_workers = 0; c.persistent_workers = False; c.pin_memory = False; c.prefetch_factor = None

    model, _enc = initialize_model(cfg.model)
    model.to(DEV)
    n_syn = set_flags(model, *flags)

    main = cfg.training.main
    settings = _resolve_main_training_settings(main)
    param_groups = normalize_param_groups_config(settings.param_groups)
    model_param_groups = model.get_param_groups(param_groups)
    base_opt = create_optimizer(model_param_groups, settings.optimizer_config)
    optimizer = CustomWeightDecayOptimizer(
        model=model, optimizer=base_opt,
        weight_decay=settings.weight_decay_rate, weight_boosting=settings.weight_boosting)

    trainer_cfg = prepare_trainer_config(main, cfg.experiment, save_path="/tmp/none")
    trainer_cfg["task"] = cfg.model.task
    trainer = get_trainer(strategy=settings.strategy, optimizer=optimizer,
                          trainer_config_dict=trainer_cfg, analysis_manager=None)
    trainer.enable_amp = False; trainer.plot_losses = False; trainer.load_best_state_dict = False
    trainer.suppress_prints = True

    # capture per-batch TRAIN loss (grad enabled) via the loss-fn wrap
    per_batch = []; step = [0]; orig = trainer.loss_function
    def rec(m, x, y=None):
        l = orig(m, x, y)
        if torch.is_grad_enabled():
            v = l.detach().item(); per_batch.append(v); step[0] += 1
            if step[0] == 1 or step[0] % args.print_every == 0:
                print(f"    [{name}] step {step[0]:4d}  loss {v:.4f}  min {min(per_batch):.4f}", flush=True)
        return l
    trainer.loss_function = rec

    print(f"  === {name}: {n_syn} sparse layers, epochs={args.epochs} batch={args.batch} "
          f"train={len(train_ds)} (fp32, weight_boost={settings.weight_boosting}, wd={settings.weight_decay_rate}) ===", flush=True)
    trainer.train(model=model, train_data=train_ds, valid_data=valid_small)
    print(f"  {name}: {len(per_batch)} steps  {per_batch[0]:.4f} -> {per_batch[-1]:.4f}  min {min(per_batch):.4f}", flush=True)
    del model, optimizer, base_opt; torch.cuda.empty_cache()
    return per_batch


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG, help="config yaml (default: 10k profiling config)")
    ap.add_argument("--series", default="on", help="comma list: off,fused,on")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--subset", type=int, default=0, help="limit train set to first N (0 = full)")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    CFG = args.config
    assert torch.cuda.is_available(), "need CUDA"
    print(f"gpu: {torch.cuda.get_device_name(0)}\nconfig: {CFG}\nseries: {args.series}\n", flush=True)
    out = {}
    for name in args.series.split(","):
        name = name.strip()
        out[name] = run_series(name, SERIES[name], args)
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(
            REPO, "profilings/weight_transform_opt", args.out)
        json.dump({"config": CFG, "epochs": args.epochs, "batch": args.batch,
                   "subset": args.subset, "losses": out}, open(path, "w"), indent=2)
        print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
