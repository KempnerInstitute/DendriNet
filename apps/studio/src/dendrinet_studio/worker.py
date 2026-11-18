"""Construct a CPU model in an isolated process and report core resource counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = load_config(str(args.config))
    torch.set_num_threads(1)
    seed = cfg.experiment.model_seed
    torch.manual_seed(cfg.experiment.seed if seed is None else seed)
    model, _ = initialize_model(cfg.model)
    summary = model_resource_summary(model)
    summary["scope"] = (
        "Constructed model before data-dependent gate calibration; no training or dataset loaded"
    )
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
