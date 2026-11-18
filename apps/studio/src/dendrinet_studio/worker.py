"""Construct the model with the trainer's dataset and report its resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = load_config(str(args.config))
    torch.set_num_threads(1)
    # The trainer derives the encoder width from a sample and resolves topology
    # and initialization seeds here. Inspection does not create a training run.
    _, _, _, _, model, _ = setup_environment(
        cfg.model, cfg.training, cfg.data, cfg.outputs, cfg.experiment, is_main=False
    )
    summary = model_resource_summary(model)
    summary["input_shape"] = cfg.model.encoder.params.input_shape
    summary["scope"] = (
        "Built from the configured dataset and training seeds, before gate calibration."
    )
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
