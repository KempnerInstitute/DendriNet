"""Smoke-check an installed wheel outside its source checkout."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import io
import sys
from pathlib import Path

import torch
import yaml

import dendritic_modeling
from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    imported = Path(dendritic_modeling.__file__).resolve()
    if imported.is_relative_to(source):
        raise RuntimeError(
            f"Imported the source checkout instead of the wheel: {imported}"
        )
    distribution = importlib.metadata.distribution("dendritic_modeling")
    assert distribution.version == dendritic_modeling.__version__
    assert imported.is_relative_to(Path(distribution.locate_file("")).resolve())
    for entry in distribution.entry_points:
        if entry.group != "console_scripts":
            continue
        saved = sys.argv
        try:
            sys.argv = [entry.name, "--help"]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                try:
                    entry.load()()
                except SystemExit as error:
                    assert error.code in (None, 0), (entry.name, error.code)
            assert "usage:" in output.getvalue(), entry.name
        finally:
            sys.argv = saved
    for name in example_names():
        cfg = config_from_mapping(yaml.safe_load(example_text(name)))
        torch.manual_seed(42)
        model, _ = initialize_model(cfg.model)
        output = model(torch.rand(2, cfg.model.encoder.params.input_dim))
        if isinstance(output, (list, tuple)):
            output = output[0]
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
    print(
        f"Installed package {distribution.version}: CLI and example checks passed ({imported})"
    )


if __name__ == "__main__":
    main()
