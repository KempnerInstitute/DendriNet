"""Tests for reproducible frozen-checkpoint analysis."""

from pathlib import Path

import torch
import yaml

from dendritic_modeling.config import Config, load_analysis_config
from dendritic_modeling.scripts.analysis import run_checkpoint_analysis as runner


def test_load_standalone_analysis_config(tmp_path: Path) -> None:
    path = tmp_path / "analysis.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "analysis": {
                    "representation_accessibility": {
                        "enabled": True,
                        "training": False,
                        "params": {
                            "representations": ["excitatory", "joint"],
                            "projections": [
                                {
                                    "name": "matched8",
                                    "type": "balanced_countsketch",
                                    "output_dim": 8,
                                }
                            ],
                            "probes": [
                                {
                                    "name": "linear",
                                    "type": "ridge",
                                    "n_label_shuffles": 2,
                                }
                            ],
                        },
                    }
                }
            }
        )
    )

    config = load_analysis_config(str(path))

    section = config.representation_accessibility_analysis
    assert section.enabled is True
    assert section.params.projections[0].type == "balanced_countsketch"
    assert section.params.projections[0].output_dim == 8
    assert section.params.probes[0].type == "ridge"


def test_checkpoint_state_dict_layouts() -> None:
    tensor = torch.ones(2)
    assert runner._extract_state_dict({"model_state_dict": {"weight": tensor}}) == {
        "weight": tensor
    }
    normalized = runner._normalize_state_dict_keys(
        {"module.weight": tensor, "module.bias": tensor}
    )
    assert set(normalized) == {"weight", "bias"}


def test_run_checkpoint_analysis_records_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    analysis_path = tmp_path / "analysis.yaml"
    checkpoint_path = tmp_path / "model.pt"
    output_dir = tmp_path / "output"
    config_path.write_text("model: {}\n")
    analysis_path.write_text("analysis: {}\n")

    model = torch.nn.Linear(2, 2)
    torch.save(model.state_dict(), checkpoint_path)
    config = Config()

    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(
        runner,
        "load_analysis_config",
        lambda *_args, **_kwargs: config.analysis,
    )
    monkeypatch.setattr(
        runner,
        "setup_environment",
        lambda **_kwargs: (
            str(output_dir),
            [(torch.zeros(2), 0)],
            [(torch.zeros(2), 0)],
            [(torch.zeros(2), 0)],
            model,
            None,
        ),
    )

    class FakeManager:
        def __init__(self, **_kwargs):
            pass

        def run_analysis(self, **_kwargs):
            return {"ok": True}

    monkeypatch.setattr(runner, "AnalysisManager", FakeManager)

    result = runner.run_checkpoint_analysis(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        analysis_config_path=analysis_path,
        output_dir=output_dir,
        device="cpu",
    )

    provenance = result["provenance"]
    assert provenance["training_performed"] is False
    assert provenance["state_dict_key_count"] == 2
    assert provenance["dataset_sizes"] == {"train": 1, "valid": 1, "test": 1}
    assert (output_dir / "checkpoint_analysis_provenance.json").is_file()
