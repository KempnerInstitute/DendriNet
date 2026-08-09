"""Tests for matched pretrained vision-boundary diagnostics."""

from pathlib import Path

import numpy as np
import torch
import yaml

from dendritic_modeling.analysis.managers.analysis_manager import AnalysisManager
from dendritic_modeling.analysis.tools import vision_boundary_diagnostics as module
from dendritic_modeling.analysis.tools.vision_boundary_diagnostics import (
    VisionBoundaryDiagnosticsAnalyzer,
    _linear_cka,
)
from dendritic_modeling.config import (
    AnalysisConfig,
    EvaluationRuntimeConfig,
    VisionBoundaryDiagnosticsAnalysisConfig,
    VisionBoundaryDiagnosticsAnalysisParams,
    load_analysis_config,
)


class _NamedSegment(torch.nn.Module):
    def __init__(self, layer_names: list[str], layer: torch.nn.Module):
        super().__init__()
        self._layer_names = layer_names
        self.layer = layer

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layer(values)


class _ReplacementModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_network = _NamedSegment([], torch.nn.Identity())
        self.core_network = torch.nn.Linear(4, 3, bias=False)
        self.decoder_network = _NamedSegment([], torch.nn.Identity())
        with torch.no_grad():
            self.core_network.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder_network(self.core_network(self.encoder_network(values)))


def _dataset() -> torch.utils.data.TensorDataset:
    values = torch.tensor(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 2.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 2.0, 0.0],
        ]
    )
    return torch.utils.data.TensorDataset(values, torch.arange(4))


def test_linear_cka_is_one_for_scaled_matched_geometry() -> None:
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    assert np.isclose(_linear_cka(values, 4.0 * values), 1.0)


def test_load_vision_boundary_diagnostics_config(tmp_path: Path) -> None:
    path = tmp_path / "analysis.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "analysis": {
                    "vision_boundary_diagnostics": {
                        "enabled": True,
                        "training": False,
                        "params": {
                            "backbone": "alexnet",
                            "projection_dim": 32,
                            "projection_seed": 7,
                            "max_samples": 12,
                        },
                    }
                }
            }
        )
    )

    config = load_analysis_config(str(path))

    section = config.vision_boundary_diagnostics_analysis
    assert section.enabled is True
    assert section.params.projection_dim == 32
    assert section.params.projection_seed == 7
    assert section.params.max_samples == 12


def test_analyzer_compares_matched_boundary_and_saves_json(
    tmp_path: Path, monkeypatch
) -> None:
    model = _ReplacementModel()
    exact = torch.nn.Linear(4, 3, bias=False)
    exact.load_state_dict(model.core_network.state_dict())
    monkeypatch.setattr(
        module,
        "_exact_reference",
        lambda *_args, **_kwargs: (exact, ["features.0"]),
    )
    analyzer = VisionBoundaryDiagnosticsAnalyzer(
        VisionBoundaryDiagnosticsAnalysisParams(
            projection_dim=3,
            max_samples=4,
        )
    )

    result = analyzer.analyze(
        model,
        _dataset(),
        save_path=str(tmp_path),
        filename="final",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert result["contract"]["target_modules"] == ["features.0"]
    assert result["contract"]["sample_count"] == 4
    assert np.isclose(result["comparison"]["relative_mse"], 0.0)
    assert np.isclose(result["comparison"]["mean_sample_cosine"], 1.0)
    assert np.isclose(result["comparison"]["projected_linear_cka"], 1.0)
    assert (tmp_path / "final.json").is_file()


def test_analysis_manager_runs_vision_boundary_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    model = _ReplacementModel()
    exact = torch.nn.Linear(4, 3, bias=False)
    exact.load_state_dict(model.core_network.state_dict())
    monkeypatch.setattr(
        module,
        "_exact_reference",
        lambda *_args, **_kwargs: (exact, ["features.0"]),
    )
    analysis = AnalysisConfig(
        vision_boundary_diagnostics_analysis=(
            VisionBoundaryDiagnosticsAnalysisConfig(
                enabled=True,
                training=False,
                params=VisionBoundaryDiagnosticsAnalysisParams(
                    projection_dim=3,
                    max_samples=4,
                ),
            )
        )
    )
    analysis.performance_analysis.enabled = False
    manager = AnalysisManager(
        model,
        {"train": _dataset(), "valid": _dataset(), "test": _dataset()},
        analysis,
        save_root=str(tmp_path),
        device="cpu",
    )

    manager.run_analysis(filename="final", training=False)

    assert (tmp_path / "vision_boundary_diagnostics" / "final.json").is_file()
