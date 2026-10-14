"""AnalysisManager records an explicit completion status for every analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.managers.analysis_status import (
    ANALYSIS_STATUS_FILE,
    ANALYSIS_STATUS_SCHEMA_VERSION,
    AnalysisIncompleteError,
)
from dendritic_modeling.analysis.registry import register_analyzer, unregister_analyzer
from dendritic_modeling.config import (
    AnalysisConfig,
    TransferFunctionConfig,
    load_analysis_config,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity
from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    require_completed_analyses,
)
from dendritic_modeling.utils.reproducibility import isolated_random_seed


def _model() -> Classifier:
    torch.manual_seed(0)
    return Classifier(
        encoder_network=Identity(input_dim=8),
        core_network=ExcitationInhibitionNetwork(
            input_dim=8,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[0],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[1],
            ei_synapses_per_branch_per_layer=[0],
            ie_synapses_per_branch_per_layer=[1],
            ii_synapses_per_branch_per_layer=[0],
        ),
        decoder_network=Identity(input_dim=4),
    )


def _data() -> dict:
    generator = torch.Generator().manual_seed(1)

    def split(n):
        return torch.utils.data.TensorDataset(
            torch.rand(n, 8, generator=generator),
            torch.randint(0, 4, (n,), generator=generator),
        )

    return {"train": split(30), "valid": split(20), "test": split(25)}


def _config(**sections) -> AnalysisConfig:
    config = AnalysisConfig()
    config.performance.enabled = True
    config.performance.final_splits = ["test"]
    for name, enabled in sections.items():
        getattr(config, name).enabled = enabled
    return config


def _manager(tmp_path: Path, config: AnalysisConfig) -> AnalysisManager:
    return AnalysisManager(
        model=_model(),
        data=_data(),
        analysis_config=config,
        save_root=str(tmp_path),
        device="cpu",
        evaluation_seed=5,
        probe_seed=7,
    )


def _status_file(tmp_path: Path) -> dict:
    return json.loads((tmp_path / ANALYSIS_STATUS_FILE).read_text())


def test_a_completed_pass_is_recorded_and_returned(tmp_path):
    manager = _manager(tmp_path, _config())
    record = manager.run_analysis()

    assert record["finished"] is True and record["complete"] is True
    assert record["requested"] == ["performance"]
    assert record["analyses"]["performance"] == {
        "status": "completed",
        "output_dir": "performance",
    }
    payload = _status_file(tmp_path)
    assert payload["schema_version"] == ANALYSIS_STATUS_SCHEMA_VERSION
    assert payload["passes"]["final"] == record
    assert manager.last_analysis_status == record


def _file_digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ANALYSIS_STATUS_FILE
    }


def test_status_recording_leaves_result_files_byte_identical(tmp_path):
    """The analyzers write the same bytes as a pass without status recording."""
    config = _config(weight=True)
    recorded, plain = tmp_path / "recorded", tmp_path / "plain"
    _manager(recorded, config).run_analysis()

    manager = _manager(plain, _config(weight=True))
    with isolated_random_seed(5):
        manager._run_ordered_analysis_sequence(
            filename="final",
            training=False,
            runtime=manager._analysis_runtime(False),
        )
    assert not (plain / ANALYSIS_STATUS_FILE).exists()
    digests = _file_digests(recorded)
    assert digests and digests == _file_digests(plain)


def test_an_analyzer_that_writes_nothing_is_empty(tmp_path):
    manager = _manager(tmp_path, _config())
    manager.performance_analyzer.analyze = lambda **_kwargs: None

    record = manager.run_analysis()

    status = record["analyses"]["performance"]
    assert status["status"] == "empty"
    assert "without writing any file into performance" in status["reason"]
    assert record["finished"] is True and record["complete"] is False


def test_reported_skips_and_deferred_outputs(tmp_path):
    manager = _manager(tmp_path, _config())
    analyzer = manager.performance_analyzer
    analyzer.analyze = lambda **_kwargs: analyzer.report_skip("not applicable")

    record = manager.run_analysis()
    assert record["analyses"]["performance"]["status"] == "skipped"
    assert record["analyses"]["performance"]["reason"] == "not applicable"

    # The reports of one call do not leak into the next.
    analyzer.analyze = lambda **_kwargs: analyzer.report_deferred_output("later")
    record = manager.run_analysis()
    assert record["analyses"]["performance"] == {
        "status": "completed",
        "output_dir": "performance",
        "deferred_output": "later",
    }


def test_a_failure_is_recorded_and_raised(tmp_path):
    manager = _manager(tmp_path, _config(weight=True))

    def fail(**_kwargs):
        raise RuntimeError("boom")

    manager.performance_analyzer.analyze = fail
    with pytest.raises(RuntimeError, match="boom"):
        manager.run_analysis()

    record = _status_file(tmp_path)["passes"]["final"]
    assert record["finished"] is False and record["complete"] is False
    assert record["analyses"]["performance"] == {
        "status": "failed",
        "output_dir": "performance",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    assert record["analyses"]["weight"] == {
        "status": "skipped",
        "reason": "not reached: the pass stopped with RuntimeError",
    }


def test_require_completion_fails_an_incomplete_pass(tmp_path):
    config = _config()
    config.require_completion = True
    manager = _manager(tmp_path, config)
    manager.performance_analyzer.analyze = lambda **_kwargs: None

    with pytest.raises(AnalysisIncompleteError, match="performance: empty") as error:
        manager.run_analysis()
    assert error.value.incomplete["performance"]["status"] == "empty"
    record = _status_file(tmp_path)["passes"]["final"]
    assert record["require_completion"] is True and record["complete"] is False

    # A complete pass passes the requirement.
    complete = _manager(tmp_path / "complete", config)
    assert complete.run_analysis()["complete"] is True


def test_training_passes_keep_only_the_latest_record(tmp_path):
    config = _config()
    config.performance.training = True
    config.performance.training_splits = ["valid"]
    manager = _manager(tmp_path, config)

    manager.run_analysis(filename="epoch1", training=True)
    manager.run_analysis(filename="epoch2", training=True)
    manager.run_analysis(filename="final", training=False)

    passes = _status_file(tmp_path)["passes"]
    assert sorted(passes) == ["epoch2", "final"]
    assert passes["epoch2"]["training"] is True
    assert passes["epoch2"]["analyses"]["performance"]["output_dir"] == (
        "performance/epochs"
    )


class _NeverRun(AbstractAnalyzer):
    def analyze(self, model, data, device="cpu", **kwargs):
        raise AssertionError("an analyzer without a phase is never run")


def test_a_plugin_analyzer_without_a_phase_is_recorded_as_skipped(tmp_path):
    register_analyzer(
        "status_probe",
        config_field="status_probe",
        class_path=f"{__name__}._NeverRun",
        attribute_name="status_probe_analyzer",
        save_subdir="status_probe",
        constructor_style="none",
    )
    try:
        config = _config()
        config.extensions["status_probe"] = SimpleNamespace(
            enabled=True, training=False, params=None
        )
        record = _manager(tmp_path, config).run_analysis()
    finally:
        unregister_analyzer("status_probe")
    assert record["analyses"]["status_probe"]["status"] == "skipped"
    assert "registered without a phase" in record["analyses"]["status_probe"]["reason"]
    assert record["complete"] is False


def test_require_completion_is_a_typed_option(tmp_path):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({"analysis": {"require_completion": True}}))
    assert load_analysis_config(str(path)).require_completion is True
    assert AnalysisConfig().require_completion is False
    with pytest.raises(ValueError, match="require_completion must be true or false"):
        AnalysisConfig(require_completion="yes")


def test_collector_rejects_recorded_incomplete_analyses():
    completed = {"analyses": {"weight": {"status": "completed"}}}
    require_completed_analyses({"analysis_status": completed}, ["weight"])
    # Runs written before the status record existed are not checked.
    require_completed_analyses({}, ["weight"])
    empty = {"analyses": {"weight": {"status": "empty", "reason": "no file"}}}
    with pytest.raises(ValueError, match=r"weight: empty \(no file\)"):
        require_completed_analyses({"analysis_status": empty}, ["weight"])


def test_a_pass_without_requested_analyzers_writes_no_status_file(tmp_path):
    """A training epoch without training analyzers leaves the run directory as is."""
    manager = _manager(tmp_path, _config())  # performance runs only after training
    record = manager.run_analysis(filename="epoch1", training=True)
    assert record["requested"] == [] and record["complete"] is True
    assert not (tmp_path / ANALYSIS_STATUS_FILE).exists()


def test_collectors_read_the_status_file_before_the_provenance_copy(tmp_path):
    from dendritic_modeling.analysis.managers.analysis_status import (
        incomplete_analyses,
        read_status_file,
    )
    from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
        incomplete_recorded_analyses,
    )

    assert read_status_file(tmp_path) == {}
    completed = {"finished": True, "analyses": {"weight": {"status": "completed"}}}
    provenance = {"analysis_status": completed, "filename": "custom"}
    assert incomplete_recorded_analyses(provenance, ["weight"], run_dir=tmp_path) == {}

    failed = {
        "finished": False,
        "analyses": {"weight": {"status": "failed", "error_type": "OSError"}},
    }
    (tmp_path / ANALYSIS_STATUS_FILE).write_text(
        json.dumps(
            {
                "schema_version": ANALYSIS_STATUS_SCHEMA_VERSION,
                "passes": {"custom": failed},
            }
        )
    )
    incomplete = incomplete_recorded_analyses(provenance, ["weight"], run_dir=tmp_path)
    assert incomplete == {"weight": {"status": "failed", "error_type": "OSError"}}
    # Another pass name in the file leaves the provenance copy in charge.
    other = {**provenance, "filename": "final"}
    assert incomplete_recorded_analyses(other, ["weight"], run_dir=tmp_path) == {}
    assert incomplete_analyses({"finished": True, "analyses": {}}, ["weight"]) == {
        "weight": {
            "status": "not recorded",
            "reason": "the recorded pass did not run it",
        }
    }
    (tmp_path / ANALYSIS_STATUS_FILE).write_text(json.dumps({"passes": {}}))
    with pytest.raises(ValueError, match="is not an analysis_status_v1 record"):
        read_status_file(tmp_path)
