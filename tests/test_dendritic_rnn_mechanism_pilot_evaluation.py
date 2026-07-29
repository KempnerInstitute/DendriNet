from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from dendritic_modeling.datasets.sequence.neuro_tasks import RomoDelayComparisonDataset
from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_mechanism_pilot as mechanism_analysis,
)
from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_mechanism_pilot import (
    DEFAULT_ANALYSIS_CONTRACT,
    EXPECTED_DELAY_STRATA,
    FAST_DIST,
    FAST_PROX,
    METRICS,
    SLOW_DIST,
    SLOW_PROX,
    _load_analysis_contract,
    _summarize_delay_strata,
    _summarize_metric,
    analyze,
    compute_statistics,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_mechanism_pilot import (
    DEFAULT_CONFIG,
    MechanismPilotEvaluationError,
    _pair_audit,
    load_config,
)


def test_mechanism_pilot_evaluation_contract_loads() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    assert path == DEFAULT_CONFIG.resolve()
    assert config["bank"]["n_trials"] == 4096
    assert config["bank"]["min_delay"] == 240
    assert config["resources"]["checkpoints_per_job"] == 4


def test_mechanism_pilot_analysis_contract_is_hash_bound(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_ANALYSIS_CONTRACT.read_text())
    analysis_source = Path(mechanism_analysis.__file__).resolve()
    payload["analysis_source_sha256"] = hashlib.sha256(
        analysis_source.read_bytes()
    ).hexdigest()
    contract_path = tmp_path / "analysis_contract.json"
    contract_path.write_text(json.dumps(payload))

    contract = _load_analysis_contract(contract_path)
    assert contract["frozen_before_any_training_complete"] is True
    assert contract["delay_strata_in_aggregate_gate"] is False

    payload["analysis_source_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(payload))
    with pytest.raises(mechanism_analysis.MechanismPilotAnalysisError, match="changed"):
        _load_analysis_contract(contract_path)


def test_analysis_uses_external_protocol_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_payload = json.loads(DEFAULT_ANALYSIS_CONTRACT.read_text())
    analysis_source = Path(mechanism_analysis.__file__).resolve()
    contract_payload["analysis_source_sha256"] = hashlib.sha256(
        analysis_source.read_bytes()
    ).hexdigest()
    configured_contract = tmp_path / "configured_contract.json"
    configured_contract.write_text(json.dumps(contract_payload))
    override_contract = tmp_path / "override_contract.json"
    override_contract.write_text(json.dumps(contract_payload))

    config_payload = yaml.safe_load(DEFAULT_CONFIG.read_text())
    analysis_output = tmp_path / "external_analysis"
    config_payload["study"]["analysis_contract"] = str(configured_contract)
    config_payload["study"]["analysis_output_root"] = str(analysis_output)
    config_path = tmp_path / "external_evaluation.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))

    rows = []
    for seed_index, seed in enumerate(("r000", "r001", "r002")):
        for condition, base in (
            (SLOW_PROX, 0.60),
            (SLOW_DIST, 0.72),
            (FAST_PROX, 0.64),
            (FAST_DIST, 0.66),
        ):
            value = base + 0.01 * seed_index
            rows.append(
                {
                    "pairing_group": seed,
                    "condition": condition,
                    **dict.fromkeys(METRICS, value),
                    "delay_strata": [
                        {
                            "delay_low": low,
                            "delay_high": high,
                            "n_trials": 100,
                            **dict.fromkeys(METRICS, value),
                        }
                        for low, high in EXPECTED_DELAY_STRATA
                    ],
                }
            )
    monkeypatch.setattr(
        mechanism_analysis,
        "_load_results",
        lambda _config, _config_path: (rows, []),
    )

    configured = compute_statistics(config_path)
    assert configured["config"] == str(config_path.resolve())
    assert configured["analysis_contract"]["path"] == str(configured_contract.resolve())
    assert configured["analysis_source"]["path"] == str(analysis_source)

    overridden = compute_statistics(config_path, override_contract)
    assert overridden["analysis_contract"]["path"] == str(override_contract.resolve())

    monkeypatch.setattr(
        mechanism_analysis,
        "compute_statistics",
        lambda _config_path, _contract_path=None: {"go_no_go": {"decision": "EXPAND"}},
    )
    monkeypatch.setattr(
        mechanism_analysis,
        "render_figure",
        lambda _statistics, output: output.write_bytes(b"figure"),
    )
    outputs = analyze(config_path)
    assert Path(outputs["statistics"]).parent == analysis_output
    assert Path(outputs["figure"]).parent == analysis_output


def test_analysis_rejects_execution_outside_required_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_source = (
        snapshot_root / "src/dendritic_modeling/scripts/analysis/"
        "analyze_dendritic_rnn_mechanism_pilot.py"
    )
    snapshot_source.parent.mkdir(parents=True)
    snapshot_source.write_bytes(Path(mechanism_analysis.__file__).read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    manifest = {
        "source_snapshot": {
            "root": str(snapshot_root),
            "package_sha256": "1" * 64,
            "training_runtime_source_sha256": "2" * 64,
        }
    }
    monkeypatch.setattr(
        mechanism_analysis,
        "_manifest_groups",
        lambda _config: (manifest, manifest_path, {}),
    )

    with pytest.raises(
        mechanism_analysis.MechanismPilotAnalysisError,
        match="not imported from the frozen source",
    ):
        mechanism_analysis._source_snapshot_provenance(
            {"study": {"require_source_snapshot": True}}
        )


def test_pair_audit_detects_post_stimulus_mismatch() -> None:
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(91)
        dataset = RomoDelayComparisonDataset(
            n_samples=8,
            frequency_values=(2.0, 2.8, 3.6, 4.4, 5.2, 6.0),
            stimulus_duration=20,
            response_duration=8,
            noise_std=0.15,
            positive_input_encoding=True,
            tuning_dim=8,
            pair_sampling="balanced_second_stimulus_v1",
            min_delay=240,
            max_delay=320,
        )
    _pair_audit(dataset)
    pair_id = int(dataset.pair_ids[0])
    mate = int((dataset.pair_ids == pair_id).nonzero(as_tuple=False)[1].item())
    dataset.inputs[mate, dataset.stimulus_duration, 0] += 1.0
    with pytest.raises(MechanismPilotEvaluationError, match="differ after delay onset"):
        _pair_audit(dataset)


def test_primary_contrast_is_paired_difference_of_differences() -> None:
    nested = {}
    for seed_index, seed in enumerate(("r000", "r001", "r002")):
        offset = 0.01 * seed_index
        nested[seed] = {
            SLOW_PROX: dict.fromkeys(METRICS, 0.60 + offset),
            SLOW_DIST: dict.fromkeys(METRICS, 0.75 + offset),
            FAST_PROX: dict.fromkeys(METRICS, 0.65 + offset),
            FAST_DIST: dict.fromkeys(METRICS, 0.67 + offset),
        }
    summary = _summarize_metric(nested, "accuracy")
    assert summary["primary_contrast_seed_values"] == pytest.approx([0.13] * 3)
    assert summary["primary_contrast_mean"] == pytest.approx(0.13)
    assert summary["primary_contrast_positive_seed_count"] == 3


def test_delay_strata_preserve_paired_interaction() -> None:
    rows = []
    for seed_index, seed in enumerate(("r000", "r001", "r002")):
        for condition, base in (
            (SLOW_PROX, 0.60),
            (SLOW_DIST, 0.70),
            (FAST_PROX, 0.62),
            (FAST_DIST, 0.64),
        ):
            rows.append(
                {
                    "pairing_group": seed,
                    "condition": condition,
                    "delay_strata": [
                        {
                            "delay_low": low,
                            "delay_high": high,
                            "n_trials": 100 + stratum_index,
                            **dict.fromkeys(
                                METRICS,
                                base + 0.01 * seed_index + 0.02 * stratum_index,
                            ),
                        }
                        for stratum_index, (low, high) in enumerate(
                            EXPECTED_DELAY_STRATA
                        )
                    ],
                }
            )
    summary = _summarize_delay_strata(rows)
    assert [row["primary_contrast_mean"] for row in summary["strata"]] == pytest.approx(
        [0.08] * 4
    )
    assert summary["longest_minus_shortest_mean"] == pytest.approx(0.0)
