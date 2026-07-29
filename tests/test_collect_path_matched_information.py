from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dendritic_modeling.analysis.tools.path_matched_intervention import (
    build_nested_path_supports,
)
from dendritic_modeling.scripts.analysis.collect_path_matched_information import (
    _load_collection_config,
    collect_path_matched_information,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    profile = tmp_path / "analysis.yaml"
    for path, content in (
        (config, "model: {}\n"),
        (checkpoint, "frozen checkpoint\n"),
        (profile, "analysis: {}\n"),
    ):
        path.write_text(content, encoding="utf-8")

    support = build_nested_path_supports(
        n_somas=2,
        branch_factors=[2, 2],
        draws=2,
        seed=17,
        namespace="path-information-collector-test",
    )
    soma_indices = np.arange(2, dtype=np.int64)
    somatic_support = {
        "support_draw": 0,
        "selected_indices": soma_indices.tolist(),
        "selected_indices_sha256": _sha256_array(soma_indices),
    }
    available = {
        "joint_EI_currents": [1, 2],
        "upstream_child_current": [0, 1],
        "post_gate_output": [0, 1, 2],
    }
    records = []
    for signal, depths in available.items():
        for depth in depths:
            draws = range(2) if depth else range(1)
            for draw in draws:
                support_entry = (
                    support["depths"][str(depth)][draw] if depth else somatic_support
                )
                observed = 0.2 + 0.01 * depth + 0.001 * draw
                null = [0.01, 0.02]
                records.append(
                    {
                        "soma_relative_depth": depth,
                        "signal": signal,
                        "support_draw": draw,
                        "technical_draw_role": (
                            "coordinate-support sensitivity; not an independent "
                            "trained-model replicate"
                        ),
                        "selected_index_supports_sha256": support[
                            "selected_index_supports_sha256"
                        ],
                        "support_plan_sha256": support["support_plan_sha256"],
                        "feature_selection_policy": (
                            "shared_nested_anatomical_coordinate_support"
                        ),
                        "selected_features": support_entry["selected_indices"],
                        "selected_features_sha256": support_entry[
                            "selected_indices_sha256"
                        ],
                        "selected_coordinate_count_per_component": 2,
                        "joint_feature_count": (
                            4 if signal == "joint_EI_currents" else 2
                        ),
                        "n_label_shuffles": 2,
                        "label_shuffle_mi_bits": null,
                        "observed_mi_bits": observed,
                        "label_shuffle_mi_mean_bits": 0.015,
                        "label_shuffle_mi_std_bits": 0.005,
                        "excess_mi_over_label_shuffle_bits": observed - 0.015,
                        "preprocessing_uses_labels": False,
                    }
                )
    result = {
        "schema_version": "path_matched_information_v1",
        "analysis_split": "validation",
        "test_examples_or_labels_used": False,
        "feature_selection_uses_labels_or_activations": False,
        "technical_draws_are_model_replicates": False,
        "sample_count": 20,
        "information_seed_effective": 23,
        "tree": {"n_somas": 2, "branch_factors": [2, 2]},
        "available_depths_by_signal": available,
        "support_plan": support,
        "somatic_support": somatic_support,
        "records": records,
    }
    output = tmp_path / "run"
    result_dir = output / "path_matched_information"
    result_dir.mkdir(parents=True)
    (result_dir / "final.json").write_text(json.dumps(result), encoding="utf-8")
    provenance = {
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(output.resolve()),
        "enabled_analyzers": ["path_matched_information"],
        "code_identity": {"commit": "test"},
    }
    for name, path in (
        ("config", config),
        ("checkpoint", checkpoint),
        ("analysis_config", profile),
    ):
        provenance[f"{name}_path"] = str(path.resolve())
        provenance[f"{name}_sha256"] = _sha256(path)
    (output / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    job = {
        "lineage": "mechanism_selected_recipe",
        "mechanism": "shunting",
        "pairing_seed": 0,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "analysis_config": str(profile),
        "output_dir": str(output),
    }
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
    return jobs, support["selected_index_supports_sha256"]


def test_path_information_collector_preserves_and_averages_support_draws(
    tmp_path,
) -> None:
    jobs, support_hash = _fixture(tmp_path)
    release = collect_path_matched_information(
        jobs_paths=[jobs],
        output_dir=tmp_path / "collected",
        expected_count=1,
        expected_support_hash=support_hash,
    )

    assert release["collected_job_count"] == 1
    with (tmp_path / "collected/support_draw_records.csv").open() as handle:
        support_rows = list(csv.DictReader(handle))
    with (tmp_path / "collected/checkpoint_records.csv").open() as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    assert len(support_rows) == 12
    assert len(checkpoint_rows) == 7
    soma_rows = [row for row in checkpoint_rows if row["soma_relative_depth"] == "0"]
    assert {row["technical_support_draw_count"] for row in soma_rows} == {"1"}
    assert {row["job_mechanism"] for row in checkpoint_rows} == {"shunting"}


def test_path_information_collector_fails_closed_on_support_drift(tmp_path) -> None:
    jobs, _support_hash = _fixture(tmp_path)
    with pytest.raises(ValueError, match="support differs"):
        collect_path_matched_information(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collected",
            expected_count=1,
            expected_support_hash="0" * 64,
        )


def test_path_information_collector_rejects_record_level_support_drift(
    tmp_path: Path,
) -> None:
    jobs, support_hash = _fixture(tmp_path)
    job = json.loads(jobs.read_text())
    result_path = Path(job["output_dir"]) / "path_matched_information" / "final.json"
    result = json.loads(result_path.read_text())
    result["records"][0]["selected_features"] = [1, 0]
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="exact support"):
        collect_path_matched_information(
            jobs_paths=[jobs],
            output_dir=tmp_path / "selected_drift",
            expected_count=1,
            expected_support_hash=support_hash,
        )

    record = result["records"][0]
    depth = int(record["soma_relative_depth"])
    draw = int(record["support_draw"])
    support_entry = (
        result["support_plan"]["depths"][str(depth)][draw]
        if depth
        else result["somatic_support"]
    )
    record["selected_features"] = support_entry["selected_indices"]
    record["selected_features_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="selected-feature digest"):
        collect_path_matched_information(
            jobs_paths=[jobs],
            output_dir=tmp_path / "digest_drift",
            expected_count=1,
            expected_support_hash=support_hash,
        )


def test_path_information_collector_requires_label_independent_selection(
    tmp_path: Path,
) -> None:
    jobs, support_hash = _fixture(tmp_path)
    job = json.loads(jobs.read_text())
    result_path = Path(job["output_dir"]) / "path_matched_information" / "final.json"
    result = json.loads(result_path.read_text())
    result["feature_selection_uses_labels_or_activations"] = True
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="used data or labels"):
        collect_path_matched_information(
            jobs_paths=[jobs],
            output_dir=tmp_path / "label_selected",
            expected_count=1,
            expected_support_hash=support_hash,
        )


def test_path_information_collection_config_resolves_paths(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}\n")
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: path_matched_information_collection_config_v1",
                "jobs: [jobs.jsonl]",
                "output_dir: collected",
                "expected_count: 10",
                "expected_support_hash: abc123",
            ]
        )
        + "\n"
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "collected").resolve()
    assert loaded["expected_count"] == 10
    assert loaded["expected_support_hash"] == "abc123"
