from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dendritic_modeling.analysis.tools.path_matched_intervention import (
    build_nested_path_supports,
)
from dendritic_modeling.scripts.analysis.collect_path_matched_intervention import (
    collect_path_matched_interventions,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    selected_depths: tuple[int, ...] = (1, 2),
) -> tuple[Path, str]:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    profile = tmp_path / "analysis.yaml"
    for path, content in (
        (config, "model: {}\n"),
        (checkpoint, "frozen checkpoint\n"),
        (profile, "analysis: {}\n"),
    ):
        path.write_text(content, encoding="utf-8")

    output = tmp_path / "run"
    result_dir = output / "path_matched_intervention"
    result_dir.mkdir(parents=True)
    support = build_nested_path_supports(
        n_somas=2,
        branch_factors=[2, 2],
        draws=2,
        seed=17,
        namespace="collector-test",
    )
    records = []
    for depth in selected_depths:
        for target, signals in (
            ("excitation_current", ["excitation_current"]),
            ("joint_EI_currents", ["excitation_current", "inhibition_current"]),
        ):
            for method in ("zero", "global_test_shuffle"):
                for draw in range(2):
                    entry = support["depths"][str(depth)][draw]
                    accuracy = 0.7 - 0.01 * depth - 0.005 * draw
                    records.append(
                        {
                            "soma_relative_depth": depth,
                            "target": target,
                            "target_signals": signals,
                            "method": method,
                            "support_draw": draw,
                            "technical_draw_is_model_replicate": False,
                            "support_plan_sha256": support["support_plan_sha256"],
                            "selected_indices_sha256": entry["selected_indices_sha256"],
                            "selected_compartment_coordinate_count": 2,
                            "producer_stream_count": len(signals),
                            "replaced_tensor_coordinate_count": 2 * len(signals),
                            "available_compartment_coordinate_count": 2 * (2**depth),
                            "intervention_scope": (
                                "path_matched_fixed_count_compartment_coordinates"
                            ),
                            "baseline_accuracy": 0.8,
                            "accuracy": accuracy,
                            "accuracy_drop": 0.8 - accuracy,
                            "chance_accuracy": 0.1,
                            "normalized_above_chance_accuracy_loss": (
                                (0.8 - accuracy) / 0.7
                            ),
                            "baseline_mean_cross_entropy": 0.6,
                            "mean_cross_entropy": 0.7 + 0.01 * depth,
                            "mean_cross_entropy_increase": 0.1 + 0.01 * depth,
                            "module_names": [f"tree.depth_{depth}"] * len(signals),
                        }
                    )
    result = {
        "schema_version": 1,
        "analysis_type": "path_matched_depth_intervention",
        "reference_split": "training",
        "evaluation_split": "test",
        "tree": {"n_somas": 2, "branch_factors": [2, 2]},
        "support_plan": support,
        "shuffle": {
            "seed": 3,
            "permutation_sha256": "a" * 64,
        },
        "reference_sample_count": 20,
        "baseline": {
            "sample_count": 10,
            "n_classes": 2,
            "accuracy": 0.8,
            "mean_cross_entropy": 0.6,
        },
        "records": records,
    }
    (result_dir / "final.json").write_text(json.dumps(result), encoding="utf-8")
    provenance = {
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(output.resolve()),
        "enabled_analyzers": ["path_matched_intervention"],
    }
    for name, path in (
        ("config", config),
        ("checkpoint", checkpoint),
        ("analysis_config", profile),
    ):
        provenance[f"{name}_path"] = str(path.resolve())
        provenance[f"{name}_sha256"] = _sha256(path)
    (output / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    job = {
        "lineage": "rule_only_control",
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


def test_collector_writes_draw_audit_and_checkpoint_inference_tables(tmp_path):
    jobs, support_hash = _fixture(tmp_path)
    collection = collect_path_matched_interventions(
        jobs_paths=[jobs],
        output_dir=tmp_path / "collected",
        expected_count=1,
        expected_support_hash=support_hash,
    )

    assert collection["collected_job_count"] == 1
    with (tmp_path / "collected/support_draw_records.csv").open() as handle:
        support_rows = list(csv.DictReader(handle))
    with (tmp_path / "collected/checkpoint_records.csv").open() as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    assert len(support_rows) == 16
    assert len(checkpoint_rows) == 8
    assert {row["technical_support_draw_count"] for row in checkpoint_rows} == {"2"}
    assert {row["job_lineage"] for row in checkpoint_rows} == {"rule_only_control"}


def test_collector_fails_closed_on_wrong_support_contract(tmp_path):
    jobs, _support_hash = _fixture(tmp_path)
    with pytest.raises(ValueError, match="support differs"):
        collect_path_matched_interventions(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collected",
            expected_count=1,
            expected_support_hash="0" * 64,
        )


def test_collector_accepts_a_declared_subset_of_support_depths(tmp_path):
    jobs, support_hash = _fixture(tmp_path, selected_depths=(2,))
    collection = collect_path_matched_interventions(
        jobs_paths=[jobs],
        output_dir=tmp_path / "collected",
        expected_count=1,
        expected_support_hash=support_hash,
    )

    assert collection["collected_job_count"] == 1
    with (tmp_path / "collected/checkpoint_records.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["soma_relative_depth"] for row in rows} == {"2"}
