import csv
import json

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.config.analysis import WeightAnalysisParams
from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import _sha256
from dendritic_modeling.scripts.analysis.collect_synaptic_support import (
    RESULT_SUBDIRECTORY,
    _array_sha256,
    _checkpoint_source_activity_summary_rows,
    _checkpoint_spatial_region_summary_rows,
    _checkpoint_summary_rows,
    _profile_rows,
    _validate_region_alignment,
    _validate_support,
    collect_synaptic_support,
)


def _support_result():
    structural = {
        "network_layer_index": 0,
        "population_name": "e0",
        "target_polarity": "excitatory",
        "dendritic_depth": 2,
        "depth_label": "2",
        "pathway": "ff_inhibitory",
        "source_polarity": "inhibitory",
        "n_soma": 2,
        "branches_per_soma": 3,
        "source_dim": 4,
        "selection_policy": "standard",
        "mask_source": "deterministic_current",
    }
    record = {
        **structural,
        "soma_index": 0,
        "realized_contacts_per_branch_mean": 1.0,
        "active_effective_conductance_mean": 0.5,
        "source_coverage_fraction": 0.75,
        "eligible_source_fraction": 1.0,
        "source_coverage_of_eligible": 0.75,
        "source_entropy_bits": 1.5,
        "normalized_source_entropy": 0.75,
        "effective_source_fraction": 0.5,
        "mean_pairwise_branch_jaccard": None,
    }
    metadata = {
        **structural,
        "contact_frequency_key": "contact",
        "mean_effective_conductance_key": "conductance",
    }
    return {
        "source_support": {
            "quantity_contract": {"not_activity_or_information": True},
            "n_scalar_records": 1,
            "records": [record],
            "profile_metadata": [metadata],
            "profiles_file": "final_source_support_profiles.npz",
        }
    }


def _source_region_alignment():
    analyzer = WeightAnalyzer(
        WeightAnalysisParams(
            compute_source_support=True,
            source_support_image_shape=(2, 2),
            compute_source_region_alignment=True,
            source_region_soma_to_class=[0, 1],
            source_region_foreground_thresholds=[0.5],
        )
    )
    structural = {
        "network_layer_index": 0,
        "population_name": "e0",
        "target_polarity": "excitatory",
        "dendritic_depth": 1,
        "depth_label": "1",
        "pathway": "ff_excitatory",
        "source_polarity": "excitatory",
        "n_soma": 2,
        "branches_per_soma": 1,
        "source_dim": 4,
        "selection_policy": "standard",
        "mask_source": "test",
        "contact_frequency_key": "contact",
        "mean_effective_conductance_key": "conductance",
    }
    return analyzer._compute_source_region_alignment(
        profiles={
            "contact": np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "conductance": np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
        },
        profile_metadata=[structural],
        reference_dataset=TensorDataset(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([0, 1]),
        ),
        reference_split="validation",
        runtime=None,
    )


def test_validate_support_accepts_null_undefined_pairwise_overlap(tmp_path):
    result = _support_result()

    support, records, metadata = _validate_support(
        result,
        result_path=tmp_path / "final",
    )

    assert support["n_scalar_records"] == 1
    assert records[0]["mean_pairwise_branch_jaccard"] is None
    assert metadata[0]["pathway"] == "ff_inhibitory"


def test_profile_rows_validate_declared_shapes_and_hashes(tmp_path):
    profiles_path = tmp_path / "profiles.npz"
    contact = np.arange(8, dtype=np.float64).reshape(2, 4)
    conductance = contact / 10.0
    np.savez_compressed(
        profiles_path,
        contact=contact,
        conductance=conductance,
    )
    job = {
        "output_dir": str(tmp_path),
        "config": "config.yaml",
        "checkpoint": "model.pt",
        "analysis_config": "analysis.yaml",
    }

    rows = _profile_rows(
        job_index=0,
        job=job,
        metadata=_support_result()["source_support"]["profile_metadata"],
        profiles_path=profiles_path,
    )

    assert len(rows) == 2
    assert {row["quantity"] for row in rows} == {
        "contact_frequency",
        "mean_effective_conductance",
    }
    assert rows[0]["profile_shape"] == json.dumps([2, 4])
    assert {row["profile_sha256"] for row in rows} == {
        _array_sha256(contact),
        _array_sha256(conductance),
    }


def test_profile_rows_reject_undeclared_arrays(tmp_path):
    profiles_path = tmp_path / "profiles.npz"
    array = np.zeros((2, 4), dtype=np.float64)
    np.savez_compressed(
        profiles_path,
        contact=array,
        conductance=array,
        unexpected=array,
    )

    with pytest.raises(ValueError, match="undeclared"):
        _profile_rows(
            job_index=0,
            job={},
            metadata=_support_result()["source_support"]["profile_metadata"],
            profiles_path=profiles_path,
        )


def test_checkpoint_summary_averages_somas_without_promoting_them():
    base = _support_result()["source_support"]["records"][0]
    rows = []
    for soma_index, entropy in enumerate((0.5, 0.9)):
        row = {
            **base,
            "n_soma": 2,
            "soma_index": soma_index,
            "normalized_source_entropy": entropy,
            "technical_record_is_independent_replicate": False,
            "trained_checkpoint_is_inference_unit": True,
        }
        rows.append(row)

    summary = _checkpoint_summary_rows(rows)

    assert len(summary) == 1
    assert summary[0]["technical_soma_count"] == 2
    assert summary[0]["normalized_source_entropy_mean"] == pytest.approx(0.7)
    assert summary[0]["trained_checkpoint_is_inference_unit"] is True


def test_validate_region_alignment_accepts_package_native_records(tmp_path):
    alignment = _source_region_alignment()

    payload, activity, spatial, skipped = _validate_region_alignment(
        {"source_region_alignment": alignment},
        result_path=tmp_path / "final",
    )

    assert payload is alignment
    assert len(activity) == 4
    assert len(spatial) == 4
    assert skipped == []
    assert activity[0]["reference_split"] == "validation"


def _with_job_fields(records):
    return [
        {
            **record,
            "job_index": 0,
            "job_seed": 17,
            "technical_record_is_independent_replicate": False,
            "trained_checkpoint_is_inference_unit": True,
        }
        for record in records
    ]


def _write_collector_run(tmp_path):
    run_dir = tmp_path / "run"
    result_dir = run_dir / RESULT_SUBDIRECTORY
    result_dir.mkdir(parents=True)

    result = _support_result()
    support = result["source_support"]
    support["records"].append({**support["records"][0], "soma_index": 1})
    support["n_scalar_records"] = 2
    np.savez_compressed(
        result_dir / support["profiles_file"],
        contact=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        conductance=np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
    )

    alignment = _source_region_alignment()
    for field, filename in (
        ("source_activity_records_file", "final_source_activity.csv"),
        ("spatial_region_records_file", "final_spatial_region.csv"),
    ):
        alignment[field] = filename
        (result_dir / filename).write_text(
            "package-native artifact\n", encoding="utf-8"
        )
    result["source_region_alignment"] = alignment
    result_path = result_dir / "final"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inputs = {}
    for field, content in (
        ("config", "model: test\n"),
        ("checkpoint", "checkpoint bytes\n"),
        ("analysis_config", "analysis: test\n"),
    ):
        path = tmp_path / f"{field}.dat"
        path.write_text(content, encoding="utf-8")
        inputs[field] = path.resolve()
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(run_dir.resolve()),
        "enabled_analyzers": ["weight"],
    }
    for field, path in inputs.items():
        provenance[f"{field}_path"] = str(path)
        provenance[f"{field}_sha256"] = _sha256(path)
    (run_dir / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job = {
        "output_dir": str(run_dir.resolve()),
        "config": str(inputs["config"]),
        "checkpoint": str(inputs["checkpoint"]),
        "analysis_config": str(inputs["analysis_config"]),
        "seed": 17,
    }
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job, sort_keys=True) + "\n", encoding="utf-8")
    return jobs_path


def test_collector_emits_region_checkpoint_summaries_and_provenance(tmp_path):
    jobs_path = _write_collector_run(tmp_path)
    output_dir = tmp_path / "collection"

    release = collect_synaptic_support(
        jobs_paths=[jobs_path],
        output_dir=output_dir,
        expected_count=1,
    )

    with (output_dir / "checkpoint_source_activity_alignment_summary.csv").open(
        encoding="utf-8"
    ) as handle:
        activity_summaries = list(csv.DictReader(handle))
    with (output_dir / "checkpoint_spatial_region_alignment_summary.csv").open(
        encoding="utf-8"
    ) as handle:
        spatial_summaries = list(csv.DictReader(handle))

    assert len(activity_summaries) == 2
    assert len(spatial_summaries) == 2
    assert {row["quantity"] for row in activity_summaries} == {
        "contact_frequency",
        "mean_effective_conductance",
    }
    assert activity_summaries[0]["job_seed"] == "17"
    assert activity_summaries[0]["foreground_threshold"] == "0.5"
    assert activity_summaries[0]["technical_soma_count"] == "2"
    assert spatial_summaries[0]["center_row_start"] == "0"
    assert release["source_activity_checkpoint_summary_count"] == 2
    assert release["spatial_region_checkpoint_summary_count"] == 2
    assert release["aggregation_policy"] == "equal_soma_mean_within_checkpoint_only"
    assert release["within_checkpoint_soma_weighting"] == "equal"
    assert release["technical_soma_aggregation_performed"] is True
    assert release["trained_seed_aggregation_performed"] is False
    assert {
        "checkpoint_source_activity_alignment_summary.csv",
        "checkpoint_spatial_region_alignment_summary.csv",
    }.issubset(release["outputs"])


def test_source_activity_checkpoint_summary_is_equal_soma_and_group_preserving():
    activity = _with_job_fields(_source_region_alignment()["source_activity_records"])
    contact_rows = [row for row in activity if row["quantity"] == "contact_frequency"]
    contact_rows[0]["preferred_rest_foreground_contrast"] = 0.2
    contact_rows[1]["preferred_rest_foreground_contrast"] = 0.6
    contact_rows[1]["center_preferred_foreground_occupancy_enrichment"] = None

    summaries = _checkpoint_source_activity_summary_rows(activity)

    assert len(summaries) == 2
    summary = next(row for row in summaries if row["quantity"] == "contact_frequency")
    assert summary["job_seed"] == 17
    assert summary["network_layer_index"] == 0
    assert summary["dendritic_depth"] == 1
    assert summary["pathway"] == "ff_excitatory"
    assert summary["foreground_threshold"] == pytest.approx(0.5)
    assert summary["technical_soma_count"] == 2
    assert summary["soma_weighting"] == "equal"
    assert summary["technical_somas_are_independent_replicates"] is False
    assert summary["trained_checkpoint_is_inference_unit"] is True
    assert summary["preferred_rest_foreground_contrast_defined_count"] == 2
    assert summary["preferred_rest_foreground_contrast_mean"] == pytest.approx(0.4)
    assert summary["preferred_rest_foreground_contrast_sd"] == pytest.approx(0.2)
    assert (
        summary["center_preferred_foreground_occupancy_enrichment_defined_count"] == 1
    )


def test_spatial_checkpoint_summary_preserves_center_bounds_and_population_sd():
    spatial = _with_job_fields(_source_region_alignment()["spatial_region_records"])
    contact_rows = [row for row in spatial if row["quantity"] == "contact_frequency"]
    contact_rows[0]["center_surround_enrichment_contrast"] = -0.2
    contact_rows[1]["center_surround_enrichment_contrast"] = 0.6

    summaries = _checkpoint_spatial_region_summary_rows(spatial)

    assert len(summaries) == 2
    summary = next(row for row in summaries if row["quantity"] == "contact_frequency")
    assert summary["center_row_start"] == 0
    assert summary["center_row_stop"] == 1
    assert summary["center_col_start"] == 0
    assert summary["center_col_stop"] == 1
    assert summary["dendritic_depth"] == 1
    assert summary["pathway"] == "ff_excitatory"
    assert summary["technical_soma_count"] == 2
    assert summary["center_surround_enrichment_contrast_defined_count"] == 2
    assert summary["center_surround_enrichment_contrast_mean"] == pytest.approx(0.2)
    assert summary["center_surround_enrichment_contrast_sd"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "summarize,records_key,table_name",
    [
        (
            _checkpoint_source_activity_summary_rows,
            "source_activity_records",
            "source-activity alignment",
        ),
        (
            _checkpoint_spatial_region_summary_rows,
            "spatial_region_records",
            "spatial-region alignment",
        ),
    ],
)
def test_region_checkpoint_summaries_reject_missing_or_duplicate_somas(
    summarize,
    records_key,
    table_name,
):
    records = _with_job_fields(_source_region_alignment()[records_key])
    contact_rows = [row for row in records if row["quantity"] == "contact_frequency"]

    with pytest.raises(ValueError, match="exactly one record for every soma"):
        summarize(contact_rows[:1])

    duplicate = {**contact_rows[0]}
    with pytest.raises(ValueError, match=f"{table_name}.*duplicate somas"):
        summarize([contact_rows[0], duplicate])
