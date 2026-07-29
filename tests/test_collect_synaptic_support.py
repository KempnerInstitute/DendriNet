import json

import numpy as np
import pytest

from dendritic_modeling.scripts.analysis.collect_synaptic_support import (
    _array_sha256,
    _profile_rows,
    _validate_support,
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
