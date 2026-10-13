"""Checkpoint version rules: legacy fallbacks are recorded, current files must be complete."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.activations.parametric import ParametricTanhOnlyM
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_reactivation import (
    _apply_reactivation_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.checkpoints import (
    extract_model_state_dict,
    load_initial_model_checkpoint,
    load_state_dict_with_record,
)
from dendritic_modeling.networks.state_versions import compare_state_dict_keys
from dendritic_modeling.scripts.analysis.run_checkpoint_analysis import (
    _normalize_state_dict_keys,
)


def _save(state, path):
    torch.save(state, path)
    return path


def _without(state, key, *, version=None, module=""):
    """``state`` minus ``key``, with the metadata version of ``module`` set."""
    reduced = OrderedDict((k, v) for k, v in state.items() if k != key)
    reduced._metadata = OrderedDict(
        (name, dict(meta)) for name, meta in state._metadata.items()
    )
    if version is not None:
        reduced._metadata[module]["version"] = version
    return reduced


def test_a_current_file_missing_a_versioned_key_fails_through_the_loader(tmp_path):
    model = nn.Sequential(ParametricTanhOnlyM(3, fixed_b=0.123))
    path = _save(_without(model.state_dict(), "0.fixed_b_buffer"), tmp_path / "a.pt")
    with pytest.raises(RuntimeError, match="fixed_b_buffer"):
        load_initial_model_checkpoint(nn.Sequential(ParametricTanhOnlyM(3)), path)


def test_a_legacy_file_keeps_the_constructed_value_and_records_it(tmp_path):
    model = nn.Sequential(ParametricTanhOnlyM(3, fixed_b=0.123))
    legacy = _without(model.state_dict(), "0.fixed_b_buffer", version=1, module="0")
    path = _save(legacy, tmp_path / "legacy.pt")
    target = nn.Sequential(ParametricTanhOnlyM(3, fixed_b=0.5))
    record = load_initial_model_checkpoint(target, path)
    assert record["missing_keys"] == []
    assert [item["key"] for item in record["legacy_state"]] == ["0.fixed_b_buffer"]
    assert target[0].fixed_b == 0.5


def test_ddp_prefixes_and_metadata_survive_both_loaders():
    model = nn.Sequential(ParametricTanhOnlyM(3))
    wrapped = OrderedDict((f"module.{k}", v) for k, v in model.state_dict().items())
    wrapped._metadata = OrderedDict(
        [("", {"version": 1})]
        + [
            (f"module.{k}" if k else "module", dict(v))
            for k, v in model.state_dict()._metadata.items()
        ]
    )
    for state in (
        extract_model_state_dict({"model_state_dict": wrapped}),
        _normalize_state_dict_keys(wrapped),
    ):
        assert set(state) == set(model.state_dict())
        assert state._metadata["0"]["version"] == 2


def test_rewiring_seed_and_mask_follow_the_version_rule():
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 0] = False
    layer = IndexedRewireLinear(8, 4, 3, connection_mask=mask, seed=5)
    state = layer.state_dict()
    target = IndexedRewireLinear(8, 4, 3, connection_mask=mask, seed=9)

    legacy = _without(state, "rewire_seed", version=1)
    record = load_state_dict_with_record(target, legacy)
    assert [item["key"] for item in record["legacy_state"]] == ["rewire_seed"]

    current = _without(state, "rewire_seed")
    with pytest.raises(RuntimeError, match="rewire_seed"):
        load_state_dict_with_record(target, current)

    other_mask = torch.ones(4, 8, dtype=torch.bool)
    other_mask[:, 1:4] = False
    kept = IndexedRewireLinear(8, 4, 3, connection_mask=other_mask, seed=9)
    record = load_state_dict_with_record(
        kept, _without(state, "_allowed_connection_mask", version=1)
    )
    (entry,) = record["legacy_state"]
    assert entry["key"] == "_allowed_connection_mask"
    outside = int((~torch.gather(other_mask, 1, state["connection_indices"])).sum())
    assert outside > 0 and f"{outside} of 12 loaded contacts" in entry["note"]


def test_deepst_counter_follows_the_version_rule():
    layer = DeepstLinear(6, 3, target_density=0.5)
    state = layer.state_dict()
    with pytest.raises(RuntimeError, match="rewire_step"):
        DeepstLinear(6, 3, target_density=0.5).load_state_dict(
            _without(state, "rewire_step")
        )
    record = load_state_dict_with_record(
        DeepstLinear(6, 3, target_density=0.5),
        _without(state, "rewire_step", version=1),
    )
    assert [item["key"] for item in record["legacy_state"]] == ["rewire_step"]


def test_version_1_release_files_record_their_contact_age_unit():
    layer = IndexedReleaseLinear(5, 2, 4, seed=8)
    state = layer.state_dict()
    legacy = _without(state, "__none__", version=1)
    record = load_state_dict_with_record(IndexedReleaseLinear(5, 2, 4, seed=8), legacy)
    assert [item["key"] for item in record["legacy_state"]] == ["contact_age"]
    current = load_state_dict_with_record(IndexedReleaseLinear(5, 2, 4, seed=8), state)
    assert current["legacy_state"] == []


def test_key_comparison_separates_version_added_keys():
    mask = torch.ones(4, 8, dtype=torch.bool)
    model = nn.Sequential(IndexedRewireLinear(8, 4, 3, connection_mask=mask, seed=5))
    state = model.state_dict()
    old = _without(state, "0.rewire_seed", version=1, module="0")
    assert compare_state_dict_keys(model, old) == {
        "missing": [],
        "version_added": ["0.rewire_seed"],
        "unexpected": [],
    }
    assert compare_state_dict_keys(model, _without(state, "0.rewire_seed")) == {
        "missing": ["0.rewire_seed"],
        "version_added": [],
        "unexpected": [],
    }


def test_branch_layer_priors_and_anchors_are_version_2_keys():
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=8,
        excitatory_synapses_per_branch=2,
        reactivate=True,
        reactivation_type="param_tanh",
    )
    _apply_reactivation_init(layer, 2.0, 0.5)
    model = nn.Sequential(layer)
    added = [
        "0._additive_tangent_is_set",
        "0._additive_tangent_n0",
        "0._additive_tangent_t0",
        "0.reactivation_analytical_init_b",
        "0.reactivation_analytical_init_m",
    ]
    state = model.state_dict()
    assert set(added) <= set(state)
    old = state
    for key in added:
        old = _without(old, key, version=1, module="0")
    assert compare_state_dict_keys(model, old) == {
        "missing": [],
        "version_added": added,
        "unexpected": [],
    }
    current = state
    for key in added:
        current = _without(current, key)
    assert compare_state_dict_keys(model, current)["missing"] == added


def test_numerics_comparison_reports_version_added_keys_as_layout():
    from dendritic_modeling.scripts.verification.compare_develop_numerics import (
        compare_payloads,
    )

    tensor = {"shape": [1], "l2": 1.0, "first8": [1.0]}
    current = {
        "state": {
            "tensors": {"layer.w": tensor, "layer.rewire_seed": tensor},
            "module_versions": {"": 1, "layer": 2},
            "version_added": {"layer.rewire_seed": 2},
        }
    }
    develop = {
        "state": {
            "tensors": {"layer.w": tensor},
            "module_versions": {"": 1, "layer": 1},
            "version_added": {},
        }
    }
    notes: list[str] = []
    assert (
        compare_payloads(
            current, develop, atol=0, rtol=0, max_failures=10, layout_notes=notes
        )
        == []
    )
    assert notes == ["$.state.layer.rewire_seed: added by a newer module version"]
    develop["state"]["tensors"]["layer.w"] = {**tensor, "l2": 2.0}
    assert compare_payloads(current, develop, atol=0, rtol=0, max_failures=10)
