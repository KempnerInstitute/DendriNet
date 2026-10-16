"""The recipe runner binds source bytes and tests actual model computations."""

import hashlib

import pytest
import yaml
from test_biological_constraints import _population

from dendritic_modeling.config.biological_adoption import prepare_biological_config
from dendritic_modeling.scripts.tools.biological_recipes import (
    compare_models,
    read_experiments,
)
from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping


def test_collection_hashes_are_checked_before_use(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: {core: {type: MLP}}\n")
    entry = {
        "format": "collection",
        "sources": [
            {
                "path": "config.yaml",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    assert list(read_experiments(entry, tmp_path)) == [yaml.safe_load(path.read_text())]
    path.write_text("model: {}\n")
    with pytest.raises(ValueError, match="hash changed"):
        list(read_experiments(entry, tmp_path))


def test_adoption_comparison_covers_state_and_gradients_at_zero():
    data = {
        "config_version": 3,
        "experiment": {"seed": 7},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": 8}},
            "core": {
                "type": "population_network",
                **_population(input_transform="identity"),
            },
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 2}},
        },
    }
    old, _, unresolved = upgrade_mapping(data)
    assert not unresolved
    new = prepare_biological_config(data, input_activation="nonnegative")
    result = compare_models(old, new, 8)
    assert result["exact"] and result["tensors"] > 10
    assert result["biological_violations"] == []


def test_dotted_overrides_index_lists_at_any_depth():
    from dendritic_modeling.scripts.tools.biological_recipes import _set_dotted

    layer = {"excitatory": {"branch_factors": [2, 3]}}
    mapping = {"model": {"core": {"unified_ei": {"layers": [layer]}}}}
    prefix = "model.core.unified_ei.layers.0.excitatory."
    _set_dotted(mapping, prefix + "branch_factors.0", 4)
    _set_dotted(mapping, prefix + "weight_transform", "softplus")
    assert layer == {
        "excitatory": {"branch_factors": [4, 3], "weight_transform": "softplus"}
    }
