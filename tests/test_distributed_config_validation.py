"""``distributed`` choices are checked when the config loads."""

from __future__ import annotations

import pytest

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.config import (
    FSDP_DTYPE_NAMES,
    DistributedConfig,
    FSDPConfig,
)


def _distributed(section: dict) -> DistributedConfig:
    return config_from_mapping({"distributed": section}).distributed


@pytest.mark.parametrize(
    "mode, expected", [("none", "none"), ("DDP", "ddp"), (" fsdp ", "fsdp")]
)
def test_modes_are_normalized(mode, expected):
    assert _distributed({"mode": mode}).mode == expected


@pytest.mark.parametrize("mode", ["fdsp", "gpu", "data_parallel"])
def test_unknown_modes_raise_at_load(mode):
    with pytest.raises(
        ValueError, match=r"distributed\.mode must be one of none, ddp, fsdp"
    ):
        _distributed({"mode": mode})


def test_a_close_misspelling_gets_a_hint():
    with pytest.raises(ValueError, match="did you mean 'fsdp'"):
        _distributed({"mode": "fdsp"})


def test_fsdp_choices_are_normalized_and_checked():
    fsdp = _distributed(
        {
            "mode": "fsdp",
            "fsdp": {
                "sharding_strategy": "full_shard",
                "backward_prefetch": "backward_post",
                "mixed_precision_dtype": "BF16",
                "buffer_dtype": "keep",
            },
        }
    ).fsdp
    assert fsdp.sharding_strategy == "FULL_SHARD"
    assert fsdp.backward_prefetch == "BACKWARD_POST"
    assert fsdp.mixed_precision_dtype == "bf16"
    assert fsdp.buffer_dtype == "keep"


@pytest.mark.parametrize(
    "field, value",
    [
        ("sharding_strategy", "SHARDED"),
        ("backward_prefetch", "BEFORE"),
        ("mixed_precision_dtype", "bf8"),
        ("param_dtype", "int8"),
        ("reduce_dtype", "double"),
        ("buffer_dtype", "copy"),
    ],
)
def test_unknown_fsdp_choices_raise(field, value):
    with pytest.raises(ValueError, match=rf"distributed\.fsdp\.{field} must be one of"):
        FSDPConfig(**{field: value})


def test_the_fsdp_builder_accepts_the_same_dtype_names():
    fsdp_utils = pytest.importorskip("dendritic_modeling.training.fsdp_utils")

    assert set(fsdp_utils._FSDP_DTYPE_ALIASES) == set(FSDP_DTYPE_NAMES)
