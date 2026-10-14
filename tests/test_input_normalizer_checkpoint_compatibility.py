"""Historical range buffers must agree exactly with the configured partition."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.divisive import (
    DivisiveInputNormalizer,
)


@pytest.mark.parametrize(
    "mode,ranges", [("none", None), ("global_l1", None), ("group_l1", [[0, 2], [2, 4]])]
)
def test_legacy_checkpoint_preserves_current_defaults_and_outputs(mode, ranges):
    model = DivisiveInputNormalizer(4, mode=mode, group_ranges=ranges)
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    before = model(x).clone()
    legacy = {
        "group_starts": model.group_starts.clone(),
        "group_stops": model.group_stops.clone(),
    }
    model.load_state_dict(legacy, strict=True)
    assert torch.equal(model(x), before)
    assert not model.state_dict()  # Current serialization remains unchanged.
    assert set(legacy) == {"group_starts", "group_stops"}  # Caller data intact.
    model.load_state_dict({}, strict=True)  # Earlier checkpoints also load.


@pytest.mark.parametrize("bad", ["value", "shape", "dtype", "incomplete"])
def test_different_legacy_partition_is_rejected_even_without_strict_loading(bad):
    model = DivisiveInputNormalizer(4, mode="group_l1", group_ranges=[[0, 2], [2, 4]])
    state = {
        "group_starts": model.group_starts.clone(),
        "group_stops": model.group_stops.clone(),
    }
    if bad == "value":
        state["group_starts"][1] = 1
    if bad == "shape":
        state["group_starts"] = torch.tensor([], dtype=torch.long)
    if bad == "dtype":
        state["group_starts"] = state["group_starts"].float()
    if bad == "incomplete":
        del state["group_stops"]
    with pytest.raises(RuntimeError, match="input-normalization ranges"):
        model.load_state_dict(state, strict=False)
