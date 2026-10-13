"""The bitmask topology encoding keeps every per-contact tensor on its contact.

A bitmask stores which inputs a row contacts, not their slots. Layers whose
per-contact state or training updates depend on the slot order (indexed
release, rewire and dynamic) keep their slot order exactly; fixed indexed
layers may be stored with their contacts in ascending order, as before.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from test_operator_config_contract import _config

from dendritic_modeling.config.compression import (
    ModelCompressionConfig,
    SparseTopologyCompressionConfig,
)
from dendritic_modeling.deployment.compression import (
    bitmask_slot_order_keys,
    compact_model_checkpoint,
    load_compact_model_checkpoint,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.degree_grouped_indexed import (
    DegreeGroupedIndexedLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    CellwiseImpactRewiring,
    ImpactRewiringConfig,
    population_rewiring_groups,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.checkpoints import (
    compact_sparse_bitmask_state_dict,
    decode_sparse_bitmask_state_dict,
)


def _pre_fix_bitmask_encoder(state_dict):
    """The encoder before 2026-10-13: every row reordered, only ``pre_w`` follows."""
    compact = dict(state_dict)
    encoding = {}
    for key, value in tuple(compact.items()):
        if not key.endswith("connection_indices") or value.ndim != 2:
            continue
        weight_key = f"{key[: -len('connection_indices')]}pre_w"
        weight = compact.get(weight_key)
        if weight is None or tuple(weight.shape) != tuple(value.shape):
            continue
        sorted_indices, order = value.to(torch.long).sort(dim=1)
        n_rows, k = sorted_indices.shape
        n_inputs = int(sorted_indices.max().item()) + 1 if k else 0
        packed = torch.zeros((n_rows, (n_inputs + 7) // 8), dtype=torch.int16)
        if k:
            packed.scatter_add_(
                1,
                torch.div(sorted_indices, 8, rounding_mode="floor"),
                torch.bitwise_left_shift(
                    torch.ones_like(sorted_indices, dtype=torch.int16),
                    torch.remainder(sorted_indices, 8),
                ).to(torch.int16),
            )
        compact[weight_key] = weight.gather(1, order)
        del compact[key]
        compact[f"{key}.__bitmask"] = packed.to(torch.uint8)
        compact[f"{key}.__bitmask_metadata"] = torch.tensor(
            [n_rows, k, n_inputs], dtype=torch.int64
        )
        encoding[key] = "bitmask"
    return compact, encoding


def _assert_same_state(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].dtype == right[key].dtype, key
        assert torch.equal(left[key], right[key]), key


def _not_ascending_rows(layer) -> int:
    return int((layer.connection_indices.long().diff(dim=1) < 0).any(dim=1).sum())


def _bitmask_policy(**options):
    return ModelCompressionConfig(
        sparse_topology=SparseTopologyCompressionConfig(
            topology_encoding="bitmask", **options
        )
    )


# --- A trained indexed_release population with cellwise impact rewiring ---

_REWIRING = ImpactRewiringConfig(
    frequency=1,
    maturity_steps=1,
    eligible_fraction=0.5,
    segment_fraction=0.5,
    threshold=1.0,
    seed=3,
)


def _release_config():
    config = _config("population_network", "shunting", "endpoint_tanh")
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"].update(
        topk_type="indexed_release",
        indexed_seed=5,
        weight_transform="exp",
        dbl_init_method="fan_in_lognormal",
        leak_conductance=0.03,
        ff_excitatory_synapses=3,
    )
    layer["populations"][0]["n_neurons"] = 4
    return config


def _build_release_model():
    torch.manual_seed(0)
    return get_architecture("population_network", _release_config(), input_dim=12)


def _release_banks(model):
    return [m for m in model.modules() if isinstance(m, IndexedReleaseLinear)]


def _trained_release_model():
    model = _build_release_model()
    controller = CellwiseImpactRewiring(population_rewiring_groups(model), _REWIRING)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(16, 12, generator=torch.Generator().manual_seed(1))
    model.train()
    torch.manual_seed(2)
    for _ in range(6):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        controller.step(optimizer)
    return model, controller


def _rewire_once(model, controller_state):
    controller = CellwiseImpactRewiring(population_rewiring_groups(model), _REWIRING)
    controller.load_state_dict(deepcopy(controller_state))
    return controller.step(torch.optim.SGD(model.parameters(), lr=0.0))


def test_indexed_release_bitmask_round_trip_keeps_every_contact_and_next_update(
    tmp_path: Path,
) -> None:
    model, controller = _trained_release_model()
    banks = _release_banks(model)
    assert banks and all(_not_ascending_rows(bank) > 0 for bank in banks)
    assert all(bank.contact_age.gt(0).any() for bank in banks)
    source = tmp_path / "source.pt"
    torch.save(model.state_dict(), source)
    output = tmp_path / "compact.pt"

    report = compact_model_checkpoint(model, source, output, _bitmask_policy())
    assert report["converted_modules"] == 0
    payload = torch.load(output, map_location="cpu", weights_only=True)
    keys = {
        f"{path}.connection_indices"
        for path, module in model.named_modules()
        if isinstance(module, IndexedReleaseLinear)
    }
    assert payload["sparse_index_encoding"] == dict.fromkeys(keys, "bitmask_slot_order")
    for key in keys:
        assert f"{key}.__bitmask" in payload["state_dict"]
        assert payload["state_dict"][f"{key}.__bitmask_slot_order"].dtype == (
            torch.uint8
        )

    restored = _build_release_model()
    load = load_compact_model_checkpoint(restored, output)
    assert load["contact_order_warnings"] == []
    # Every tensor, contact by contact, in its original slot.
    _assert_same_state(restored.state_dict(), model.state_dict())

    # The next structural update and a training forward match the model that
    # was never encoded.
    controller_state = controller.state_dict()
    assert _rewire_once(model, controller_state) > 0
    assert _rewire_once(restored, controller_state) > 0
    _assert_same_state(restored.state_dict(), model.state_dict())
    x = torch.rand(8, 12, generator=torch.Generator().manual_seed(9))
    model.train()
    restored.train()
    torch.manual_seed(13)
    expected = model(x)
    torch.manual_seed(13)
    observed = restored(x)
    assert torch.equal(observed, expected)
    _assert_same_state(restored.state_dict(), model.state_dict())


def test_ascending_layout_refuses_to_misalign_per_contact_state() -> None:
    model, _ = _trained_release_model()
    state = model.state_dict()
    with pytest.raises(ValueError, match=r"contact_age, .*impact"):
        compact_sparse_bitmask_state_dict(state)

    encoded, encoding = compact_sparse_bitmask_state_dict(state, keep_slot_order=True)
    assert set(encoding.values()) == {"bitmask_slot_order"}
    _assert_same_state(decode_sparse_bitmask_state_dict(encoded), dict(state))

    # Ascending rows need no reordering, so nothing can be misaligned.
    bank = IndexedReleaseLinear(12, 4, 4, seed=8)
    with torch.no_grad():
        bank.connection_indices.copy_(bank.connection_indices.sort(dim=1).values)
        bank.impact.uniform_()
    encoded, encoding = compact_sparse_bitmask_state_dict(bank.state_dict())
    assert encoding == {"connection_indices": "bitmask"}
    _assert_same_state(decode_sparse_bitmask_state_dict(encoded), bank.state_dict())


# --- Indexed layers kept unconverted by the compact export ---


def _layer_zoo() -> nn.ModuleDict:
    torch.manual_seed(21)
    zoo = nn.ModuleDict(
        {
            "rewire": IndexedRewireLinear(
                40,
                6,
                8,
                seed=4,
                rewire_frequency=1,
                rewire_quantile=0.4,
                weight_transform="identity",
            ),
            "dynamic": IndexedDynamicTopKLinear(
                24,
                5,
                2,
                candidate_size=6,
                seed=3,
                selection="standard",
                noise_level=0.0,
                weight_transform="identity",
            ),
            "fixed": IndexedSparseLinear(30, 5, 6, seed=7, weight_transform="identity"),
            "grouped": DegreeGroupedIndexedLinear(10, 6, [0, 2, 3, 3, 5, 2], seed=4),
        }
    )
    return zoo


def _trained_zoo() -> nn.ModuleDict:
    zoo = _layer_zoo()
    for _ in range(3):
        zoo["rewire"].apply_rewiring()
    with torch.no_grad():
        # Tied candidate weights: the Top-K choice depends on the slot order.
        zoo["dynamic"].pre_w.copy_(
            torch.tensor([1.0, 1.0, 0.5, 0.5, 0.25, 1.0]).repeat(5, 1)
        )
    return zoo


def _contact_map(layer):
    indices = layer.connection_indices.long()
    weights = layer.pre_w.detach()
    return [
        {int(i): float(w) for i, w in zip(row_i, row_w)}
        for row_i, row_w in zip(indices, weights)
    ]


def test_kept_rewire_and_dynamic_layers_round_trip_exactly(tmp_path: Path) -> None:
    zoo = _trained_zoo()
    assert _not_ascending_rows(zoo["rewire"]) > 0
    assert _not_ascending_rows(zoo["dynamic"]) > 0
    assert _not_ascending_rows(zoo["fixed"]) > 0
    original = deepcopy(zoo)
    source = tmp_path / "source.pt"
    torch.save(zoo.state_dict(), source)
    output = tmp_path / "compact.pt"
    policy = _bitmask_policy(allow_indexed_dynamic=False, allow_indexed_rewire=False)

    assert bitmask_slot_order_keys(zoo) == {
        "rewire.connection_indices",
        "dynamic.connection_indices",
    }
    compact_model_checkpoint(zoo, source, output, policy)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["sparse_index_encoding"] == {
        "rewire.connection_indices": "bitmask_slot_order",
        "dynamic.connection_indices": "bitmask_slot_order",
        "fixed.connection_indices": "bitmask",
        "grouped.groups.0.projection.connection_indices": "bitmask",
        "grouped.groups.1.projection.connection_indices": "bitmask",
        "grouped.groups.2.projection.connection_indices": "bitmask",
    }

    restored = _layer_zoo()
    assert (
        load_compact_model_checkpoint(restored, output)["contact_order_warnings"] == []
    )
    for name in ("rewire", "dynamic"):
        _assert_same_state(restored[name].state_dict(), original[name].state_dict())
    # Fixed layers are stored with ascending contacts: same contacts and
    # weights, other slots.
    assert torch.equal(
        restored["fixed"].connection_indices,
        original["fixed"].connection_indices.sort(dim=1).values,
    )
    assert _contact_map(restored["fixed"]) == _contact_map(original["fixed"])
    x = torch.rand(7, 30, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(restored["fixed"](x), original["fixed"](x))
    grouped_x = torch.rand(7, 10, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(
        restored["grouped"](grouped_x), original["grouped"](grouped_x)
    )

    # Next structural updates match the layers that were never encoded.
    original["rewire"].apply_rewiring()
    restored["rewire"].apply_rewiring()
    _assert_same_state(restored["rewire"].state_dict(), original["rewire"].state_dict())
    for layer in (original["dynamic"], restored["dynamic"]):
        layer.eval()
    assert torch.equal(
        restored["dynamic"].active_sparse_mask(),
        original["dynamic"].active_sparse_mask(),
    )
    dynamic_x = torch.rand(3, 24, generator=torch.Generator().manual_seed(6))
    assert torch.equal(restored["dynamic"](dynamic_x), original["dynamic"](dynamic_x))


def test_ascending_layout_of_fixed_layers_is_unchanged() -> None:
    zoo = _trained_zoo()
    del zoo["rewire"], zoo["dynamic"]
    state = zoo.state_dict()

    expected, expected_encoding = _pre_fix_bitmask_encoder(state)
    for keep in (False, bitmask_slot_order_keys(zoo)):
        observed, encoding = compact_sparse_bitmask_state_dict(
            state, keep_slot_order=keep
        )
        assert encoding == expected_encoding
        _assert_same_state(observed, expected)


def test_contact_slot_declarations_name_every_per_contact_tensor() -> None:
    layers = {
        IndexedSparseLinear: IndexedSparseLinear(30, 5, 6, seed=7),
        IndexedRewireLinear: IndexedRewireLinear(30, 5, 6, seed=7),
        IndexedReleaseLinear: IndexedReleaseLinear(
            30, 5, 6, seed=7, release_track_source_activity=True
        ),
        IndexedDynamicTopKLinear: IndexedDynamicTopKLinear(
            30, 5, 6, candidate_size=9, seed=7
        ),
    }
    for cls, layer in layers.items():
        state = layer.state_dict()
        shape = tuple(state["connection_indices"].shape)
        aligned = {
            name
            for name, value in state.items()
            if name != "connection_indices"
            and value.ndim >= 2
            and tuple(value.shape[:2]) == shape
        }
        assert aligned == set(cls.CONTACT_SLOT_STATE), cls.__name__
    assert IndexedSparseLinear.SLOT_ORDER_MATTERS is False
    assert IndexedRewireLinear.SLOT_ORDER_MATTERS is True
    assert IndexedReleaseLinear.SLOT_ORDER_MATTERS is True
    assert IndexedDynamicTopKLinear.SLOT_ORDER_MATTERS is True

    class Subclass(IndexedSparseLinear):
        """A layer type defined elsewhere keeps its slot order by default."""

    model = nn.ModuleDict({"own": Subclass(30, 5, 6, seed=7), **layers_by_name(layers)})
    assert bitmask_slot_order_keys(model) == {
        "own.connection_indices",
        "IndexedRewireLinear.connection_indices",
        "IndexedReleaseLinear.connection_indices",
        "IndexedDynamicTopKLinear.connection_indices",
    }


def layers_by_name(layers):
    return {cls.__name__: layer for cls, layer in layers.items()}


@pytest.mark.parametrize("k", [5, 300])
def test_slot_order_round_trip_restores_any_contact_order(k: int) -> None:
    generator = torch.Generator().manual_seed(k)
    indices = torch.stack(
        [torch.randperm(2 * k, generator=generator)[:k] for _ in range(4)]
    )
    state = {
        "layer.connection_indices": indices.to(torch.int32),
        "layer.pre_w": torch.randn(4, k, generator=generator),
        "layer.impact": torch.rand(4, k, generator=generator),
    }
    encoded, encoding = compact_sparse_bitmask_state_dict(
        state, keep_slot_order={"layer.connection_indices"}
    )
    assert encoding == {"layer.connection_indices": "bitmask_slot_order"}
    assert encoded["layer.connection_indices.__bitmask_slot_order"].dtype == (
        torch.uint8 if k <= 256 else torch.uint16
    )
    assert encoded["layer.connection_indices.__bitmask_metadata"].tolist()[3] == 1
    assert torch.equal(encoded["layer.pre_w"], state["layer.pre_w"])
    decoded = decode_sparse_bitmask_state_dict(encoded)
    assert torch.equal(decoded["layer.connection_indices"], indices.long())
    assert torch.equal(decoded["layer.impact"], state["layer.impact"])
    # Encoding the decoded state reproduces the encoded state.
    again, _ = compact_sparse_bitmask_state_dict(decoded, keep_slot_order=True)
    _assert_same_state(again, encoded)


def test_malformed_slot_orders_are_rejected() -> None:
    state = {
        "connection_indices": torch.tensor([[3, 0, 2], [1, 4, 0]]),
        "pre_w": torch.randn(2, 3),
    }
    encoded, _ = compact_sparse_bitmask_state_dict(state, keep_slot_order=True)
    slots = "connection_indices.__bitmask_slot_order"
    metadata = "connection_indices.__bitmask_metadata"

    def decode_with(**changes):
        candidate = dict(encoded)
        for key, value in changes.items():
            if value is None:
                del candidate[key]
            else:
                candidate[key] = value
        return decode_sparse_bitmask_state_dict(candidate)

    with pytest.raises(RuntimeError, match="Missing bitmask slot order"):
        decode_with(**{slots: None})
    with pytest.raises(RuntimeError, match="not a permutation"):
        decode_with(**{slots: torch.tensor([[0, 0, 1], [2, 1, 0]], dtype=torch.uint8)})
    with pytest.raises(RuntimeError, match="without slot-order metadata"):
        decode_with(**{metadata: encoded[metadata][:3]})
    with pytest.raises(RuntimeError, match="Invalid bitmask metadata"):
        decode_with(**{metadata: torch.tensor([2, 3, 5, 7])})
    with pytest.raises(RuntimeError, match="without a bitmask"):
        decode_sparse_bitmask_state_dict({f"other.{slots}": encoded[slots]})


def test_pre_fix_bitmask_files_report_reordered_release_layers(
    tmp_path: Path, caplog
) -> None:
    model, _ = _trained_release_model()
    state, encoding = _pre_fix_bitmask_encoder(model.state_dict())
    path = tmp_path / "pre_fix.pt"
    torch.save(
        {
            "format": "dendritic_modeling.compact_checkpoint",
            "format_version": 1,
            "state_dict": state,
            "sparse_topology_manifest": [],
            "sparse_index_encoding": encoding,
        },
        path,
    )
    restored = _build_release_model()
    with caplog.at_level(logging.WARNING):
        report = load_compact_model_checkpoint(restored, path)
    warnings = report["contact_order_warnings"]
    assert len(warnings) == len(_release_banks(model))
    assert all("impact" in item and "contact_age" in item for item in warnings)
    assert all(item in caplog.text for item in warnings)
    # The weights and indices are consistent; impact and ages are not moved.
    for old, new in zip(_release_banks(model), _release_banks(restored)):
        order = old.connection_indices.long().sort(dim=1).indices
        assert torch.equal(new.pre_w, old.pre_w.gather(1, order))
        assert torch.equal(new.impact, old.impact)


def test_compact_checkpoint_cli_refuses_a_report_before_writing(
    tmp_path: Path,
) -> None:
    from dendritic_modeling.scripts.deployment.compact_checkpoint import main

    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "compact.pt"
    arguments = [
        str(tmp_path / "config.yaml"),
        str(tmp_path / "source.pt"),
        str(output),
    ]
    with pytest.raises(FileExistsError):
        main([*arguments, "--report", str(report)])
    with pytest.raises(ValueError, match="--report must differ"):
        main([*arguments, "--report", str(output)])
    assert not output.exists()
    assert report.read_text(encoding="utf-8") == "{}\n"
