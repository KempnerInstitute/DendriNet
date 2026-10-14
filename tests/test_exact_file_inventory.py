import copy
import hashlib
from pathlib import Path

import pytest

from dendritic_modeling.utils import exact_file_inventory as exact


def _file_row(path: Path, role: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "resolved_path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "roles": [role],
    }


def _inventory(tmp_path: Path) -> dict[str, object]:
    paths = []
    for index, size in enumerate((11, 7, 5, 3, 2)):
        path = tmp_path / f"input_{index}.bin"
        path.write_bytes(bytes([index + 1]) * size)
        paths.append(path)
    return exact.build_inventory(
        [_file_row(path, f"role_{index}") for index, path in enumerate(paths)],
        generation_binding={"repository_commit": "a" * 40},
    )


def test_balanced_assignment_is_deterministic_and_byte_aware(tmp_path):
    inventory = _inventory(tmp_path)

    first = exact.balanced_assignment(inventory, world_size=3)
    second = exact.balanced_assignment(copy.deepcopy(inventory), world_size=3)

    assert first == second
    by_id = {row["task_id"]: row for row in inventory["files"]}
    loads = [0, 0, 0]
    for task, rank in first.items():
        loads[rank] += int(by_id[task]["bytes"])
    assert loads == [11, 9, 8]


def test_all_rank_shards_merge_to_one_exact_live_receipt(tmp_path):
    inventory = _inventory(tmp_path)
    shards = [
        exact.verify_shard(inventory, rank=rank, world_size=3) for rank in range(3)
    ]

    receipt, snapshots = exact.merge_verified_shards(
        inventory,
        shards,
        world_size=3,
    )

    assert receipt["status"] == "verified_before_execution"
    assert receipt["file_count"] == 5
    assert receipt["total_bytes"] == 28
    assert receipt["all_live_file_sha256_matched"] is True
    assert receipt["stat_only_cache_used"] is False
    assert set(snapshots) == {row["task_id"] for row in inventory["files"]}


def test_shard_rejects_same_size_content_drift(tmp_path):
    inventory = _inventory(tmp_path)
    target = Path(inventory["files"][0]["resolved_path"])
    target.write_bytes(b"x" * int(inventory["files"][0]["bytes"]))

    assignment = exact.balanced_assignment(inventory, world_size=3)
    rank = assignment[inventory["files"][0]["task_id"]]
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        exact.verify_shard(inventory, rank=rank, world_size=3)


def test_merge_rejects_missing_duplicate_and_wrong_rank_tasks(tmp_path):
    inventory = _inventory(tmp_path)
    shards = [
        exact.verify_shard(inventory, rank=rank, world_size=3) for rank in range(3)
    ]

    missing = copy.deepcopy(shards)
    removed = missing[0]["files"].pop()
    missing[0]["file_count"] -= 1
    missing[0]["total_bytes"] -= int(removed["bytes"])
    with pytest.raises(ValueError, match="missed tasks"):
        exact.merge_verified_shards(inventory, missing, world_size=3)

    wrong = copy.deepcopy(shards)
    donor = next(index for index, shard in enumerate(wrong) if shard["files"])
    receiver = (donor + 1) % 3
    moved = wrong[donor]["files"].pop()
    wrong[donor]["file_count"] -= 1
    wrong[donor]["total_bytes"] -= int(moved["bytes"])
    wrong[receiver]["files"].append(moved)
    wrong[receiver]["file_count"] += 1
    wrong[receiver]["total_bytes"] += int(moved["bytes"])
    with pytest.raises(ValueError, match="wrong rank"):
        exact.merge_verified_shards(inventory, wrong, world_size=3)


def test_merge_rejects_mutation_after_worker_hash(tmp_path):
    inventory = _inventory(tmp_path)
    shards = [
        exact.verify_shard(inventory, rank=rank, world_size=3) for rank in range(3)
    ]
    target = Path(inventory["files"][-1]["resolved_path"])
    target.write_bytes(b"z" * int(inventory["files"][-1]["bytes"]))

    with pytest.raises(ValueError, match="changed before merge"):
        exact.merge_verified_shards(inventory, shards, world_size=3)


def test_inventory_identity_rejects_metadata_tampering(tmp_path):
    inventory = _inventory(tmp_path)
    changed = copy.deepcopy(inventory)
    changed["files"][0]["roles"].append("unbound_role")

    with pytest.raises(ValueError, match="identity drifted"):
        exact.validate_inventory(changed)
