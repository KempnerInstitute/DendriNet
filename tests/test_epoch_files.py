import json

from dendritic_modeling.utils.epoch_files import (
    epoch_file_paths_by_number,
    epoch_files_by_number,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_epoch_files_by_number_requires_json_by_default(tmp_path):
    _write_json(tmp_path / "epoch10.json", {"epoch": 10})
    _write_json(tmp_path / "epoch2.json", {"epoch": 2})
    _write_json(tmp_path / "epoch1", {"epoch": 1})
    _write_json(tmp_path / "epoch_bad.json", {"epoch": "bad"})
    _write_json(tmp_path / "metrics.json", {"ignored": True})

    assert epoch_files_by_number(tmp_path) == [
        (2, "epoch2.json"),
        (10, "epoch10.json"),
    ]


def test_epoch_files_by_number_can_include_legacy_extensionless_files(tmp_path):
    _write_json(tmp_path / "epoch10.json", {"epoch": 10})
    _write_json(tmp_path / "epoch2.json", {"epoch": 2})
    _write_json(tmp_path / "epoch1", {"epoch": 1})
    _write_json(tmp_path / "epoch4epoch.json", {"epoch": 4})
    _write_json(tmp_path / "epoch_bad.json", {"epoch": "bad"})
    _write_json(tmp_path / "metrics.json", {"ignored": True})

    assert epoch_files_by_number(tmp_path, require_json=False) == [
        (1, "epoch1"),
        (2, "epoch2.json"),
        (4, "epoch4epoch.json"),
        (10, "epoch10.json"),
    ]


def test_epoch_files_by_number_missing_directory_returns_empty(tmp_path):
    assert epoch_files_by_number(tmp_path / "missing") == []


def test_epoch_file_paths_by_number_returns_sorted_paths(tmp_path):
    _write_json(tmp_path / "epoch3.json", {"epoch": 3})
    _write_json(tmp_path / "epoch1.json", {"epoch": 1})

    assert list(epoch_file_paths_by_number(tmp_path)) == [
        (1, str(tmp_path / "epoch1.json")),
        (3, str(tmp_path / "epoch3.json")),
    ]
