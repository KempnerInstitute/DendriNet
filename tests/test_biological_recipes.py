"""The recipe runner binds source bytes and tests actual model computations."""

import hashlib
import json
import subprocess

import pytest
import yaml
from test_biological_constraints import _population

from dendritic_modeling.config.biological_adoption import prepare_biological_config
from dendritic_modeling.scripts.tools import biological_recipes
from dendritic_modeling.scripts.tools.biological_recipes import (
    check_source,
    compare_models,
    main,
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


def _experiment():
    return {
        "config_version": 3,
        "experiment": {"seed": 7, "checkpointing": {"checkpoint_dir": "old"}},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": 8}},
            "core": {
                "type": "population_network",
                **_population(input_transform="identity"),
            },
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 2}},
        },
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _study(tmp_path, entries=None):
    """A committed study folder holding the recipe sources and manifest."""
    root = tmp_path / "study"
    (root / "sources").mkdir(parents=True)
    source = root / "sources" / "experiment.yaml"
    source.write_text(yaml.safe_dump(_experiment(), sort_keys=False))
    sweep = root / "sources" / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base_config": _experiment(),
                "sweep_config": {"model.decoder.params.output_dim": [2, 3]},
                "filter_config": {},
                "experiment_settings": {"seeds_per_condition": 1, "base_seed": 7},
            },
            sort_keys=False,
        )
    )
    common = {"input_dim": 8, "adoption": {"input_activation": "nonnegative"}}
    if entries is None:
        entries = [
            {
                "id": "identical",
                "format": "collection",
                "sources": [
                    {"path": "sources/experiment.yaml", "sha256": _sha(source)}
                ],
                "runs": 1,
                "check_indices": [0],
                "expected_computation": "identical",
                **common,
            },
            {
                "id": "revised",
                "source": "sources/experiment.yaml",
                "sha256": _sha(source),
                "overrides": {
                    "model.core.population_network.layers.0.populations.0.n_neurons": 4
                },
                "runs": 1,
                "check_indices": [0],
                "expected_computation": "retraining_required",
                **common,
            },
            {
                "id": "sweep",
                "format": "sweep",
                "source": "sources/sweep.yaml",
                "sha256": _sha(sweep),
                "composition_sha256": {"sources/sweep.yaml": _sha(sweep)},
                "runs": 2,
                "check_indices": [1],
                "expected_computation": "identical",
                **common,
            },
        ]
    manifest = root / "recipes.json"
    manifest.write_text(json.dumps({"library_commit": "c" * 40, "entries": entries}))
    git = ["git", "-C", str(root), "-c", "user.name=recipes"]
    git += ["-c", "user.email=recipes@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "recipes"], check=True)
    return manifest


@pytest.fixture
def checked_sources(monkeypatch):
    """Record library-source checks; the test library need not be pinned."""
    calls = []
    monkeypatch.setattr(
        biological_recipes, "check_source", lambda root, commit: calls.append(commit)
    )
    return calls


def _run(manifest, tmp_path, *extra):
    output = tmp_path / "out"
    code = main(
        [
            "--manifest",
            str(manifest),
            "--library-root",
            str(tmp_path / "library"),
            "--output-dir",
            str(output),
            *extra,
        ]
    )
    return code, json.loads((output / "receipt.json").read_text()), output


def test_main_validates_compares_and_materializes_a_study(tmp_path, checked_sources):
    manifest = _study(tmp_path)
    data_root = tmp_path / "data"
    code, receipt, output = _run(
        manifest, tmp_path, "--materialize", "--data-root", str(data_root)
    )
    assert code == 0 and receipt["status"] == "passed"
    assert checked_sources == ["c" * 40, "c" * 40]
    head = subprocess.run(
        ["git", "-C", str(manifest.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert receipt["paper_commit"] == head and receipt["paper_dirty"] is False
    entries = {item["id"]: item for item in receipt["entries"]}
    assert {key: item["count"] for key, item in entries.items()} == {
        "identical": 1,
        "revised": 1,
        "sweep": 2,
    }
    assert not any(item["errors"] for item in entries.values())
    assert entries["identical"]["model_checks"][0]["exact"]
    assert entries["sweep"]["model_checks"][0]["index"] == 1
    revised = entries["revised"]["model_checks"][0]
    assert not revised["exact"] and revised["differences"]
    config = yaml.safe_load(
        (output / "configs" / "identical" / "00000.yaml").read_text()
    )
    run = output / "runs" / "identical" / "0"
    assert config["outputs"]["results_dir"] == str(run)
    assert config["data"]["base_dir"] == str(data_root.resolve())
    assert config["experiment"]["checkpointing"]["checkpoint_dir"] == str(
        run / "checkpoints"
    )
    assert config["model"]["core"]["biological_neuron"] is True
    assert len(list((output / "configs" / "sweep").glob("*.yaml"))) == 2


def test_main_selects_entries_and_records_failed_recipes(tmp_path, checked_sources):
    manifest = _study(tmp_path)
    entries = json.loads(manifest.read_text())["entries"]
    entries[0]["overrides"] = entries[1]["overrides"]
    entries[2]["runs"] = 3
    entries[2]["check_indices"] = [0]
    manifest.write_text(json.dumps({"library_commit": "c" * 40, "entries": entries}))
    code, receipt, _ = _run(
        manifest, tmp_path, "--entry", "identical", "--entry", "sweep"
    )
    assert code == 1 and receipt["status"] == "failed" and receipt["paper_dirty"]
    items = {item["id"]: item for item in receipt["entries"]}
    assert set(items) == {"identical", "sweep"}
    assert "changed diagnostic tensors" in items["identical"]["errors"][0]["error"]
    assert items["sweep"]["errors"] == [{"expected_runs": 3, "validated_runs": 2}]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"runs": 0}, "runs must be a positive integer"),
        ({"check_indices": [5]}, "check_indices"),
        ({"expected_computation": "unknown"}, "expected_computation"),
        ({"id": "../escape"}, "single safe directory names"),
    ],
)
def test_main_refuses_malformed_entries(tmp_path, checked_sources, change, message):
    manifest = _study(tmp_path)
    entries = json.loads(manifest.read_text())["entries"]
    entries[0].update(change)
    manifest.write_text(json.dumps({"library_commit": "c" * 40, "entries": entries}))
    with pytest.raises(ValueError, match=message):
        _run(manifest, tmp_path)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["status"] == "failed"


def test_main_refuses_unknown_entries_outputs_in_the_study_and_changed_sources(
    tmp_path, checked_sources
):
    manifest = _study(tmp_path)
    with pytest.raises(ValueError, match="Unknown entries"):
        _run(manifest, tmp_path, "--entry", "missing")
    with pytest.raises(ValueError, match="outside library and recipe roots"):
        main(
            [
                "--manifest",
                str(manifest),
                "--library-root",
                str(tmp_path / "library"),
                "--output-dir",
                str(manifest.parent / "out"),
            ]
        )
    (manifest.parent / "sources" / "experiment.yaml").write_text("changed: true\n")
    with pytest.raises(ValueError, match="hash changed"):
        main(
            [
                "--manifest",
                str(manifest),
                "--library-root",
                str(tmp_path / "library"),
                "--output-dir",
                str(tmp_path / "second"),
            ]
        )


def test_source_check_requires_the_selected_library_checkout(tmp_path):
    with pytest.raises(ValueError, match="real pinned library Git checkout"):
        check_source(tmp_path, "c" * 40)
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="Imported library differs"):
        check_source(tmp_path, "c" * 40)
