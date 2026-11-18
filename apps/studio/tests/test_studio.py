"""Studio preserves configs and delegates construction/execution to core."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest
import torch
import yaml
from dendrinet_studio.model import example, inspect_document, parse_document
from dendrinet_studio.server import Runs, make_server

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.conversion import to_plain_dict


def test_yaml_roundtrip_retains_unexposed_settings_and_random_state():
    raw = example("synthetic_population")
    raw["metadata"] = {"study": "my study", "untouched": [3, 5]}
    raw["training"]["main"]["common"]["param_groups"]["blocklinear_lr"] = 0.0037
    torch.manual_seed(739)
    state = torch.get_rng_state().clone()
    result = inspect_document(raw)
    assert torch.equal(state, torch.get_rng_state())
    assert yaml.safe_load(result["yaml"]) == raw
    assert result["resolved_config"] == to_plain_dict(config_from_mapping(raw))
    assert result["nodes"][1]["levels_including_soma"] == [1, 2, 4]
    assert result["nodes"][1]["compartments_per_neuron"] == 7


def test_soma_only_preview_is_distinct_from_one_branch():
    raw = example("synthetic_population")
    raw["model"]["core"]["population_network"]["layers"][0]["populations"][1][
        "branch_factors"
    ] = []
    result = inspect_document(raw)
    assert result["nodes"][1]["levels_including_soma"] == [1]
    assert result["nodes"][1]["compartments_per_neuron"] == 1


def test_preview_keeps_projection_aliases_in_yaml_and_draws_the_resolved_pathway():
    raw = example("synthetic_population")
    core = raw["model"]["core"]
    core["biological_neuron"] = False
    network = core["population_network"]
    network["use_transfer"] = True
    layer = network["layers"][0]
    layer["connections"][0]["source"] = "input_exc"
    layer["connections"][2].update(
        source="input_inh", pathway="ff_excitatory", p_connect=0.8
    )
    preview = inspect_document(raw)
    assert preview["edges"][0]["source_id"] == "input_e"
    edge = preview["edges"][2]
    assert (edge["source_id"], edge["pathway"], edge["probability"]) == (
        "input_i",
        "ff_excitatory",
        0.8,
    )
    assert yaml.safe_load(preview["yaml"]) == raw


def test_invalid_keys_and_unresolved_composition_are_rejected():
    raw = example("synthetic_population")
    raw["model"]["core"]["biological_neuronn"] = True
    with pytest.raises(ValueError):
        inspect_document(raw)
    with pytest.raises(ValueError, match="standalone"):
        parse_document("extends: private.yaml\nmodel: {}")
    with pytest.raises(ValueError, match="Invalid YAML"):
        parse_document("model: [")


def test_server_requires_token_and_serves_only_known_assets(tmp_path):
    prior = tmp_path / "failed"
    prior.mkdir()
    (prior / "studio_run.json").write_text(
        json.dumps({"id": "failed", "status": "failed"})
    )
    results = prior / "results"
    results.mkdir()
    (results / "training_summary.json").write_text(
        json.dumps({"train_losses": [0.5, float("nan"), float("inf")]})
    )
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/api/schema")
        assert error.value.code == 403
        req = urllib.request.Request(
            url + "/api/example/synthetic_population",
            headers={"X-Studio-Token": server.studio_token},
        )
        with urllib.request.urlopen(req) as response:
            data = json.load(response)
        assert data["config"]["config_version"] == 4
        invalid = urllib.request.Request(
            url + "/api/preview",
            data=json.dumps({"yaml": "model: ["}).encode(),
            headers={"X-Studio-Token": server.studio_token},
        )
        with pytest.raises(urllib.error.HTTPError) as invalid_reply:
            urllib.request.urlopen(invalid)
        assert invalid_reply.value.code == 400
        assert "Invalid YAML" in json.load(invalid_reply.value)["error"]
        request = urllib.request.Request(
            url + "/api/runs", headers={"X-Studio-Token": server.studio_token}
        )
        with urllib.request.urlopen(request) as response:
            record = json.load(response)[0]
        assert record["status"] == "failed"
        assert record["summary"]["train_losses"] == [0.5, None, None]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/../../pyproject.toml")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_construction_job_uses_imported_core_and_reports_actual_counts(
    tmp_path, monkeypatch
):
    import time

    monkeypatch.setattr(
        "dendrinet_studio.server.socket.gethostname", lambda: "workstation"
    )
    runs = Runs(tmp_path)
    try:
        row = runs.start(example("synthetic_population"), kind="inspect", device="cpu")
        with pytest.raises(ValueError, match="already running"):
            runs.start(example("synthetic_population"), kind="train", device="cpu")
        until = time.monotonic() + 120
        while time.monotonic() < until:
            result = runs.snapshot()[0]
            if result["status"] != "running":
                break
            time.sleep(0.2)
        assert result["status"] == "completed", result.get("log")
        assert result["resources"]["active_synapses"] > 0
        assert (
            result["resources"]["total_parameters"]
            >= result["resources"]["trainable_parameters"]
        )
        saved = json.loads((tmp_path / row["id"] / "studio_run.json").read_text())
        assert saved["config_sha256"] == row["config_sha256"]
        assert saved["exit_code"] == 0
    finally:
        runs.close()


def test_login_node_execution_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dendrinet_studio.server.socket.gethostname", lambda: "mylogin01"
    )
    with pytest.raises(ValueError, match="login node"):
        Runs(tmp_path).start(
            example("synthetic_population"), kind="train", device="cpu"
        )


@pytest.mark.parametrize("explicit_core_width", [False, True])
def test_inspection_matches_training_with_inferred_input_width(
    tmp_path, explicit_core_width
):
    raw = example("synthetic_population")
    del raw["model"]["encoder"]["params"]["input_dim"]
    network = raw["model"]["core"]["population_network"]
    if not explicit_core_width:
        del network["input_dim"]
    defaults = network["layers"][0]["population_defaults"]
    defaults["topk_type"] = "standard"
    for key in list(defaults):
        if key.startswith("indexed_"):
            del defaults[key]
    raw["training"]["main"]["common"]["epochs"] = 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    inspected = tmp_path / "resources.json"
    subprocess.run(
        [sys.executable, "-m", "dendrinet_studio.worker", str(config), str(inspected)],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dendritic_modeling.scripts.training.train_experiments",
            str(config),
            "--output_dir",
            str(tmp_path / "training"),
        ],
        cwd=tmp_path,
        check=True,
    )
    summary = json.loads(inspected.read_text())
    (trained_path,) = (tmp_path / "training").rglob("model_resources.json")
    trained = json.loads(trained_path.read_text())
    assert summary["input_shape"] == [16]
    for key in (
        "total_parameters",
        "trainable_parameters",
        "candidate_synapse_slots",
        "active_synapses",
    ):
        assert summary[key] == trained[key]


def test_unfinished_old_record_is_not_reported_as_success(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    (run / "studio_run.json").write_text(json.dumps({"id": "old", "status": "running"}))
    assert Runs(tmp_path).snapshot()[0]["status"] == "unknown after server restart"
