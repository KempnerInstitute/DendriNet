"""Studio preserves configurations and delegates all model semantics to DendriNet."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
import yaml
from dendrinet_studio.model import (
    dump_document,
    example,
    example_catalog,
    inspect_document,
    parse_document,
    parse_value,
)
from dendrinet_studio.runs import Runs, progress_from_log
from dendrinet_studio.schema import choices_for, describe_config_schema
from dendrinet_studio.server import _bind, make_server

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.examples import example_names
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

REPOSITORY = Path(__file__).resolve().parents[3]


# Inspection -------------------------------------------------------------------------


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


def _two_layer_document():
    raw = example("synthetic_population")
    layers = raw["model"]["core"]["population_network"]["layers"]
    first = layers[0]
    del first["connections"][0]["source"]
    del first["populations"][0]["branch_factors"]
    first["population_defaults"]["ff_inhibitory_synapses_by_level"] = [4, 4, 6]
    first["populations"][0]["population"]["ff_inhibitory_synapses_by_level"] = []
    second = yaml.safe_load(yaml.safe_dump(first))
    second["name"] = "layer1"
    second["connections"] = [
        {"target": "i"},
        {"source": "layer0.e", "target": "e"},
        {"source": "layer0.i", "target": "e"},
        {"source": "i", "target": "e"},
    ]
    layers.append(second)
    return raw


@pytest.mark.parametrize(
    "document",
    [
        lambda: example("synthetic_population"),
        _two_layer_document,
        lambda: example("recurrent_decision_population"),
        lambda: example("adding_problem_population"),
    ],
)
def test_configured_contacts_equal_the_synapses_of_the_built_model(document):
    """The drawing's counts are the builder's, including delayed pathways."""
    raw = document()
    preview = inspect_document(raw)
    model, _ = initialize_model(config_from_mapping(raw).model)
    assert preview["synapses_total"] == model_resource_summary(model)["active_synapses"]


def test_invalid_keys_and_unresolved_composition_are_rejected():
    raw = example("synthetic_population")
    raw["model"]["core"]["biological_neuronn"] = True
    with pytest.raises(ValueError):
        inspect_document(raw)
    with pytest.raises(ValueError, match="standalone"):
        parse_document("extends: private.yaml\nmodel: {}")
    with pytest.raises(ValueError, match=r"Invalid YAML \(line 2\)"):
        parse_document("model:\n  task: [")
    assert parse_value("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        parse_value("x: .nan")
    assert yaml.safe_load(dump_document(raw)) == raw


@pytest.mark.parametrize("name", example_names())
def test_every_installed_example_is_described(name):
    preview = inspect_document(example(name))
    assert preview["family"]
    if preview["family"] == "population_network":
        assert preview["nodes"] and preview["edges"]
        assert preview["synapses_total"] > 0
    else:
        assert preview["overview"]["decoder"]


def test_example_catalog_lists_the_tutorial_first_and_marks_downloads():
    catalog = example_catalog()
    assert catalog[0]["name"] == "synthetic_population"
    assert {item["name"] for item in catalog} == set(example_names())
    by_name = {item["name"]: item for item in catalog}
    assert by_name["mnist_population"]["download"] is True
    assert by_name["synthetic_population"]["download"] is False
    assert by_name["adding_problem_gru"]["task"] == "regression"
    assert by_name["recurrent_decision_population"]["recurrent"] is True
    assert all(item["title"] and item["description"] for item in catalog)


def _shipped_experiment_configs():
    folder = REPOSITORY / "configs" / "examples"
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if isinstance(raw, dict) and "model" in raw and "extends" not in raw:
            found.append(path)
    return found


@pytest.mark.requires_git_checkout
@pytest.mark.parametrize(
    "path", _shipped_experiment_configs(), ids=lambda path: path.stem
)
def test_shipped_experiment_configs_open_in_the_studio(path):
    """Every standalone example of the repository can be opened and drawn."""
    raw = parse_document(path.read_text())
    preview = inspect_document(raw)
    assert yaml.safe_load(preview["yaml"]) == raw


# Schema ------------------------------------------------------------------------------


def test_schema_reads_options_and_choices_from_the_library():
    schema = describe_config_schema()
    json.dumps(schema, allow_nan=False)
    common = schema["sections"]["training.main.common"]
    names = {field["name"]: field for field in common["fields"]}
    assert names["epochs"]["kind"] == "int"
    assert "ce" in names["loss_function"]["choices"]
    core_types = choices_for("model.core.type", "type")
    assert {"population_network", "gru", "dendritic_shunting"} <= set(core_types)
    assert schema["core_families"]["gru"] == "baseline_rnn"
    assert schema["core_families"]["dendritic_shunting"] is None
    assert "classification" in schema["loss_tasks"]["ce"]
    assert schema["loss_tasks"]["mse"] == []
    population = {f["name"]: f for f in schema["payloads"]["population"]["fields"]}
    assert "xavier_normal" in population["topk_init_method"]["choices"]
    assert population["use_shunting"]["kind"] == "bool"
    rnn = {f["name"]: f for f in schema["payloads"]["baseline_rnn"]["fields"]}
    assert {"gru", "lstm"} <= set(rnn["cell_type"]["choices"])
    assert any(
        "linear_separable" in names for names in schema["dataset_groups"].values()
    )


# Server --------------------------------------------------------------------------------


class Studio:
    def __init__(self, root, **options):
        self.server = make_server(root, 0, **options)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path, data=None, *, token=True, host=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Studio-Token"] = self.server.studio_token
        if host:
            headers["Host"] = host
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            kind = response.headers.get("Content-Type", "")
            return (json.loads(raw) if "json" in kind else raw), response.headers

    def close(self):
        self.server.studio_runs.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def studio(tmp_path):
    server = Studio(tmp_path)
    yield server
    server.close()


def test_server_requires_token_and_a_local_host(studio):
    with pytest.raises(urllib.error.HTTPError) as error:
        studio.request("/api/schema", token=False)
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error:
        studio.request("/api/schema", host="attacker.example")
    assert error.value.code == 403
    schema, _ = studio.request("/api/schema")
    assert schema["examples"][0]["name"] == "synthetic_population"
    assert schema["execution"] is True


def test_server_serves_the_client_with_strict_headers(studio):
    page, headers = studio.request("/", token=False)
    assert b'src="js/app.js"' in page
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    _, headers = studio.request("/js/app.js", token=False)
    assert headers["Content-Type"].startswith("text/javascript")
    with pytest.raises(urllib.error.HTTPError) as error:
        studio.request("/../../pyproject.toml", token=False)
    assert error.value.code == 404


def test_preview_parse_and_dump_endpoints(studio):
    preview, _ = studio.request("/api/example/adding_problem_gru")
    assert preview["family"] == "gru" and preview["task"] == "regression"
    with pytest.raises(urllib.error.HTTPError) as error:
        studio.request("/api/preview", {"yaml": "model:\n  task: ["})
    assert error.value.code == 400
    assert "line 2" in json.load(error.value)["error"]
    value, _ = studio.request("/api/parse", {"yaml": "{a: 1, b: [2]}"})
    assert value["value"] == {"a": 1, "b": [2]}
    dumped, _ = studio.request("/api/dump", {"config": {"b": 1, "a": 2}})
    assert dumped["yaml"].startswith("b: 1")


def test_run_records_are_read_tolerantly_and_values_stay_honest(tmp_path):
    failed = tmp_path / "failed"
    (failed / "results").mkdir(parents=True)
    (failed / "studio_run.json").write_text(
        json.dumps({"id": "failed", "status": "failed"})
    )
    (failed / "results" / "training_summary.json").write_text(
        json.dumps({"train_losses": [0.5, float("nan"), float("inf")]})
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "studio_run.json").write_text("{not json")
    old = tmp_path / "old"
    old.mkdir()
    (old / "studio_run.json").write_text(json.dumps({"id": "old", "status": "running"}))
    server = Studio(tmp_path)
    try:
        snapshot, _ = server.request("/api/runs")
    finally:
        server.close()
    rows = {row["id"]: row for row in snapshot["runs"]}
    assert rows["failed"]["summary"]["train_losses"] == [0.5, None, None]
    assert rows["old"]["status"] == "unknown after server restart"
    assert snapshot["problems"] and "broken" in snapshot["problems"][0]["path"]
    assert (broken / "studio_run.json").read_text() == "{not json"


def test_design_only_mode_never_starts_a_worker(tmp_path):
    server = Studio(tmp_path, allow_execution=False)
    try:
        schema, _ = server.request("/api/schema")
        assert schema["execution"] is False
        with pytest.raises(urllib.error.HTTPError) as error:
            server.request(
                "/api/run",
                {"config": example("synthetic_population"), "kind": "inspect"},
            )
        assert "--no-execution" in json.load(error.value)["error"]
    finally:
        server.close()
    assert not list(tmp_path.glob("*/studio_run.json"))


def test_a_taken_port_falls_back_to_the_next_free_one(tmp_path):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        taken = blocker.getsockname()[1]
        server = _bind(tmp_path, taken)
        try:
            assert server.server_port != taken
        finally:
            server.studio_runs.close()
            server.server_close()


def test_progress_lines_become_partial_curves():
    log = (
        "INFO [Standard] epoch=10/30, train_loss=0.6836 (base=0.6836, reg=0.0), "
        "valid_loss=0.7087, elapsed=0.3s\n"
        "INFO [Standard] epoch=20/30, train_loss=nan (base=nan), valid_loss=0.7, x\n"
    )
    progress = progress_from_log(log)
    assert (progress["epoch"], progress["epochs"]) == (20, 30)
    assert progress["points"][0] == {
        "epoch": 10,
        "train_loss": 0.6836,
        "valid_loss": 0.7087,
    }
    assert progress["points"][1]["train_loss"] is None
    assert progress_from_log("no progress here") is None


def test_run_configurations_are_read_only_for_known_runs(tmp_path):
    runs = Runs(tmp_path)
    with pytest.raises(ValueError, match="No run"):
        runs.config("../../etc")


# Workers ------------------------------------------------------------------------------


def _wait(runs, key, seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        row = next(r for r in runs.snapshot()["runs"] if r["id"] == key)
        if row["status"] != "running":
            return row
        time.sleep(0.25)
    raise AssertionError(f"run {key} did not finish in {seconds} s")


@pytest.mark.slow
def test_model_check_builds_with_the_imported_core_and_reports_counts(tmp_path):
    runs = Runs(tmp_path, threads=2)
    try:
        row = runs.start(example("synthetic_population"), kind="inspect", device="cpu")
        with pytest.raises(ValueError, match="already running"):
            runs.start(example("synthetic_population"), kind="train", device="cpu")
        result = _wait(runs, row["id"], 180)
        assert result["status"] == "completed", result.get("log")
        assert result["resources"]["active_synapses"] == 1472
        assert result["resources"]["trainable_parameters"] == 1986
        saved = json.loads((tmp_path / row["id"] / "studio_run.json").read_text())
        assert saved["config_sha256"] == row["config_sha256"]
        assert saved["exit_code"] == 0
        assert runs.config(row["id"]) == example("synthetic_population")
    finally:
        runs.close()


@pytest.mark.slow
def test_a_training_run_can_be_named_annotated_and_stopped(tmp_path):
    raw = example("synthetic_population")
    raw["training"]["main"]["common"]["epochs"] = 500
    runs = Runs(tmp_path, threads=1)
    try:
        row = runs.start(raw, kind="train", device="cpu", name="long run")
        runs.annotate(row["id"], notes="stopped by the test")
        time.sleep(1.0)
        runs.stop(row["id"])
        result = next(r for r in runs.snapshot()["runs"] if r["id"] == row["id"])
        assert result["status"] == "stopped"
        assert (result["name"], result["notes"]) == ("long run", "stopped by the test")
        assert not runs.busy()
    finally:
        runs.close()


@pytest.mark.slow
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
