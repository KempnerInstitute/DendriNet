"""Real-browser workflows: import, visual edits, export, model check and training."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import pytest
import yaml

playwright = pytest.importorskip("playwright.sync_api")

from dendrinet_studio.model import example  # noqa: E402
from dendrinet_studio.server import make_server  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def studio(tmp_path):
    server = make_server(tmp_path / "runs", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.studio_runs.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def page(studio):
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100}, accept_downloads=True
        )
        tab = context.new_page()
        tab.errors = []
        tab.on("pageerror", lambda error: tab.errors.append(str(error)))
        # A rejected configuration is an HTTP 400 reply, which the browser logs.
        tab.on(
            "console",
            lambda message: (
                tab.errors.append(message.text)
                if message.type == "error"
                and "Failed to load resource" not in message.text
                else None
            ),
        )
        tab.goto(f"http://127.0.0.1:{studio.server_port}/#{studio.studio_token}")
        tab.locator("svg.network .card").first.wait_for()
        yield tab
        browser.close()


def exported(page):
    page.get_by_role("tab", name="YAML").click()
    with page.expect_download() as event:
        page.get_by_role("button", name="Export").click()
    return yaml.safe_load(Path(event.value.path()).read_text())


def edge(page, source, target):
    return page.locator(
        f'svg.network path.edge[data-source="{source}"][data-target="{target}"]'
    )


def two_layer_document():
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


def test_imported_file_keeps_its_omissions_and_draws_every_pathway(page):
    raw = two_layer_document()
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    playwright.expect(page.locator("svg.network .card")).to_have_count(4)
    # Input reaching layer1 is the readout of layer0; inhibition ends in a bar.
    playwright.expect(edge(page, "layer0.e", "layer1.i")).to_have_count(1)
    inhibitory = edge(page, "layer0.i", "layer1.e")
    playwright.expect(inhibitory).to_have_attribute("marker-end", "url(#m-inh)")
    playwright.expect(inhibitory).to_have_attribute("data-pathway", "ff_inhibitory")
    # The neuron table shows the per-level inhibitory counts from the layer defaults.
    page.get_by_role("button", name="Edit population layer0.e").click()
    rows = page.locator("table.levels tbody tr")
    playwright.expect(rows.first).to_contain_text("4")
    edited = exported(page)
    first = edited["model"]["core"]["population_network"]["layers"][0]
    assert "source" not in first["connections"][0]
    assert "branch_factors" not in first["populations"][0]
    assert first["population_defaults"]["ff_inhibitory_synapses_by_level"] == [4, 4, 6]
    assert not page.errors, page.errors


@pytest.mark.parametrize("second_enabled", [True, False])
def test_converging_inhibitory_connections_show_the_target_budget_once(
    page, second_enabled
):
    raw = example("synthetic_population")
    layer = raw["model"]["core"]["population_network"]["layers"][0]
    layer["populations"].append({**layer["populations"][0], "name": "i2"})
    layer["connections"].extend(
        [
            {"source": "input", "target": "i2"},
            {"source": "i2", "target": "e", "enabled": second_enabled},
        ]
    )
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()

    target = page.get_by_role("button", name="Edit population layer0.e")
    labels = page.locator("svg.network .chip-edge.inh text")
    if second_enabled:
        playwright.expect(target).to_contain_text("I 4 shared")
        playwright.expect(labels).to_have_text(["I", "I"])
    else:
        playwright.expect(target).not_to_contain_text("shared")
        playwright.expect(labels).to_have_text(["I 4/comp", "I · off"])


def test_soma_only_projection_does_not_label_every_branch_with_soma_contacts(page):
    raw = example("synthetic_population")
    layer = raw["model"]["core"]["population_network"]["layers"][0]
    layer["connections"][-1]["target_levels"] = ["soma"]
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()

    card = page.get_by_role("button", name="Edit population layer0.e")
    playwright.expect(card).to_contain_text("I by level")
    playwright.expect(page.locator("svg.network .chip-edge.inh text")).to_have_text(
        "I 4/comp · → soma"
    )


def test_visual_edits_undo_and_export(page):
    page.get_by_role("button", name="Edit population layer0.e").click()
    page.locator(".presets .chip", has_text="[3,3,3]").click()
    playwright.expect(page.locator(".inspector")).to_contain_text(
        "40 compartments per neuron"
    )
    neurons = page.locator('[data-key$="populations.1.n_neurons"]')
    neurons.fill("12")
    neurons.press("Enter")
    playwright.expect(page.locator(".summary-strip")).to_contain_text("20")
    page.get_by_role("radio", name="Additive").first.click()
    playwright.expect(page.locator(".equation").first).to_contain_text(
        "V = E \u2212 I + C"
    )
    page.keyboard.press("Control+z")
    playwright.expect(page.locator(".equation").first).to_contain_text("Shunting")
    edited = exported(page)
    population = edited["model"]["core"]["population_network"]["layers"][0][
        "populations"
    ][1]
    assert population["branch_factors"] == [3, 3, 3]
    assert population["n_neurons"] == 12
    assert "use_shunting" not in population.get("population", {})
    # A new layer is wired from the previous layer's readout and still builds.
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Layer", exact=True).click()
    playwright.expect(page.locator("svg.network .card")).to_have_count(4)
    playwright.expect(page.locator(".status.ok")).to_contain_text("Valid")
    # An invalid edit is reported and the drawing keeps the last valid design.
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill("model:\n  task: [")
    page.get_by_role("button", name="Apply").click()
    playwright.expect(page.locator(".yaml-status.error")).to_contain_text("line 2")
    assert not page.errors, page.errors


def test_other_families_open_with_their_own_forms(page):
    for title, family in (
        ("GRU baseline for regression", "GRU"),
        ("Structured dendritic tree network", "dendritic shunting"),
    ):
        page.locator(".example", has_text=title).click()
        dialog = page.locator("dialog.dialog")
        if dialog.count():
            dialog.get_by_role("button", name="Open example").click()
        playwright.expect(page.locator(".detail-panel h3")).to_contain_text(family)
    page.get_by_role("tab", name="Data & task").click()
    playwright.expect(
        page.get_by_role("radio", name="Classification")
    ).to_have_attribute("aria-checked", "true")
    page.get_by_role("tab", name="Training").click()
    playwright.expect(page.locator("#view-training .form-page")).to_contain_text(
        "Initialization"
    )
    assert not page.errors, page.errors


def test_check_model_then_train_and_compare(page, studio):
    page.get_by_role("button", name="Check model").click()
    playwright.expect(page.locator(".topbar .status")).to_contain_text(
        "model checked", timeout=180_000
    )
    playwright.expect(page.locator(".summary-strip")).to_contain_text("1,986")
    page.get_by_role("tab", name="Training").click()
    epochs = page.locator('[data-key="training.main.common.epochs"]')
    epochs.fill("2")
    epochs.press("Enter")
    page.get_by_role("button", name="Train").click()
    name = page.locator("dialog.dialog input")
    name.fill("two epochs")
    name.press("Enter")
    card = page.locator(".run-card", has_text="two epochs")
    playwright.expect(card.locator(".status")).to_contain_text(
        "Completed", timeout=300_000
    )
    card.locator("input[type=checkbox]").check()
    playwright.expect(page.locator("#loss-chart svg.chart")).to_have_count(1)
    playwright.expect(page.locator("#metric-table")).to_contain_text("Accuracy")
    rows = [r for r in studio.studio_runs.snapshot()["runs"] if r["kind"] == "train"]
    assert (
        rows[0]["status"] == "completed"
        and len(rows[0]["summary"]["train_losses"]) == 2
    )
    saved = yaml.safe_load((Path(rows[0]["cwd"]) / "config.yaml").read_text())
    assert saved["training"]["main"]["common"]["epochs"] == 2
    artifacts = os.environ.get("STUDIO_BROWSER_ARTIFACTS")
    if artifacts:
        Path(artifacts).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(artifacts) / "runs.png"), full_page=False)
    assert not page.errors, page.errors
    time.sleep(0.1)


@pytest.mark.parametrize("mode, keep_e", [("paired", True), ("split", False)])
def test_external_routes_are_per_target_and_build_with_core(page, mode, keep_e):
    if not keep_e:
        raw = example("synthetic_population")
        layer = raw["model"]["core"]["population_network"]["layers"][0]
        layer["populations"].insert(
            0,
            {
                "name": "relay",
                "polarity": "excitatory",
                "n_neurons": 12,
                "branch_factors": [1],
            },
        )
        layer["connections"].extend(
            [{"source": "input", "target": "relay"}, {"source": "relay", "target": "e"}]
        )
        page.get_by_role("tab", name="YAML").click()
        page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
        page.get_by_role("button", name="Apply").click()
        page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Edit population layer0.e").click()
    table = page.get_by_role("region", name="Synapse sources per neuron")
    rows = table.locator("tbody tr")
    playwright.expect(rows.nth(0)).to_contain_text("External input")
    playwright.expect(rows.nth(0)).to_contain_text("soma + depths 1\u20132")
    playwright.expect(rows.nth(0).locator("td").last).to_have_text("56")
    playwright.expect(rows.nth(1)).to_contain_text("layer0.i")
    playwright.expect(rows.nth(1).locator("td").last).to_have_text("28")
    page.get_by_role("switch", name="Direct external I").check()
    warning = page.get_by_text(
        "Both signs currently read shared input channels.", exact=False
    )
    playwright.expect(warning).to_be_visible()
    page.get_by_label("External afferents for the first layer").select_option(mode)
    playwright.expect(warning).to_have_count(0)
    playwright.expect(rows.nth(1)).to_contain_text("External input_i")
    playwright.expect(rows.nth(1).locator("td").last).to_have_text("28")
    if keep_e:
        playwright.expect(
            page.get_by_role("switch", name="Direct external E")
        ).to_be_disabled()
    else:
        page.get_by_role("switch", name="Direct external E").uncheck()
        playwright.expect(rows.nth(0)).to_contain_text("(off)")
        playwright.expect(rows.nth(0)).to_contain_text("layer0.relay")
        playwright.expect(rows.nth(0).locator("td").last).to_have_text("56")
    page.get_by_role("button", name="Check model").click()
    playwright.expect(page.locator(".topbar .status")).to_contain_text(
        "model checked", timeout=180_000
    )
    net = exported(page)["model"]["core"]["population_network"]
    assert net["use_transfer"] is True
    params = net["transfer_params"]
    assert params["independent_pathways"] is (mode == "split")
    assert params["input_source_identity"] == (
        "shared" if mode == "split" else "paired_afferents"
    )
    edges = net["layers"][0]["connections"]
    assert edges[0] == {"source": "input", "target": "i"}
    assert edges[1].get("enabled", True) is keep_e
    assert edges[2] == {"source": "i", "target": "e"}
    assert edges[-1] == {"source": "input_i", "target": "e"}
    assert not page.errors


def test_external_route_switch_preserves_custom_placement(page):
    raw = example("synthetic_population")
    net = raw["model"]["core"]["population_network"]
    net["use_transfer"] = True
    net["transfer_params"] = {
        "input_source_identity": "paired_afferents",
        "output_activation": "relu",
    }
    routes = net["layers"][0]["connections"]
    routes.append(
        {"source": "input_inh", "target": "e", "target_levels": ["soma"], "seed": 11}
    )
    routes[2]["target_levels"] = [1, 2]
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Edit population layer0.e").click()
    table = page.get_by_role("region", name="Synapse sources per neuron")
    playwright.expect(table).to_contain_text("layer0.i · depths 1\u20132")
    playwright.expect(table).to_contain_text("External input_i · soma")
    page.get_by_role("switch", name="Direct external I").uncheck()
    result = exported(page)["model"]["core"]["population_network"]
    routes[-1]["enabled"] = False
    assert result == net


def test_structured_feedforward_sources_follow_presence_of_inhibitory_cells(page):
    raw = example("structured_tree_synthetic")
    for local_inhibition in (True, False):
        core = raw["model"]["core"]
        if not local_inhibition:
            core["architecture"]["inhibitory_layer_sizes"] = []
            core["connectivity"]["ei_synapses_per_branch_per_layer"] = [0]
            core["transfer"] = {"independent_pathways": True}
        page.get_by_role("tab", name="YAML").click()
        page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
        page.get_by_role("button", name="Apply").click()
        page.get_by_role("tab", name="Model").click()
        rows = page.get_by_role(
            "region", name="First-layer feedforward sources"
        ).locator("tbody tr")
        playwright.expect(rows).to_have_count(4 if local_inhibition else 2)
        playwright.expect(rows.nth(1).locator("td")).to_have_text(
            [
                "E cells",
                "I",
                "Local I cells" if local_inhibition else "External I stream",
                "2",
            ]
        )
    assert not page.errors


def delay_next_response(page, endpoint):
    """Deliver one real API response after a second UI action has taken place."""
    page.evaluate(
        """endpoint => {
          const fetch = window.fetch.bind(window);
          window.fetch = async (...args) => {
            if (new URL(args[0], location.href).pathname !== "/api/" + endpoint) return fetch(...args);
            window.fetch = fetch;
            const response = await fetch(...args);
            const json = response.json.bind(response);
            response.json = async () => {
              const payload = await json();
              await new Promise(resolve => {
                window.resumeStudioResponse = resolve;
                document.documentElement.dataset.responseReady = "yes";
              });
              return payload;
            };
            return response;
          };
        }""",
        endpoint,
    )


def release_response(page):
    # Let the promise handlers settle before checking the resulting UI.
    page.evaluate(
        """async () => {
        window.resumeStudioResponse();
        delete window.resumeStudioResponse;
        delete document.documentElement.dataset.responseReady;
        await new Promise(done => setTimeout(done, 0));
    }"""
    )


def test_morphology_drawing_is_bounded_and_keeps_the_real_level_counts(page):
    raw = example("synthetic_population")
    target = raw["model"]["core"]["population_network"]["layers"][0]["populations"][1]
    for factors in ([3] * 4, [4] * 5, [6] * 8, [6] * 25, []):
        target["branch_factors"] = factors
        page.get_by_role("tab", name="YAML").click()
        page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
        page.get_by_role("button", name="Apply").click()
        playwright.expect(page.locator(".yaml-status")).to_contain_text("Matches")
        page.get_by_role("tab", name="Model").click()
        neuron = page.locator("svg.neuron")
        levels = [1]
        for daughters in factors:
            levels.append(levels[-1] * daughters)
        total = sum(levels)
        drawn = 1 + neuron.locator(".dendrite").count()
        assert drawn <= 160
        table = page.locator("table.levels")
        playwright.expect(
            table.locator("tbody tr").first.locator("td").nth(1)
        ).to_have_text(f"{levels[-1]:,}")
        playwright.expect(
            table.locator("tfoot tr").first.locator("th").nth(1)
        ).to_have_text(f"{total:,}")
        if factors in ([3] * 4, []):
            assert drawn == total
        else:
            assert drawn < total
            playwright.expect(neuron).to_have_attribute(
                "aria-label", re.compile(f"{drawn} of {total:,} compartments")
            )
    label = page.evaluate(
        """async () => {
          const { drawNeuron } = await import("./js/morphology.js");
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          drawNeuron(svg, {counts: {ff_excitatory: [2000000]}});
          return [svg.querySelector(".count-label").textContent, svg.querySelectorAll(".bouton").length];
        }"""
    )
    assert label == ["2000000 E each", 0]
    assert not page.errors, page.errors


@pytest.mark.parametrize("invalid", [False, True])
def test_adopting_an_example_supersedes_inflight_model_validation(page, invalid):
    delay_next_response(page, "preview")
    page.evaluate(
        """async invalid => {
        const { edit } = await import("./js/store.js");
        void edit(config => {
          if (invalid) config.unknown_option = true;
          else config.experiment.seed = 102;
        });
    }""",
        invalid,
    )
    page.locator("html[data-response-ready]").wait_for()
    page.locator(".example", has_text="GRU baseline for regression").click()
    page.get_by_role("button", name="Open example").click()
    playwright.expect(page.locator(".detail-panel h3")).to_contain_text("GRU")
    playwright.expect(
        page.get_by_role("button", name="Train", exact=True)
    ).to_be_enabled()
    release_response(page)
    assert page.evaluate(
        """async () => {
        const { state } = await import("./js/store.js");
        return !state.validating && !state.error && JSON.stringify(state.config) === JSON.stringify(state.preview.config);
    }"""
    )
    assert not page.errors, page.errors


def test_restored_design_survives_repeated_reloads_until_discarded(page):
    page.get_by_role("button", name="Edit population layer0.e").click()
    neurons = page.locator('[data-key$="populations.1.n_neurons"]')
    neurons.fill("12")
    neurons.press("Enter")
    for _ in range(2):
        page.reload()
        page.get_by_role("button", name="Restore", exact=True).click()
        playwright.expect(neurons).to_have_value("12")
        assert (
            page.evaluate(
                "JSON.parse(localStorage.getItem('dendrinet-studio-draft-v1')).config.model.core.population_network.layers[0].populations[1].n_neurons"
            )
            == 12
        )
    page.reload()
    page.get_by_role("button", name="Discard", exact=True).click()
    playwright.expect(neurons).to_have_value("16")
    assert page.evaluate("localStorage.getItem('dendrinet-studio-draft-v1')") is None


def test_prompt_confirms_on_enter_and_preserves_cancel_and_escape(page):
    doc = page.locator("button.doc-name")
    original = doc.inner_text()
    for action in ("cancel", "Escape", "Enter"):
        doc.click()
        dialog = page.get_by_role("dialog", name="Rename this design")
        field = dialog.get_by_role("textbox")
        field.fill("updated design")
        if action == "cancel":
            dialog.get_by_role("button", name="Cancel").click()
        else:
            field.press(action)
        playwright.expect(dialog).to_have_count(0)
        playwright.expect(doc).to_have_text(
            "updated design.yaml" if action == "Enter" else original
        )


@pytest.mark.parametrize("intervening_action", ["type", "adopt"])
def test_delayed_yaml_dump_preserves_newer_text_or_document(page, intervening_action):
    page.get_by_role("tab", name="YAML").click()
    editor = page.locator("textarea.yaml-text")
    delay_next_response(page, "dump")
    page.evaluate(
        """async () => {
        const { edit } = await import("./js/store.js");
        await edit(config => { config.unknown_option = true; });
    }"""
    )
    page.locator("html[data-response-ready]").wait_for()
    if intervening_action == "type":
        expected = "# keep my edit\n" + editor.input_value()
        editor.fill(expected)
    else:
        page.get_by_role("tab", name="Model").click()
        page.locator(".example", has_text="GRU baseline for regression").click()
        page.get_by_role("button", name="Open example").click()
        playwright.expect(page.locator(".detail-panel h3")).to_contain_text("GRU")
        page.get_by_role("tab", name="YAML").click()
        expected = editor.input_value()
        assert "gru" in expected.lower()
    release_response(page)
    playwright.expect(editor).to_have_value(expected)


@pytest.mark.parametrize("invalid_submission", [False, True])
def test_typing_during_apply_preserves_the_new_yaml(page, invalid_submission):
    page.get_by_role("tab", name="YAML").click()
    editor = page.locator("textarea.yaml-text")
    raw = yaml.safe_load(editor.input_value())
    original_seed = raw["experiment"]["seed"]
    raw["experiment"]["seed"] = 101
    submitted = "model: [" if invalid_submission else yaml.safe_dump(raw)
    editor.fill(submitted)
    delay_next_response(page, "preview")
    page.get_by_role("button", name="Apply").click()
    page.locator("html[data-response-ready]").wait_for()
    raw["experiment"]["seed"] = 102
    new_text = yaml.safe_dump(raw)
    editor.fill(new_text)
    release_response(page)
    playwright.expect(editor).to_have_value(new_text)
    playwright.expect(page.locator(".yaml-status")).to_contain_text("Edited")
    read_seed = (
        'async () => (await import("./js/store.js")).state.config.experiment.seed'
    )
    assert page.evaluate(read_seed) == original_seed
    page.get_by_role("button", name="Apply").click()
    playwright.expect(page.locator(".yaml-status")).to_contain_text("Matches")
    assert page.evaluate(read_seed) == 102
