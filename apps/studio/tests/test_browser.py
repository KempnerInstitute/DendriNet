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


def open_design(page, raw):
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    playwright.expect(page.locator(".yaml-status")).to_contain_text(
        "Matches the current design"
    )
    page.get_by_role("tab", name="Model").click()


def test_imported_file_keeps_its_omissions_and_draws_every_pathway(page):
    open_design(page, two_layer_document())
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
    layer["populations"][1].setdefault("population", {})[
        "ff_inhibitory_synapses_by_level"
    ] = [0, 0, 4]
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
    page.get_by_role("dialog").get_by_role("button", name="Add layer").click()
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
    layer = net["layers"][0]
    # A second somatic source keeps the receiver buildable when the external one is off.
    layer["populations"].append({**layer["populations"][0], "name": "backup_i"})
    routes = layer["connections"]
    routes.extend(
        [
            {"source": "input", "target": "backup_i"},
            {"source": "backup_i", "target": "e", "target_levels": ["soma"]},
        ]
    )
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


def wire(page, receiver, source, timing="same_step"):
    page.get_by_role("button", name="Connect", exact=True).click()
    dialog = page.get_by_role("dialog", name="Connect populations")
    dialog.get_by_label("Receiving population").select_option(receiver)
    dialog.get_by_label("Timing", exact=True).select_option(timing)
    dialog.get_by_label("Source", exact=True).select_option(source)
    return dialog


def test_add_a_second_feedforward_excitatory_source_and_build_the_export(page):
    import torch

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    raw = two_layer_document()
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Edit population layer0.e", exact=True).click()
    page.get_by_role("button", name="E population", exact=True).first.click()
    page.get_by_role("button", name="Edit population layer0.e2").wait_for()
    target = page.get_by_role("button", name="Edit population layer1.e", exact=True)
    target.click()
    source_row = page.get_by_role(
        "region", name="Synapse sources per neuron"
    ).get_by_role("row", name=re.compile(r"^E "))
    original_budget = source_row.get_by_role("cell").last.inner_text()
    page.get_by_role("button", name="Add input source").click()
    dialog = page.get_by_role("dialog", name="Connect populations")
    playwright.expect(dialog.get_by_label("Receiving population")).to_have_value(
        "layer1.e"
    )
    dialog.get_by_label("Source", exact=True).select_option("layer0.e2")
    playwright.expect(dialog.get_by_role("status")).to_contain_text("layer0.e.")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "layer0.e2", "layer1.e")).to_have_attribute(
        "data-pathway", "ff_excitatory"
    )
    playwright.expect(
        page.get_by_text("Source delay: 0 steps.", exact=False)
    ).to_be_visible()
    target.click()
    playwright.expect(
        source_row.get_by_role("button", name="layer0.e", exact=True)
    ).to_be_visible()
    playwright.expect(
        source_row.get_by_role("button", name="layer0.e2", exact=True)
    ).to_be_visible()
    assert source_row.get_by_role("cell").last.inner_text() == original_budget
    saved = exported(page)
    model, _ = initialize_model(config_from_mapping(saved).model)
    first, second = model.core_network.layers
    assert not first.is_recurrent and not second.is_recurrent
    assert second._stream_sources("e", "same_step", "ff_excitatory") == [
        "layer0.e",
        "layer0.e2",
    ]
    received = []
    handle = second.populations["e"].register_forward_pre_hook(
        lambda module, inputs: received.append(inputs[0])
    )
    model.core_network(torch.ones(2, 16))
    handle.remove()
    expected = torch.cat([first._last_outputs["e"], first._last_outputs["e2"]], dim=-1)
    torch.testing.assert_close(received[0], expected, rtol=0, atol=0)
    assert not page.errors


def test_response_times_and_route_delay_survive_visual_edits_and_export(page):
    import torch

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    raw = example("recurrent_decision_population")
    layer = raw["model"]["core"]["population_network"]["layers"][0]
    layer["dt"] = 0.5
    layer["connections"][-2]["target_levels"] = ["distal"]
    layer["populations"][0]["population"] = {
        "rec_excitatory_synapses_by_level": [2, 0, 0]
    }
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    loop = page.get_by_role(
        "button",
        name="Connection circuit.pyr to circuit.pyr, rec_excitatory, delayed",
        exact=True,
    )
    loop.press("Enter")
    table = page.get_by_role("table", name="Response times by depth")
    playwright.expect(
        page.get_by_text("Source delay: 1 step.", exact=False)
    ).to_be_visible()
    playwright.expect(table.locator("tbody tr")).to_have_count(1)
    playwright.expect(table.get_by_role("cell")).to_have_text(["1", "40"])
    page.get_by_role("button", name="Edit receiving population").click()
    soma = table.get_by_role("spinbutton", name="Response time at soma")
    primary = table.get_by_role("spinbutton", name="Response time at 1 · primary")
    distal = table.get_by_role("spinbutton", name="Response time at 2 · distal")
    for control, value in [(soma, "10"), (primary, "20"), (distal, "40")]:
        playwright.expect(control).to_have_value(value)
    distal.fill("55")
    distal.press("Enter")
    page.get_by_role("button", name="Use base and ratio").wait_for()
    primary.fill("17.5")
    primary.press("Enter")
    page.get_by_label("Learn one time constant per depth").click()
    playwright.expect(
        table.get_by_role("columnheader", name="Initial τ (learned)")
    ).to_be_visible()
    result = exported(page)
    layer = result["model"]["core"]["population_network"]["layers"][0]
    assert layer["populations"][0]["population"]["level_taus"] == [55, 17.5, 10]
    assert layer["dt"] == 0.5 and layer["connections"][-2]["timing"] == "delayed"
    model, _ = initialize_model(config_from_mapping(result).model)
    pyr = model.core_network.layers[0].populations["pyr"]
    torch.testing.assert_close(
        pyr.current_taus.detach(), torch.tensor([55.0, 17.5, 10.0])
    )
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Use base and ratio").click()
    playwright.expect(distal).to_have_value("40")
    assert (
        "level_taus"
        not in exported(page)["model"]["core"]["population_network"]["layers"][0][
            "populations"
        ][0]["population"]
    )
    assert not page.errors


def test_set_travel_delays_in_the_connection_editor(page):
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    page.locator(".example", has_text="Recurrent decision population network").click()
    loop = page.get_by_role(
        "button",
        name="Connection circuit.pyr to circuit.pyr, rec_excitatory, delayed",
        exact=True,
    )
    loop.press("Enter")
    page.get_by_role("button", name="Edit connection", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit connection")
    dialog.get_by_label("Travel delay", exact=True).select_option("levels")
    dialog.get_by_label("Travel steps at 1 · primary").fill("2")
    dialog.get_by_label("Travel steps at 2 · distal").fill("3")
    dialog.get_by_role("button", name="Save connection").click()
    table = page.get_by_role("table", name="Response times by depth")
    playwright.expect(table.locator("tbody td")).to_have_text(
        ["1", "10", "2", "20", "3", "40"]
    )
    playwright.expect(loop.locator("title")).to_contain_text(
        "1\N{EN DASH}3 steps of travel"
    )
    selected = page.locator('.route-row[aria-current="true"] .meta')
    playwright.expect(selected).to_have_text("E · 1\N{EN DASH}3 Δt")
    raw = exported(page)
    route = raw["model"]["core"]["population_network"]["layers"][0]["connections"][-2]
    assert route["delay_steps_by_level"] == [3, 2, 1] and "delay_steps" not in route
    model, _ = initialize_model(config_from_mapping(raw).model)
    assert model.core_network.layers[0].delay_history_lengths == {"pyr": 2}
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Edit receiving population").click()
    page.get_by_role("button", name="level", exact=True).click()
    playwright.expect(table.locator("tbody tr")).to_have_count(4)
    expanded = exported(page)["model"]["core"]["population_network"]["layers"][0]
    assert expanded["connections"][-2]["delay_steps_by_level"] == [1, 3, 2, 1]
    page.get_by_role("tab", name="Model").click()
    presets = page.get_by_role("group", name="Morphology presets")
    presets.get_by_text("[2,2]", exact=True).click()
    playwright.expect(table.locator("tbody tr")).to_have_count(3)
    loop.press("Enter")
    page.get_by_role("button", name="Edit connection", exact=True).click()
    dialog.get_by_label("Travel delay", exact=True).select_option("uniform")
    dialog.get_by_label("Travel delay (steps)").fill("4")
    dialog.get_by_role("button", name="Save connection").click()
    playwright.expect(table.locator("tbody td")).to_have_text(
        ["4", "10", "4", "20", "4", "40"]
    )
    saved = exported(page)["model"]["core"]["population_network"]["layers"][0][
        "connections"
    ][-2]
    assert saved["delay_steps"] == 4 and "delay_steps_by_level" not in saved
    assert not page.errors


def test_create_rewire_and_undo_without_invalidating_the_document(page):
    # The published Add Connection button immediately duplicated input -> e.
    dialog = wire(page, "layer0.e", "input")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(dialog.get_by_role("alert")).to_contain_text("duplicate")
    dialog.get_by_role("button", name="Cancel").click()
    playwright.expect(page.locator("svg.network path.edge[data-source]")).to_have_count(
        3
    )
    playwright.expect(page.locator(".banner")).to_be_hidden()
    page.get_by_role("button", name="E population", exact=True).first.click()
    playwright.expect(
        page.get_by_role("button", name="Edit population layer0.e2")
    ).to_be_visible()
    dialog = wire(page, "layer0.e", "layer0.e2")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "layer0.e2", "layer0.e")).to_have_attribute(
        "data-pathway", "ff_excitatory"
    )
    page.get_by_role("button", name="Edit connection", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit connection")
    dialog.get_by_label("Timing", exact=True).select_option("delayed")
    dialog.get_by_label("Contacts per eligible compartment").fill("2")
    dialog.get_by_role("button", name="Save connection").click()
    playwright.expect(edge(page, "layer0.e2", "layer0.e")).to_have_attribute(
        "data-pathway", "rec_excitatory"
    )
    result = exported(page)["model"]["core"]["population_network"]["layers"][0]
    assert (
        result["recurrent"]
        and result["populations"][1]["population"]["rec_excitatory_synapses"] == 2
    )
    assert result["connections"][-1] == {
        "source": "e2",
        "target": "e",
        "timing": "delayed",
    }
    page.get_by_role("tab", name="Model").click()
    page.keyboard.press("Control+z")
    playwright.expect(edge(page, "layer0.e2", "layer0.e")).to_have_attribute(
        "data-pathway", "ff_excitatory"
    )
    assert not page.errors


def test_connecting_both_external_signs_selects_biological_afferents(page):
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    dialog = wire(page, "layer0.e", "input_i")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(dialog.get_by_role("alert")).to_contain_text("afferent")
    dialog.get_by_label("External afferents for both signs").select_option("paired")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "input_i", "layer0.e")).to_have_attribute(
        "data-pathway", "ff_inhibitory"
    )
    cfg = config_from_mapping(exported(page))
    assert (
        cfg.model.core.population_network["transfer_params"]["input_source_identity"]
        == "paired_afferents"
    )
    initialize_model(cfg.model)


def test_hybrid_graph_runs_with_forward_and_delayed_feedback(page, tmp_path):
    import torch

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment

    page.get_by_role("button", name="Layer", exact=True).click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_label("Layer name").fill("memory")
    dialog.get_by_label("Layer dynamics").select_option("recurrent")
    dialog.get_by_role("button", name="Add layer").click()
    loop = edge(page, "memory.e", "memory.e")
    playwright.expect(loop).to_have_attribute("marker-end", "url(#m-exc)")
    # The second layer may receive first-layer outputs now and send its
    # previous inhibitory output back, even if the sending layer is later.
    dialog = wire(page, "memory.e", "layer0.i")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "layer0.i", "memory.e")).to_have_attribute(
        "data-pathway", "ff_inhibitory"
    )
    dialog = wire(page, "layer0.e", "memory.i", "delayed")
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "memory.i", "layer0.e")).to_have_attribute(
        "data-pathway", "rec_inhibitory"
    )
    page.get_by_role("button", name="Layer", exact=True).click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_label("Layer name").fill("output")
    dialog.get_by_role("button", name="Add layer").click()
    playwright.expect(
        page.get_by_role("button", name="Edit population output.e")
    ).to_be_visible()
    raw = exported(page)
    net = raw["model"]["core"]["population_network"]
    assert [layer.get("recurrent", False) for layer in net["layers"]] == [
        True,
        True,
        False,
    ]
    assert net["layers"][0]["connections"][-1]["source"] == "memory.i"
    assert net["layers"][1]["connections"][-1]["source"] == "layer0.i"
    raw["outputs"]["results_dir"] = str(tmp_path)
    cfg = config_from_mapping(raw)
    _, _, _, _, model, _ = setup_environment(
        cfg.model, cfg.training, cfg.data, cfg.outputs, cfg.experiment, is_main=False
    )
    x = torch.linspace(0, 1, 3 * 4 * 16).reshape(3, 4, 16).requires_grad_()
    core = model.core_network
    y, hidden = core(x, return_hidden=True)
    assert hidden is not None and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[:, :-1].abs().sum() > 0
    assert not page.errors


def test_canvas_feedback_and_sequence_training_in_a_mixed_network(page, studio):
    import math

    raw = example("recurrent_decision_population")
    task = raw["data"]["dataset_params"][raw["data"]["dataset_name"]]
    task["n_samples"] = 96
    raw["training"]["main"]["common"].update(epochs=1, batch_size=16)
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply", exact=True).click()
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Layer", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Add layer").click()
    card = page.get_by_role("button", name="Edit population layer1.pv")
    card.click()
    page.get_by_role("button", name="Connect from this population").click()
    page.get_by_role("button", name="Edit population circuit.pyr").click()
    dialog = page.get_by_role("dialog")
    playwright.expect(dialog.get_by_label("Source", exact=True)).to_have_value(
        "layer1.pv"
    )
    playwright.expect(dialog.get_by_label("Timing", exact=True)).to_have_value(
        "delayed"
    )
    dialog.get_by_role("button", name="Add connection").click()
    playwright.expect(edge(page, "layer1.pv", "circuit.pyr")).to_have_attribute(
        "data-timing", "delayed"
    )
    page.get_by_role("button", name="Train", exact=True).click()
    name = page.get_by_role("dialog").locator("input")
    name.fill("mixed sequence")
    name.press("Enter")
    card = page.locator(".run-card", has_text="mixed sequence")
    playwright.expect(card.locator(".status")).to_contain_text(
        "Completed", timeout=240_000
    )
    run = next(r for r in studio.studio_runs.snapshot()["runs"] if r["kind"] == "train")
    (loss,) = run["summary"]["train_losses"]
    assert math.isfinite(loss)
    saved = yaml.safe_load((Path(run["cwd"]) / "config.yaml").read_text())
    layers = saved["model"]["core"]["population_network"]["layers"]
    assert [layer.get("recurrent", False) for layer in layers] == [True, False]
    assert layers[0]["connections"][-1] == {
        "source": "layer1.pv",
        "target": "pyr",
        "timing": "delayed",
    }
    assert not page.errors


def test_adding_populations_and_changing_depth_keep_imported_layer_defaults(page):
    raw = two_layer_document()
    net = raw["model"]["core"]["population_network"]
    net["readout_layer"] = "layer0"
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    page.get_by_role("button", name="Edit population layer0.e").click()
    page.get_by_role("button", name="E population", exact=True).first.click()
    card = page.get_by_role("button", name="Edit population layer0.e2")
    playwright.expect(card).to_be_visible()
    page.locator(".presets .chip", has_text="[3,3,3]").click()
    playwright.expect(card).to_contain_text("[3, 3, 3]")
    page.get_by_role("button", name="Layer", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Add layer").click()
    playwright.expect(
        page.get_by_role("button", name="Edit population layer2.e")
    ).to_be_visible()
    net = exported(page)["model"]["core"]["population_network"]
    assert net["readout_layer"] == "layer0"
    assert net["layers"][0]["population_defaults"][
        "ff_inhibitory_synapses_by_level"
    ] == [4, 4, 6]
    new = net["layers"][0]["populations"][-1]["population"]
    assert new["ff_inhibitory_synapses_by_level"] == [4, 4, 4, 6]
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


def test_readout_selection_updates_both_explicit_readouts(page):
    raw = example("synthetic_population")
    raw["model"]["core"]["population_network"]["readout_population"] = "e"
    page.get_by_role("tab", name="YAML").click()
    page.locator("textarea.yaml-text").fill(yaml.safe_dump(raw, sort_keys=False))
    page.get_by_role("button", name="Apply").click()
    page.get_by_role("tab", name="Model").click()
    page.locator(".tree-row.layer-row").first.click()
    page.get_by_label("Readout population", exact=True).select_option("i")
    playwright.expect(
        page.get_by_role("button", name="Edit population layer0.i")
    ).to_contain_text("readout")
    net = exported(page)["model"]["core"]["population_network"]
    assert net["readout_population"] == net["layers"][0]["readout_population"] == "i"
    assert not page.errors


def test_cancel_while_checking_a_route_keeps_draft_and_history(page):
    dialog = wire(page, "layer0.e", "layer0.e", "delayed")
    delay_next_response(page, "preview")
    dialog.get_by_role("button", name="Add connection").click()
    page.wait_for_selector("html[data-response-ready]")
    dialog.get_by_role("button", name="Cancel").click()
    release_response(page)
    playwright.expect(edge(page, "layer0.e", "layer0.e")).to_have_count(0)
    before = example("synthetic_population")
    assert exported(page) == before
    assert not page.errors


def release_response(page):
    # Let the promise handlers settle before checking the resulting UI.
    page.evaluate("""async () => {
        window.resumeStudioResponse();
        delete window.resumeStudioResponse;
        delete document.documentElement.dataset.responseReady;
        await new Promise(done => setTimeout(done, 0));
    }""")


def test_zero_budget_add_layer_omits_inactive_inhibition(page):
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    raw = example("synthetic_population")
    first = raw["model"]["core"]["population_network"]["layers"][0]
    first["population_defaults"]["ff_inhibitory_synapses"] = 0
    open_design(page, raw)
    page.get_by_role("button", name="Layer", exact=True).click()
    dialog = page.get_by_role("dialog", name="Add layer")
    playwright.expect(dialog).to_contain_text("no eligible inhibitory contacts")
    dialog.get_by_role("button", name="Add layer").click()
    playwright.expect(page.locator("svg.network .card")).to_have_count(4)
    assert edge(page, "layer1.i", "layer1.e").count() == 0
    result = exported(page)
    new = result["model"]["core"]["population_network"]["layers"][1]
    assert new["population_defaults"]["ff_inhibitory_synapses"] == 0
    assert new["connections"] == [
        {"source": "input", "target": "i"},
        {"source": "input", "target": "e"},
    ]
    initialize_model(config_from_mapping(result).model)
    assert not page.errors


def test_add_layer_rejects_broadcast_credit_for_stepped_template(page):
    raw = example("synthetic_population")
    layer = raw["model"]["core"]["population_network"]["layers"][0]
    layer["population_defaults"]["autograd_credit_mode"] = "soma_broadcast"
    open_design(page, raw)
    page.get_by_role("button", name="Layer", exact=True).click()
    dialog = page.get_by_role("dialog", name="Add layer")
    dialog.get_by_label("Layer dynamics").select_option("recurrent")
    dialog.get_by_role("button", name="Add layer").click()
    playwright.expect(dialog.get_by_role("alert")).to_contain_text("exact")
    assert page.locator("svg.network .card").count() == 2
    dialog.get_by_role("button", name="Cancel").click()
    assert exported(page) == raw
    assert not page.errors


def test_retimed_restricted_route_only_funds_the_selected_depth(page):
    import torch

    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    raw = two_layer_document()
    open_design(page, raw)
    route = page.get_by_role(
        "button",
        name="Connection layer0.i to layer1.e, ff_inhibitory, same_step",
        exact=True,
    )
    route.press("Enter")
    page.get_by_role("switch", name="All levels").click()
    page.locator("#lvl-0").uncheck()
    page.locator("#lvl-1").uncheck()
    page.get_by_role("button", name="Edit connection", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit connection")
    dialog.get_by_label("Timing", exact=True).select_option("delayed")
    dialog.get_by_label("Contacts per eligible compartment").fill("3")
    dialog.get_by_role("button", name="Save connection").click()
    playwright.expect(edge(page, "layer0.i", "layer1.e")).to_have_attribute(
        "data-pathway", "rec_inhibitory"
    )
    result = exported(page)
    layer = result["model"]["core"]["population_network"]["layers"][1]
    saved = next(c for c in layer["connections"] if c.get("source") == "layer0.i")
    assert saved["target_levels"] == [2] and saved["timing"] == "delayed"
    assert layer["populations"][1]["population"][
        "rec_inhibitory_synapses_by_level"
    ] == [3, 0, 0]
    model, _ = initialize_model(config_from_mapping(result).model)
    assert torch.isfinite(model.core_network(torch.ones(2, 5, 16))).all()
    assert not page.errors


@pytest.mark.parametrize("field", ["response time", "contact count"])
def test_pending_depth_edit_survives_adding_a_morphology_level(page, field):
    page.locator(".example", has_text="Recurrent decision population network").click()
    page.get_by_role("button", name="Edit population circuit.pyr", exact=True).click()
    if field == "response time":
        label, value, key, expected = (
            "Response time at 1 · primary",
            "25",
            "level_taus",
            [80, 40, 25, 10],
        )
    else:
        page.get_by_text("Set counts per level", exact=True).click()
        label, value, key, expected = (
            "Recurrent excitatory contacts at level 1 · primary",
            "3",
            "rec_excitatory_synapses_by_level",
            [2, 2, 3, 2],
        )

    delay_next_response(page, "preview")
    control = page.get_by_role("spinbutton", name=label, exact=True)
    control.fill(value)
    control.press("Enter")
    page.wait_for_selector("html[data-response-ready]")
    page.get_by_role("button", name="level", exact=True).click()
    release_response(page)
    table = page.get_by_role("table", name="Response times by depth")
    playwright.expect(table.locator("tbody tr")).to_have_count(4)
    saved = exported(page)["model"]["core"]["population_network"]["layers"][0][
        "populations"
    ][0]
    assert saved["population"][key] == expected
    assert not page.errors


def test_queued_edits_to_connection_controls_survive_in_order(page):
    page.locator(".example", has_text="Recurrent decision population network").click()
    page.get_by_role(
        "button",
        name="Connection circuit.pyr to circuit.pyr, rec_excitatory, delayed",
        exact=True,
    ).press("Enter")
    delay_next_response(page, "preview")
    probability = page.get_by_label("Connection probability", exact=True)
    probability.fill("0.4")
    probability.press("Enter")
    page.wait_for_selector("html[data-response-ready]")
    seed = page.get_by_label("Seed", exact=True)
    seed.fill("27")
    seed.press("Enter")
    release_response(page)
    # Wait for the validated draft itself, independent of the values typed into the form.
    page.wait_for_function(
        """async () => {
        const {state} = await import('./js/store.js');
        const c = state.config.model.core.population_network.layers[0].connections.at(-2);
        return c.probability === 0.4 && c.seed === 27;
    }""",
        timeout=5000,
    )
    page.get_by_role("button", name="Undo (Ctrl+Z)", exact=True).click()
    playwright.expect(seed).to_have_value("")
    saved = exported(page)["model"]["core"]["population_network"]["layers"][0][
        "connections"
    ][-2]
    assert saved["probability"] == 0.4 and "seed" not in saved
    assert not page.errors


def test_stale_topology_dialog_waits_for_validation_then_rejects_later_edits(page):
    page.get_by_role("button", name="Edit population layer0.e", exact=True).click()
    delay_next_response(page, "preview")
    contacts = page.locator(
        'input[data-key$="populations.1.population.ff_inhibitory_synapses"]'
    )
    contacts.fill("0")
    contacts.press("Enter")
    page.wait_for_selector("html[data-response-ready]")
    page.get_by_role("button", name="Layer", exact=True).click()
    dialog = page.get_by_role("dialog", name="Add layer")
    assert not dialog.count()
    release_response(page)
    playwright.expect(dialog).to_contain_text("no eligible inhibitory contacts")
    page.evaluate("""async () => {
        const {edit} = await import('./js/store.js');
        await edit(cfg => { cfg.experiment.seed = 753; });
    }""")
    dialog.get_by_role("button", name="Add layer").click()
    playwright.expect(dialog.get_by_role("alert")).to_contain_text("design changed")
    dialog.get_by_role("button", name="Cancel").click()
    saved = exported(page)
    assert saved["experiment"]["seed"] == 753
    assert len(saved["model"]["core"]["population_network"]["layers"]) == 1
    assert not page.errors


def test_queued_edits_rebase_morphology_on_a_later_field_edit(page):
    page.locator(".example", has_text="Recurrent decision population network").click()
    page.get_by_role("button", name="Edit population circuit.pyr", exact=True).click()
    delay_next_response(page, "preview")
    add = page.get_by_role("button", name="More branches at level 1", exact=True)
    add.click()
    page.wait_for_selector("html[data-response-ready]")
    add.click()
    primary = page.get_by_role("spinbutton", name="Response time at 1 · primary")
    primary.fill("25")
    primary.press("Enter")
    release_response(page)
    playwright.expect(
        page.get_by_role("button", name="Edit population circuit.pyr")
    ).to_contain_text("[4, 2]")
    saved = exported(page)["model"]["core"]["population_network"]["layers"][0][
        "populations"
    ][0]
    assert saved["population"]["level_taus"] == [40, 25, 10]
    assert not page.errors


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
    label = page.evaluate("""async () => {
          const { drawNeuron } = await import("./js/morphology.js");
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          drawNeuron(svg, {counts: {ff_excitatory: [2000000]}});
          return [svg.querySelector(".count-label").textContent, svg.querySelectorAll(".bouton").length];
        }""")
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
    assert page.evaluate("""async () => {
        const { state } = await import("./js/store.js");
        return !state.validating && !state.error && JSON.stringify(state.config) === JSON.stringify(state.preview.config);
    }""")
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
    page.evaluate("""async () => {
        const { edit } = await import("./js/store.js");
        await edit(config => { config.unknown_option = true; });
    }""")
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


@pytest.mark.parametrize("inherited", [False, True])
def test_uniform_reset_preserves_later_depth_edit(page, inherited):
    raw = example("synthetic_population")
    layer = raw["model"]["core"]["population_network"]["layers"][0]
    field = "ff_inhibitory_synapses_by_level"
    settings = (
        layer["population_defaults"]
        if inherited
        else layer["populations"][1].setdefault("population", {})
    )
    settings[field] = [2, 3, 6]
    if inherited:
        # The inhibitory population has a shallower tree than the receiver.
        layer["populations"][0].setdefault("population", {})[field] = []
    open_design(page, raw)
    page.get_by_role("button", name="Edit population layer0.e", exact=True).click()
    page.get_by_text("Set counts per level", exact=True).click()
    delay_next_response(page, "preview")
    page.get_by_role("button", name="Use one count for all levels").click()
    page.wait_for_selector("html[data-response-ready]")
    primary = page.get_by_role(
        "spinbutton", name="Inhibitory contacts at level 1 · primary"
    )
    primary.fill("7")
    primary.press("Enter")
    release_response(page)
    page.evaluate("async () => (await import('./js/store.js')).topologyReady()")
    saved = exported(page)["model"]["core"]["population_network"]["layers"][0]
    assert saved["populations"][1]["population"][field] == [4, 7, 4]
    undo = page.get_by_role("button", name="Undo (Ctrl+Z)", exact=True)
    undo.click()
    rolled_back = exported(page)["model"]["core"]["population_network"]["layers"][0]
    assert rolled_back["populations"][1].get("population", {}).get(field) == (
        [] if inherited else None
    )
    undo.click()
    assert exported(page) == raw
    assert not page.errors


@pytest.mark.parametrize("primary_source", [False, True])
def test_retimed_route_requires_coverage_for_existing_contacts(page, primary_source):
    from dendritic_modeling.config import config_from_mapping
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    raw = two_layer_document()
    layer = raw["model"]["core"]["population_network"]["layers"][1]
    layer["connections"][2]["target_levels"] = ["distal"]
    field = "rec_inhibitory_synapses_by_level"
    layer["populations"][1].setdefault("population", {})[field] = [0, 2, 0]
    if primary_source:
        layer["recurrent"] = True
        layer["connections"].append(
            {
                "source": "i",
                "target": "e",
                "timing": "delayed",
                "target_levels": ["primary"],
            }
        )
    open_design(page, raw)
    page.get_by_role(
        "button",
        name="Connection layer0.i to layer1.e, ff_inhibitory, same_step",
        exact=True,
    ).press("Enter")
    page.get_by_role("button", name="Edit connection", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit connection")
    dialog.get_by_label("Timing", exact=True).select_option("delayed")
    dialog.get_by_label("Contacts per eligible compartment").fill("1")
    dialog.get_by_role("button", name="Save connection").click()
    if primary_source:
        playwright.expect(dialog).not_to_be_visible()
        result = exported(page)
        saved = result["model"]["core"]["population_network"]["layers"][1]
        assert saved["populations"][1]["population"][field] == [1, 2, 0]
        initialize_model(config_from_mapping(result).model)
    else:
        playwright.expect(dialog.get_by_role("alert")).to_contain_text("depth 1")
        dialog.get_by_role("button", name="Cancel").click()
        assert exported(page) == raw
    assert not page.errors
