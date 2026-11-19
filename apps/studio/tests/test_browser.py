"""Real-browser workflows: import, visual edits, export, model check and training."""

from __future__ import annotations

import os
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
    page.locator("dialog.dialog input").fill("two epochs")
    page.get_by_role("button", name="Start training").click()
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
    playwright.expect(rows.nth(0).locator("td").last).to_have_text("56")
    playwright.expect(rows.nth(1)).to_contain_text("layer0.i")
    playwright.expect(rows.nth(1).locator("td").last).to_have_text("28")
    page.get_by_label("External afferents for the first layer").select_option(mode)
    page.get_by_role("switch", name="Direct external I").check()
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
    page.get_by_role("switch", name="Direct external I").uncheck()
    result = exported(page)["model"]["core"]["population_network"]
    routes[-1]["enabled"] = False
    assert result == net
