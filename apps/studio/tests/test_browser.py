"""Optional real-browser workflow, including a download-free training epoch."""

from __future__ import annotations

import json
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


def test_visual_edit_export_and_train(tmp_path):
    server = make_server(tmp_path / "runs", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as engine:
            browser = engine.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/#{server.studio_token}")
            page.locator("#metrics strong").first.wait_for()
            assert page.locator("#graph g.node").count() == 2
            assert page.locator('#graph path[marker-end="url(#bar)"]').count() == 1
            page.get_by_role("button", name="Edit layer0.e").click()
            morphology = page.get_by_label("Branching factors (empty = soma only)")
            morphology.fill("2.5")
            morphology.press("Tab")
            page.locator("#export").click()
            playwright.expect(page.locator("#status")).to_have_class("error")
            morphology.fill("3, 2")
            morphology.press("Tab")
            playwright.expect(page.locator("#treeCount")).to_have_text(
                "10 compartments including soma"
            )
            assert page.locator("#tree circle").count() == 9
            page.get_by_label("Neuron count", exact=True).fill("12")
            page.get_by_label("Neuron count", exact=True).press("Tab")
            page.get_by_role("button", name="Export YAML", exact=True).wait_for()
            with page.expect_download() as download:
                page.get_by_role("button", name="Export YAML", exact=True).click()
            exported = yaml.safe_load(Path(download.value.path()).read_text())
            population = exported["model"]["core"]["population_network"]["layers"][0][
                "populations"
            ][1]
            assert population["branch_factors"] == [3, 2]
            assert population["n_neurons"] == 12
            assert (
                exported["training"]["main"]["common"]["param_groups"]["blocklinear_lr"]
                == 0.0001
            )

            # New layer wiring builds through the same core constructor.
            page.locator("#addLayer").click()
            playwright.expect(page.locator("#graph g.node")).to_have_count(4)
            page.locator("#inspect").click()
            until = time.monotonic() + 150
            while time.monotonic() < until:
                rows = server.studio_runs.snapshot()
                if rows and rows[0]["status"] != "running":
                    break
                page.wait_for_timeout(300)
            assert rows[0]["status"] == "completed", rows[0].get("log")

            # Training uses a small real synthetic dataset; preserve seed zero.
            raw = example("synthetic_population")
            raw["experiment"]["model_seed"] = 0
            raw["training"]["main"]["common"]["epochs"] = 1
            raw["metadata"] = {"browser_roundtrip": "preserved"}
            page.get_by_role("button", name="Configuration", exact=True).click()
            page.locator("#yaml").fill(yaml.safe_dump(raw, sort_keys=False))
            page.get_by_role("button", name="Design", exact=True).click()
            page.locator("#train").click()
            playwright.expect(page.locator("#status")).to_have_text(
                "Apply YAML before running."
            )
            assert len(server.studio_runs.snapshot()) == 1
            page.get_by_role("button", name="Configuration", exact=True).click()
            page.locator("#applyYaml").click()
            playwright.expect(page.locator("#status")).to_have_text(
                re.compile("^Config checked")
            )
            page.get_by_role("button", name="Design", exact=True).click()
            page.get_by_role("button", name="Edit layer0.e").click()
            artifact_dir = os.environ.get("STUDIO_BROWSER_ARTIFACTS")
            if artifact_dir:
                destination = Path(artifact_dir)
                destination.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(destination / "design.png"), full_page=True)
            page.locator("#train").click()
            until = time.monotonic() + 180
            while time.monotonic() < until:
                rows = server.studio_runs.snapshot()
                if (
                    rows
                    and rows[0]["kind"] == "train"
                    and rows[0]["status"] != "running"
                ):
                    break
                page.wait_for_timeout(500)
            result = rows[0]
            assert result["kind"] == "train"
            assert result["status"] == "completed", result.get("log")
            assert result.get("summary"), result.get("log")
            page.locator("#refreshRuns").click()
            page.locator("#runList .curve-key").wait_for()
            if artifact_dir:
                page.screenshot(path=str(destination / "results.png"), full_page=True)
                (destination / "training_summary.json").write_text(
                    json.dumps(result["summary"], indent=2)
                )
            saved = yaml.safe_load((Path(result["cwd"]) / "config.yaml").read_text())
            assert saved["metadata"] == raw["metadata"]
            assert saved["experiment"]["model_seed"] == 0
            assert not errors, errors
            browser.close()
    finally:
        server.studio_runs.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
