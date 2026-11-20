"""Exercise the browser launch through an authenticated, path-prefixed Jupyter proxy."""

import csv
import json
import math
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import yaml
from playwright.sync_api import expect, sync_playwright


@contextmanager
def personal_server(root):
    root.mkdir()
    runtime = root / "runtime"
    token = secrets.token_urlsafe()
    config = root / "config"
    config.mkdir()
    (config / "jupyter_server_config.json").write_text(
        json.dumps(
            {
                "ServerApp": {
                    "jpserver_extensions": {
                        "dendrinet_binder": True,
                        "jupyter_server_proxy": True,
                    }
                }
            }
        )
    )
    env = {
        **os.environ,
        "JUPYTER_CONFIG_DIR": str(config),
        "JUPYTER_RUNTIME_DIR": str(runtime),
        "JUPYTER_TOKEN": token,
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
        ),
    }
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jupyter_server",
                "--no-browser",
                "--ServerApp.base_url=/user/visitor/",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            info = runtime / f"jpserver-{proc.pid}.json"
            for _ in range(300):
                if info.exists() and info.stat().st_size:
                    break
                if proc.poll() is not None:
                    raise RuntimeError("Personal Jupyter server exited during startup")
                time.sleep(0.1)
            base = json.loads(info.read_text())["url"]
            yield urljoin(base, "open-studio"), token
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def test_personal_session_training_and_exports(tmp_path):
    with personal_server(tmp_path / "first") as (entry, token):
        with sync_playwright() as engine:
            browser = engine.chromium.launch()
            page = browser.new_page(accept_downloads=True)
            response = page.request.get(entry, max_redirects=0)
            assert response.status in (302, 403)
            assert "/proxy/" not in response.headers.get("location", "")
            page.goto(f"{entry}?token={token}")
            page.locator("svg.network .card").first.wait_for()
            address = urlsplit(page.url)
            assert address.path.startswith("/user/visitor/proxy/")
            first_secret = address.fragment.split("&")[0]
            assert first_secret

            page.get_by_role("tab", name="Training").click()
            epochs = page.locator('[data-key="training.main.common.epochs"]')
            epochs.fill("1")
            epochs.press("Enter")
            keys = page.evaluate("Object.keys(localStorage)")
            assert f"dendrinet-studio-draft-v1:{address.path}" in keys
            page.get_by_role("button", name="Train").click()
            run_name = page.locator("dialog.dialog input")
            run_name.fill("Personal example")
            run_name.press("Enter")
            card = page.locator(".run-card", has_text="Personal example")
            expect(card.locator(".status")).to_contain_text(
                "Completed", timeout=180_000
            )
            card.locator("input[type=checkbox]").check()
            with page.expect_download() as curves:
                page.get_by_role("button", name="Curves (CSV)").click()
            with curves.value.path().open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1 and rows[0]["epoch"] == "1"
            for split in ("train", "validation"):
                assert math.isfinite(float(rows[0][f"Personal example {split}"]))
            page.get_by_role("tab", name="YAML").click()
            with page.expect_download() as config:
                page.get_by_role("button", name="Export").click()
            raw = yaml.safe_load(config.value.path().read_text())
            assert raw["training"]["main"]["common"]["epochs"] == 1
            assert raw["data"]["dataset_name"] == "linear_separable"
            reopened = page.request.get(entry, max_redirects=0)
            assert urlsplit(reopened.headers["location"]).fragment == first_secret
            browser.close()

    with (
        personal_server(tmp_path / "second") as (entry, token),
        sync_playwright() as engine,
    ):
        client = engine.request.new_context()
        reply = client.get(f"{entry}?token={token}", max_redirects=0)
        assert reply.status == 302
        address = urlsplit(urljoin(entry, reply.headers["location"]))
        assert address.fragment and address.fragment != first_secret
        runs = address._replace(path=address.path + "api/runs", fragment="").geturl()
        assert client.get(runs, headers={"X-Studio-Token": first_secret}).status == 403
        assert (
            client.get(runs, headers={"X-Studio-Token": address.fragment}).json()[
                "runs"
            ]
            == []
        )
        client.dispose()
