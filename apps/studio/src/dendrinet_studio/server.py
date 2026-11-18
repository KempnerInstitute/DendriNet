"""Serve DendriNet Studio locally, with explicit subprocess-based execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from dendrinet_studio.model import example, inspect_document, parse_document, schema

import dendritic_modeling

MAX_REQUEST_BYTES = 2 * 1024 * 1024


class Runs:
    """One local worker at a time; never execute a user-supplied shell command."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen] = {}
        self.records: dict[str, dict] = {}
        self.lock = threading.RLock()
        for record in sorted(self.root.glob("*/studio_run.json")):
            data = json.loads(record.read_text())
            if data.get("status") == "running":
                data["status"] = "unknown after server restart"
            self.records[record.parent.name] = data

    def _write(self, key: str) -> None:
        path = self.root / key / "studio_run.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.records[key], indent=2), encoding="utf-8")
        temporary.replace(path)

    def refresh(self) -> None:
        for key, process in self.processes.items():
            result = process.poll()
            if result is not None and self.records[key]["status"] == "running":
                self.records[key].update(
                    status="completed" if result == 0 else "failed",
                    exit_code=result,
                    finished=time.time(),
                )
                self._write(key)

    def start(self, raw: dict, *, kind: str, device: str) -> dict:
        if "login" in socket.gethostname().lower():
            raise ValueError(
                "Run construction/training on a workstation or allocated compute node, not a login node"
            )
        if kind not in {"inspect", "train"} or device not in {"cpu", "auto"}:
            raise ValueError("Unknown execution kind or device")
        preview = inspect_document(raw)
        if raw.get("distributed", {}).get("mode", "none") != "none":
            raise ValueError(
                "Studio launches single-process runs; export distributed configs for torchrun"
            )
        with self.lock:
            self.refresh()
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError("A worker is already running; wait or stop it first")
            key = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            folder = self.root / key
            folder.mkdir()
            config = folder / "config.yaml"
            config.write_text(preview["yaml"], encoding="utf-8")
            if kind == "inspect":
                command = [
                    sys.executable,
                    "-m",
                    "dendrinet_studio.worker",
                    str(config),
                    str(folder / "resources.json"),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "dendritic_modeling.scripts.training.train_experiments",
                    str(config),
                    "--output_dir",
                    str(folder / "results"),
                    "--run_name",
                    "model",
                ]
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1")
            if device == "cpu" or kind == "inspect":
                env["CUDA_VISIBLE_DEVICES"] = ""
            # Reuse exactly the imported package paths, even with a source
            # checkout instead of an editable install. No implicit fallback.
            roots = [
                str(Path(dendritic_modeling.__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1]),
            ]
            env["PYTHONPATH"] = os.pathsep.join(roots)
            record = {
                "id": key,
                "kind": kind,
                "status": "running",
                "started": time.time(),
                "device": device,
                "command": command,
                "cwd": str(folder),
                "python": sys.executable,
                "core_version": dendritic_modeling.__version__,
                "core_path": dendritic_modeling.__file__,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "overrides": (
                    {"output_dir": str(folder / "results"), "run_name": "model"}
                    if kind == "train"
                    else {}
                ),
            }
            self.records[key] = record
            try:
                with (folder / "worker.log").open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=folder,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
            except OSError as error:
                record.update(status="failed to start", error=str(error))
                self._write(key)
                raise
            self.processes[key] = process
            self._write(key)
            return dict(record)

    def stop(self, key: str) -> None:
        with self.lock:
            process = self.processes.get(key)
            if process is None or process.poll() is not None:
                raise ValueError("No live worker owned by this Studio session")
            # Terminate data-loader children too, without touching other runs.
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
            self.records[key].update(
                status="stopped", exit_code=process.returncode, finished=time.time()
            )
            self._write(key)

    def snapshot(self) -> list[dict]:
        with self.lock:
            self.refresh()
            rows = []
            for key, record in reversed(list(self.records.items())):
                folder = self.root / key
                row = dict(record)
                log = folder / "worker.log"
                if log.is_file():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 20000))
                        row["log"] = stream.read().decode("utf-8", errors="replace")
                for field, paths in (
                    ("resources", [folder / "resources.json"]),
                    (
                        "summary",
                        list((folder / "results").glob("**/training_summary.json")),
                    ),
                ):
                    for path in paths:
                        if path.is_file():
                            try:
                                row[field] = json.loads(path.read_text())
                            except json.JSONDecodeError:
                                pass  # The worker may still be writing it.
                rows.append(row)
            return rows

    def close(self) -> None:
        for key, process in list(self.processes.items()):
            if process.poll() is None:
                self.stop(key)


def make_server(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)
    runs = Runs(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log the access token or uploaded configurations.

        def reply(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def json_reply(self, data, status=200):
            self.reply(
                status, json.dumps(data).encode(), "application/json; charset=utf-8"
            )

        def authorized(self) -> bool:
            supplied = self.headers.get("X-Studio-Token", "")
            return secrets.compare_digest(supplied, token)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                if not self.authorized():
                    self.json_reply(
                        {"error": "Open the Studio URL printed by the launcher"}, 403
                    )
                    return
                if path == "/api/schema":
                    self.json_reply(schema())
                elif path == "/api/runs":
                    self.json_reply(runs.snapshot())
                elif path.startswith("/api/example/"):
                    try:
                        self.json_reply(
                            inspect_document(example(path.rsplit("/", 1)[1]))
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        self.json_reply({"error": str(error)}, 400)
                else:
                    self.json_reply({"error": "Unknown endpoint"}, 404)
                return
            name = "index.html" if path == "/" else path.lstrip("/")
            if name not in {"index.html", "app.js", "style.css"}:
                self.reply(404, b"Not found", "text/plain")
                return
            asset = files("dendrinet_studio").joinpath("web", name)
            self.reply(
                200,
                asset.read_bytes(),
                mimetypes.guess_type(name)[0] or "application/octet-stream",
            )

        def do_POST(self):
            if not self.authorized():
                self.json_reply({"error": "Invalid Studio session"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Request must be between 1 byte and 2 MB")
                data = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/preview":
                    raw = (
                        parse_document(data["yaml"])
                        if "yaml" in data
                        else data["config"]
                    )
                    self.json_reply(inspect_document(raw))
                elif path == "/api/run":
                    self.json_reply(
                        runs.start(
                            data["config"],
                            kind=data["kind"],
                            device=data.get("device", "cpu"),
                        )
                    )
                elif path == "/api/stop":
                    runs.stop(data["id"])
                    self.json_reply({"stopped": data["id"]})
                else:
                    self.json_reply({"error": "Unknown endpoint"}, 404)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.json_reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.studio_token = token
    server.studio_runs = runs
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=Path("studio-runs"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.workspace, args.port)
    url = f"http://127.0.0.1:{server.server_port}/#{server.studio_token}"
    print(f"DendriNet Studio: {url}\nRuns: {args.workspace.resolve()}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.studio_runs.close()
        server.server_close()


if __name__ == "__main__":
    main()
