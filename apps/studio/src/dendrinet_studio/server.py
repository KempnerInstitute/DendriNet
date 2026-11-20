"""Serve DendriNet Studio on this computer and open it in the browser.

The server listens on the loopback interface only. Every API request carries a
random session token that the launcher prints as part of the URL, so other
local web pages cannot drive it. Configurations are validated by DendriNet and
runs are ordinary ``dendritic-train`` processes recorded in a workspace folder.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import secrets
import signal
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dendrinet_studio import __version__ as studio_version
from dendrinet_studio.runs import Runs, default_threads, json_values

MAX_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_PORT = 8765
# Explicit types: the platform's MIME registry can map .js to text/plain, which
# a browser refuses to execute under ``nosniff``.
ASSET_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def _assets() -> dict[str, bytes]:
    """The web client's files, read once from the installed package."""
    web = files("dendrinet_studio").joinpath("web")
    found = {}

    def collect(folder, prefix=""):
        for item in folder.iterdir():
            name = prefix + item.name
            if item.is_dir():
                collect(item, name + "/")
            elif Path(item.name).suffix in ASSET_TYPES:
                found[name] = item.read_bytes()

    collect(web)
    return found


def make_server(
    root: Path,
    port: int = DEFAULT_PORT,
    *,
    threads: int | None = None,
    allow_execution: bool = True,
    allowed_hosts: tuple[str, ...] = (),
) -> ThreadingHTTPServer:
    from dendrinet_studio.model import (
        dump_document,
        example,
        example_catalog,
        inspect_document,
        inspect_wiring,
        parse_document,
        parse_value,
    )
    from dendrinet_studio.schema import describe_config_schema

    import dendritic_modeling

    token = secrets.token_urlsafe(32)
    hosts = {"127.0.0.1", "localhost", "::1", *allowed_hosts}
    runs = Runs(root, threads=threads, allow_execution=allow_execution)
    assets = _assets()

    def schema_payload():
        return {
            "core_version": dendritic_modeling.__version__,
            "studio_version": studio_version,
            "examples": example_catalog(),
            "execution": allow_execution,
            "threads": runs.threads,
            "workspace": str(runs.root),
            **describe_config_schema(),
        }

    def preview(data):
        raw = parse_document(data["yaml"]) if "yaml" in data else data["config"]
        if not isinstance(raw, dict):
            raise ValueError("A configuration must be a mapping")
        return inspect_document(raw)

    def run_summary(raw):
        described = inspect_document(raw)
        decoder = described["resolved_config"]["model"].get("decoder") or {}
        return {
            "family": described["family"],
            "task": described["task"],
            "dataset": described["dataset"],
            "strategy": described["strategy"],
            "recurrent": described["recurrent"],
            "outputs": (decoder.get("params") or {}).get("output_dim"),
        }

    class Handler(BaseHTTPRequestHandler):
        server_version = "DendriNetStudio"
        sys_version = ""

        def log_message(self, *args):
            pass  # Never log the access token or uploaded configurations.

        def reply(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(payload)

        def json_reply(self, data, status=200):
            self.reply(
                status,
                json.dumps(json_values(data), allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def host_allowed(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in hosts

        def authorized(self) -> bool:
            supplied = self.headers.get("X-Studio-Token", "")
            return self.host_allowed() and secrets.compare_digest(supplied, token)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                if not self.authorized():
                    self.json_reply(
                        {"error": "Open the Studio URL the launcher printed"}, 403
                    )
                    return
                try:
                    if path == "/api/schema":
                        self.json_reply(schema_payload())
                    elif path == "/api/runs":
                        self.json_reply(runs.snapshot())
                    elif path.startswith("/api/example/"):
                        self.json_reply(
                            inspect_document(example(path.rsplit("/", 1)[1]))
                        )
                    elif path.startswith("/api/runs/") and path.endswith("/config"):
                        key = unquote(path[len("/api/runs/") : -len("/config")])
                        self.json_reply({"config": runs.config(key)})
                    else:
                        self.json_reply({"error": "Unknown endpoint"}, 404)
                except (ValueError, TypeError, KeyError) as error:
                    self.json_reply({"error": _message(error)}, 400)
                return
            if not self.host_allowed():
                self.reply(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            name = "index.html" if path == "/" else path.lstrip("/")
            if name not in assets:
                self.reply(404, b"Not found", "text/plain; charset=utf-8")
                return
            self.reply(200, assets[name], ASSET_TYPES[Path(name).suffix])

        def do_POST(self):
            if not self.authorized():
                self.json_reply({"error": "Invalid Studio session"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Request must be between 1 byte and 2 MB")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object")
                path = urlsplit(self.path).path
                if path == "/api/preview":
                    self.json_reply(preview(data))
                elif path == "/api/wiring":
                    self.json_reply(inspect_wiring(data["config"]))
                elif path == "/api/parse":
                    self.json_reply({"value": parse_value(str(data["yaml"]))})
                elif path == "/api/dump":
                    self.json_reply({"yaml": dump_document(data["config"])})
                elif path == "/api/run":
                    config = data["config"]
                    self.json_reply(
                        runs.start(
                            config,
                            kind=data["kind"],
                            device=data.get("device", "cpu"),
                            name=str(data.get("name", "")),
                            summary=run_summary(config),
                        )
                    )
                elif path == "/api/stop":
                    runs.stop(str(data["id"]))
                    self.json_reply({"stopped": data["id"]})
                elif path == "/api/annotate":
                    self.json_reply(
                        runs.annotate(
                            str(data["id"]),
                            name=data.get("name"),
                            notes=data.get("notes"),
                        )
                    )
                else:
                    self.json_reply({"error": "Unknown endpoint"}, 404)
            except KeyError as error:
                self.json_reply({"error": f"Missing field {error.args[0]!r}"}, 400)
            except (ValueError, TypeError, OSError) as error:
                self.json_reply({"error": _message(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.studio_token = token
    server.studio_runs = runs
    return server


def _message(error: Exception) -> str:
    text = str(error).strip() or type(error).__name__
    return text if len(text) < 4000 else text[:4000] + " …"


def _bind(workspace: Path, port: int, **options) -> ThreadingHTTPServer:
    """Bind the requested port, or the next free one when it is taken."""
    candidates = [port] if port == 0 else [port, *range(port + 1, port + 20), 0]
    for candidate in candidates:
        try:
            return make_server(workspace, candidate, **options)
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
    raise OSError("No free local port found")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dendrinet-studio",
        description="Design, check, train and compare DendriNet models in the browser.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("studio-runs"),
        help="folder for run records and results (default: ./studio-runs)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"local port (default {DEFAULT_PORT}; the next free port if taken; 0 = any)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"CPU threads per run (default {default_threads()} on this computer)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the URL without opening a browser (for remote machines)",
    )
    parser.add_argument(
        "--no-execution",
        action="store_true",
        default=os.environ.get("DENDRINET_STUDIO_NO_EXECUTION", "") not in ("", "0"),
        help="design and export only; never build or train models here "
        "(also set by DENDRINET_STUDIO_NO_EXECUTION=1)",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="also accept this Host name (for a trusted proxy in front of the "
        "local port); repeatable",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {studio_version}"
    )
    args = parser.parse_args(argv)
    server = _bind(
        args.workspace,
        args.port,
        threads=args.threads,
        allow_execution=not args.no_execution,
        allowed_hosts=tuple(args.allow_host),
    )
    url = f"http://127.0.0.1:{server.server_port}/#{server.studio_token}"
    print(f"DendriNet Studio: {url}\nRuns: {args.workspace.resolve()}", flush=True)
    if args.no_browser:
        print(
            "On another computer, forward the port with "
            f"'ssh -L {server.server_port}:127.0.0.1:{server.server_port} <host>' "
            "and open the URL there.",
            flush=True,
        )
    else:
        webbrowser.open(url)

    def shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Workers are stopped with their server, so none keeps training unseen.
        server.studio_runs.close()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
