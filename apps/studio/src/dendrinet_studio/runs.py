"""Local runs: one worker at a time, honest records, no shell commands.

Each run lives in its own folder of the workspace with the exact YAML it used,
the command, the package identity and the worker's log. A run is reported as
completed only when its worker exited successfully; a run whose server went
away is never assumed to have finished.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

import dendritic_modeling

RECORD = "studio_run.json"
KINDS = {"inspect": "Model check", "train": "Training"}
DEVICES = {"cpu", "auto"}
LOG_TAIL_BYTES = 40_000
# Trainer progress lines, e.g. "epoch=10/30, train_loss=0.68 (...), valid_loss=0.71".
_PROGRESS = re.compile(
    r"epoch=(\d+)/(\d+),\s*train_loss=([^\s,(]+).*?valid_loss=([^\s,(]+)"
)


def json_values(value: Any) -> Any:
    """Represent non-finite results as missing, never as a successful zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    return value


def default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _process_start(pid: int) -> str | None:
    """A token that changes when a process ID is reused (Linux only)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Field 22 (start time) follows the parenthesised command name.
    return stat.rsplit(")", 1)[-1].split()[19]


def _alive(pid: int | None, start: str | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    except OSError:
        return False
    current = _process_start(pid)
    return start is None or current is None or current == start


def _package_roots() -> list[str]:
    """Folders that make the imported packages importable in a worker.

    Installed packages are already on the interpreter's path; adding their
    site-packages folder to PYTHONPATH would put it ahead of the standard
    library. Only source checkouts need an entry.
    """
    import dendrinet_studio

    roots = []
    for module in (dendritic_modeling, dendrinet_studio):
        root = Path(module.__file__).resolve().parents[1]
        if root.name in {"site-packages", "dist-packages"}:
            continue
        if str(root) not in roots:
            roots.append(str(root))
    return roots


def worker_environment(device: str, threads: int) -> dict[str, str]:
    env = os.environ.copy()
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    roots = _package_roots()
    env["PYTHONPATH"] = os.pathsep.join(roots + [p for p in existing if p not in roots])
    if not env["PYTHONPATH"]:
        del env["PYTHONPATH"]
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def progress_from_log(text: str) -> dict | None:
    """Epochs reported so far by the trainer's progress lines."""
    points = {}
    epochs = None
    for match in _PROGRESS.finditer(text):
        epoch, total = int(match.group(1)), int(match.group(2))
        epochs = total
        points[epoch] = {
            "epoch": epoch,
            "train_loss": _number(match.group(3)),
            "valid_loss": _number(match.group(4)),
        }
    if not points:
        return None
    latest = max(points)
    return {
        "epoch": latest,
        "epochs": epochs,
        "points": [points[k] for k in sorted(points)],
    }


def _newest(folder: Path, pattern: str) -> Path | None:
    found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


class Runs:
    """Start, stop and describe local workers for one workspace folder."""

    def __init__(
        self,
        root: Path,
        *,
        threads: int | None = None,
        allow_execution: bool = True,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.threads = int(threads or default_threads())
        self.allow_execution = allow_execution
        self.processes: dict[str, subprocess.Popen] = {}
        self.records: dict[str, dict] = {}
        self.problems: list[dict[str, str]] = []
        self.lock = threading.RLock()
        for path in sorted(self.root.glob(f"*/{RECORD}")):
            self._load(path)

    # Records -----------------------------------------------------------------

    def _load(self, path: Path) -> None:
        data = _read_json(path)
        if not isinstance(data, dict):
            self.problems.append(
                {"path": str(path), "error": "unreadable run record; left untouched"}
            )
            return
        key = path.parent.name
        data.setdefault("id", key)
        data.setdefault("kind", "train")
        data.setdefault("status", "unknown")
        data.setdefault("command", [])
        data.setdefault("name", "")
        data.setdefault("notes", "")
        if data["status"] == "running":
            if _alive(data.get("pid"), data.get("pid_start")):
                data["status"] = "running"
                data["detached"] = True
            else:
                data["status"] = "unknown after server restart"
        self.records[key] = data

    def _write(self, key: str) -> None:
        path = self.root / key / RECORD
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(json_values(self.records[key]), indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def refresh(self) -> None:
        with self.lock:
            for key, process in self.processes.items():
                result = process.poll()
                if result is not None and self.records[key]["status"] == "running":
                    self.records[key].update(
                        status="completed" if result == 0 else "failed",
                        exit_code=result,
                        finished=time.time(),
                    )
                    self._write(key)
            for key, record in self.records.items():
                if record.get("detached") and record["status"] == "running":
                    if not _alive(record.get("pid"), record.get("pid_start")):
                        record.update(status="unknown (worker ended while detached)")
                        record.pop("detached", None)
                        self._write(key)

    def busy(self) -> bool:
        with self.lock:
            self.refresh()
            live = any(p.poll() is None for p in self.processes.values())
            detached = any(
                r.get("detached") and r["status"] == "running"
                for r in self.records.values()
            )
            return live or detached

    # Execution -----------------------------------------------------------------

    def start(
        self,
        raw: dict,
        *,
        kind: str,
        device: str = "cpu",
        name: str = "",
        summary: dict | None = None,
    ) -> dict:
        if not self.allow_execution:
            raise ValueError(
                "This Studio was started with --no-execution: export the YAML and "
                "run it with dendritic-train"
            )
        if kind not in KINDS:
            raise ValueError(f"Unknown run kind {kind!r}; use one of {sorted(KINDS)}")
        if device not in DEVICES:
            raise ValueError(f"Unknown device {device!r}; use one of {sorted(DEVICES)}")
        if raw.get("distributed", {}).get("mode", "none") != "none":
            raise ValueError(
                "Studio runs one process; export distributed configurations for torchrun"
            )
        with self.lock:
            if self.busy():
                raise ValueError("A worker is already running; wait for it or stop it")
            key = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            folder = self.root / key
            folder.mkdir()
            config = folder / "config.yaml"
            config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            if kind == "inspect":
                command = [
                    sys.executable,
                    "-m",
                    "dendrinet_studio.worker",
                    str(config),
                    str(folder / "resources.json"),
                    "--threads",
                    str(self.threads),
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
            record = {
                "id": key,
                "kind": kind,
                "name": name.strip()[:120],
                "notes": "",
                "status": "running",
                "started": time.time(),
                "device": device,
                "threads": self.threads,
                "command": command,
                "cwd": str(folder),
                "python": sys.executable,
                "core_version": dendritic_modeling.__version__,
                "core_path": dendritic_modeling.__file__,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                **(summary or {}),
            }
            self.records[key] = record
            try:
                with (folder / "worker.log").open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=folder,
                        env=worker_environment(device, self.threads),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
            except OSError as error:
                record.update(status="failed to start", error=str(error))
                self._write(key)
                raise
            record.update(pid=process.pid, pid_start=_process_start(process.pid))
            self.processes[key] = process
            self._write(key)
            return dict(record)

    def stop(self, key: str) -> None:
        with self.lock:
            record = self.records.get(key)
            process = self.processes.get(key)
            if process is not None and process.poll() is None:
                pid = process.pid
            elif record and record.get("detached") and record["status"] == "running":
                pid = record.get("pid")
                if not _alive(pid, record.get("pid_start")):
                    raise ValueError("That worker is no longer running")
            else:
                raise ValueError("No running worker with that ID")
            # A worker owns its process group, so its data-loader children stop too.
            self._signal(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _alive(pid, record.get("pid_start")):
                if process is not None and process.poll() is not None:
                    break
                time.sleep(0.1)
            if _alive(pid, record.get("pid_start")) and (
                process is None or process.poll() is None
            ):
                self._signal(pid, signal.SIGKILL)
                if process is not None:
                    process.wait(timeout=5)
            record.update(
                status="stopped",
                exit_code=process.returncode if process is not None else None,
                finished=time.time(),
            )
            record.pop("detached", None)
            self._write(key)

    @staticmethod
    def _signal(pid: int, sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(pid, sig)
            else:  # pragma: no cover - Windows
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    def annotate(self, key: str, *, name: str | None = None, notes: str | None = None):
        with self.lock:
            record = self.records.get(key)
            if record is None:
                raise ValueError("No run with that ID")
            if name is not None:
                record["name"] = str(name).strip()[:120]
            if notes is not None:
                record["notes"] = str(notes)[:4000]
            self._write(key)
            return dict(record)

    def config(self, key: str) -> dict:
        """The configuration a run used, read from its own folder."""
        with self.lock:
            if key not in self.records:
                raise ValueError("No run with that ID")
            path = self.root / key / "config.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f"The run's configuration could not be read: {error}"
            ) from error
        if not isinstance(data, dict):
            raise ValueError("The run's configuration is not a mapping")
        return data

    def close(self) -> None:
        with self.lock:
            for key, process in list(self.processes.items()):
                if process.poll() is None:
                    self.stop(key)

    # Description ---------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            self.refresh()
            rows = []
            for key, record in reversed(list(self.records.items())):
                folder = self.root / key
                row = dict(record)
                row["kind_label"] = KINDS.get(row.get("kind"), str(row.get("kind")))
                end = row.get("finished") or (
                    time.time() if row["status"] == "running" else None
                )
                if row.get("started") and end:
                    row["duration_s"] = round(end - row["started"], 1)
                log = folder / "worker.log"
                if log.is_file():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - LOG_TAIL_BYTES))
                        row["log"] = stream.read().decode("utf-8", errors="replace")
                    if row["status"] == "running" and row.get("kind") == "train":
                        progress = progress_from_log(row["log"])
                        if progress:
                            row["progress"] = progress
                resources = _read_json(folder / "resources.json")
                results = folder / "results"
                if resources is None and results.is_dir():
                    path = _newest(results, "**/model_resources.json")
                    resources = _read_json(path) if path else None
                if isinstance(resources, dict):
                    row["resources"] = resources
                if results.is_dir():
                    summary = _newest(results, "**/training_summary.json")
                    if summary is not None and isinstance(
                        data := _read_json(summary), dict
                    ):
                        row["summary"] = data
                    final = _newest(results, "**/performance/final.json")
                    if final is not None and isinstance(
                        data := _read_json(final), dict
                    ):
                        row["metrics"] = data
                    run_folder = summary.parent if summary is not None else None
                    if run_folder is not None:
                        row["results_dir"] = str(run_folder)
                rows.append(row)
            return {
                "runs": json_values(rows),
                "problems": list(self.problems),
                "busy": self.busy(),
                "workspace": str(self.root),
                "threads": self.threads,
                "execution": self.allow_execution,
            }
