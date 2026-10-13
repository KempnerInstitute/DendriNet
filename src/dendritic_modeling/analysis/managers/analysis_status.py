"""Completion status of the analyses a run requested.

:class:`~dendritic_modeling.analysis.managers.analysis_manager.AnalysisManager`
records, for every analyzer requested in a pass (``run_analysis``), whether it

``completed``
   returned and wrote output (or reported that it keeps its output for a
   later pass);
``skipped``
   did not run, with the reason: the analyzer reported that the model or the
   data does not apply, or the manager did not call it;
``failed``
   raised, with the error type and message (the error still propagates);
``empty``
   returned without writing any file into its output directory.

The records go to ``analysis_status.json`` in the analysis output directory,
keyed by the pass name (``final``, a stage name, ``epoch<N>``); per-epoch
passes keep only the most recent record. A pass that did not finish (the
process stopped) keeps ``"finished": false``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

ANALYSIS_STATUS_FILE = "analysis_status.json"
ANALYSIS_STATUS_SCHEMA_VERSION = "analysis_status_v1"

COMPLETED = "completed"
SKIPPED = "skipped"
FAILED = "failed"
EMPTY = "empty"
ANALYSIS_STATUSES = (COMPLETED, SKIPPED, FAILED, EMPTY)
# When an analyzer runs more than once in a pass, the worst status counts.
_SEVERITY = {COMPLETED: 0, EMPTY: 1, SKIPPED: 2, FAILED: 3}


class AnalysisIncompleteError(RuntimeError):
    """Raised when ``analysis.require_completion`` is set and an analysis
    requested in a pass did not complete."""

    def __init__(self, pass_name: str, incomplete: Mapping[str, Mapping[str, Any]]):
        self.pass_name = pass_name
        self.incomplete = {name: dict(record) for name, record in incomplete.items()}
        details = "; ".join(
            f"{name}: {record.get('status')}"
            + (
                f" ({record['reason']})"
                if record.get("reason")
                else (
                    f" ({record.get('error_type')}: {record.get('error_message')})"
                    if record.get("error_type")
                    else ""
                )
            )
            for name, record in self.incomplete.items()
        )
        super().__init__(
            f"analysis pass {pass_name!r} did not complete every requested "
            f"analysis (analysis.require_completion is true): {details}"
        )


def list_output_files(directory: str) -> set[str]:
    """Return the files under ``directory``, relative to it (none if absent)."""
    found: set[str] = set()
    if not os.path.isdir(directory):
        return found
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                stack.append(entry.path)
            else:
                found.add(os.path.relpath(entry.path, directory))
    return found


def modified_since(
    directory: str, names: Iterable[str], reference_ns: int
) -> list[str]:
    """Return the files among ``names`` whose modification time is not
    earlier than ``reference_ns`` (a time stamp of the same file system)."""
    modified = []
    for name in names:
        try:
            if os.stat(os.path.join(directory, name)).st_mtime_ns >= reference_ns:
                modified.append(name)
        except OSError:
            continue
    return sorted(modified)


def _read_status_file(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logger.warning("Replacing unreadable analysis status file %s: %s", path, error)
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ANALYSIS_STATUS_SCHEMA_VERSION
        or not isinstance(payload.get("passes"), dict)
    ):
        logger.warning("Replacing analysis status file %s of another schema", path)
        return {}
    return payload


def write_pass_record(directory: str, pass_name: str, record: Mapping[str, Any]) -> str:
    """Store ``record`` as the pass ``pass_name`` in ``analysis_status.json``.

    A training-phase pass replaces earlier training-phase passes. The file is
    replaced atomically. Returns its path.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, ANALYSIS_STATUS_FILE)
    payload = _read_status_file(path)
    passes = dict(payload.get("passes", {}))
    if record.get("training"):
        passes = {
            name: previous
            for name, previous in passes.items()
            if not (isinstance(previous, Mapping) and previous.get("training"))
        }
    passes[str(pass_name)] = dict(record)
    payload = {"schema_version": ANALYSIS_STATUS_SCHEMA_VERSION, "passes": passes}
    handle, temporary = tempfile.mkstemp(
        prefix=f".{ANALYSIS_STATUS_FILE}.", dir=directory, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def read_status_file(directory: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Return the pass records of ``analysis_status.json`` in ``directory``.

    A directory without the file (a run written before status records
    existed) gives an empty mapping. A file that is not a status record
    raises ``ValueError``, since what it records cannot be checked.
    """
    path = os.path.join(os.fspath(directory), ANALYSIS_STATUS_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ANALYSIS_STATUS_SCHEMA_VERSION
        or not isinstance(payload.get("passes"), dict)
    ):
        raise ValueError(f"{path} is not an {ANALYSIS_STATUS_SCHEMA_VERSION} record")
    return {
        str(name): dict(record)
        for name, record in payload["passes"].items()
        if isinstance(record, Mapping)
    }


def incomplete_analyses(
    record: Mapping[str, Any], analyzers: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """The analyzers among ``analyzers`` that the pass ``record`` shows did not
    complete, with their recorded status.

    An analyzer the pass did not record at all did not run in it: the pass
    either was not asked to run it or stopped before reaching it.
    """
    analyses = record.get("analyses")
    analyses = analyses if isinstance(analyses, Mapping) else {}
    finished = record.get("finished", True) is not False
    incomplete: dict[str, dict[str, Any]] = {}
    for name in analyzers:
        entry = analyses.get(name)
        if isinstance(entry, Mapping):
            if entry.get("status") != COMPLETED:
                incomplete[name] = dict(entry)
        else:
            incomplete[name] = {
                "status": "not recorded",
                "reason": (
                    "the recorded pass did not run it"
                    if finished
                    else "the recorded pass stopped before it"
                ),
            }
    return incomplete


def describe_incomplete(incomplete: Mapping[str, Mapping[str, Any]]) -> str:
    """One line per analyzer: its status and the recorded reason or error."""
    parts = []
    for name, record in sorted(incomplete.items()):
        detail = record.get("reason")
        if not detail and record.get("error_type"):
            detail = f"{record['error_type']}: {record.get('error_message')}"
        parts.append(
            f"{name}: {record.get('status')}" + (f" ({detail})" if detail else "")
        )
    return "; ".join(parts)


def combine_records(
    previous: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge two records of one analyzer in one pass; the worst status counts."""
    if previous is None:
        return dict(current)
    worse = (
        current
        if _SEVERITY[current["status"]] >= _SEVERITY[previous["status"]]
        else previous
    )
    merged = dict(worse)
    merged["written_files"] = int(previous.get("written_files", 0)) + int(
        current.get("written_files", 0)
    )
    missing = list(previous.get("missing_outputs", [])) + list(
        current.get("missing_outputs", [])
    )
    if missing:
        merged["missing_outputs"] = missing
    merged["calls"] = int(previous.get("calls", 1)) + int(current.get("calls", 1))
    return merged


__all__ = [
    "ANALYSIS_STATUSES",
    "ANALYSIS_STATUS_FILE",
    "ANALYSIS_STATUS_SCHEMA_VERSION",
    "COMPLETED",
    "EMPTY",
    "FAILED",
    "SKIPPED",
    "AnalysisIncompleteError",
    "combine_records",
    "describe_incomplete",
    "incomplete_analyses",
    "list_output_files",
    "modified_since",
    "read_status_file",
    "write_pass_record",
]
