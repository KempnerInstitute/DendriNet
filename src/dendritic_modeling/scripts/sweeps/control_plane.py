"""Reproducibility and scheduler policy for generated sweeps.

The functions in this module deliberately have no dependency on Slurm itself.
They validate scheduler account/partition pairs against a site policy and
freeze the inputs that determine a generated sweep before any job is submitted.

A site policy is a YAML file listing the approved pairs::

    profiles:
      - {id: lab_gpu, account: my_lab, partition: gpu}

``DENDRITIC_SCHEDULER_PROFILES`` names it (``none`` for no policy); otherwise
``$XDG_CONFIG_HOME/dendritic_modeling/scheduler_profiles.yaml`` (default
``~/.config``) is used when it exists. Without a policy any explicit pair is
accepted and recorded as unvalidated.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dendritic_modeling.utils.source_identity import (
    GIT_TIMEOUT_SECONDS,
    UNTRACKED_SOURCE_ROOT,
    discover_source_identity,
    is_git_checkout,
    package_tree_digest,
    sha256_file,
    untracked_source_digest,
)

SCHEDULER_PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True)
class SchedulerProfile:
    """One resolved account/partition pairing.

    ``policy`` is the site policy file that approved it (``None`` when no
    policy is configured, with ``validated`` false).
    """

    profile_id: str
    account: str
    partition: str
    validated: bool = True
    placeholder: bool = False
    policy: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return the stable JSON/YAML representation used in manifests."""
        return asdict(self)


#: Environment variable naming the site scheduler policy file.
SCHEDULER_POLICY_ENV = "DENDRITIC_SCHEDULER_PROFILES"
_POLICY_ENTRY_KEYS = frozenset({"id", "account", "partition"})


def scheduler_policy_path() -> Path | None:
    """The site scheduler policy file in use, or ``None`` without one."""
    configured = os.environ.get(SCHEDULER_POLICY_ENV, "").strip()
    if configured.lower() == "none":
        return None
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{SCHEDULER_POLICY_ENV}={configured} does not name a file"
            )
        return path
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    path = Path(config_home) / "dendritic_modeling" / "scheduler_profiles.yaml"
    return path if path.is_file() else None


def load_scheduler_profiles(
    path: str | Path | None = None,
) -> tuple[SchedulerProfile, ...] | None:
    """The approved profiles of a site policy (default: the one in use)."""
    import yaml

    policy = scheduler_policy_path() if path is None else Path(path)
    if policy is None:
        return None
    data = yaml.safe_load(policy.read_text(encoding="utf-8")) or {}
    entries = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{policy}: expected a non-empty 'profiles' list")
    profiles = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _POLICY_ENTRY_KEYS:
            raise ValueError(
                f"{policy}: each profile needs exactly id, account and partition; "
                f"got {entry!r}"
            )
        profiles.append(
            SchedulerProfile(
                profile_id=_required_scheduler_value(entry["id"], "profile id"),
                account=_required_scheduler_value(entry["account"], "account"),
                partition=_required_scheduler_value(entry["partition"], "partition"),
                policy=str(policy.resolve()),
            )
        )
    pairs = [(profile.account, profile.partition) for profile in profiles]
    ids = [profile.profile_id for profile in profiles]
    if len(set(pairs)) != len(pairs) or len(set(ids)) != len(ids):
        raise ValueError(f"{policy}: profile ids and account/partition pairs repeat")
    return tuple(profiles)


def _required_scheduler_value(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Scheduler {field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Scheduler {field_name} is required")
    return normalized


def validate_scheduler_profile(
    account: Any,
    partition: Any,
    *,
    allow_placeholder: bool = False,
) -> SchedulerProfile:
    """Validate and name an approved scheduler account/partition pair.

    A placeholder is accepted only for non-submitting generation paths. An
    explicit, non-placeholder pair is always validated, including during a dry
    run, so a dry run remains a faithful submission preflight.
    """
    normalized_account = _required_scheduler_value(account, "account")
    normalized_partition = _required_scheduler_value(partition, "partition")

    has_placeholder = SCHEDULER_PLACEHOLDER in {
        normalized_account,
        normalized_partition,
    }
    if has_placeholder:
        if not allow_placeholder:
            raise ValueError(
                "Scheduler placeholders are permitted only for non-submitting "
                "generate-only or dry-run workflows"
            )
        return SchedulerProfile(
            profile_id="non_submitting_placeholder",
            account=normalized_account,
            partition=normalized_partition,
            validated=False,
            placeholder=True,
        )

    profiles = load_scheduler_profiles()
    if profiles is None:
        # No site policy: the scheduler decides access; the pair is recorded.
        return SchedulerProfile(
            profile_id="unlisted",
            account=normalized_account,
            partition=normalized_partition,
            validated=False,
        )
    for profile in profiles:
        if (profile.account, profile.partition) == (
            normalized_account,
            normalized_partition,
        ):
            return profile
    allowed = ", ".join(f"({item.account}, {item.partition})" for item in profiles)
    raise ValueError(
        "Unapproved scheduler account/partition pair "
        f"({normalized_account}, {normalized_partition}) under the site policy "
        f"{profiles[0].policy}. Allowed pairs: {allowed}"
    )


def _manifest_path(path: Path, *, repo_root: Path, output_dir: Path) -> str:
    resolved = path.resolve()
    for base, prefix in ((output_dir.resolve(), ""), (repo_root.resolve(), "repo:")):
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        return f"{prefix}{relative.as_posix()}"
    return str(resolved)


def write_frozen_sweep_manifest(
    *,
    output_dir: str | Path,
    input_config_path: str | Path,
    resolved_config_path: str | Path,
    generated_config_paths: Iterable[str | Path],
    expected_config_count: int,
    scheduler_profile: SchedulerProfile,
    repo_root: str | Path,
    source_files: Iterable[str | Path] = (),
    generator_identity: str | None = None,
    generation_mode: str,
) -> Path:
    """Write an immutable-by-convention manifest for one generated sweep."""
    output_root = Path(output_dir).resolve()
    repository_root = Path(repo_root).resolve()
    generated_paths = [Path(path).resolve() for path in generated_config_paths]
    expected = int(expected_config_count)
    if expected < 0:
        raise ValueError("expected_config_count must be non-negative")
    if len(generated_paths) != expected:
        raise ValueError(
            "Generated config count does not match the frozen expectation: "
            f"expected {expected}, observed {len(generated_paths)}"
        )

    input_path = Path(input_config_path).resolve()
    resolved_path = Path(resolved_config_path).resolve()
    generated_records = [
        {
            "index": index,
            "path": _manifest_path(
                path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(path),
        }
        for index, path in enumerate(generated_paths)
    ]

    payload = {
        "schema_version": 1,
        "manifest_type": "dendritic_modeling.frozen_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_mode": str(generation_mode),
        "expected_config_count": expected,
        "original_yaml": {
            "path": _manifest_path(
                input_path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(input_path),
        },
        "resolved_original_yaml": {
            "path": _manifest_path(
                resolved_path,
                repo_root=repository_root,
                output_dir=output_root,
            ),
            "sha256": sha256_file(resolved_path),
        },
        "generated_configs": generated_records,
        "scheduler_profile": scheduler_profile.as_dict(),
        "generator": generator_identity,
        "source_identity": discover_source_identity(
            repository_root,
            source_files=source_files,
            include_package=True,
        ),
    }

    manifest_path = output_root / "frozen_sweep_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Frozen sweep manifest already exists: {manifest_path}")
    temporary_path = output_root / f".{manifest_path.name}.tmp-{os.getpid()}"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


__all__ = [
    "GIT_TIMEOUT_SECONDS",
    "SCHEDULER_PLACEHOLDER",
    "SCHEDULER_POLICY_ENV",
    "UNTRACKED_SOURCE_ROOT",
    "SchedulerProfile",
    "discover_source_identity",
    "is_git_checkout",
    "load_scheduler_profiles",
    "package_tree_digest",
    "scheduler_policy_path",
    "sha256_file",
    "untracked_source_digest",
    "validate_scheduler_profile",
    "write_frozen_sweep_manifest",
]
