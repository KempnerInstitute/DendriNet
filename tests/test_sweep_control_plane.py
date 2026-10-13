"""Tests for sweep scheduler policy and frozen generation manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from dendritic_modeling.scripts.sweeps import sweep_manager as sweep_manager_module
from dendritic_modeling.scripts.sweeps.control_plane import (
    SCHEDULER_PLACEHOLDER,
    SCHEDULER_POLICY_ENV,
    load_scheduler_profiles,
    scheduler_policy_path,
    sha256_file,
    validate_scheduler_profile,
    write_frozen_sweep_manifest,
)

EXPECTED_PROFILE_PAIRS = {
    ("lab_a", "gpu_priority"),
    ("lab_a", "gpu_shared"),
    ("lab_a", "gpu_requeue"),
    ("lab_a", "gpu_small"),
    ("lab_b", "gpu_large"),
    ("lab_b", "gpu_large_priority"),
}


class _StubGenerator:
    def validate_config(self, config) -> bool:
        return True

    def create_output_directory(self, base_dir: str, prefix: str) -> str:
        output_dir = Path(base_dir) / f"{prefix}_test"
        for name in ("configs", "results", "plots", "jobs"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def generate_configs(self, config) -> list:
        return [OmegaConf.create({"experiment": {"seed": seed}}) for seed in (11, 12)]

    def save_configs(
        self,
        configs: list,
        output_dir: str,
        execution_results_dir: str | None = None,
    ) -> list[str]:
        paths = []
        for index, config in enumerate(configs):
            if execution_results_dir is not None:
                config.outputs = {
                    "results_dir": str(Path(execution_results_dir) / f"config_{index}")
                }
            path = Path(output_dir) / "configs" / f"stub_{index}.yaml"
            OmegaConf.save(config, path)
            paths.append(str(path))
        return paths


class _StubJobManager:
    def __init__(self, slurm_config):
        self.slurm_config = slurm_config

    def create_job_scripts(
        self,
        config_paths,
        output_dir,
        split_jobs,
        use_array_jobs,
    ) -> list[str]:
        return []


def _write_sweep_yaml(
    path: Path,
    *,
    slurm_config: dict | None = None,
    expected_config_count: int = 2,
) -> None:
    payload = {
        "outputs": {"run_name": "control_plane"},
        "sweep_contract": {"expected_config_count": expected_config_count},
        "provenance_probe": "${sweep_contract.expected_config_count}",
        "sweep_config": {"experiment.seed": [11, 12]},
        "base_config": {"experiment": {"seed": 11}},
    }
    if slurm_config is not None:
        payload["slurm_config"] = slurm_config
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_sweep_config_extends_deep_merges_and_tracks_sources(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child_dir = tmp_path / "project"
    child_dir.mkdir()
    child = child_dir / "child.yaml"
    parent.write_text(
        yaml.safe_dump(
            {
                "slurm_config": {"time": "01:00:00", "gpus_per_node": 1},
                "sweep_config": {"model.core.type": ["einet"]},
                "base_config": {
                    "data": {"dataset_name": "imagenet", "base_dir": ""},
                    "outputs": {"results_dir": "outputs"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    child.write_text(
        yaml.safe_dump(
            {
                "extends": "../parent.yaml",
                "slurm_config": {"time": "05:00:00"},
                "base_config": {
                    "data": {"base_dir": "/datasets/imagenet"},
                    "outputs": {"results_dir": "/results/fc6"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config, sources = sweep_manager_module.SweepManager._load_composed_config(
        str(child)
    )

    assert sources == [parent.resolve(), child.resolve()]
    assert config.slurm_config.time == "05:00:00"
    assert config.slurm_config.gpus_per_node == 1
    assert config.sweep_config["model.core.type"] == ["einet"]
    assert config.base_config.data.dataset_name == "imagenet"
    assert config.base_config.data.base_dir == "/datasets/imagenet"
    assert config.base_config.outputs.results_dir == "/results/fc6"


def test_sweep_config_extends_cycle_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extends cycle"):
        sweep_manager_module.SweepManager._load_composed_config(str(first))


@pytest.fixture
def site_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    policy = tmp_path / "scheduler_profiles.yaml"
    policy.write_text(
        "profiles:\n"
        + "".join(
            f"  - {{id: p{index}, account: {account}, partition: {partition}}}\n"
            for index, (account, partition) in enumerate(sorted(EXPECTED_PROFILE_PAIRS))
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SCHEDULER_POLICY_ENV, str(policy))
    return policy


def test_scheduler_policy_contains_exactly_the_approved_pairs(site_policy) -> None:
    profiles = load_scheduler_profiles()
    assert {(profile.account, profile.partition) for profile in profiles} == (
        EXPECTED_PROFILE_PAIRS
    )
    for profile in profiles:
        assert profile.policy == str(site_policy.resolve())
        assert validate_scheduler_profile(profile.account, profile.partition) == profile


@pytest.mark.usefixtures("site_policy")
@pytest.mark.parametrize(
    ("account", "partition"),
    [
        ("lab_b", "gpu_requeue"),
        ("lab_a", "gpu_large"),
        ("lab_a", "gpu_shared,gpu_requeue"),
        ("other", "gpu_priority"),
    ],
)
def test_scheduler_policy_rejects_every_unapproved_pair(
    account: str,
    partition: str,
) -> None:
    with pytest.raises(ValueError, match="Unapproved scheduler"):
        validate_scheduler_profile(account, partition)


def test_without_a_site_policy_any_pair_is_recorded_unvalidated() -> None:
    assert scheduler_policy_path() is None
    profile = validate_scheduler_profile("my_lab", "gpu")
    assert (profile.profile_id, profile.validated, profile.policy) == (
        "unlisted",
        False,
        None,
    )


def test_the_user_config_location_holds_the_default_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = tmp_path / "dendritic_modeling" / "scheduler_profiles.yaml"
    policy.parent.mkdir()
    policy.write_text("profiles:\n  - {id: a, account: x, partition: y}\n")
    monkeypatch.delenv(SCHEDULER_POLICY_ENV)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert scheduler_policy_path() == policy
    assert validate_scheduler_profile("x", "y").profile_id == "a"


@pytest.mark.parametrize(
    "text, match",
    [
        ("profiles: []\n", "non-empty 'profiles' list"),
        ("profiles:\n  - {id: a, account: x}\n", "exactly id, account"),
        (
            "profiles:\n  - {id: a, account: x, partition: y}\n"
            "  - {id: b, account: x, partition: y}\n",
            "repeat",
        ),
    ],
)
def test_malformed_site_policies_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, match: str
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(text)
    monkeypatch.setenv(SCHEDULER_POLICY_ENV, str(policy))
    with pytest.raises(ValueError, match=match):
        validate_scheduler_profile("x", "y")


def test_scheduler_placeholder_is_non_submitting_only() -> None:
    with pytest.raises(ValueError, match="non-submitting"):
        validate_scheduler_profile(SCHEDULER_PLACEHOLDER, SCHEDULER_PLACEHOLDER)

    profile = validate_scheduler_profile(
        SCHEDULER_PLACEHOLDER,
        SCHEDULER_PLACEHOLDER,
        allow_placeholder=True,
    )
    assert profile.profile_id == "non_submitting_placeholder"
    assert profile.placeholder is True
    assert profile.validated is False


def test_frozen_manifest_hashes_resolved_and_generated_yaml(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    output_dir.mkdir()
    input_config = tmp_path / "input.yaml"
    resolved_config = output_dir / "original_config.yaml"
    generated = [output_dir / "config_0.yaml", output_dir / "config_1.yaml"]
    input_config.write_text("value: ${answer}\nanswer: 42\n", encoding="utf-8")
    resolved_config.write_text("value: 42\nanswer: 42\n", encoding="utf-8")
    generated[0].write_text("seed: 1\n", encoding="utf-8")
    generated[1].write_text("seed: 2\n", encoding="utf-8")
    profile = validate_scheduler_profile("lab_a", "gpu_shared")

    manifest_path = write_frozen_sweep_manifest(
        output_dir=output_dir,
        input_config_path=input_config,
        resolved_config_path=resolved_config,
        generated_config_paths=generated,
        expected_config_count=2,
        scheduler_profile=profile,
        repo_root=tmp_path,
        source_files=[resolved_config],
        generator_identity="tests.StubGenerator",
        generation_mode="generate_only",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["manifest_type"] == "dendritic_modeling.frozen_sweep"
    assert manifest["expected_config_count"] == 2
    assert manifest["resolved_original_yaml"]["sha256"] == sha256_file(resolved_config)
    assert [record["sha256"] for record in manifest["generated_configs"]] == [
        sha256_file(path) for path in generated
    ]
    assert manifest["scheduler_profile"] == profile.as_dict()
    assert manifest["generator"] == "tests.StubGenerator"
    assert manifest["source_identity"]["git"]["available"] is False

    with pytest.raises(FileExistsError, match="already exists"):
        write_frozen_sweep_manifest(
            output_dir=output_dir,
            input_config_path=input_config,
            resolved_config_path=resolved_config,
            generated_config_paths=generated,
            expected_config_count=2,
            scheduler_profile=profile,
            repo_root=tmp_path,
            generation_mode="generate_only",
        )


def test_sweep_manager_generate_only_keeps_placeholders_and_freezes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path)
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _StubJobManager)
    manager = sweep_manager_module.SweepManager()
    manager.generator = _StubGenerator()

    sweep = manager.generate_sweep(
        str(config_path),
        output_dir=str(tmp_path / "outputs"),
        generate_only=True,
    )

    manifest = json.loads(Path(sweep["frozen_manifest"]).read_text(encoding="utf-8"))
    assert sweep["num_configs"] == 2
    assert manifest["generation_mode"] == "generate_only"
    assert manifest["expected_config_count"] == 2
    assert manifest["scheduler_profile"]["placeholder"] is True
    assert manifest["scheduler_profile"]["account"] == SCHEDULER_PLACEHOLDER
    resolved = OmegaConf.load(Path(sweep["output_dir"]) / "original_config.yaml")
    assert resolved.provenance_probe == 2
    assert resolved.slurm_config.account == SCHEDULER_PLACEHOLDER
    assert resolved.slurm_config.partition == SCHEDULER_PLACEHOLDER
    assert (
        manifest["original_yaml"]["sha256"]
        != manifest["resolved_original_yaml"]["sha256"]
    )


def test_sweep_manager_separates_staging_from_execution_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path)
    execution_root = tmp_path / "large-results"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["execution_results_dir"] = str(execution_root)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _StubJobManager)
    manager = sweep_manager_module.SweepManager()
    manager.generator = _StubGenerator()

    sweep = manager.generate_sweep(
        str(config_path),
        output_dir=str(tmp_path / "staging"),
        generate_only=True,
    )

    assert Path(sweep["output_dir"]).is_relative_to(tmp_path / "staging")
    assert sweep["execution_results_dir"] == str(execution_root)
    generated = OmegaConf.load(sweep["config_paths"][1])
    assert generated.outputs.results_dir == str(execution_root / "config_1")


def test_sweep_manager_dry_run_keeps_missing_scheduler_as_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path)
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _StubJobManager)
    manager = sweep_manager_module.SweepManager()
    manager.generator = _StubGenerator()

    sweep = manager.generate_sweep(
        str(config_path),
        output_dir=str(tmp_path / "outputs"),
        dry_run=True,
    )

    manifest = json.loads(Path(sweep["frozen_manifest"]).read_text(encoding="utf-8"))
    assert manifest["generation_mode"] == "dry_run"
    assert manifest["scheduler_profile"]["profile_id"] == ("non_submitting_placeholder")


def test_sweep_manager_refuses_to_submit_placeholder_bundle() -> None:
    manager = sweep_manager_module.SweepManager()
    with pytest.raises(ValueError, match="Cannot submit"):
        manager.submit_jobs(
            {
                "job_scripts": [],
                "output_dir": "/not-used",
                "scheduler_profile": {"placeholder": True},
            }
        )


@pytest.mark.usefixtures("site_policy")
def test_sweep_manager_dry_run_validates_explicit_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(
        config_path,
        slurm_config={
            "account": "lab_b",
            "partition": "gpu_requeue",
        },
    )
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _StubJobManager)
    manager = sweep_manager_module.SweepManager()
    manager.generator = _StubGenerator()

    with pytest.raises(ValueError, match="Unapproved scheduler"):
        manager.generate_sweep(
            str(config_path),
            output_dir=str(tmp_path / "outputs"),
            dry_run=True,
        )
    assert not (tmp_path / "outputs").exists()


def test_sweep_manager_rejects_declared_factorial_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path, expected_config_count=3)
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _StubJobManager)
    manager = sweep_manager_module.SweepManager()
    manager.generator = _StubGenerator()

    with pytest.raises(ValueError, match="expected 3, observed 2"):
        manager.generate_sweep(
            str(config_path),
            output_dir=str(tmp_path / "outputs"),
            generate_only=True,
        )
    assert not (tmp_path / "outputs").exists()


def test_generated_configs_are_loaded_before_submission() -> None:
    good = OmegaConf.create(
        {"data": {"dataset_name": "mnist"}, "_sweep_config_id": "config_0"}
    )
    sweep_manager_module.SweepManager._validate_generated_configs([good])

    typo = OmegaConf.create(
        {"data": {"datset_name": "mnist"}, "_sweep_config_id": "config_1"}
    )
    with pytest.raises(ValueError, match="config_1 does not load"):
        sweep_manager_module.SweepManager._validate_generated_configs([good, typo])


def test_a_generated_config_that_does_not_load_leaves_no_sweep_directory(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {"run_name": "typo"},
                "sweep_config": {"data.datset_name": ["mnist", "cifar10"]},
                "base_config": {"data": {"dataset_name": "mnist"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manager = sweep_manager_module.SweepManager()
    with pytest.raises(ValueError, match="does not load"):
        manager.generate_sweep(
            str(config_path),
            output_dir=str(tmp_path / "outputs"),
            generate_only=True,
        )
    assert not (tmp_path / "outputs").exists()


class _FailingJobManager(_StubJobManager):
    def create_job_scripts(self, config_paths, output_dir, split_jobs, use_array_jobs):
        raise ValueError("job scripts failed")


@pytest.mark.parametrize("existing_base", [False, True])
def test_a_failed_write_removes_only_what_generation_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_base: bool
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path)
    base = tmp_path / "new_root" / "outputs"
    if existing_base:
        base.mkdir(parents=True)
        (base / "earlier_sweep").mkdir()
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _FailingJobManager)
    manager = sweep_manager_module.SweepManager()

    with pytest.raises(ValueError, match="job scripts failed"):
        manager.generate_sweep(
            str(config_path), output_dir=str(base), generate_only=True
        )
    if existing_base:
        assert sorted(path.name for path in base.iterdir()) == ["earlier_sweep"]
    else:
        assert not (tmp_path / "new_root").exists()


def test_a_failed_write_keeps_the_execution_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "sweep.yaml"
    _write_sweep_yaml(config_path)
    execution_root = tmp_path / "large-results"
    (execution_root / "config_0").mkdir(parents=True)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["execution_results_dir"] = str(execution_root)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(sweep_manager_module, "SweepJobManager", _FailingJobManager)

    with pytest.raises(ValueError, match="job scripts failed"):
        sweep_manager_module.SweepManager().generate_sweep(
            str(config_path), output_dir=str(tmp_path / "staging"), generate_only=True
        )
    assert not (tmp_path / "staging").exists()
    assert (execution_root / "config_0").is_dir()
