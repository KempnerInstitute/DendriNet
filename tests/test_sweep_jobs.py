"""Sweep, rerun and single job scripts share one Slurm profile."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import yaml
from omegaconf import OmegaConf

from dendritic_modeling.scripts.sweeps import (
    rerun_missing_configs as rerun,
    sweep_manager,
)
from dendritic_modeling.scripts.sweeps.control_plane import (
    discover_source_identity,
    untracked_source_digest,
)
from dendritic_modeling.scripts.sweeps.slurm_profile import (
    DEFAULT_TRAINING_SCRIPT,
    SlurmProfile,
    normalize_slurm_time,
)
from dendritic_modeling.scripts.sweeps.unified.base_sweep import (
    REPO_ROOT,
    BaseSweepGenerator,
    SweepJobManager,
    _config_mount_points,
    _execution_source_guard,
    _mount_canary,
    read_submitted_jobs,
)

SLURM = {
    "account": "lab_account",
    "partition": "gpu_partition",
    "time": "04:00:00",
    "constraint": "h100",
    "gpus_per_node": 1,
    "cpus_per_task": 8,
    "mem": "64G",
    "run_name": "unit_sweep",
    "max_concurrent_jobs": 2,
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("02:00:00", "02:00:00"),
        ("1-00:00:00", "1-00:00:00"),
        (7200, "2:00:00"),
        (OmegaConf.create("time: 12:00:00").time, "12:00:00"),
    ],
)
def test_slurm_times_keep_their_duration(value, expected):
    assert normalize_slurm_time(value) == expected


@pytest.mark.parametrize("value", ["120", True, -5, "2 hours"])
def test_ambiguous_or_invalid_slurm_times_raise(value):
    with pytest.raises(ValueError):
        normalize_slurm_time(value)


def _sbatch_lines(path):
    return [
        line for line in path.read_text().splitlines() if line.startswith("#SBATCH")
    ]


def test_installed_launchers_use_native_modules_and_resolve_custom_paths(
    tmp_path, monkeypatch
):
    from dendritic_modeling.scripts.sweeps import slurm_profile

    monkeypatch.setattr(slurm_profile, "CHECKOUT_ROOT", None)
    monkeypatch.chdir(tmp_path)
    profile = SlurmProfile.from_config(SLURM)
    assert slurm_profile.job_working_directory("sweep") == tmp_path / "sweep"
    assert "$(pwd)/src" not in profile.environment()
    assert profile.training_command().endswith(
        "-m dendritic_modeling.scripts.training.train_experiments"
    )
    fsdp = SlurmProfile.from_config({**SLURM, "launcher": "torchrun"})
    assert "-m torch.distributed.run" in fsdp.training_command()
    assert fsdp.training_command(
        DEFAULT_TRAINING_SCRIPT.replace(".py", "_fsdp.py")
    ).endswith("-m dendritic_modeling.scripts.training.train_experiments_fsdp")
    assert str(tmp_path / "custom.py") in profile.training_command("custom.py")


def test_single_and_array_scripts_request_the_same_resources(tmp_path):
    (tmp_path / "jobs").mkdir()
    configs = [tmp_path / f"config_{index}.yaml" for index in range(3)]
    for config in configs:
        config.write_text("experiment: {}\n")
    manager = SweepJobManager(dict(SLURM))
    single = tmp_path / "jobs" / "single.sh"
    array = tmp_path / "jobs" / "array.sh"
    manager._create_single_job_script(
        str(single), [str(c) for c in configs], str(tmp_path)
    )
    manager._create_array_job_script(
        str(array), [str(c) for c in configs], str(tmp_path)
    )

    def resources(path):
        return [
            line
            for line in _sbatch_lines(path)
            if not line.startswith(
                ("#SBATCH --array", "#SBATCH --output", "#SBATCH --error")
            )
        ]

    assert resources(single) == resources(array)
    assert "#SBATCH --constraint=h100" in resources(array)
    assert "#SBATCH --array=0-2%2" in _sbatch_lines(array)
    assert DEFAULT_TRAINING_SCRIPT in single.read_text()
    assert (REPO_ROOT / DEFAULT_TRAINING_SCRIPT).is_file()
    assert "module purge" in single.read_text()


def test_the_mount_canary_probes_the_mounts_holding_config_paths(tmp_path, monkeypatch):
    mounted = tmp_path / "shared"
    (mounted / "data").mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n  base_dir: {mounted}/data/cifar10\n"
        "outputs:\n  results_dir: outputs/run\n"
        "notes: see https://example.org/page\n"
    )
    monkeypatch.setattr(os.path, "ismount", lambda path: str(path) == str(mounted))

    assert _config_mount_points([str(config)]) == [str(mounted)]
    canary = _mount_canary([str(config)])
    assert f"for fs_root in {mounted}; do" in canary
    assert "exit 75" in canary

    manager = SweepJobManager(dict(SLURM))
    (tmp_path / "jobs").mkdir()
    script = tmp_path / "jobs" / "array.sh"
    manager._create_array_job_script(str(script), [str(config)], str(tmp_path))
    assert canary in script.read_text()


def test_configs_on_the_root_filesystem_get_no_mount_probe(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  base_dir: {tmp_path}/data\n")
    monkeypatch.setattr(os.path, "ismount", lambda path: False)

    assert _config_mount_points([str(config)]) == []
    assert "stat" not in _mount_canary([str(config)])


def test_array_specs_list_reruns_and_cap_concurrency():
    profile = SlurmProfile.from_config(SLURM)
    assert profile.array_spec([0, 1, 2]) == "0-2%2"
    assert profile.array_spec([4, 1]) == "1,4"
    assert profile.array_spec([7]) == "7"


def _sweep(tmp_path, *, finished=(0,), partial=(2,)):
    sweep = tmp_path / "unified_sweep_20261008120000"
    (sweep / "configs").mkdir(parents=True)
    (sweep / "jobs").mkdir()
    records = []
    for index in range(4):
        path = sweep / "configs" / f"config_{index}.yaml"
        path.write_text("experiment: {}\n")
        records.append({"index": index, "path": f"configs/config_{index}.yaml"})
    OmegaConf.save(
        OmegaConf.create({"slurm_config": SLURM}), sweep / "original_config.yaml"
    )
    manifest = {
        "generated_configs": records,
        "source_identity": discover_source_identity(REPO_ROOT),
    }
    (sweep / rerun.MANIFEST_NAME).write_text(json.dumps(manifest))
    for index in range(4):
        result = sweep / "results" / f"config_{index}"
        (result / "performance").mkdir(parents=True)
        (result / "train.log").write_text("Training experiment completed\n")
        if index in finished or index in partial:
            (result / "performance" / "final.json").write_text("{}")
        if index in finished:
            (result / "training_summary.json").write_text("{}")
    return sweep


def test_unfinished_configs_are_found_from_their_artifacts(tmp_path):
    sweep = _sweep(tmp_path)
    _manifest, configs, slurm = rerun.load_sweep(sweep)
    assert rerun.find_unfinished(sweep, len(configs), slurm) == [1, 2, 3]


@pytest.mark.requires_git_checkout
def test_a_rerun_is_the_sweeps_array_script_for_the_unfinished_configs(tmp_path):
    sweep = _sweep(tmp_path)
    manifest, configs, slurm = rerun.load_sweep(sweep)
    original = sweep / "jobs" / "run_array_sweep.sh"
    SweepJobManager(dict(slurm))._create_array_job_script(
        str(original), [str(c) for c in configs], str(sweep)
    )
    script = rerun.write_rerun_script(sweep, manifest, configs, slurm, [1, 2, 3])

    def without_array(path):
        return [line for line in _sbatch_lines(path) if "--array" not in line]

    assert without_array(script) == without_array(original)
    assert "#SBATCH --array=1-3%2" in _sbatch_lines(script)
    commit = manifest["source_identity"]["git"]["commit"]
    assert f'EXPECTED_REPOSITORY_HEAD="{commit}"' in script.read_text()


def test_a_rerun_refuses_a_different_source_state(tmp_path):
    sweep = _sweep(tmp_path)
    manifest, configs, slurm = rerun.load_sweep(sweep)
    manifest["source_identity"]["git"]["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="not the source the sweep was generated"):
        rerun.write_rerun_script(sweep, manifest, configs, slurm, [1])


def test_the_command_line_finds_the_sweep_by_timestamp(tmp_path):
    sweep = _sweep(tmp_path, finished=(0, 1, 2, 3))
    assert (
        rerun.resolve_sweep_dir(timestamp="20261008120000", base_dir=tmp_path) == sweep
    )
    assert rerun.main([str(sweep)]) == 0
    with pytest.raises(ValueError, match="pass --force"):
        rerun.main([str(sweep), "--config-ids", "1"])
    with pytest.raises(FileNotFoundError, match="expected one sweep directory"):
        rerun.resolve_sweep_dir(timestamp="19990101", base_dir=tmp_path)


def test_unfinished_configs_are_read_from_the_execution_results_root(tmp_path):
    sweep = _sweep(tmp_path, finished=(0, 1, 2, 3))
    external = tmp_path / "external"
    (external / "config_1" / "performance").mkdir(parents=True)
    (external / "config_1" / "performance" / "final.json").write_text("{}")
    (external / "config_1" / "training_summary.json").write_text("{}")
    slurm = {**SLURM, "execution_results_dir": str(external)}
    assert rerun.find_unfinished(sweep, 4, slurm) == [0, 2, 3]


def _configs(tmp_path, count):
    paths = []
    for index in range(count):
        path = tmp_path / f"config_{index}.yaml"
        path.write_text("experiment: {}\n")
        paths.append(str(path))
    return paths


def _scheduled_configs(script):
    return [
        line.split()[-1]
        for line in script.read_text().splitlines()
        if line.startswith("$PYTHON")
    ]


def test_split_jobs_schedule_every_config_once(tmp_path):
    (tmp_path / "jobs").mkdir()
    configs = _configs(tmp_path, 10)
    scripts = SweepJobManager(dict(SLURM)).create_job_scripts(
        configs, str(tmp_path), split_jobs=3
    )
    jobs = [tmp_path / "jobs" / f"run_sweep_{index}.sh" for index in range(3)]
    assert scripts == [*map(str, jobs), str(tmp_path / "submit_all_jobs.sh")]
    scheduled = [_scheduled_configs(job) for job in jobs]
    assert [len(chunk) for chunk in scheduled] == [4, 3, 3]
    expected = [os.path.relpath(path, REPO_ROOT) for path in configs]
    assert [path for chunk in scheduled for path in chunk] == expected
    with pytest.raises(ValueError, match="cannot be combined"):
        SweepJobManager(dict(SLURM)).create_job_scripts(
            configs, str(tmp_path), split_jobs=3, use_array_jobs=True
        )


class _Generator(BaseSweepGenerator):
    def generate_configs(self, base_config):
        return []

    def validate_config(self, config):
        return True


def test_sweep_directories_are_new_and_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generator = _Generator("unit")
    first = generator.create_output_directory("sweeps", prefix="run")
    second = generator.create_output_directory("sweeps", prefix="run")
    assert os.path.isabs(first) and first != second
    assert os.path.dirname(first) == str(tmp_path / "sweeps")
    assert sorted(os.listdir(second)) == ["configs", "jobs", "plots", "results"]


def _fake_sbatch(tmp_path, monkeypatch, *, fail=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    body = "exit 1" if fail else 'echo "$((40000 + $(ls -1 "$0.d" | wc -l)));cluster"'
    sbatch.write_text(f'#!/bin/bash\nmkdir -p "$0.d"\n{body}\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def test_submission_records_job_ids_and_fails_loudly(tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    scripts = SweepJobManager(dict(SLURM)).create_job_scripts(
        _configs(tmp_path, 4), str(tmp_path), split_jobs=2
    )
    info = {"job_scripts": scripts, "output_dir": str(tmp_path)}
    _fake_sbatch(tmp_path, monkeypatch)
    submitted = sweep_manager.SweepManager().submit_jobs(info)
    assert [job["script"] for job in submitted] == ["run_sweep_0.sh", "run_sweep_1.sh"]
    assert all(job["job_id"] == "40000" for job in submitted)
    assert read_submitted_jobs(tmp_path) == submitted

    failing = tmp_path / "failing"
    (failing / "jobs").mkdir(parents=True)
    scripts = SweepJobManager(dict(SLURM)).create_job_scripts(
        _configs(failing, 2), str(failing)
    )
    _fake_sbatch(failing, monkeypatch, fail=True)
    with pytest.raises(subprocess.CalledProcessError):
        sweep_manager.SweepManager().submit_jobs(
            {"job_scripts": scripts, "output_dir": str(failing)}
        )
    assert read_submitted_jobs(failing) == []


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _run_guard(guard):
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + guard], capture_output=True, text=True
    ).returncode


def test_the_job_guard_covers_untracked_source_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "module.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/module.py")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        "init",
    )
    (repo / "src" / "new module.py").write_text("y = 2\n")
    (repo / "src" / "b.py").write_text("z = 3\n")

    identity = discover_source_identity(repo)
    names = [b"src/b.py", b"src/new module.py"]
    assert identity["git"]["untracked_src_files"] == ["src/b.py", "src/new module.py"]
    assert identity["git"]["untracked_src_sha256"] == untracked_source_digest(
        repo, names
    )
    guard = _execution_source_guard(repo, identity)
    assert _run_guard(guard) == 0

    (repo / "src" / "b.py").write_text("z = 4\n")
    assert _run_guard(guard) == 68
    (repo / "src" / "b.py").write_text("z = 3\n")
    (repo / "src" / "module.py").write_text("x = 2\n")
    assert _run_guard(guard) == 67


def test_git_failures_raise_instead_of_unguarding_jobs(tmp_path):
    repo = tmp_path / "broken"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /nonexistent\n")
    with pytest.raises(RuntimeError, match="git rev-parse HEAD failed"):
        discover_source_identity(repo)
    plain = tmp_path / "installed"
    plain.mkdir()
    assert discover_source_identity(plain)["git"] == {"available": False}


def test_the_command_line_forwards_its_options(tmp_path, monkeypatch):
    calls = {}

    def record(name):
        def method(self, *args, **kwargs):
            calls[name] = (args, kwargs)
            return {}

        return method

    monkeypatch.setattr(sweep_manager.SweepManager, "run_full_sweep", record("run"))
    monkeypatch.setattr(
        sweep_manager.SweepManager, "analyze_results", record("analyze")
    )
    assert sweep_manager.main(["--config", "s.yaml", "--output-dir", "out"]) == 0
    assert calls["run"][1]["output_dir"] == "out"
    assert sweep_manager.main(["--analyze", "--results-dir", "sweep"]) == 0
    assert calls["analyze"][0] == ("sweep", None)
    with pytest.raises(SystemExit):
        sweep_manager.main(["--config", "s.yaml", "--run", "--generate-only"])
    with pytest.raises(SystemExit):
        sweep_manager.main(["--analyze"])


def _sweep_file(tmp_path, **sections):
    data = {
        "slurm_config": {"account": "lab", "partition": "gpu"},
        "experiment_settings": {"seeds_per_condition": 1, "base_seed": 0},
        "sweep_config": {"experiment.seed": [1, 2]},
        "base_config": {"experiment": {"seed": 0}},
        **sections,
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    "sections, message",
    [
        (
            {"slurm_config": {"account": "lab", "partition": "gpu", "memory": "64G"}},
            "did you mean 'mem'",
        ),
        (
            {"experiment_settings": {"seed_per_condition": 2}},
            "did you mean 'seeds_per_condition'",
        ),
        (
            {"filter_config": {"mode": "ei-sum", "tolerence": 1}},
            "did you mean 'tolerance'",
        ),
        (
            {"experiment_setting": {"seeds_per_condition": 2}},
            "did you mean 'experiment_settings'",
        ),
    ],
)
def test_misspelled_sweep_keys_raise(tmp_path, sections, message):
    with pytest.raises(ValueError, match=message):
        sweep_manager.SweepManager._load_composed_config(
            str(_sweep_file(tmp_path, **sections))
        )


def test_annotations_load_with_a_warning(tmp_path, caplog):
    path = _sweep_file(
        tmp_path, study_id="s1", **{"x-anchor": {"a": 1}, "metadata": {"b": 2}}
    )
    with caplog.at_level("WARNING"):
        sweep_manager.SweepManager._load_composed_config(str(path))
    assert "do not read: study_id (" in caplog.text
