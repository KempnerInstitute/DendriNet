"""Tests for rerun_missing_configs.py and package completion-log routing."""

import logging
from pathlib import Path

from dendritic_modeling.scripts.script_utils.setup_utils import (
    attach_run_directory_file_handlers,
)
from dendritic_modeling.scripts.sweeps import rerun_missing_configs as rerun
from dendritic_modeling.scripts.training import train_experiments
from dendritic_modeling.utils.logging_config import LoggerManager, get_logger

SUCCESS = "Training experiment completed successfully"


def test_train_experiments_uses_package_logger():
    assert train_experiments.logger is get_logger()


def test_package_logger_writes_success_line_to_run_dir(tmp_path):
    manager = LoggerManager()
    original_path = manager.file_path

    try:
        attach_run_directory_file_handlers(str(tmp_path))
        get_logger().info(SUCCESS)
        logging.getLogger("__main__").info(SUCCESS)

        package_log = (tmp_path / "dendritic_modeling.log").read_text()
        train_log = (tmp_path / "train.log").read_text()
        assert SUCCESS in package_log
        assert SUCCESS in train_log
    finally:
        manager.set_log_file(original_path)


def test_main_logger_does_not_write_run_dir_logs(tmp_path):
    manager = LoggerManager()
    original_path = manager.file_path

    try:
        attach_run_directory_file_handlers(str(tmp_path))
        logging.getLogger("__main__").info(SUCCESS)

        package_log = (tmp_path / "dendritic_modeling.log").read_text()
        train_log = (tmp_path / "train.log").read_text()
        assert SUCCESS not in package_log
        assert SUCCESS not in train_log
    finally:
        manager.set_log_file(original_path)


def _write_sweep_layout(tmp_path, *, timestamp="20260101120000", run_name="sweep_run"):
    outputs = tmp_path / "outputs"
    sweep_dir = outputs / f"{run_name}_{timestamp}"
    (sweep_dir / "configs").mkdir(parents=True)
    (sweep_dir / "results").mkdir()
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base_config:",
                "  outputs:",
                f'    results_dir: "{outputs}"',
                f'    run_name: "{run_name}"',
                "",
            ]
        )
    )
    return config_path, sweep_dir


def test_resolve_timestamp_uses_run_name_prefix(tmp_path):
    config_path, sweep_dir = _write_sweep_layout(tmp_path)
    resolved = rerun.resolve_sweep_directory(
        timestamp="20260101120000",
        config_file=str(config_path),
    )
    assert Path(resolved) == sweep_dir


def test_resolve_timestamp_falls_back_to_unified_prefix(tmp_path):
    unified_dir = tmp_path / "output" / "unified_20260101120000"
    unified_dir.mkdir(parents=True)
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text("base_config:\n  outputs:\n    results_dir: missing\n")
    resolved = rerun.resolve_sweep_directory(
        timestamp="20260101120000",
        config_file=str(config_path),
        base_dir=str(tmp_path / "output"),
    )
    assert Path(resolved) == unified_dir


def test_completion_requires_success_log_not_model_file(tmp_path):
    results_dir = tmp_path / "results"
    config_dir = results_dir / "config_0"
    config_dir.mkdir(parents=True)
    (config_dir / "final_model.pt").write_bytes(b"model")
    (config_dir / "dendritic_modeling.log").write_text("Saved model to somewhere\n")

    is_completed, _, _ = rerun.check_result_for_config(str(results_dir), 0)
    assert is_completed is False


def test_completion_accepts_success_line_in_either_log(tmp_path):
    results_dir = tmp_path / "results"
    config_dir = results_dir / "config_3"
    config_dir.mkdir(parents=True)
    (config_dir / "train.log").write_text(f"noise\n{SUCCESS}\n")

    is_completed, _, _ = rerun.check_result_for_config(str(results_dir), 3)
    assert is_completed is True

    other_dir = results_dir / "config_4"
    other_dir.mkdir()
    (other_dir / "dendritic_modeling.log").write_text(f"{SUCCESS}\n")
    is_completed, _, _ = rerun.check_result_for_config(str(results_dir), 4)
    assert is_completed is True


def _make_sweep_dir(tmp_path, config_ids=(0, 8, 15), slurm_extra=None):
    sweep_dir = tmp_path / "sweep_run_20260101120000"
    configs_dir = sweep_dir / "configs"
    results_dir = sweep_dir / "results"
    configs_dir.mkdir(parents=True)
    results_dir.mkdir()
    for config_id in config_ids:
        (configs_dir / f"unified_config_{config_id}.yaml").write_text("seed: 0\n")
        (results_dir / f"config_{config_id}").mkdir()

    slurm_lines = [
        "slurm_config:",
        "  account: test_account",
        "  partition: test_partition",
        '  time: "01:00:00"',
        "  mem: 16G",
        "  cpus_per_task: 4",
        "  max_concurrent_jobs: 7",
        "  conda_env_path: /path/to/conda/env",
        "  modules_to_load:",
        "    - python/3.9",
        "  exclude_nodes:",
        "    - node1",
        "  constraint: a100",
    ]
    if slurm_extra:
        slurm_lines.extend(slurm_extra)
    (sweep_dir / "original_config.yaml").write_text("\n".join(slurm_lines) + "\n")
    return sweep_dir


def test_generate_rerun_script_matches_current_job_conventions(tmp_path):
    sweep_dir = _make_sweep_dir(tmp_path)
    output_file = tmp_path / "rerun_missing.sh"

    success = rerun.generate_rerun_script(
        [8, 15], str(sweep_dir), str(output_file), max_concurrent=None
    )
    assert success is True

    content = output_file.read_text()
    assert f"#SBATCH --chdir={rerun.REPO_ROOT}" in content
    assert 'export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"' in content
    assert "conda activate" not in content
    assert 'PYTHON="/path/to/conda/env/bin/python"' in content
    assert "#SBATCH --array=8,15%7" in content
    assert f"#SBATCH --output={sweep_dir}/jobs/%A_%a/output_%A_%a.out" in content
    assert "#SBATCH --exclude=node1" in content
    assert "#SBATCH --constraint=a100" in content
    assert "--output_dir" in content
    assert '  8) CONFIG="' in content
    assert "unified_config_8.yaml" in content
    assert "unified_config_15.yaml" in content
    assert "set -euo pipefail" in content


def test_generate_rerun_script_uses_system_python_when_conda_empty(tmp_path):
    sweep_dir = _make_sweep_dir(tmp_path)
    original = (sweep_dir / "original_config.yaml").read_text()
    (sweep_dir / "original_config.yaml").write_text(
        original.replace("  conda_env_path: /path/to/conda/env", '  conda_env_path: ""')
    )
    output_file = tmp_path / "rerun_missing.sh"

    success = rerun.generate_rerun_script([8], str(sweep_dir), str(output_file))
    assert success is True
    content = output_file.read_text()
    assert 'PYTHON="python"' in content
    assert "conda activate" not in content


def test_find_missing_configs_skips_completed_and_keeps_incomplete(tmp_path):
    sweep_dir = _make_sweep_dir(tmp_path, config_ids=(0, 1, 2))
    results_dir = sweep_dir / "results"
    (results_dir / "config_0" / "dendritic_modeling.log").write_text(f"{SUCCESS}\n")
    (results_dir / "config_1" / "final_model.pt").write_bytes(b"model")

    config_ids, completed_ids, missing_ids, _ = rerun.find_missing_configs(
        str(sweep_dir / "configs"), str(results_dir)
    )
    assert config_ids == [0, 1, 2]
    assert completed_ids == {0}
    assert missing_ids == [1, 2]
