"""SLURM job tools: input shapes, GPU plans and the scripts they write."""

import subprocess
import sys
from pathlib import Path

import pytest

from dendritic_modeling.scripts.tools import generate_slurm_job, submit_fsdp_job
from dendritic_modeling.scripts.tools.generate_slurm_job import _job_input_shape


def test_job_input_shape_uses_canonical_dataset_dimensions():
    assert _job_input_shape("mnist") == (784,)
    assert _job_input_shape("cifar10") == (3072,)
    assert _job_input_shape("cifar100") == (3072,)
    assert _job_input_shape("imagenet") == (150528,)
    assert _job_input_shape("unknown") == (784,)


def test_submit_fsdp_reuses_slurm_job_input_shape_helper():
    assert submit_fsdp_job._job_input_shape is _job_input_shape


def test_importing_submit_fsdp_preserves_package_resolution():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; before = sys.path.copy(); "
            "from dendritic_modeling.scripts.tools import submit_fsdp_job; "
            # PyTorch may append a temporary helper directory during import.
            # The helper must not prepend a package directory that shadows
            # unrelated top-level packages (notably Hugging Face datasets).
            "assert sys.path[:len(before)] == before, (before, sys.path)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


REPO = Path(__file__).resolve().parents[1]
SINGLE = str(REPO / "configs" / "config_exp.yaml")
FSDP = str(REPO / "configs" / "config_exp_fsdp.yaml")


def _args(config, tmp_path, *extra):
    return generate_slurm_job.build_parser().parse_args(
        [
            config,
            *("--account", "acct", "--partition", "part"),
            *("--output-dir", str(tmp_path)),
            *extra,
        ]
    )


def test_single_process_configs_run_on_one_gpu_with_python(tmp_path):
    generate_slurm_job.run(_args(SINGLE, tmp_path))
    (script,) = tmp_path.glob("*.sbatch")
    text = script.read_text()
    assert "#SBATCH --account=acct" in text and "#SBATCH --partition=part" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert (
        "$PYTHON -u src/dendritic_modeling/scripts/training/train_experiments.py"
        in text
    )
    assert "configs/config_exp.yaml" in text
    for debug_only in ("CUDA_LAUNCH_BLOCKING", "BLOCKING_WAIT", "NCCL_DEBUG"):
        assert debug_only not in text


def test_fsdp_jobs_take_the_smallest_gpu_count_that_fits(tmp_path, monkeypatch):
    class _Estimate:
        def fsdp_memory_per_gpu(self, count):
            return 200.0 / count

    monkeypatch.setattr(
        generate_slurm_job, "estimate_config", lambda *a, **k: _Estimate()
    )
    plan = generate_slurm_job.plan_job(FSDP, gpu_memory_gb=80.0)
    assert (plan.mode, plan.gpus_per_node, plan.memory_per_gpu_gb) == ("fsdp", 4, 50.0)
    with pytest.raises(ValueError, match="multi-node"):
        generate_slurm_job.plan_job(FSDP, gpu_memory_gb=40.0)

    generate_slurm_job.run(_args(FSDP, tmp_path))
    text = next(tmp_path.glob("*.sbatch")).read_text()
    assert "#SBATCH --gres=gpu:4" in text
    assert "--nproc_per_node=4 src/dendritic_modeling/scripts/training/" in text
    assert "train_experiments_fsdp.py configs/config_exp_fsdp.yaml" in text


def test_gpu_counts_must_match_the_distributed_mode():
    with pytest.raises(ValueError, match="one process on one GPU"):
        generate_slurm_job.plan_job(SINGLE, gpus_per_node=2)


def test_submission_returns_the_slurm_job_id(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)

        class Done:
            stdout = "12345;cluster\n"

        return Done()

    monkeypatch.setattr(generate_slurm_job.subprocess, "run", fake_run)
    job = generate_slurm_job.run(_args(SINGLE, tmp_path, "--submit"))
    assert job == "12345"
    assert calls[0][:2] == ["sbatch", "--parsable"]


def test_the_fsdp_submitter_rejects_other_modes():
    with pytest.raises(SystemExit):
        submit_fsdp_job.main([SINGLE, "--account", "a", "--partition", "p"])
