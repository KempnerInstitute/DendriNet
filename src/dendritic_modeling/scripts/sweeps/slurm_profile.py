"""One Slurm job profile for the scripts the sweep tools write.

Array jobs, sequential jobs and reruns of a sweep render their ``#SBATCH``
header, environment and launch command from the same :class:`SlurmProfile`,
built from the sweep's ``slurm_config``, so they request the same resources.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TRAINING_SCRIPT = "src/dendritic_modeling/scripts/training/train_experiments.py"
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_SOURCE_PARENT = _PACKAGE_DIR.parent.parent
CHECKOUT_ROOT = (
    _SOURCE_PARENT
    if _PACKAGE_DIR.parent.name == "src"
    and (_SOURCE_PARENT / "pyproject.toml").is_file()
    else None
)
# Source provenance always identifies the library, even if the user's sweep is
# launched inside an unrelated Git checkout.
LIBRARY_SOURCE_ROOT = CHECKOUT_ROOT or _PACKAGE_DIR
LAUNCHERS = ("python", "torchrun")
_SLURM_TIME = re.compile(r"(\d+-)?\d+(:\d+){0,2}")


def job_working_directory(output_dir: str | Path) -> Path:
    return CHECKOUT_ROOT or Path(output_dir).resolve()


def installed_training_target(script: str) -> str:
    """Retain native module entry points and freeze the path of an installed user's script."""
    if CHECKOUT_ROOT is not None or script in {
        DEFAULT_TRAINING_SCRIPT,
        "src/dendritic_modeling/scripts/training/train_experiments_fsdp.py",
    }:
        return script
    return str(Path(script).resolve())


def normalize_slurm_time(value: Any) -> str:
    """Return ``value`` as a Slurm ``--time``.

    A string in a Slurm format with a colon or a day prefix (``HH:MM:SS``,
    ``MM:SS``, ``D-HH``, ``D-HH:MM:SS``) passes unchanged. An integer is a
    number of seconds: YAML reads an unquoted ``12:00:00`` as the integer
    43200. A string of digits alone raises, because Slurm reads it as minutes
    while earlier array scripts read it as seconds.
    """
    if isinstance(value, bool):
        raise ValueError(f"slurm_config.time must be a duration, got {value!r}")
    if isinstance(value, (int, float)):
        if float(value) != int(value) or int(value) <= 0:
            raise ValueError(f"slurm_config.time must be positive seconds, got {value}")
        seconds = int(value)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    text = str(value).strip()
    if text.isdigit():
        raise ValueError(
            f"slurm_config.time={text!r} is ambiguous (minutes for Slurm, seconds "
            "in earlier sweep scripts); write it as HH:MM:SS"
        )
    if not _SLURM_TIME.fullmatch(text):
        raise ValueError(
            f"slurm_config.time={text!r} is not a Slurm duration (HH:MM:SS or "
            "D-HH:MM:SS)"
        )
    return text


def _required(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"SLURM {key} not set. Provide '{key}' in slurm_config.")
    return str(value)


@dataclass(frozen=True)
class SlurmProfile:
    """Resources, environment and launcher of the jobs of one sweep."""

    account: str
    partition: str
    job_name: str = "sweep"
    qos: str | None = None
    time: str = "02:00:00"
    constraint: str | None = None
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 4
    mem: str = "32G"
    exclude_nodes: tuple[str, ...] = ()
    modules_to_load: tuple[str, ...] = ()
    conda_env_path: str | None = None
    launcher: str = "python"
    training_script: str = DEFAULT_TRAINING_SCRIPT
    max_concurrent_jobs: int = 10
    execution_results_dir: str | None = None

    @classmethod
    def from_config(cls, slurm_config: Mapping[str, Any] | None) -> SlurmProfile:
        """Build the profile from a sweep's ``slurm_config`` mapping."""
        config = dict(slurm_config or {})
        launcher = str(config.get("launcher", "python")).lower()
        if launcher not in LAUNCHERS:
            raise ValueError("slurm_config.launcher must be 'python' or 'torchrun'")
        return cls(
            account=_required(config, "account"),
            partition=_required(config, "partition"),
            job_name=str(config.get("run_name") or config.get("job_name", "sweep")),
            qos=config.get("qos") or None,
            time=normalize_slurm_time(config.get("time", "02:00:00")),
            constraint=config.get("constraint") or None,
            nodes=int(config.get("nodes", 1)),
            ntasks_per_node=int(config.get("ntasks_per_node", 1)),
            gpus_per_node=int(config.get("gpus_per_node", 1)),
            cpus_per_task=int(config.get("cpus_per_task", 4)),
            mem=str(config.get("mem", "32G")),
            exclude_nodes=tuple(config.get("exclude_nodes") or ()),
            modules_to_load=tuple(config.get("modules_to_load") or ()),
            conda_env_path=config.get("conda_env_path") or None,
            launcher=launcher,
            training_script=str(
                config.get("training_script") or DEFAULT_TRAINING_SCRIPT
            ),
            max_concurrent_jobs=int(config.get("max_concurrent_jobs", 10)),
            execution_results_dir=config.get("execution_results_dir") or None,
        )

    def header(
        self,
        *,
        repo_root: Path,
        output: str,
        error: str,
        array: str | None = None,
    ) -> str:
        """Return the ``#SBATCH`` lines; only ``array`` and the logs vary by job."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
        ]
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        lines.append(f"#SBATCH --chdir={repo_root}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        lines += [
            f"#SBATCH --time={self.time}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --gres=gpu:{self.gpus_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.mem}",
        ]
        if self.exclude_nodes:
            lines.append(f"#SBATCH --exclude={','.join(self.exclude_nodes)}")
        if array is not None:
            lines.append(f"#SBATCH --array={array}")
        lines += [f"#SBATCH --output={output}", f"#SBATCH --error={error}"]
        return "\n".join(lines) + "\n\n"

    def array_spec(self, indices: list[int]) -> str:
        """Return the ``--array`` value running ``indices`` with the job limit."""
        if not indices:
            raise ValueError("an array job needs at least one index")
        ordered = sorted(int(index) for index in indices)
        if len(ordered) == 1:
            spec = str(ordered[0])
        elif ordered == list(range(ordered[0], ordered[-1] + 1)):
            spec = f"{ordered[0]}-{ordered[-1]}"
        else:
            spec = ",".join(str(index) for index in ordered)
        if len(ordered) > self.max_concurrent_jobs:
            spec += f"%{self.max_concurrent_jobs}"
        return spec

    def environment(self) -> str:
        """Return the module, interpreter and ``PYTHONPATH`` setup as ``$PYTHON``."""
        lines = (
            ["module purge"]
            if self.modules_to_load
            else ["if type module >/dev/null 2>&1; then module purge; fi"]
        )
        lines += [f"module load {module}" for module in self.modules_to_load]
        lines.append("")
        if self.conda_env_path:
            lines += [
                f"# Using conda environment at: {self.conda_env_path}",
                f'PYTHON="{self.conda_env_path}/bin/python"',
                "",
                "# Set library path for conda environment",
                f'export LD_LIBRARY_PATH="{self.conda_env_path}/lib:'
                '${LD_LIBRARY_PATH:-}"',
                "",
                "# Verify environment works",
                "$PYTHON -c \"import torch; print(f'PyTorch version: "
                "{torch.__version__}')\"",
                "",
            ]
        elif CHECKOUT_ROOT is None and not self.modules_to_load:
            lines += [f"PYTHON={shlex.quote(sys.executable)}", ""]
        else:
            lines += ["# Using system python", 'PYTHON="python"', ""]
        if CHECKOUT_ROOT is not None:
            lines += ['export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"', ""]
        return "\n".join(lines) + "\n"

    def launch_prefix(self) -> str:
        """Return the command that runs the training script with ``$PYTHON``."""
        prefix = "$PYTHON -u" if CHECKOUT_ROOT is not None else '"$PYTHON" -u'
        if self.launcher == "torchrun":
            prefix += (
                " -m torch.distributed.run --standalone "
                f"--nproc_per_node={self.gpus_per_node}"
            )
        return prefix

    def training_command(self, script: str | None = None) -> str:
        """Use the pinned checkout's script or its installed module."""
        script = self.training_script if script is None else script
        target = installed_training_target(script)
        if CHECKOUT_ROOT is None and not Path(target).is_absolute():
            module = script.removeprefix("src/").removesuffix(".py").replace("/", ".")
            return f"{self.launch_prefix()} -m {module}"
        return f"{self.launch_prefix()} {shlex.quote(target)}"


__all__ = [
    "DEFAULT_TRAINING_SCRIPT",
    "LAUNCHERS",
    "SlurmProfile",
    "normalize_slurm_time",
]
