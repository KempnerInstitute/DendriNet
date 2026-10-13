"""
Base classes for the unified sweep framework.

This module provides abstract base classes that define the interface
for different sweep types (EI, branch, noise, general).
"""

import logging
import os
import re
import shlex
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.config import CONFIG_VERSION
from dendritic_modeling.scripts.sweeps.control_plane import (
    UNTRACKED_SOURCE_ROOT,
    discover_source_identity,
)
from dendritic_modeling.scripts.sweeps.slurm_profile import SlurmProfile

logger = logging.getLogger(__name__)

# The checkout containing this module; jobs run its src/.
REPO_ROOT = Path(__file__).resolve().parents[5]
SUBMITTED_JOBS_NAME = "submitted_jobs.tsv"


def _execution_source_guard(
    repo_root: Path, identity: dict[str, Any] | None = None
) -> str:
    """Return a shell preflight that binds a job to a source identity.

    ``identity`` defaults to the checkout's current one (sweep generation); a
    rerun passes the identity frozen in the sweep's manifest. The job exits
    when the commit, the tracked changes or the untracked files under
    ``src/`` differ from it.
    """
    if identity is None:
        identity = discover_source_identity(repo_root)
    git_identity = identity.get("git", {})
    if not git_identity.get("available"):
        logger.warning(
            "%s is not a Git checkout: the sweep's jobs are not bound to a source "
            "state",
            repo_root,
        )
        return "# Source guard unavailable: the library is not a Git checkout.\n"

    expected_head = str(git_identity["commit"])
    expected_diff = git_identity.get("tracked_diff_sha256")
    if not expected_diff:
        raise RuntimeError("could not fingerprint the tracked generation worktree")

    quoted_root = shlex.quote(str(repo_root))
    guard = (
        "# Refuse execution after the source checkout changes.\n"
        f"REPOSITORY_ROOT={quoted_root}\n"
        f'EXPECTED_REPOSITORY_HEAD="{expected_head}"\n'
        f'EXPECTED_TRACKED_DIFF_SHA256="{expected_diff}"\n'
        'ACTUAL_REPOSITORY_HEAD="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"\n'
        'ACTUAL_TRACKED_DIFF_SHA256="$(git -C "$REPOSITORY_ROOT" '
        "diff --binary HEAD | sha256sum | awk '{print $1}')\"\n"
        'if [ "$ACTUAL_REPOSITORY_HEAD" != "$EXPECTED_REPOSITORY_HEAD" ]; then\n'
        '  echo "Source commit changed after sweep generation: expected '
        '$EXPECTED_REPOSITORY_HEAD, observed $ACTUAL_REPOSITORY_HEAD" >&2\n'
        "  exit 66\n"
        "fi\n"
        'if [ "$ACTUAL_TRACKED_DIFF_SHA256" != "$EXPECTED_TRACKED_DIFF_SHA256" ]; then\n'
        '  echo "Tracked source changes differ from the frozen generation worktree" >&2\n'
        "  exit 67\n"
        "fi\n"
    )
    expected_untracked = git_identity.get("untracked_src_sha256")
    if expected_untracked:
        # Same digest as control_plane.untracked_source_digest.
        guard += (
            f'EXPECTED_UNTRACKED_SRC_SHA256="{expected_untracked}"\n'
            'ACTUAL_UNTRACKED_SRC_SHA256="$(cd "$REPOSITORY_ROOT" && '
            f"git ls-files --others --exclude-standard -z -- {UNTRACKED_SOURCE_ROOT} "
            "| LC_ALL=C sort -z | while IFS= read -r -d '' name; do "
            "printf '%s\\0%s\\n' \"$name\" "
            '"$(sha256sum < "$name" | awk \'{print $1}\')"; done '
            "| sha256sum | awk '{print $1}')\"\n"
            'if [ "$ACTUAL_UNTRACKED_SRC_SHA256" != "$EXPECTED_UNTRACKED_SRC_SHA256" ]; '
            "then\n"
            '  echo "Untracked files under src/ differ from the frozen generation '
            'worktree" >&2\n'
            "  exit 68\n"
            "fi\n"
        )
    return guard


class BaseSweepGenerator(ABC):
    """
    Abstract base class for sweep configuration generators.

    Each sweep type (EI, branch, noise, general) should inherit from this
    and implement the required methods.
    """

    def __init__(self, sweep_type: str):
        self.sweep_type = sweep_type
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    @abstractmethod
    def generate_configs(self, base_config: DictConfig) -> list[DictConfig]:
        """
        Generate sweep configurations from base configuration.

        Args:
            base_config: Base configuration with sweep parameters

        Returns:
            List of configurations for each sweep condition
        """

    @abstractmethod
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate that the configuration is valid for this sweep type.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, raises ValueError if invalid
        """

    def create_output_directory(
        self, base_dir: str, prefix: Optional[str] = None
    ) -> str:
        """Create a new sweep directory ``<base_dir>/<prefix>_<timestamp>``.

        The timestamp is taken now and the path is absolute. A directory that
        already exists (a second sweep in the same second) is never reused:
        the new one gets a numeric suffix, so one sweep cannot overwrite the
        configs that another sweep's queued jobs will read.

        Args:
            base_dir: Base directory for outputs
            prefix: Optional prefix for directory name

        Returns:
            Path to created output directory
        """
        if prefix is None:
            prefix = self.sweep_type
        base = os.path.abspath(os.path.expanduser(base_dir))
        os.makedirs(base, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{prefix}_{self.timestamp}"
        output_dir = os.path.join(base, name)
        suffix = 1
        while True:
            try:
                os.mkdir(output_dir)
                break
            except FileExistsError:
                suffix += 1
                output_dir = os.path.join(base, f"{name}_{suffix}")

        for subdir in ["configs", "results", "plots", "jobs"]:
            os.makedirs(os.path.join(output_dir, subdir))

        return output_dir

    def save_configs(
        self,
        configs: list[DictConfig],
        output_dir: str,
        execution_results_dir: Optional[str] = None,
    ) -> list[str]:
        """
        Save generated configurations to files.

        Args:
            configs: List of configurations to save
            output_dir: Directory to save configurations
            execution_results_dir: Optional root used by training jobs for
                checkpoints and metrics. This permits lightweight sweep
                manifests and job scripts to be staged separately from large
                execution outputs.

        Returns:
            List of paths to saved configuration files
        """
        config_dir = os.path.join(output_dir, "configs")
        results_base_dir = execution_results_dir or os.path.join(output_dir, "results")
        if execution_results_dir is not None:
            staged_results = Path(output_dir) / "results"
            if staged_results.is_symlink():
                if os.readlink(staged_results) != execution_results_dir:
                    raise FileExistsError(
                        f"staged results link has a different target: {staged_results}"
                    )
            else:
                if staged_results.exists():
                    if any(staged_results.iterdir()):
                        raise FileExistsError(
                            "cannot replace non-empty staged results directory: "
                            f"{staged_results}"
                        )
                    staged_results.rmdir()
                staged_results.symlink_to(
                    execution_results_dir, target_is_directory=True
                )
        config_paths = []

        for i, config in enumerate(configs):
            # Set the results_dir for this config to point to its specific subdirectory
            config_results_dir = os.path.join(results_base_dir, f"config_{i}")

            # Ensure outputs section exists and set results_dir
            if "outputs" not in config:
                config["outputs"] = {}
            config["outputs"]["results_dir"] = config_results_dir
            # The collectors read results/config_<i>/ directly.
            config["outputs"]["exact_run_dir"] = True
            # Record the configuration format the generated file was validated
            # in, so dendritic-upgrade-config never reads it as an older one.
            if config.get("config_version") is None:
                config["config_version"] = CONFIG_VERSION

            # Save the config
            config_path = os.path.join(config_dir, f"{self.sweep_type}_config_{i}.yaml")
            OmegaConf.save(config, config_path)
            config_paths.append(config_path)

        # Save metadata
        metadata = {
            "sweep_type": self.sweep_type,
            "timestamp": self.timestamp,
            "num_configs": len(configs),
            "config_files": [os.path.basename(p) for p in config_paths],
        }
        metadata_path = os.path.join(config_dir, "sweep_metadata.yaml")
        OmegaConf.save(OmegaConf.create(metadata), metadata_path)

        return config_paths


class BaseSweepAnalyzer(ABC):
    """
    Abstract base class for sweep result analyzers.
    """

    def __init__(self, sweep_type: str):
        self.sweep_type = sweep_type

    @abstractmethod
    def collect_results(self, results_dir: str) -> dict[str, Any]:
        """
        Collect and aggregate results from sweep experiments.

        Args:
            results_dir: Directory containing sweep results

        Returns:
            Dictionary with aggregated results
        """

    @abstractmethod
    def generate_plots(self, results: dict[str, Any], output_dir: str) -> list[str]:
        """
        Generate analysis plots for sweep results.

        Args:
            results: Aggregated results from collect_results
            output_dir: Directory to save plots

        Returns:
            List of paths to generated plot files
        """

    def find_best_configs(
        self, results: dict[str, Any], metric: str = "test_acc"
    ) -> list[dict[str, Any]]:
        """
        Find the best performing configurations.

        Args:
            results: Aggregated results
            metric: Metric to optimize (default: test_acc)

        Returns:
            List of best configurations sorted by performance
        """
        # Default implementation - can be overridden by subclasses
        if "configs" not in results or metric not in results:
            return []

        configs_with_scores = []
        for i, config in enumerate(results["configs"]):
            if i < len(results[metric]):
                score = results[metric][i]
                configs_with_scores.append(
                    {"config": config, "score": score, "config_id": i}
                )

        # Sort by score (descending for accuracy, may need to adjust for loss)
        configs_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return configs_with_scores[:10]  # Return top 10

    def generate_summary_report(self, results: dict[str, Any], output_dir: str) -> str:
        """
        Generate a summary report of sweep results.

        Args:
            results: Aggregated results
            output_dir: Directory to save report

        Returns:
            Path to generated report file
        """
        report_path = os.path.join(
            output_dir, "plots", f"{self.sweep_type}_sweep_summary.txt"
        )

        with open(report_path, "w") as f:
            f.write(f"{self.sweep_type.upper()} Sweep Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sweep type: {self.sweep_type}\n\n")

            if "configs" in results:
                f.write(f"Total configurations: {len(results['configs'])}\n")

            # Find best configs if possible
            try:
                best_configs = self.find_best_configs(results)
                if best_configs:
                    f.write("\nTop 5 configurations:\n")
                    f.write("-" * 30 + "\n")
                    for i, config_info in enumerate(best_configs[:5]):
                        f.write(
                            f"{i + 1}. Score: {config_info['score']:.4f} "
                            f"(Config ID: {config_info['config_id']})\n"
                        )
            except Exception as e:
                f.write(f"\nCould not determine best configurations: {e}\n")

        return report_path


# An absolute path written as a YAML value: at the start of a line or after
# whitespace, a quote, ``[``, ``,``, ``=`` or ``:``.
_CONFIG_ABSOLUTE_PATH = re.compile(
    r"""(?:^|(?<=[\s"'\[,=:]))(/[^\s"'\],#]+)""", re.MULTILINE
)


def _containing_mount(path: Path) -> Path | None:
    """The mount point holding ``path`` (its deepest existing ancestor), or
    ``None`` when that is the root filesystem."""
    for candidate in (path, *path.parents):
        if candidate == Path(candidate.anchor):
            return None
        if os.path.ismount(candidate):
            return candidate
    return None


def _config_mount_points(config_paths: list[str]) -> list[str]:
    """Mount points (other than ``/``) of the absolute paths the configs name."""
    mounts: set[str] = set()
    seen: set[str] = set()
    for config_path in config_paths:
        try:
            text = Path(config_path).read_text(encoding="utf-8")
        except OSError:
            continue
        for match in _CONFIG_ABSOLUTE_PATH.finditer(text):
            path = match.group(1).rstrip("/") or "/"
            if path in seen:
                continue
            seen.add(path)
            mount = _containing_mount(Path(path))
            if mount is not None:
                mounts.add(str(mount))
    return sorted(mounts)


def _mount_canary(config_paths: list[str]) -> str:
    """Return a preflight that fails fast when a mount the configs use hangs.

    The mount points holding the absolute paths named in the configs (found
    when the script is generated; the root filesystem is not probed) are each
    checked with a 60 s ``stat`` on the allocated node: a hung network mount
    would otherwise stall the first path access inside Python until the job's
    time limit, and a missing one would fail only then. Configs without such
    paths get no probe.
    """
    mounts = _config_mount_points(config_paths)
    if not mounts:
        return "# Mount canary: the configs name no paths on separate mounts\n"
    roots = " ".join(shlex.quote(mount) for mount in mounts)
    return (
        "# Mount canary (fail fast on wedged or missing mounts)\n"
        f"for fs_root in {roots}; do\n"
        '    if ! timeout 60 stat -t "$fs_root" > /dev/null 2>&1; then\n'
        '        echo "MOUNT CANARY FAILED on $(hostname): $fs_root unreachable" >&2\n'
        "        exit 75\n"
        "    fi\n"
        "done\n"
        'echo "Mount canary passed on $(hostname)"\n'
    )


class SweepJobManager:
    """
    Manages SLURM job creation and submission for sweeps.
    """

    def __init__(self, slurm_config: DictConfig):
        self.slurm_config = slurm_config

    def create_job_scripts(
        self,
        config_paths: list[str],
        output_dir: str,
        split_jobs: Optional[int] = None,
        use_array_jobs: Optional[bool] = None,
    ) -> list[str]:
        """Write the sweep's job scripts and ``submit_all_jobs.sh``.

        A sweep of several configs is one array job by default (task ``i``
        runs config ``i``). ``split_jobs=N`` writes up to ``N`` sequential jobs
        instead, whose config lists differ in length by at most one and
        together cover every config; ``use_array_jobs=False`` writes one
        sequential job.

        Args:
            config_paths: List of configuration file paths
            output_dir: Output directory for sweep
            split_jobs: Number of sequential jobs to split the configs into
            use_array_jobs: Array job (default unless ``split_jobs`` is given)

        Returns:
            List of paths to created job scripts
        """
        if not config_paths:
            raise ValueError("a sweep without configs has no jobs to write")
        if split_jobs is not None:
            if int(split_jobs) < 1:
                raise ValueError("split_jobs must be a positive number of jobs")
            if use_array_jobs:
                raise ValueError(
                    "split_jobs writes sequential jobs; it cannot be combined with "
                    "use_array_jobs=True"
                )
        if use_array_jobs is None:
            use_array_jobs = split_jobs is None
        jobs_dir = os.path.join(output_dir, "jobs")
        job_scripts = []

        if use_array_jobs and len(config_paths) > 1:
            script_path = os.path.join(jobs_dir, "run_array_sweep.sh")
            self._create_array_job_script(script_path, config_paths, output_dir)
            job_scripts.append(script_path)
        elif split_jobs is None or int(split_jobs) == 1:
            script_path = os.path.join(jobs_dir, "run_sweep.sh")
            self._create_single_job_script(script_path, config_paths, output_dir)
            job_scripts.append(script_path)
        else:
            chunks = _split_evenly(list(config_paths), int(split_jobs))
            for i, job_configs in enumerate(chunks):
                script_path = os.path.join(jobs_dir, f"run_sweep_{i}.sh")
                self._create_single_job_script(script_path, job_configs, output_dir)
                job_scripts.append(script_path)

        main_script = self._create_main_submission_script(job_scripts, output_dir)

        return [*job_scripts, main_script]

    def _create_single_job_script(
        self, script_path: str, config_paths: list[str], output_dir: str
    ):
        """Create a SLURM job script that runs ``config_paths`` in sequence."""
        profile = SlurmProfile.from_config(self.slurm_config)
        with open(script_path, "w") as f:
            f.write(
                profile.header(
                    repo_root=REPO_ROOT,
                    output=f"{output_dir}/jobs/%j.out",
                    error=f"{output_dir}/jobs/%j.err",
                )
            )
            f.write("set -euo pipefail\n\n")
            f.write(_execution_source_guard(REPO_ROOT))
            f.write("\n")
            f.write(profile.environment())
            f.write(_mount_canary(list(config_paths)))
            f.write("\n# Run experiments\n")
            for config_path in config_paths:
                rel_config_path = os.path.relpath(
                    os.path.abspath(config_path), start=REPO_ROOT
                )
                f.write(
                    f"{profile.launch_prefix()} {profile.training_script} "
                    f"{rel_config_path}\n"
                )
        os.chmod(script_path, 0o755)

    def _create_array_job_script(
        self,
        script_path: str,
        config_paths: list[str],
        output_dir: str,
        *,
        indices: list[int] | None = None,
        source_identity: dict[str, Any] | None = None,
    ):
        """Create a SLURM array job script for parallel execution.

        Task ``i`` runs ``config_paths[i]``. ``indices`` restricts the array to
        those configs (a rerun) and ``source_identity`` binds it to a frozen
        source identity; the header otherwise equals the sweep's array script.
        """
        profile = SlurmProfile.from_config(self.slurm_config)
        array = profile.array_spec(
            list(range(len(config_paths))) if indices is None else list(indices)
        )
        with open(script_path, "w") as f:
            # Logs go to the existing jobs directory: vanilla Slurm opens them
            # before the job script could create a per-task directory.
            f.write(
                profile.header(
                    repo_root=REPO_ROOT,
                    output=f"{output_dir}/jobs/output_%A_%a.out",
                    error=f"{output_dir}/jobs/error_%A_%a.err",
                    array=array,
                )
            )
            # IMPORTANT: SBATCH directives must appear before any shell commands.
            f.write("set -euo pipefail\n\n")
            f.write(_execution_source_guard(REPO_ROOT, source_identity))
            f.write("\n")
            f.write(
                "# ================================================================\n"
                "# This file has been generated by the unified sweep system\n"
                "#                            ***\n"
                "#       Manual changes to this file may be overwritten.\n"
                "# ================================================================\n\n"
            )
            f.write(profile.environment())

            config_dir = (
                os.path.dirname(config_paths[0])
                if config_paths
                else os.path.join(output_dir, "configs")
            )
            f.write("# Path to the config file directory\n")
            f.write(f'CONFIG_DIR="{config_dir}"\n\n')
            f.write(_mount_canary(list(config_paths)))
            f.write("\n# Path to output directory\n")
            f.write(f'OUTPUT_DIR="{output_dir}"\n\n')
            if profile.execution_results_dir:
                f.write("# External root for checkpoints and metrics\n")
                f.write(f'EXECUTION_RESULTS_DIR="{profile.execution_results_dir}"\n\n')

            f.write("# Array job: select config based on SLURM_ARRAY_TASK_ID\n")
            f.write("CONFIG_FILES=(\n")
            for config_path in config_paths:
                rel_config_path = os.path.relpath(
                    os.path.abspath(config_path), start=REPO_ROOT
                )
                f.write(f'    "{rel_config_path}"\n')
            f.write(")\n\n")
            f.write("# Get the config file for this array task\n")
            f.write('CONFIG="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"\n')
            f.write('RUN_NAME="sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"\n\n')
            f.write("# Path to the folder to save outputs for this specific run\n")
            results_root = (
                "${EXECUTION_RESULTS_DIR}"
                if profile.execution_results_dir
                else "${OUTPUT_DIR}/results"
            )
            f.write(
                f'OUTPUT_FOLDER_PATH="{results_root}/config_${{SLURM_ARRAY_TASK_ID}}"\n\n'
            )
            f.write("# Create output directory\n")
            f.write("mkdir -p ${OUTPUT_FOLDER_PATH}\n\n")
            f.write('echo "Starting training job at $(date)"\n')
            f.write('echo "Array task $SLURM_ARRAY_TASK_ID processing: $CONFIG"\n')
            f.write("start_time=$(date +%s)\n\n")

            training_script = profile.training_script
            if not training_script.startswith("/"):
                training_script = f"$(pwd)/{training_script}"
            f.write("# Run experiment with srun\n")
            f.write("srun \\\n")
            f.write("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\\n")
            f.write("  --kill-on-bad-exit \\\n")
            f.write(
                f"  {profile.launch_prefix()} {training_script} "
                '"$CONFIG" --output_dir "$OUTPUT_FOLDER_PATH"\n\n'
            )
            f.write("end_time=$(date +%s)\n")
            f.write('echo "Done with training job at $(date)"\n')
            f.write('echo "Total duration: $((end_time - start_time)) seconds."\n')
        os.chmod(script_path, 0o755)

    def _create_main_submission_script(
        self, job_scripts: list[str], output_dir: str
    ) -> str:
        """Write ``submit_all_jobs.sh``, which submits the job scripts.

        It stops at the first failed ``sbatch`` and appends each job ID, the
        script name and the submission time to ``jobs/submitted_jobs.tsv``.
        """
        main_script = os.path.join(output_dir, "submit_all_jobs.sh")
        names = [
            os.path.basename(script)
            for script in job_scripts
            if not script.endswith("submit_all_jobs.sh")
        ]

        with open(main_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -euo pipefail\n")
            f.write("# Submit the sweep's jobs and record their job IDs.\n\n")
            f.write(f"# Sweep output directory: {output_dir}\n\n")
            f.write(f"REPO_ROOT={shlex.quote(str(REPO_ROOT))}\n")
            f.write('SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
            f.write(f'LEDGER="${{SWEEP_DIR}}/jobs/{SUBMITTED_JOBS_NAME}"\n\n')
            f.write(
                "submit() {\n"
                "    local job_id\n"
                '    job_id="$(sbatch --parsable --chdir "${REPO_ROOT}" '
                '"${SWEEP_DIR}/jobs/$1")"\n'
                '    job_id="${job_id%%;*}"\n'
                '    printf \'%s\\t%s\\t%s\\n\' "${job_id}" "$1" '
                '"$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LEDGER}"\n'
                '    echo "Submitted $1 as job ${job_id}"\n'
                "}\n\n"
            )
            f.write('echo "Submitting sweep jobs from ${SWEEP_DIR}"\n')
            for index, name in enumerate(names):
                if index:
                    f.write("sleep 2  # Brief delay between submissions\n")
                f.write(f"submit {shlex.quote(name)}\n")
            f.write('\necho "Job IDs recorded in ${LEDGER}"\n')

        os.chmod(main_script, 0o755)
        return main_script


def _split_evenly(items: list, parts: int) -> list[list]:
    """Split ``items`` into at most ``parts`` consecutive, non-empty chunks.

    Chunk lengths differ by at most one and the chunks cover ``items`` in order.
    """
    parts = max(1, min(int(parts), len(items)))
    size, extra = divmod(len(items), parts)
    chunks, start = [], 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        chunks.append(items[start:stop])
        start = stop
    if [item for chunk in chunks for item in chunk] != list(items):
        raise AssertionError("split jobs do not cover the sweep's configs")
    return chunks


def read_submitted_jobs(sweep_dir: str | Path) -> list[dict[str, str]]:
    """Return the job IDs ``submit_all_jobs.sh`` recorded for a sweep."""
    ledger = Path(sweep_dir) / "jobs" / SUBMITTED_JOBS_NAME
    if not ledger.is_file():
        return []
    records = []
    for line in ledger.read_text().splitlines():
        job_id, script, submitted_utc = line.split("\t")
        records.append(
            {"job_id": job_id, "script": script, "submitted_utc": submitted_utc}
        )
    return records
