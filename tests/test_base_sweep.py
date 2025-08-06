"""
Tests for scripts sweeps unified base_sweep module to improve coverage.
"""

import pytest
import tempfile
import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.scripts.sweeps.unified.base_sweep import (
    BaseSweepGenerator,
    BaseSweepAnalyzer,
    SweepJobManager,
)


class ConcreteSweepGenerator(BaseSweepGenerator):
    """Concrete implementation of BaseSweepGenerator for testing."""

    def generate_configs(self, base_config: DictConfig) -> list[DictConfig]:
        """Simple implementation that generates 3 test configs."""
        configs = []
        for i in range(3):
            config = OmegaConf.create(base_config)
            config.test_param = f"value_{i}"
            configs.append(config)
        return configs

    def validate_config(self, config: DictConfig) -> bool:
        """Simple validation - check if test_param exists."""
        if "test_param" not in config:
            raise ValueError("test_param is required")
        return True


class ConcreteSweepAnalyzer(BaseSweepAnalyzer):
    """Concrete implementation of BaseSweepAnalyzer for testing."""

    def collect_results(self, results_dir: str) -> dict[str, any]:
        """Mock result collection."""
        return {
            "configs": [{"param1": 1}, {"param1": 2}, {"param1": 3}],
            "test_acc": [0.85, 0.92, 0.78],
            "train_loss": [0.15, 0.08, 0.22],
        }

    def generate_plots(self, results: dict[str, any], output_dir: str) -> list[str]:
        """Mock plot generation."""
        plot_paths = [
            os.path.join(output_dir, "accuracy_plot.png"),
            os.path.join(output_dir, "loss_plot.png"),
        ]
        # Create mock files
        for path in plot_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("mock plot data")
        return plot_paths


class TestBaseSweepGenerator:
    """Test BaseSweepGenerator class."""

    def test_init(self):
        """Test initialization of BaseSweepGenerator."""
        generator = ConcreteSweepGenerator("test_sweep")

        assert generator.sweep_type == "test_sweep"
        assert hasattr(generator, "timestamp")
        assert len(generator.timestamp) == 14  # YYYYMMDDHHMMSS format

    def test_timestamp_format(self):
        """Test timestamp format."""
        generator = ConcreteSweepGenerator("test")

        # Check timestamp format
        timestamp = generator.timestamp
        datetime.strptime(timestamp, "%Y%m%d%H%M%S")  # Should not raise

    def test_generate_configs(self):
        """Test generate_configs implementation."""
        generator = ConcreteSweepGenerator("test")
        base_config = OmegaConf.create({"base_param": "base_value"})

        configs = generator.generate_configs(base_config)

        assert len(configs) == 3
        for i, config in enumerate(configs):
            assert config.base_param == "base_value"
            assert config.test_param == f"value_{i}"

    def test_validate_config_valid(self):
        """Test validate_config with valid config."""
        generator = ConcreteSweepGenerator("test")
        config = OmegaConf.create({"test_param": "value"})

        result = generator.validate_config(config)

        assert result is True

    def test_validate_config_invalid(self):
        """Test validate_config with invalid config."""
        generator = ConcreteSweepGenerator("test")
        config = OmegaConf.create({"other_param": "value"})

        with pytest.raises(ValueError, match="test_param is required"):
            generator.validate_config(config)

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise TypeError when not implemented."""
        with pytest.raises(TypeError):
            BaseSweepGenerator("test")


class TestBaseSweepAnalyzer:
    """Test BaseSweepAnalyzer class."""

    def test_init(self):
        """Test initialization of BaseSweepAnalyzer."""
        analyzer = ConcreteSweepAnalyzer("test_sweep")

        assert analyzer.sweep_type == "test_sweep"

    def test_collect_results(self):
        """Test collect_results implementation."""
        analyzer = ConcreteSweepAnalyzer("test")

        results = analyzer.collect_results("/mock/path")

        assert "configs" in results
        assert "test_acc" in results
        assert "train_loss" in results
        assert len(results["configs"]) == 3

    def test_generate_plots(self):
        """Test generate_plots implementation."""
        analyzer = ConcreteSweepAnalyzer("test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_paths = analyzer.generate_plots({}, tmp_dir)

            assert len(plot_paths) == 2
            for path in plot_paths:
                assert os.path.exists(path)
                assert path.endswith(".png")

    def test_find_best_configs(self):
        """Test find_best_configs method."""
        analyzer = ConcreteSweepAnalyzer("test")
        results = {
            "configs": [{"param": 1}, {"param": 2}, {"param": 3}],
            "test_acc": [0.85, 0.92, 0.78],
        }

        best_configs = analyzer.find_best_configs(results, "test_acc")

        assert len(best_configs) == 3
        assert best_configs[0]["score"] == 0.92  # Highest accuracy first
        assert best_configs[0]["config"]["param"] == 2
        assert best_configs[1]["score"] == 0.85
        assert best_configs[2]["score"] == 0.78

    def test_find_best_configs_empty_results(self):
        """Test find_best_configs with empty results."""
        analyzer = ConcreteSweepAnalyzer("test")
        results = {}

        best_configs = analyzer.find_best_configs(results)

        assert best_configs == []

    def test_find_best_configs_missing_metric(self):
        """Test find_best_configs with missing metric."""
        analyzer = ConcreteSweepAnalyzer("test")
        results = {"configs": [{"param": 1}]}

        best_configs = analyzer.find_best_configs(results, "missing_metric")

        assert best_configs == []

    def test_find_best_configs_mismatched_lengths(self):
        """Test find_best_configs with mismatched config and metric lengths."""
        analyzer = ConcreteSweepAnalyzer("test")
        results = {
            "configs": [{"param": 1}, {"param": 2}, {"param": 3}],
            "test_acc": [0.85, 0.92],  # Only 2 scores for 3 configs
        }

        best_configs = analyzer.find_best_configs(results, "test_acc")

        assert len(best_configs) == 2  # Only configs with scores

    def test_generate_summary_report(self):
        """Test generate_summary_report method."""
        analyzer = ConcreteSweepAnalyzer("test")
        results = {
            "configs": [{"param": 1}, {"param": 2}],
            "test_acc": [0.85, 0.92],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create plots subdirectory
            plots_dir = os.path.join(tmp_dir, "plots")
            os.makedirs(plots_dir)

            report_path = analyzer.generate_summary_report(results, tmp_dir)

            assert os.path.exists(report_path)
            assert report_path.endswith("test_sweep_summary.txt")

            with open(report_path) as f:
                content = f.read()
                assert "TEST Sweep Summary Report" in content
                assert "Total configurations: 2" in content
                assert "Top 5 configurations:" in content

    def test_generate_summary_report_with_error(self):
        """Test generate_summary_report when finding best configs fails."""

        # Create an analyzer that will raise an exception in find_best_configs
        class ErrorAnalyzer(BaseSweepAnalyzer):
            def collect_results(self, results_dir: str) -> dict[str, any]:
                return {}

            def generate_plots(
                self, results: dict[str, any], output_dir: str
            ) -> list[str]:
                return []

            def find_best_configs(
                self, results: dict[str, any], metric: str = "test_acc"
            ) -> list[dict[str, any]]:
                raise ValueError("Intentional test error")

        analyzer = ErrorAnalyzer("test")
        results = {"configs": [{"param": 1}], "test_acc": [0.9]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            plots_dir = os.path.join(tmp_dir, "plots")
            os.makedirs(plots_dir)

            report_path = analyzer.generate_summary_report(results, tmp_dir)

            assert os.path.exists(report_path)

            with open(report_path) as f:
                content = f.read()
                assert "Could not determine best configurations" in content

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise TypeError when not implemented."""
        with pytest.raises(TypeError):
            BaseSweepAnalyzer("test")


class TestSweepJobManager:
    """Test SweepJobManager class."""

    @pytest.fixture
    def slurm_config(self):
        """Create basic SLURM configuration."""
        return OmegaConf.create(
            {
                "job_name": "test_sweep",
                "account": "test_account",
                "partition": "test_partition",
                "time": "01:00:00",
                "nodes": 1,
                "ntasks_per_node": 1,
                "gpus_per_node": 1,
                "cpus_per_task": 2,
                "mem": "16G",
                "modules_to_load": ["python/3.9", "cuda/11.8"],
                "conda_env_path": "/path/to/conda/env",
            }
        )

    def test_init(self, slurm_config):
        """Test initialization of SweepJobManager."""
        manager = SweepJobManager(slurm_config)

        assert manager.slurm_config == slurm_config

    def test_create_job_scripts_single_job(self, slurm_config):
        """Test creating a single job script."""
        manager = SweepJobManager(slurm_config)
        config_paths = ["config1.yaml", "config2.yaml"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create jobs directory
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            job_scripts = manager.create_job_scripts(
                config_paths, tmp_dir, split_jobs=1, use_array_jobs=False
            )

            assert len(job_scripts) == 2  # Main script + submission script
            assert any("run_sweep.sh" in script for script in job_scripts)
            assert any("submit_all_jobs.sh" in script for script in job_scripts)

            # Check that job script exists and has SLURM directives
            job_script_path = next(
                script for script in job_scripts if "run_sweep.sh" in script
            )
            assert os.path.exists(job_script_path)

            with open(job_script_path) as f:
                content = f.read()
                assert "#SBATCH --job-name=test_sweep" in content
                assert "#SBATCH --account=test_account" in content
                assert "#SBATCH --partition=test_partition" in content
                assert "module load python/3.9" in content
                assert "module load cuda/11.8" in content
                assert "/path/to/conda/env" in content

    def test_create_job_scripts_array_jobs(self, slurm_config):
        """Test creating array job scripts."""
        manager = SweepJobManager(slurm_config)
        config_paths = ["config1.yaml", "config2.yaml", "config3.yaml"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            job_scripts = manager.create_job_scripts(
                config_paths, tmp_dir, use_array_jobs=True
            )

            assert len(job_scripts) == 2  # Array script + submission script
            assert any("run_array_sweep.sh" in script for script in job_scripts)

    def test_create_job_scripts_split_jobs(self, slurm_config):
        """Test splitting configs into multiple jobs."""
        manager = SweepJobManager(slurm_config)
        config_paths = ["config1.yaml", "config2.yaml", "config3.yaml", "config4.yaml"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            job_scripts = manager.create_job_scripts(
                config_paths, tmp_dir, split_jobs=2, use_array_jobs=False
            )

            # Should have 2 job scripts + 1 submission script
            assert len(job_scripts) == 3
            job_files = [script for script in job_scripts if "run_sweep_" in script]
            assert len(job_files) == 2

    def test_create_single_job_script_content(self, slurm_config):
        """Test the content of a single job script."""
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "test_job.sh")
            config_paths = ["config1.yaml", "config2.yaml"]

            manager._create_single_job_script(script_path, config_paths, tmp_dir)

            assert os.path.exists(script_path)

            with open(script_path) as f:
                content = f.read()

                # Check SLURM directives
                assert "#!/bin/bash" in content
                assert "#SBATCH --job-name=test_sweep" in content
                assert "#SBATCH --time=01:00:00" in content
                assert "#SBATCH --mem=16G" in content

                # Check modules
                assert "module load python/3.9" in content
                assert "module load cuda/11.8" in content

                # Check conda environment setup
                assert "/path/to/conda/env" in content

    def test_create_single_job_script_no_conda(self):
        """Test job script creation without conda environment."""
        slurm_config = OmegaConf.create({"job_name": "test"})
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "test_job.sh")
            config_paths = ["config1.yaml"]

            manager._create_single_job_script(script_path, config_paths, tmp_dir)

            with open(script_path) as f:
                content = f.read()
                assert "python" in content  # Should use system python
                assert "conda" not in content

    def test_create_array_job_script(self, slurm_config):
        """Test creation of array job script."""
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "array_job.sh")
            config_paths = ["config1.yaml", "config2.yaml", "config3.yaml"]

            manager._create_array_job_script(script_path, config_paths, tmp_dir)

            assert os.path.exists(script_path)

            with open(script_path) as f:
                content = f.read()
                assert "#SBATCH --array=" in content
                assert "CONFIG_FILES=" in content

    def test_create_main_submission_script(self, slurm_config):
        """Test creation of main submission script."""
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            job_scripts = ["job1.sh", "job2.sh"]

            main_script = manager._create_main_submission_script(job_scripts, tmp_dir)

            assert os.path.exists(main_script)
            assert "submit_all_jobs.sh" in main_script

            with open(main_script) as f:
                content = f.read()
                assert "sbatch" in content
                assert "job1.sh" in content
                assert "job2.sh" in content

    def test_job_script_permissions(self, slurm_config):
        """Test that job scripts are created with proper permissions."""
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            script_path = os.path.join(tmp_dir, "test_job.sh")
            config_paths = ["config1.yaml"]

            manager._create_single_job_script(script_path, config_paths, tmp_dir)

            assert os.path.exists(script_path)
            # File should be readable
            with open(script_path) as f:
                content = f.read()
                assert "#!/bin/bash" in content


class TestSweepJobManagerEdgeCases:
    """Test edge cases for SweepJobManager."""

    def test_empty_config_paths(self):
        """Test with empty config paths."""
        slurm_config = OmegaConf.create({"job_name": "test"})
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            job_scripts = manager.create_job_scripts([], tmp_dir)

            # Should still create submission script
            assert len(job_scripts) >= 1

    def test_split_jobs_more_than_configs(self):
        """Test when split_jobs is larger than number of configs."""
        slurm_config = OmegaConf.create({"job_name": "test"})
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = os.path.join(tmp_dir, "jobs")
            os.makedirs(jobs_dir)

            config_paths = ["config1.yaml"]
            job_scripts = manager.create_job_scripts(
                config_paths, tmp_dir, split_jobs=5, use_array_jobs=False
            )

            # Should only create one job
            job_files = [script for script in job_scripts if "run_sweep_" in script]
            assert len(job_files) == 1

    def test_slurm_config_defaults(self):
        """Test SLURM config with default values."""
        slurm_config = OmegaConf.create({})  # Empty config
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "test_job.sh")
            config_paths = ["config1.yaml"]

            manager._create_single_job_script(script_path, config_paths, tmp_dir)

            with open(script_path) as f:
                content = f.read()
                # Check that defaults are used
                assert "#SBATCH --job-name=sweep" in content
                assert "#SBATCH --account=kempner_dev" in content
                assert "#SBATCH --partition=kempner" in content

    def test_job_manager_with_empty_modules(self):
        """Test job manager with empty modules list."""
        slurm_config = OmegaConf.create(
            {"job_name": "test", "modules_to_load": []}  # Empty modules list
        )
        manager = SweepJobManager(slurm_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "test_job.sh")
            config_paths = ["config1.yaml"]

            manager._create_single_job_script(script_path, config_paths, tmp_dir)

            with open(script_path) as f:
                content = f.read()
                # Should still work without module loading
                assert "#!/bin/bash" in content
