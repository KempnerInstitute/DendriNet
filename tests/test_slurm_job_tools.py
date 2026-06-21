"""Tests for SLURM job helper resource-estimation defaults."""

from dendritic_modeling.scripts.tools import submit_fsdp_job
from dendritic_modeling.scripts.tools.generate_slurm_job import _job_input_shape


def test_job_input_shape_uses_canonical_dataset_dimensions():
    assert _job_input_shape("mnist") == (784,)
    assert _job_input_shape("cifar10") == (3072,)
    assert _job_input_shape("cifar100") == (3072,)
    assert _job_input_shape("imagenet") == (150528,)
    assert _job_input_shape("unknown") == (784,)


def test_submit_fsdp_reuses_slurm_job_input_shape_helper():
    assert submit_fsdp_job._job_input_shape is _job_input_shape
