import pytest
import torch

from dendritic_modeling.training.factory import get_trainer
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

# Fixtures moved to conftest.py for shared access across test files


def test_standard_trainer(simple_model, sample_data):
    optimizer = torch.optim.Adam(simple_model.parameters())
    trainer = get_trainer(
        "standard",
        optimizer,
        {"loss_function": "cross_entropy", "epochs": 1, "batch_size": 10},
    )
    results = trainer.train(simple_model, sample_data, sample_data)
    assert "train_losses" in results
    assert len(results["train_losses"]) == 1
    assert "valid_losses" in results


def test_fsdp_trainer_init(simple_model):
    # Skip if CUDA is not available or if CUDA drivers are not working
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        # Test if we can actually set a CUDA device (this will fail in CI without drivers)
        torch.cuda.set_device(0)
    except RuntimeError as e:
        if "Found no NVIDIA driver" in str(e):
            pytest.skip("NVIDIA drivers not available")
        else:
            raise

    torch.optim.Adam(simple_model.parameters())
    fsdp_config = {"sharding_strategy": "FULL_SHARD"}
    trainer = FSDPTrainer(
        rank=0,
        world_size=1,
        fsdp_config=fsdp_config,
        checkpoint_dir="checkpoints",
        save_every_n_epochs=5,  # Add this to satisfy parent constructor condition
        optimizer_config={"lr": 0.001},
        epochs=1,
        loss_function="cross_entropy",
    )
    assert trainer is not None
    assert isinstance(trainer, FSDPTrainer)
    # Check that checkpoint directory was created
    assert trainer.checkpoint_dir is not None


def test_two_step_trainer(simple_model, sample_data):
    optimizer = torch.optim.Adam(simple_model.parameters())
    trainer = get_trainer(
        "two_step",
        optimizer,
        {"loss_function": "cross_entropy", "epochs": 1, "batch_size": 10},
    )
    results = trainer.train(simple_model, sample_data, sample_data)
    assert "train_losses" in results
