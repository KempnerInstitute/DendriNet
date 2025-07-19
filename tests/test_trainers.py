import pytest
import torch
from torch.utils.data import TensorDataset
from dendritic_modeling.training.factory import get_trainer
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

@pytest.fixture
def simple_model():
    encoder = Identity(10)
    core = get_architecture('MLP', {'input_dim': 10, 'hidden_dims': [5], 'output_dim': 2})
    decoder = Identity(2)
    return Classifier(encoder, core, decoder)

@pytest.fixture
def sample_data():
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)

def test_standard_trainer(simple_model, sample_data):
    optimizer = torch.optim.Adam(simple_model.parameters())
    trainer = get_trainer('standard', optimizer, {'epochs': 1, 'batch_size': 10})
    results = trainer.train(simple_model, sample_data, sample_data)
    assert 'train_losses' in results
    assert len(results['train_losses']) == 1
    assert 'valid_losses' in results

def test_fsdp_trainer_init(simple_model):
    optimizer = torch.optim.Adam(simple_model.parameters())
    fsdp_config = {'sharding_strategy': 'FULL_SHARD'}
    trainer = FSDPTrainer(0, 1, fsdp_config, 'checkpoints/', optimizer_config={'lr': 0.001}, epochs=1)
    assert trainer is not None
    assert isinstance(trainer, FSDPTrainer)
