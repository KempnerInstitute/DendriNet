import pytest
import torch

from dendritic_modeling.analysis.utils.tau_interventions import (
    temporary_level_tau_override,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationConfig,
    StatefulDendriNet,
)


def _population(*, learnable_tau: bool) -> StatefulDendriNet:
    torch.manual_seed(19)
    config = PopulationConfig(
        n_neurons=4,
        branch_factors=[2, 2],
        ff_excitatory_synapses=3,
        level_taus=[125.0, 25.0, 5.0],
        learnable_tau=learnable_tau,
        use_noise=False,
    )
    population = StatefulDendriNet(
        pop_config=config,
        excitatory_input_dim=6,
        dt=1.0,
    )
    population.eval()
    return population


def _unroll(population: StatefulDendriNet) -> torch.Tensor:
    generator = torch.Generator().manual_seed(41)
    inputs = torch.rand(2, 12, 6, generator=generator)
    state = None
    outputs = []
    with torch.inference_mode():
        for timestep in range(inputs.shape[1]):
            output, state = population.step(inputs[:, timestep], state=state)
            outputs.append(output)
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("learnable_tau", [False, True])
def test_native_tau_override_is_bitwise_noop_and_restores_state(learnable_tau):
    population = _population(learnable_tau=learnable_tau)
    before = {
        key: tensor.detach().clone() for key, tensor in population.state_dict().items()
    }
    baseline = _unroll(population)

    with temporary_level_tau_override(population, [125.0, 25.0, 5.0]) as audit:
        intervened = _unroll(population)
        assert audit.state_tensor_noop is True
        assert audit.state_field == ("log_tau" if learnable_tau else "decays")
        assert torch.equal(baseline, intervened)

    after = population.state_dict()
    assert set(after) == set(before)
    assert all(torch.equal(after[key], before[key]) for key in before)


@pytest.mark.parametrize("learnable_tau", [False, True])
def test_tau_override_changes_dynamics_and_restores_after_exception(learnable_tau):
    population = _population(learnable_tau=learnable_tau)
    before = {
        key: tensor.detach().clone() for key, tensor in population.state_dict().items()
    }
    baseline = _unroll(population)

    with pytest.raises(RuntimeError, match="deliberate"):
        with temporary_level_tau_override(population, [5.0, 25.0, 125.0]) as audit:
            intervened = _unroll(population)
            assert audit.state_tensor_noop is False
            assert not torch.equal(baseline, intervened)
            raise RuntimeError("deliberate")

    after = population.state_dict()
    assert all(torch.equal(after[key], before[key]) for key in before)


@pytest.mark.parametrize(
    "taus, error",
    [
        ([5.0, 25.0], "Expected 3"),
        ([5.0, 0.0, 125.0], "positive finite"),
        ([5.0, float("inf"), 125.0], "positive finite"),
    ],
)
def test_tau_override_rejects_invalid_schedules(taus, error):
    population = _population(learnable_tau=False)
    with pytest.raises(ValueError, match=error):
        with temporary_level_tau_override(population, taus):
            pass


def test_tau_override_rejects_other_module_types():
    with pytest.raises(TypeError, match="requires StatefulDendriNet"):
        with temporary_level_tau_override(torch.nn.Linear(2, 2), [1.0]):
            pass
