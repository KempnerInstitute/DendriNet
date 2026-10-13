"""Hot paths that avoid duplicate work and host synchronizations."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)


@pytest.mark.parametrize(
    "activation, shared", [("relu", True), (None, True), ("sigmoid", False)]
)
def test_transfer_activates_a_shared_input_once(activation, shared):
    layer = TransferLayer(
        input_dim=4, transfer_params={"output_activation": activation}
    )
    excitatory, inhibitory = layer(torch.rand(3, 4))

    assert (excitatory is inhibitory) is shared
    assert torch.equal(excitatory, inhibitory)


def test_variance_statistics_follow_the_batch_then_the_moving_average():
    layer = VarianceTopKLinear(4, 2, 2)
    first, second = torch.rand(5, 4), torch.rand(5, 4)
    layer.update_stats(first)
    assert torch.equal(layer.activation_mean, first.mean(0))
    layer.update_stats(second)
    m = layer.momentum
    expected = first.mean(0) * m + second.mean(0) * (1 - m)
    assert torch.equal(layer.activation_mean, expected)


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_variance_statistics_update_without_a_host_sync():
    layer = VarianceTopKLinear(16, 8, 4).cuda()
    x = torch.rand(6, 16, device="cuda")
    layer.update_stats(x)  # first update
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer.update_stats(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_local_learning_step_helpers_run_without_host_syncs():
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
        reduce_error_to_scalar,
    )
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
        _accumulate_local_logging_loss,
    )

    delta = torch.randn(8, 5, device="cuda")
    loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reduce_error_to_scalar(delta)
        _accumulate_local_logging_loss(loss_sum, delta.square().mean())
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_forbidden_inputs_are_masked_at_their_positions():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
        TopKLinear,
    )

    layer = TopKLinear(5, 3, 2, forbidden_input_index_per_output=[1, -1, 4])
    scores = layer._apply_forbidden_scores(torch.zeros(3, 5))
    expected = torch.zeros(3, 5)
    expected[0, 1] = expected[2, 4] = float("-inf")
    assert torch.equal(scores, expected)
    assert "_forbidden_rows" not in layer.state_dict()  # checkpoints unchanged


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_recurrent_step_helpers_run_without_host_syncs():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
        TopKLinear,
    )
    from dendritic_modeling.networks.architectures.recurrent.spiking_soma import LIFSoma

    soma = LIFSoma(output_mode="rate").cuda()
    layer = TopKLinear(5, 3, 2, forbidden_input_index_per_output=[1, -1, 4]).cuda()
    voltage = torch.rand(4, 3, device="cuda")
    scores = torch.rand(3, 5, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        soma(voltage, None, None, None)
        layer._apply_forbidden_scores(scores)
        layer._apply_forbidden_mask(scores)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_dense_to_sparse_schedule_reads_no_gpu_scalars():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
        DenseToSparseLinear,
    )

    layer = DenseToSparseLinear(
        12, 6, 3, initial_density=1.0, start_step=1, end_step=4, update_interval=1
    ).cuda()
    assert layer.current_k == 12  # the host mirrors are in place
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer.advance_sparsity_schedule(2)  # past the start step
        assert layer.current_k < 12
        layer.weight_mask()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_dense_to_sparse_mirrors_follow_state_loads():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
        DenseToSparseLinear,
    )

    def layer():
        return DenseToSparseLinear(
            12, 6, 3, initial_density=1.0, start_step=0, end_step=4, update_interval=1
        )

    trained, fresh = layer(), layer()
    trained.advance_sparsity_schedule(3)
    fresh.load_state_dict(trained.state_dict())
    assert fresh.current_k == trained.current_k < 12
    with torch.no_grad():
        fresh.sparsity_step.zero_()  # a direct buffer write is read back
    assert fresh.current_k == 12


def _deepst(**options):
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
        DeepstLinear,
    )

    return DeepstLinear(20, 6, target_density=0.3, seed=5, **options)


def test_deepst_rewired_record_and_step_mirrors_follow_every_writer():
    layer = _deepst()
    assert layer.consume_rewired_mask() is None
    with torch.no_grad():
        layer.pre_w.fill_(-10.0)  # every contact falls below threshold
    layer.apply_rewiring()
    rewired = layer.consume_rewired_mask()
    assert rewired is not None and bool(rewired.any())
    assert layer.consume_rewired_mask() is None  # consumed and cleared

    layer._last_rewired_mask[0, 0] = True  # a writer outside the event
    assert torch.equal(layer.consume_rewired_mask().nonzero(), torch.tensor([[0, 0]]))

    state = layer.state_dict()
    state["rewire_step"] = torch.tensor(41)
    layer.load_state_dict(state)
    layer.apply_rewiring()
    assert int(layer.rewire_step) == 42 and layer._rewire_step_value() == 42


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_deepst_steps_between_rewiring_events_read_nothing_from_the_gpu():
    layer = _deepst(rewire_frequency=4).cuda()
    layer.apply_rewiring()  # step 1: the mirrors read the moved buffers once
    layer.consume_rewired_mask()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(2):  # steps 2 and 3 do not rewire
            layer.apply_rewiring()
            assert layer.consume_rewired_mask() is None
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert int(layer.rewire_step) == 3


def test_validation_sums_match_host_sums_of_batch_losses():
    from dendritic_modeling.training.strategies.standard import _ValidationSums

    losses = [(0.1, 16), (2.0 / 3.0, 16), (1e-3, 5)]
    sums = _ValidationSums("cpu")
    batch_sum = weighted_sum = 0.0
    for value, batch_size in losses:
        loss = torch.tensor(value)
        sums.add(loss, batch_size)
        batch_sum += loss.item()
        weighted_sum += loss.item() * batch_size

    stats = sums.stats()
    assert (stats.batch_loss_sum, stats.sample_weighted_loss_sum) == (
        batch_sum,
        weighted_sum,
    )
    assert (stats.batch_count, stats.sample_count) == (3, 37)


@pytest.mark.requires_gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_validation_sums_accumulate_without_host_syncs():
    from dendritic_modeling.training.strategies.standard import _ValidationSums

    sums = _ValidationSums("cuda")
    loss = torch.rand((), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sums.add(loss, 7)
        sums.add(loss, 3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert sums.stats().sample_count == 10
