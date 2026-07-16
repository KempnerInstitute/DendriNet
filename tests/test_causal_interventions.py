"""Tests for phase/pathway/depth interventions on current population RNNs."""

from __future__ import annotations

import math

import pytest
import torch

from dendritic_modeling.analysis.utils.causal_interventions import (
    CausalInterventionBatch,
    FixedMeanCalibration,
    PathwayInterventionTarget,
    calibrate_fixed_mean_pathways,
    evaluate_same_checkpoint_intervention,
    manual_population_classifier_forward,
    resolve_pathway_targets,
    romo_phase_masks,
)
from dendritic_modeling.models import Classifier, RecurrentClassifier
from dendritic_modeling.networks.architectures.classical import Identity
from dendritic_modeling.networks.architectures.classical.mlp import MLP
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _population_classifier(
    *, output_mode: str = "last", learned_global_scale: bool = False
) -> Classifier:
    torch.manual_seed(3)
    core = PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=5,
            input_projection_dims=[4],
            output_mode=output_mode,
            layers=[
                PopulationLayerConfig(
                    name="microcircuit",
                    recurrent=True,
                    readout_population="pyr",
                    population_defaults={
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 2,
                        "rec_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                        "weight_transform": "identity",
                        "topk_type": "standard",
                        "allow_self_recurrence": True,
                    },
                    populations=[
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 3,
                            "branch_factors": [2],
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [],
                        },
                    ],
                    connections=[
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "pv"},
                        {
                            "source": "pyr",
                            "target": "pyr",
                            "timing": "delayed",
                        },
                        {
                            "source": "pv",
                            "target": "pyr",
                            "timing": "delayed",
                        },
                        {
                            "source": "pyr",
                            "target": "pv",
                            "timing": "delayed",
                        },
                        {
                            "source": "pv",
                            "target": "pv",
                            "timing": "delayed",
                        },
                    ],
                )
            ],
        )
    )
    encoder = MLP(input_dim=4, hidden_dims=[5], activation="relu")
    if learned_global_scale:
        decoder = Identity(3)
        model = Classifier(
            encoder,
            core,
            decoder,
            fixed_output_scale=1.4,
            output_scale_mode="global",
        )
        with torch.no_grad():
            model.log_global_output_scale.fill_(math.log(2.2))
    else:
        decoder = MLP(input_dim=3, hidden_dims=[4], activation="relu", output_dim=2)
        model = Classifier(
            encoder,
            core,
            decoder,
            learned_output_scale=False,
            fixed_output_scale=1.7,
        )

    # Make every feedforward path observably nonzero for direct hook assertions.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.15)
                if module.bias is not None:
                    module.bias.fill_(0.05)
            pre_w = getattr(module, "pre_w", None)
            if isinstance(pre_w, torch.Tensor):
                pre_w.fill_(0.2)
    return model


def _romo_batch(offset: float = 0.0):
    delays = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    lengths = 2 * 2 + delays + 1
    generator = torch.Generator().manual_seed(19)
    inputs = torch.rand(4, 7, 4, generator=generator) + 0.2 + offset
    masks = romo_phase_masks(
        delays,
        max_sequence_length=7,
        stimulus_duration=2,
        response_duration=1,
        seq_lengths=lengths,
    )
    return inputs, lengths, delays, masks


def _target(depth: int = 1, pathway: str = "ff_excitatory"):
    return PathwayInterventionTarget(
        layer="microcircuit",
        target_population="pyr",
        soma_relative_depth=depth,
        pathway=pathway,
    )


def _capture_outputs(module, callback):
    return module.register_forward_hook(
        lambda _module, _inputs, output: callback(output.detach().clone())
    )


def test_variable_delay_romo_phase_masks_partition_each_valid_trial() -> None:
    _, lengths, _, masks = _romo_batch()
    combined = torch.zeros_like(masks["all_valid"], dtype=torch.int64)
    for phase in ("first_stimulus", "delay", "second_stimulus", "response"):
        combined += masks[phase].long()

    assert torch.equal(combined.bool(), masks["all_valid"])
    assert int(combined.max()) == 1
    assert masks["delay"][0].sum() == 1
    assert masks["delay"][1].sum() == 2
    assert torch.equal(masks["all_valid"].sum(dim=1), lengths)


@pytest.mark.parametrize("output_mode", ["last", "mean"])
@pytest.mark.parametrize("learned_global_scale", [False, True])
def test_noop_manual_unroll_matches_full_classifier_forward(
    output_mode: str, learned_global_scale: bool
) -> None:
    model = _population_classifier(
        output_mode=output_mode,
        learned_global_scale=learned_global_scale,
    )
    inputs, lengths, _, _ = _romo_batch()
    model.eval()
    with torch.no_grad():
        expected = model(inputs, seq_lengths=lengths)

    actual = manual_population_classifier_forward(
        model,
        inputs,
        seq_lengths=lengths,
    ).logits

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("output_mode", ["last", "mean"])
def test_real_recurrent_classifier_noop_and_intervention_path(output_mode: str) -> None:
    generic = _population_classifier(output_mode=output_mode)
    model = RecurrentClassifier(
        generic.encoder_network,
        generic.core_network,
        generic.decoder_network,
    )
    inputs, lengths, _, masks = _romo_batch()
    labels = torch.tensor([0, 1, 0, 1])
    model.eval()
    with torch.no_grad():
        expected = model(inputs, seq_lengths=lengths)

    no_op = manual_population_classifier_forward(
        model,
        inputs,
        seq_lengths=lengths,
    )
    intervention = evaluate_same_checkpoint_intervention(
        model,
        inputs,
        labels,
        targets=[_target(depth=0)],
        phase_mask=masks["delay"],
        mode="zero",
        seq_lengths=lengths,
    )

    assert torch.equal(no_op.logits, expected)
    assert intervention.baseline_logits.shape == expected.shape
    assert intervention.intervened_logits.shape == expected.shape


def test_zero_hook_changes_only_selected_phase_pathway_and_depth() -> None:
    model = _population_classifier()
    inputs, lengths, _, masks = _romo_batch()
    target = _target(depth=1)
    resolved = resolve_pathway_targets(model.core_network, [target])[0]
    non_target = resolve_pathway_targets(model.core_network, [_target(depth=0)])[0]
    recurrent_excitation = resolve_pathway_targets(
        model.core_network, [_target(depth=1, pathway="rec_excitatory")]
    )[0]
    recurrent_inhibition = resolve_pathway_targets(
        model.core_network, [_target(depth=1, pathway="rec_inhibitory")]
    )[0]
    target_branch = model.core_network.layers[0].populations["pyr"].branch_layers[0]
    assert recurrent_excitation.module is target_branch.branch_recurrent
    assert recurrent_inhibition.module is target_branch.branch_rec_inhibition
    baseline_target: list[torch.Tensor] = []
    baseline_non_target: list[torch.Tensor] = []
    intervened_target: list[torch.Tensor] = []
    intervened_non_target: list[torch.Tensor] = []

    handles = [
        _capture_outputs(resolved.module, baseline_target.append),
        _capture_outputs(non_target.module, baseline_non_target.append),
    ]
    try:
        manual_population_classifier_forward(model, inputs, seq_lengths=lengths)
    finally:
        for handle in handles:
            handle.remove()
    handles = [
        _capture_outputs(resolved.module, intervened_target.append),
        _capture_outputs(non_target.module, intervened_non_target.append),
    ]
    try:
        result = manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=lengths,
            targets=[target],
            phase_mask=masks["first_stimulus"],
            mode="zero",
        )
    finally:
        for handle in handles:
            handle.remove()

    baseline_target_tensor = torch.stack(baseline_target, dim=1)
    intervened_target_tensor = torch.stack(intervened_target, dim=1)
    baseline_non_target_tensor = torch.stack(baseline_non_target, dim=1)
    intervened_non_target_tensor = torch.stack(intervened_non_target, dim=1)
    phase = masks["first_stimulus"]
    assert torch.count_nonzero(intervened_target_tensor[phase]) == 0
    assert torch.equal(intervened_target_tensor[~phase], baseline_target_tensor[~phase])
    assert torch.equal(intervened_non_target_tensor, baseline_non_target_tensor)

    key = resolved.key
    assert len(result.audit.targets) == 1
    assert result.audit.targets[0]["soma_relative_depth"] == 1
    assert result.audit.targets[0]["pathway"] == "ff_excitatory"
    assert torch.equal(result.audit.touched_masks[key], phase)
    assert not bool(result.audit.directly_changed_masks[key][~phase].any())
    assert bool(result.audit.directly_changed_masks[key][phase].all())


def test_shuffle_is_deterministic_local_and_preserves_active_batch_multiset() -> None:
    model = _population_classifier()
    inputs, lengths, _, masks = _romo_batch()
    target = _target(depth=0)
    resolved = resolve_pathway_targets(model.core_network, [target])[0]

    def run_once():
        captured: list[torch.Tensor] = []
        handle = _capture_outputs(resolved.module, captured.append)
        try:
            result = manual_population_classifier_forward(
                model,
                inputs,
                seq_lengths=lengths,
                targets=[target],
                phase_mask=masks["first_stimulus"],
                mode="shuffle",
                shuffle_seed=41,
            )
        finally:
            handle.remove()
        return result, torch.stack(captured, dim=1)

    baseline_values: list[torch.Tensor] = []
    handle = _capture_outputs(resolved.module, baseline_values.append)
    try:
        manual_population_classifier_forward(model, inputs, seq_lengths=lengths)
    finally:
        handle.remove()
    baseline = torch.stack(baseline_values, dim=1)

    rng_before = torch.random.get_rng_state().clone()
    first, first_values = run_once()
    rng_after = torch.random.get_rng_state().clone()
    second, second_values = run_once()

    assert torch.equal(rng_after, rng_before)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first_values, second_values)
    assert bool((first_values != baseline).any())
    for timestep in (0, 1):
        assert torch.equal(
            torch.sort(first_values[:, timestep], dim=0).values,
            torch.sort(baseline[:, timestep], dim=0).values,
        )
    assert torch.equal(first_values[:, 2:], baseline[:, 2:])


def test_fixed_mean_uses_separate_calibration_not_evaluation_batch_mean() -> None:
    model = _population_classifier()
    inputs, lengths, _, masks = _romo_batch()
    reference_inputs, reference_lengths, _, reference_masks = _romo_batch(offset=3.0)
    target = _target(depth=0)
    resolved = resolve_pathway_targets(model.core_network, [target])[0]

    reference_outputs: list[torch.Tensor] = []
    handle = _capture_outputs(resolved.module, reference_outputs.append)
    try:
        manual_population_classifier_forward(
            model,
            reference_inputs,
            seq_lengths=reference_lengths,
        )
    finally:
        handle.remove()
    reference_output = torch.stack(reference_outputs, dim=1)
    expected_reference = (
        reference_output[reference_masks["first_stimulus"]].double().mean(dim=0)
    )

    calibration = calibrate_fixed_mean_pathways(
        model,
        [
            CausalInterventionBatch(
                reference_inputs,
                reference_masks["first_stimulus"],
                reference_lengths,
            )
        ],
        targets=[target],
        source_id="train_reference_split",
    )
    assert torch.allclose(
        calibration.references[resolved.key], expected_reference, atol=1e-12, rtol=0
    )

    baseline_outputs: list[torch.Tensor] = []
    handle = _capture_outputs(resolved.module, baseline_outputs.append)
    try:
        manual_population_classifier_forward(model, inputs, seq_lengths=lengths)
    finally:
        handle.remove()
    baseline_output = torch.stack(baseline_outputs, dim=1)
    evaluation_batch_mean = (
        baseline_output[masks["first_stimulus"]].double().mean(dim=0)
    )
    assert not torch.allclose(expected_reference, evaluation_batch_mean)

    intervened_outputs: list[torch.Tensor] = []
    handle = _capture_outputs(resolved.module, intervened_outputs.append)
    try:
        manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=lengths,
            targets=[target],
            phase_mask=masks["first_stimulus"],
            mode="fixed_mean",
            fixed_mean_calibration=calibration,
        )
    finally:
        handle.remove()
    intervened_output = torch.stack(intervened_outputs, dim=1)
    selected = intervened_output[masks["first_stimulus"]]
    expected_rows = expected_reference.to(selected).unsqueeze(0).expand_as(selected)
    assert torch.equal(selected, expected_rows)
    assert torch.equal(
        intervened_output[~masks["first_stimulus"]],
        baseline_output[~masks["first_stimulus"]],
    )

    with pytest.raises(ValueError, match="separate reference set"):
        manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=lengths,
            targets=[target],
            phase_mask=masks["first_stimulus"],
            mode="fixed_mean",
        )


def test_hooks_caches_and_training_modes_restore_after_intervention_exception() -> None:
    model = _population_classifier()
    model.train()
    model.decoder_network.eval()  # Deliberately preserve a mixed mode tree.
    inputs, lengths, _, masks = _romo_batch()
    target = _target(depth=0)
    resolved = resolve_pathway_targets(model.core_network, [target])[0]
    training_before = {id(module): module.training for module in model.modules()}
    hook_count_before = len(resolved.module._forward_hooks)
    cache_before = getattr(resolved.module, "_recurrent_weight_cache_enabled", False)
    invalid_calibration = FixedMeanCalibration(
        references={resolved.key: torch.zeros(1)},
        counts={resolved.key: 10},
        source_id="invalid_shape_reference",
    )

    with pytest.raises(ValueError, match="expected"):
        manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=lengths,
            targets=[target],
            phase_mask=masks["first_stimulus"],
            mode="fixed_mean",
            fixed_mean_calibration=invalid_calibration,
        )

    assert len(resolved.module._forward_hooks) == hook_count_before
    assert (
        getattr(resolved.module, "_recurrent_weight_cache_enabled", False)
        == cache_before
    )
    assert {
        id(module): module.training for module in model.modules()
    } == training_before


def test_same_checkpoint_result_contains_per_trial_predictions_and_accuracy_drop() -> (
    None
):
    model = _population_classifier()
    inputs, lengths, _, masks = _romo_batch()
    labels = torch.tensor([0, 1, 0, 1])
    result = evaluate_same_checkpoint_intervention(
        model,
        inputs,
        labels,
        targets=[_target(depth=0)],
        phase_mask=masks["second_stimulus"],
        mode="zero",
        seq_lengths=lengths,
    )

    assert result.baseline_logits.shape == (4, 2)
    assert result.intervened_logits.shape == (4, 2)
    assert torch.equal(
        result.per_trial_accuracy_drop,
        result.baseline_correct.float() - result.intervened_correct.float(),
    )
    assert result.accuracy_drop == pytest.approx(
        result.baseline_accuracy - result.intervened_accuracy
    )
    payload = result.to_dict()
    assert payload["analysis_type"] == "same_checkpoint_phase_pathway_intervention"
    assert len(payload["trials"]) == 4


def test_unsupported_model_and_intervention_mode_fail_closed() -> None:
    inputs, lengths, _, masks = _romo_batch()
    with pytest.raises(TypeError, match="Classifier or RecurrentClassifier"):
        manual_population_classifier_forward(torch.nn.Linear(4, 2), inputs)

    model = _population_classifier()
    with pytest.raises(ValueError, match="mode must be one of"):
        manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=lengths,
            targets=[_target(depth=0)],
            phase_mask=masks["delay"],
            mode="batch_mean",
        )
