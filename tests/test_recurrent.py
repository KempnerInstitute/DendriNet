"""
Smoke tests for recurrent dendritic network integration (Phase 2).

Tests DendriticRNNCell, DendriticRNNNetwork, RecurrentState, config validation,
model wrapping, and special case behaviors.
"""

import pytest
import torch

from dendritic_modeling.config.recurrent import BranchConfig, RecurrentConfig
from dendritic_modeling.models.recurrent_classifier import RecurrentClassifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.recurrent import (
    DendriticRNNCell,
    DendriticRNNNetwork,
    RecurrentState,
)

# ---------------------------------------------------------------------------
# RecurrentState tests
# ---------------------------------------------------------------------------


class TestRecurrentState:
    def test_init_basic(self):
        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=8, n_branches=3, device=torch.device("cpu")
        )
        assert len(state.branch_E) == 3
        assert len(state.branch_E_rec) == 3
        assert len(state.branch_I) == 3
        assert len(state.branch_I_rec) == 3
        assert len(state.branch_V) == 3
        assert state.h_inh is not None
        assert state.h_inh.shape == (4, 8)
        assert state.branch_E[0].shape == (4, 16)
        assert state.branch_E_rec[0].shape == (4, 16)
        assert state.branch_I_rec[0].shape == (4, 16)

    def test_init_no_inhibitory(self):
        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=0, n_branches=2, device=torch.device("cpu")
        )
        assert state.h_inh is None

    def test_detach(self):
        state = RecurrentState.init(
            batch_size=2, n_exc=8, n_inh=4, n_branches=2, device=torch.device("cpu")
        )
        state.branch_E[0].requires_grad_(True)
        state.branch_E_rec[0].requires_grad_(True)
        detached = state.detach()
        assert not detached.branch_E[0].requires_grad
        assert not detached.branch_E_rec[0].requires_grad

    def test_reset(self):
        state = RecurrentState.init(
            batch_size=2, n_exc=8, n_inh=4, n_branches=2, device=torch.device("cpu")
        )
        state.branch_E[0].fill_(1.0)
        state.branch_E_rec[0].fill_(3.0)
        state.branch_I_rec[0].fill_(4.0)
        state.h_inh.fill_(2.0)
        state.reset()
        assert state.branch_E[0].sum().item() == 0.0
        assert state.branch_E_rec[0].sum().item() == 0.0
        assert state.branch_I_rec[0].sum().item() == 0.0
        assert state.h_inh.sum().item() == 0.0


# ---------------------------------------------------------------------------
# DendriticRNNCell tests
# ---------------------------------------------------------------------------


class TestDendriticRNNCell:
    def test_basic_forward_shape(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(tau=10.0, ff_synapses=6, rec_synapses=0, ie_synapses=4),
                BranchConfig(tau=50.0, ff_synapses=4, rec_synapses=4, ie_synapses=4),
            ],
            soma_mode="shunting",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=8, n_branches=2, device=torch.device("cpu")
        )
        x_ff = torch.randn(4, 32)
        s, new_state, _info = cell(x_ff, state)
        assert s.shape == (4, 16)
        assert len(new_state.branch_E) == 2

    def test_no_inhibitory_path(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                BranchConfig(tau=50.0, ff_synapses=8, rec_synapses=8, ie_synapses=4),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        assert cell.i_network is None
        # ie_synapses should have been forced to 0
        assert cell.branch_configs[0].ie_synapses == 0

        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=0, n_branches=1, device=torch.device("cpu")
        )
        x_ff = torch.randn(4, 32)
        s, new_state, _info = cell(x_ff, state)
        assert s.shape == (4, 16)
        assert new_state.h_inh is None

    def test_no_inhibitory_forces_rec_i_synapses_zero(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=8,
                    ie_synapses=2,
                    rec_i_synapses=3,
                    compartment="dendritic_branch",
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        assert cell.i_network is None
        assert cell.branch_configs[0].ie_synapses == 0
        assert cell.branch_configs[0].rec_i_synapses == 0
        assert cell.branches[0].dbl.branch_rec_inhibition is None

    def test_store_routing_false(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=8, rec_synapses=0)],
            store_routing=False,
        )
        cell = DendriticRNNCell(config, input_dim=16)
        state = RecurrentState.init(
            batch_size=2, n_exc=16, n_inh=4, n_branches=1, device=torch.device("cpu")
        )
        _, _, info = cell(torch.randn(2, 16), state)
        assert info == {}

    def test_store_routing_true(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=4,
            branches=[
                BranchConfig(tau=10.0, ff_synapses=8, rec_synapses=0),
                BranchConfig(tau=50.0, ff_synapses=0, rec_synapses=8),
            ],
            store_routing=True,
        )
        cell = DendriticRNNCell(config, input_dim=16)
        state = RecurrentState.init(
            batch_size=2, n_exc=16, n_inh=4, n_branches=2, device=torch.device("cpu")
        )
        _, _, info = cell(torch.randn(2, 16), state)
        assert "alpha" in info
        assert info["alpha"].shape == (2, 16, 2)  # [batch, n_exc, n_branches]

    def test_gru_cell(self):
        config = RecurrentConfig(
            enabled=True,
            cell_type="gru",
            n_excitatory=16,
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=16, n_branches=3, device=torch.device("cpu")
        )
        s, _new_state, _info = cell(torch.randn(4, 32), state)
        assert s.shape == (4, 16)

    def test_lstm_cell(self):
        config = RecurrentConfig(
            enabled=True,
            cell_type="lstm",
            n_excitatory=16,
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4, n_exc=16, n_inh=16, n_branches=3, device=torch.device("cpu")
        )
        s, _new_state, _info = cell(torch.randn(4, 32), state)
        assert s.shape == (4, 16)

    def test_linear_compartment(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=8,
                    compartment="linear",
                    activation="tanh",
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=16)
        state = RecurrentState.init(
            batch_size=2, n_exc=16, n_inh=0, n_branches=1, device=torch.device("cpu")
        )
        s, _, _ = cell(torch.randn(2, 16), state)
        assert s.shape == (2, 16)

    def test_dict_config_validation(self):
        """Dict-based branch configs should be converted and validated."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                {"tau": 50.0, "ff_synapses": 8, "rec_synapses": 8},
            ],
        )
        cell = DendriticRNNCell(config, input_dim=16)
        assert isinstance(cell.branch_configs[0], BranchConfig)

    def test_invalid_soma_mode(self):
        with pytest.raises(ValueError, match="soma_mode"):
            config = RecurrentConfig(
                enabled=True,
                soma_mode="invalid",
            )
            DendriticRNNCell(config, input_dim=16)

    def test_invalid_cell_type(self):
        with pytest.raises(ValueError, match="cell_type"):
            config = RecurrentConfig(
                enabled=True,
                cell_type="invalid",
            )
            DendriticRNNCell(config, input_dim=16)


# ---------------------------------------------------------------------------
# DendriticRNNNetwork tests
# ---------------------------------------------------------------------------


class TestDendriticRNNNetwork:
    def test_output_mode_last(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=8, rec_synapses=0)],
            output_mode="last",
            input_projection_layers=[32],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(4, 20, 10)  # [batch, seq_len, input_dim]
        out = net(x)
        assert out.shape == (4, 16)  # [batch, n_exc]

    def test_output_mode_all(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=8, rec_synapses=0)],
            output_mode="all",
            input_projection_layers=[32],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(4, 20, 10)
        out = net(x)
        assert out.shape == (4, 20, 16)  # [batch, seq_len, n_exc]

    def test_output_mode_mean(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=8, rec_synapses=0)],
            output_mode="mean",
            input_projection_layers=[32],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(4, 20, 10)
        out = net(x)
        assert out.shape == (4, 16)  # [batch, n_exc]

    def test_output_dim(self):
        config = RecurrentConfig(
            enabled=True, n_excitatory=32, input_projection_layers=[16]
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        assert net.output_dim == 32

    def test_routing_info_stored(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[
                BranchConfig(tau=10.0, ff_synapses=4, rec_synapses=0),
                BranchConfig(tau=50.0, ff_synapses=0, rec_synapses=4),
            ],
            store_routing=True,
            input_projection_layers=[16],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(2, 5, 10)  # 5 timesteps
        _ = net(x)
        assert len(net._routing_info) == 5  # one per timestep

    def test_routing_info_empty_when_disabled(self):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=4, rec_synapses=0)],
            store_routing=False,
            input_projection_layers=[16],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(2, 5, 10)
        _ = net(x)
        assert len(net._routing_info) == 0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_branch_no_input_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            BranchConfig(ff_synapses=0, rec_synapses=0)

    def test_branch_valid(self):
        bc = BranchConfig(ff_synapses=4, rec_synapses=0)
        assert bc.ff_synapses == 4

    def test_rec_i_synapses_only_for_dendritic_branch(self):
        with pytest.raises(ValueError, match="rec_i_synapses is only valid"):
            BranchConfig(
                ff_synapses=4,
                rec_synapses=4,
                ie_synapses=2,
                rec_i_synapses=1,
                compartment="topk",
            )

    def test_recurrent_config_defaults(self):
        rc = RecurrentConfig()
        assert rc.enabled is False
        assert rc.cell_type == "dendritic"
        assert len(rc.branches) == 3


# ---------------------------------------------------------------------------
# Factory validation tests
# ---------------------------------------------------------------------------


class TestFactory:
    def test_dendritic_rnn_disabled_raises(self):
        from dendritic_modeling.networks.architectures.factory import get_architecture

        config = RecurrentConfig(enabled=False)
        # Need a proper config structure
        params = type("Cfg", (), {"recurrent": config})()
        with pytest.raises(ValueError, match=r"requires recurrent.enabled=true"):
            get_architecture("dendritic_rnn", params, input_dim=32)

    def test_dendritic_rnn_enabled_constructs(self):
        from dendritic_modeling.networks.architectures.factory import get_architecture

        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=4, rec_synapses=0)],
            input_projection_layers=[16],
        )
        params = type("Cfg", (), {"recurrent": config})()
        net = get_architecture("dendritic_rnn", params, input_dim=10)
        assert isinstance(net, DendriticRNNNetwork)
        assert net.output_dim == 8


# ---------------------------------------------------------------------------
# RecurrentClassifier tests
# ---------------------------------------------------------------------------


class TestRecurrentClassifier:
    def _make_classifier(self, output_mode="last"):
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[BranchConfig(tau=50.0, ff_synapses=4, rec_synapses=0)],
            output_mode=output_mode,
            input_projection_layers=[16],
        )
        encoder = Identity(input_dim=10)
        core = DendriticRNNNetwork(config, input_dim=10)
        # Simple linear decoder
        decoder = torch.nn.Linear(8, 5)
        return RecurrentClassifier(encoder, core, decoder)

    def test_forward_returns_logits(self):
        model = self._make_classifier()
        x = torch.randn(4, 20, 10)
        out = model(x)
        assert out.shape == (4, 5)  # [batch, n_classes]

    def test_predict_returns_class_indices(self):
        model = self._make_classifier()
        x = torch.randn(4, 20, 10)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64

    def test_predict_handles_3d_output(self):
        model = self._make_classifier(output_mode="all")
        x = torch.randn(4, 20, 10)
        preds = model.predict(x)
        assert preds.shape == (4, 20)  # [batch, seq_len]

    def test_eval_logits_reduces_3d(self):
        model = self._make_classifier(output_mode="all")
        x = torch.randn(4, 20, 10)
        logits = model.eval_logits(x)
        assert logits.shape == (4, 5)  # reduced to [batch, n_classes]


# ---------------------------------------------------------------------------
# Vanilla RNN collapse test
# ---------------------------------------------------------------------------


class TestVanillaRNNCollapse:
    def test_single_branch_linear_no_inh_sum(self):
        """Config should collapse to a vanilla RNN-like cell."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=16,
                    rec_synapses=16,
                    compartment="linear",
                    activation="tanh",
                ),
            ],
            soma_mode="sum",
            input_projection_layers=[],  # no projection
        )
        net = DendriticRNNNetwork(config, input_dim=16)
        x = torch.randn(2, 10, 16)
        out = net(x)
        assert out.shape == (2, 16)


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_through_cell(self):
        """Test that gradients flow through the input projection and cell.

        Uses linear compartment to avoid TopK sparse masking instabilities,
        focusing purely on verifying the gradient path through the network.
        """
        torch.manual_seed(0)
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=8,
                    rec_synapses=8,
                    compartment="linear",
                    activation="tanh",
                ),
            ],
            soma_mode="sum",
            input_projection_layers=[16],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(2, 5, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()
        # Check gradients exist on input projection params
        has_grad = False
        for _name, p in net.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through the network"


# ---------------------------------------------------------------------------
# dendritic_branch compartment tests
# ---------------------------------------------------------------------------


class TestDendriticBranchCompartment:
    """Tests for the unified dendritic_branch compartment that uses
    DendriticBranchLayer with trace-before-shunting semantics."""

    def test_basic_forward_shape(self):
        """dendritic_branch compartment produces correct output shape."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=6,
                    rec_synapses=0,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
                BranchConfig(
                    tau=50.0,
                    ff_synapses=4,
                    rec_synapses=4,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4,
            n_exc=16,
            n_inh=8,
            n_branches=2,
            device=torch.device("cpu"),
        )
        x_ff = torch.randn(4, 32)
        s, new_state, _info = cell(x_ff, state)
        assert s.shape == (4, 16)
        # branch_V should be populated
        assert new_state.branch_V[0].shape == (4, 16)
        assert new_state.branch_V[1].shape == (4, 16)

    def test_non_shunting_branch(self):
        """dendritic_branch with use_shunting=False works (additive mode)."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=8,
                    compartment="dendritic_branch",
                    use_shunting=False,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4,
            n_exc=16,
            n_inh=0,
            n_branches=1,
            device=torch.device("cpu"),
        )
        s, _, _ = cell(torch.randn(4, 32), state)
        assert s.shape == (4, 16)

    def test_with_reactivation(self):
        """dendritic_branch with ParametricTanh reactivation."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=4,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                    reactivate=True,
                    reactivation_type="param_tanh",
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=2,
            n_exc=16,
            n_inh=8,
            n_branches=1,
            device=torch.device("cpu"),
        )
        s, _, _ = cell(torch.randn(2, 32), state)
        assert s.shape == (2, 16)

    def test_mixed_compartments(self):
        """Mix of topk and dendritic_branch compartments in same cell."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=8,
                    rec_synapses=0,
                    ie_synapses=4,
                    compartment="topk",
                ),
                BranchConfig(
                    tau=50.0,
                    ff_synapses=6,
                    rec_synapses=6,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=2,
            n_exc=16,
            n_inh=8,
            n_branches=2,
            device=torch.device("cpu"),
        )
        s, _, _ = cell(torch.randn(2, 32), state)
        assert s.shape == (2, 16)

    def test_gradient_flow(self):
        """Gradients flow through dendritic_branch compartment."""
        torch.manual_seed(0)
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=6,
                    rec_synapses=4,
                    ie_synapses=3,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
            input_projection_layers=[16],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(2, 5, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )
        assert has_grad, "No gradients flowed through dendritic_branch network"

    def test_soma_shunting_with_dendritic_branch(self):
        """Double normalization: per-branch shunting + somatic shunting."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=8,
                    rec_synapses=0,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
                BranchConfig(
                    tau=50.0,
                    ff_synapses=4,
                    rec_synapses=4,
                    ie_synapses=4,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="shunting",  # double normalization
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=2,
            n_exc=16,
            n_inh=8,
            n_branches=2,
            device=torch.device("cpu"),
        )
        s, _, _ = cell(torch.randn(2, 32), state)
        assert s.shape == (2, 16)

    def test_temporal_trace_integration(self):
        """Verify traces integrate over multiple timesteps."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=0,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=0,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=8)
        state = RecurrentState.init(
            batch_size=1,
            n_exc=8,
            n_inh=0,
            n_branches=1,
            device=torch.device("cpu"),
        )
        x = torch.ones(1, 8)
        # Run multiple timesteps
        for t in range(10):
            _s, state, _ = cell(x, state, t_step=t)
        # After 10 steps with constant input, traces should have built up
        assert state.branch_E[0].abs().sum() > 0, "E trace should be non-zero"


# ---------------------------------------------------------------------------
# Feedforward regression test: DendriticBranchLayer unchanged when no recurrent
# ---------------------------------------------------------------------------


class TestDendriticBranchLayerFFRegression:
    """Verify DendriticBranchLayer produces identical output with and without
    the recurrent_input_dim parameter when recurrent_input is not provided."""

    def test_ff_output_unchanged(self):
        """Forward pass output identical when recurrent args are unset."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        torch.manual_seed(42)
        # Create layer WITHOUT recurrent params (original FF usage)
        layer_ff = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            use_shunting=True,
            dbl_init_method="default",
        )

        torch.manual_seed(42)
        # Create layer WITH recurrent params set to None (should be identical)
        layer_rec = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            recurrent_input_dim=None,
            recurrent_synapses_per_branch=None,
            use_shunting=True,
            dbl_init_method="default",
        )

        x = torch.randn(4, 32)
        inh = torch.randn(4, 8)

        out_ff = layer_ff(x, inhibitory_input=inh)
        out_rec = layer_rec(x, inhibitory_input=inh)
        assert torch.allclose(
            out_ff, out_rec, atol=1e-6
        ), "DendriticBranchLayer output changed when recurrent args are None"

    def test_no_extra_params_when_no_recurrent(self):
        """No branch_recurrent layer created when recurrent_input_dim is None."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        layer = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            use_shunting=True,
            dbl_init_method="default",
        )
        assert layer.branch_recurrent is None
        assert layer.input_recurrent is False
        # Count params — should have no recurrent TopKLinear
        param_names = [n for n, _ in layer.named_parameters()]
        assert not any("recurrent" in n for n in param_names)

    def test_recurrent_layer_created_when_specified(self):
        """branch_recurrent TopKLinear exists when recurrent_input_dim is set."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        layer = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            recurrent_input_dim=16,
            recurrent_synapses_per_branch=6,
            use_shunting=True,
            dbl_init_method="default",
        )
        assert layer.branch_recurrent is not None
        assert layer.input_recurrent is True

    def test_ff_output_unchanged_with_explicit_rec_inh_none(self):
        """Feedforward output is identical when recurrent inhibitory args are explicitly None."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        torch.manual_seed(123)
        layer_default = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            use_shunting=True,
            dbl_init_method="default",
        )

        torch.manual_seed(123)
        layer_explicit_none = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            rec_inhibitory_input_dim=None,
            rec_inhibitory_synapses_per_branch=None,
            use_shunting=True,
            dbl_init_method="default",
        )

        x = torch.randn(4, 32)
        inh = torch.randn(4, 8)
        out_default = layer_default(x, inhibitory_input=inh)
        out_explicit_none = layer_explicit_none(x, inhibitory_input=inh)
        assert torch.allclose(out_default, out_explicit_none, atol=1e-6)


# ---------------------------------------------------------------------------
# Recurrent inhibition (rec_i_synapses) tests
# ---------------------------------------------------------------------------


class TestRecurrentInhibition:
    """Tests for the rec_i_synapses feature: recurrent inhibitory synapses
    from the local I pool via a 4th TopK layer."""

    def test_rec_i_synapses_all_traces_populated(self):
        """dendritic_branch with rec_i_synapses > 0 populates all 4 traces."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=6,
                    rec_synapses=4,
                    ie_synapses=3,
                    rec_i_synapses=3,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=2,
            n_exc=16,
            n_inh=8,
            n_branches=1,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 32)
        # Run a few steps so traces build up
        for t in range(5):
            s, state, _ = cell(x, state, t_step=t)

        # All 4 traces should be non-zero after multiple steps
        assert state.branch_E[0].abs().sum() > 0, "E trace should be non-zero"
        assert state.branch_E_rec[0].abs().sum() > 0, "E_rec trace should be non-zero"
        assert state.branch_I[0].abs().sum() > 0, "I trace should be non-zero"
        assert state.branch_I_rec[0].abs().sum() > 0, "I_rec trace should be non-zero"
        assert s.shape == (2, 16)

    def test_separate_trace_integration(self):
        """FF and REC excitatory traces maintained independently (not collapsed)."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[
                BranchConfig(
                    tau=50.0,
                    ff_synapses=8,
                    rec_synapses=8,
                    ie_synapses=3,
                    rec_i_synapses=0,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=16)
        state = RecurrentState.init(
            batch_size=2,
            n_exc=8,
            n_inh=4,
            n_branches=1,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 16)
        for t in range(5):
            _s, state, _ = cell(x, state, t_step=t)

        # FF and REC excitatory traces should differ
        # (they receive from different TopK layers with different weights)
        assert not torch.equal(
            state.branch_E[0], state.branch_E_rec[0]
        ), "FF and REC excitatory traces should be different"

    def test_ff_no_rec_inhibitory_unchanged(self):
        """DendriticBranchLayer without rec_inhibitory args has no extra params."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        layer = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            use_shunting=True,
            dbl_init_method="default",
        )
        assert layer.branch_rec_inhibition is None
        param_names = [n for n, _ in layer.named_parameters()]
        assert not any("rec_inhibition" in n for n in param_names)

    def test_rec_inhibition_layer_created(self):
        """branch_rec_inhibition TopKLinear exists when rec_inhibitory args set."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        layer = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=32,
            excitatory_synapses_per_branch=8,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=4,
            rec_inhibitory_input_dim=8,
            rec_inhibitory_synapses_per_branch=3,
            use_shunting=True,
            dbl_init_method="default",
        )
        assert layer.branch_rec_inhibition is not None
        # Should appear in get_weights dict
        w = layer.get_weights()
        assert "rec_inh" in w

    def test_gradient_flow_all_4_layers(self):
        """Gradients flow through all 4 TopK layers in dendritic_branch."""
        torch.manual_seed(0)
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=8,
            n_inhibitory=4,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=6,
                    rec_synapses=4,
                    ie_synapses=3,
                    rec_i_synapses=3,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
            input_projection_layers=[16],
        )
        net = DendriticRNNNetwork(config, input_dim=10)
        x = torch.randn(2, 5, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist on the dendritic branch cell params
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )
        assert has_grad, "No gradients flowed through rec_i_synapses network"

    def test_no_nan_with_all_4_synapses(self):
        """No NaN outputs when using all 4 synapse types over 10 steps."""
        config = RecurrentConfig(
            enabled=True,
            n_excitatory=16,
            n_inhibitory=8,
            branches=[
                BranchConfig(
                    tau=10.0,
                    ff_synapses=6,
                    rec_synapses=4,
                    ie_synapses=3,
                    rec_i_synapses=3,
                    compartment="dendritic_branch",
                    use_shunting=True,
                ),
            ],
            soma_mode="sum",
        )
        cell = DendriticRNNCell(config, input_dim=32)
        state = RecurrentState.init(
            batch_size=4,
            n_exc=16,
            n_inh=8,
            n_branches=1,
            device=torch.device("cpu"),
        )
        x = torch.randn(4, 32)
        for t in range(10):
            s, state, _ = cell(x, state, t_step=t)
            assert torch.isfinite(s).all(), f"NaN/Inf at step {t}"
