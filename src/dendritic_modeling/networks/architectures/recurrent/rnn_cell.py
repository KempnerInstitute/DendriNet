"""
DendriticRNNCell: single-timestep recurrent cell with per-branch temporal dynamics.

Supports four compartment modes:
  - "topk": Separate E/I accumulators with soma-level normalization (legacy)
  - "dendritic_branch": Per-branch shunting via DendriticBranchLayer with
    trace-before-shunting semantics (integrate E/I/rec traces, then shunt)
  - "dendrinet": Full hierarchical DendriNet per branch
  - "linear": Simple linear projection

Optional inhibitory network, and cross-branch somatic normalization yielding
branch utilization alpha.

Phase 2 intentionally bypasses AMP and regularization (deferred to Phase 3).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.config.recurrent import BranchConfig, RecurrentConfig
from dendritic_modeling.networks.architectures.recurrent.state import RecurrentState


class _BranchModule(nn.Module):
    """Container for a single branch's learnable layers."""

    def __init__(
        self,
        branch_cfg: BranchConfig,
        input_dim: int,
        n_exc: int,
        n_inh: int,
    ):
        super().__init__()
        self.compartment = branch_cfg.compartment

        act_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
        }
        self.activation = act_fn.get(branch_cfg.activation, nn.ReLU())

        if self.compartment == "topk":
            self._init_topk(branch_cfg, input_dim, n_exc, n_inh)
        elif self.compartment == "dendritic_branch":
            self._init_dendritic_branch(branch_cfg, input_dim, n_exc, n_inh)
        elif self.compartment == "dendrinet":
            self._init_dendrinet(branch_cfg, input_dim, n_exc)
        elif self.compartment == "linear":
            self._init_linear(branch_cfg, input_dim, n_exc)
        else:
            raise ValueError(
                f"compartment must be 'topk', 'dendritic_branch', 'dendrinet', "
                f"or 'linear', got '{self.compartment}'"
            )

    def _init_topk(self, cfg: BranchConfig, input_dim: int, n_exc: int, n_inh: int):
        from dendritic_modeling.networks import TopKLinear

        self.ff_exc = (
            TopKLinear(input_dim, n_exc, K=cfg.ff_synapses)
            if cfg.ff_synapses > 0
            else None
        )
        self.rec_exc = (
            TopKLinear(n_exc, n_exc, K=cfg.rec_synapses)
            if cfg.rec_synapses > 0
            else None
        )
        self.inh = (
            TopKLinear(n_inh, n_exc, K=cfg.ie_synapses)
            if cfg.ie_synapses > 0 and n_inh > 0
            else None
        )

        # TopKLinear uses torch.empty for pre_w — must call initialize()
        # to set proper weights (matching feedforward DendriticBranchLayer behavior)
        for layer in [self.ff_exc, self.rec_exc, self.inh]:
            if layer is not None:
                layer.initialize()

    def _init_dendritic_branch(
        self, cfg: BranchConfig, input_dim: int, n_exc: int, n_inh: int
    ):
        """Create a DendriticBranchLayer with optional recurrent synapses.

        Uses the same DendriticBranchLayer as the feedforward model, extended
        with recurrent excitatory and recurrent inhibitory synapses.
        The RNN cell uses
        compute_raw_currents() + shunt_from_currents() for trace-before-shunting
        semantics (leaky integrate E/I traces, then apply shunting on traces).
        """
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
            DendriticBranchLayer,
        )

        self.dbl = DendriticBranchLayer(
            output_dim=n_exc,
            excitatory_input_dim=input_dim if cfg.ff_synapses > 0 else None,
            excitatory_synapses_per_branch=(
                cfg.ff_synapses if cfg.ff_synapses > 0 else None
            ),
            inhibitory_input_dim=n_inh if cfg.ie_synapses > 0 and n_inh > 0 else None,
            inhibitory_synapses_per_branch=(
                cfg.ie_synapses if cfg.ie_synapses > 0 and n_inh > 0 else None
            ),
            recurrent_input_dim=n_exc if cfg.rec_synapses > 0 else None,
            recurrent_synapses_per_branch=(
                cfg.rec_synapses if cfg.rec_synapses > 0 else None
            ),
            rec_inhibitory_input_dim=(
                n_inh if cfg.rec_i_synapses > 0 and n_inh > 0 else None
            ),
            rec_inhibitory_synapses_per_branch=(
                cfg.rec_i_synapses if cfg.rec_i_synapses > 0 and n_inh > 0 else None
            ),
            input_branch_factor=None,
            topk_init_method=cfg.topk_init_method,
            use_shunting=cfg.use_shunting,
            reactivate=cfg.reactivate,
            reactivation_type=cfg.reactivation_type,
            reactivation_init_m=cfg.reactivation_init_m,
            reactivation_init_b=cfg.reactivation_init_b,
            epsilon=cfg.epsilon,
            weight_transform=cfg.weight_transform,
            dbl_init_method="default",
        )

    def _init_dendrinet(self, cfg: BranchConfig, input_dim: int, n_exc: int):
        from dendritic_modeling.networks import DendriNet

        # Pre-projection to enforce FF/REC budgets
        self.ff_proj = (
            nn.Linear(input_dim, cfg.ff_synapses) if cfg.ff_synapses > 0 else None
        )
        self.rec_proj = (
            nn.Linear(n_exc, cfg.rec_synapses) if cfg.rec_synapses > 0 else None
        )
        proj_dim = cfg.ff_synapses + cfg.rec_synapses
        self.dendrinet = DendriNet(
            n_soma=n_exc,
            branch_factors=cfg.dendrinet_branch_factors,
            excitatory_input_dim=proj_dim,
            excitatory_synapses_per_branch=proj_dim,
            inhibitory_input_dim=0,
            inhibitory_synapses_per_branch=0,
        )

    def _init_linear(self, cfg: BranchConfig, input_dim: int, n_exc: int):
        self.ff_linear = nn.Linear(input_dim, n_exc) if cfg.ff_synapses > 0 else None
        self.rec_linear = nn.Linear(n_exc, n_exc) if cfg.rec_synapses > 0 else None


class DendriticRNNCell(nn.Module):
    """Single-timestep dendritic recurrent cell.

    Each cell has B independently-configurable branches. The forward pass:
    1. Optionally processes inhibitory network (skipped if n_inhibitory=0)
    2. Per-branch: routes FF/REC input through compartment, temporally integrates
    3. Cross-branch somatic normalization produces output s and routing alpha

    When cell_type="gru" or "lstm", bypasses branch system entirely.
    """

    def __init__(self, config: RecurrentConfig, input_dim: int):
        super().__init__()

        # Runtime validation (defensive against dict-based config loading)
        branches = []
        for _i, bc in enumerate(config.branches):
            if not isinstance(bc, BranchConfig):
                bc = BranchConfig(**bc)
            if config.n_inhibitory == 0:
                if bc.ie_synapses > 0:
                    bc.ie_synapses = 0
                if bc.rec_i_synapses > 0:
                    bc.rec_i_synapses = 0
            branches.append(bc)
        self.branch_configs = branches

        if config.soma_mode not in ("shunting", "additive", "sum"):
            raise ValueError(
                f"soma_mode must be 'shunting', 'additive', or 'sum', "
                f"got '{config.soma_mode}'"
            )
        if config.cell_type not in ("dendritic", "gru", "lstm"):
            raise ValueError(
                f"cell_type must be 'dendritic', 'gru', or 'lstm', "
                f"got '{config.cell_type}'"
            )

        self.cell_type = config.cell_type
        self.n_exc = config.n_excitatory
        self.n_inh = config.n_inhibitory
        self.n_branches = len(self.branch_configs)
        self.dt = config.dt
        self.g_leak = config.g_leak
        self.soma_mode = config.soma_mode
        self.store_routing = config.store_routing

        if self.cell_type == "gru":
            self.gru_cell = nn.GRUCell(input_dim, self.n_exc)
            return
        elif self.cell_type == "lstm":
            self.lstm_cell = nn.LSTMCell(input_dim, self.n_exc)
            return

        # Inhibitory network (None if n_inhibitory=0)
        if self.n_inh > 0:
            self.i_network = nn.Linear(input_dim + self.n_exc, self.n_inh)
            self.i_activation = nn.ReLU()
            i_tau = config.i_tau
            self.register_buffer("i_decay", torch.tensor(math.exp(-config.dt / i_tau)))
        else:
            self.i_network = None

        # Per-branch modules
        self.branches = nn.ModuleList()
        taus = []
        for bc in self.branch_configs:
            self.branches.append(_BranchModule(bc, input_dim, self.n_exc, self.n_inh))
            taus.append(bc.tau)

        # Per-branch decay factors
        decays = [math.exp(-config.dt / tau) for tau in taus]
        if config.learn_taus:
            # Learn log-tau to keep tau positive
            self.log_taus = nn.Parameter(torch.tensor([math.log(t) for t in taus]))
        else:
            self.register_buffer("branch_decays", torch.tensor(decays))

        self.learn_taus = config.learn_taus

    def _get_branch_decay(self, b: int) -> torch.Tensor:
        if self.learn_taus:
            tau_b = torch.exp(self.log_taus[b])
            return torch.exp(-self.dt / tau_b)
        return self.branch_decays[b]

    def _compute_somatic_output(self, state: RecurrentState) -> torch.Tensor:
        """Compute somatic output from current state (for recurrent feedback)."""
        if self.soma_mode == "shunting":
            s, _ = self._compute_soma(state)
            return s
        elif self.soma_mode == "sum":
            s, _ = self._compute_soma(state)
            return s
        else:  # additive
            s, _ = self._compute_soma(state)
            return s

    def _compute_soma(
        self, state: RecurrentState
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Cross-branch somatic normalization."""
        if self.soma_mode == "shunting":
            E_vals, I_vals = [], []
            for b in range(self.n_branches):
                if self.branch_configs[b].compartment in (
                    "dendrinet",
                    "dendritic_branch",
                ):
                    E_vals.append(F.relu(state.branch_V[b]))
                    I_vals.append(
                        torch.abs(state.branch_V[b]) - F.relu(state.branch_V[b])
                    )
                else:
                    E_vals.append(F.relu(state.branch_E[b]))
                    I_vals.append(torch.abs(state.branch_I[b]))

            E_sum = sum(E_vals)
            denom = sum(e + i for e, i in zip(E_vals, I_vals)) + self.g_leak
            s = E_sum / denom
            alpha = [e / denom for e in E_vals]
            return s, alpha

        elif self.soma_mode == "sum":
            vals = []
            for b in range(self.n_branches):
                if self.branch_configs[b].compartment in (
                    "dendrinet",
                    "dendritic_branch",
                ):
                    vals.append(state.branch_V[b])
                else:
                    vals.append(state.branch_E[b])
            s = sum(vals)
            # Normalized alpha for info only
            denom = torch.abs(s) + 1e-8
            alpha = [v / denom for v in vals]
            return s, alpha

        else:  # additive
            vals = []
            for b in range(self.n_branches):
                if self.branch_configs[b].compartment in (
                    "dendrinet",
                    "dendritic_branch",
                ):
                    vals.append(state.branch_V[b])
                else:
                    vals.append(state.branch_E[b] - state.branch_I[b])
            s = sum(vals)
            denom = torch.abs(s) + 1e-8
            alpha = [v / denom for v in vals]
            return s, alpha

    def forward(
        self,
        x_ff: torch.Tensor,
        state: RecurrentState,
        t_step: int = 0,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        """Process a single timestep.

        Args:
            x_ff: Feedforward input [batch, input_dim]
            state: Current recurrent state
            t_step: Current timestep index (for diagnostics)

        Returns:
            s: Somatic output [batch, n_exc]
            state: Updated state
            info: Dict with 'alpha' tensor if store_routing is True
        """
        # GRU/LSTM bypass
        if self.cell_type == "gru":
            h = (
                state.branch_E[0]
                if state.branch_E
                else torch.zeros(
                    x_ff.shape[0], self.n_exc, device=x_ff.device, dtype=x_ff.dtype
                )
            )
            h_new = self.gru_cell(x_ff, h)
            state.branch_E = [h_new]
            return h_new, state, {}

        if self.cell_type == "lstm":
            h = (
                state.branch_E[0]
                if state.branch_E
                else torch.zeros(
                    x_ff.shape[0], self.n_exc, device=x_ff.device, dtype=x_ff.dtype
                )
            )
            c = (
                state.branch_I[0]
                if state.branch_I
                else torch.zeros(
                    x_ff.shape[0], self.n_exc, device=x_ff.device, dtype=x_ff.dtype
                )
            )
            h_new, c_new = self.lstm_cell(x_ff, (h, c))
            state.branch_E = [h_new]
            state.branch_I = [c_new]
            return h_new, state, {}

        # Dendritic cell: compute somatic output from previous state
        s_prev = self._compute_somatic_output(state)

        # Inhibitory network
        if self.i_network is not None:
            raw_i = self.i_network(torch.cat([x_ff, s_prev], dim=-1))
            raw_i = self.i_activation(raw_i)
            state.h_inh = self.i_decay * state.h_inh + (1 - self.i_decay) * raw_i

        # Per-branch processing
        for b, branch_mod in enumerate(self.branches):
            decay_b = self._get_branch_decay(b)
            compartment = self.branch_configs[b].compartment

            if compartment == "topk":
                E_drive = torch.zeros(
                    x_ff.shape[0],
                    self.n_exc,
                    device=x_ff.device,
                    dtype=x_ff.dtype,
                )
                if branch_mod.ff_exc is not None:
                    E_drive = E_drive + branch_mod.ff_exc(x_ff)
                if branch_mod.rec_exc is not None:
                    E_drive = E_drive + branch_mod.rec_exc(s_prev)
                raw_E = branch_mod.activation(E_drive)

                raw_I = torch.zeros_like(raw_E)
                if branch_mod.inh is not None and state.h_inh is not None:
                    raw_I = branch_mod.activation(branch_mod.inh(state.h_inh))

                state.branch_E[b] = decay_b * state.branch_E[b] + (1 - decay_b) * raw_E
                state.branch_I[b] = decay_b * state.branch_I[b] + (1 - decay_b) * raw_I

            elif compartment == "dendritic_branch":
                # Per-branch dendritic computation with trace-before-shunting:
                # 1. Compute raw synaptic currents via 4 TopK layers
                # 2. Leaky-integrate 4 separate traces (FF E, REC E, FF I, REC I)
                # 3. Apply shunting (or additive) on integrated traces
                dbl = branch_mod.dbl
                raw_E, raw_E_rec, raw_I, raw_I_rec = dbl.compute_raw_currents(
                    x=x_ff,
                    inhibitory_input=state.h_inh,
                    recurrent_input=s_prev,
                    rec_inhibitory_input=state.h_inh,
                )

                # Integrate 4 separate traces with per-branch time constant
                state.branch_E[b] = decay_b * state.branch_E[b] + (1 - decay_b) * raw_E
                state.branch_E_rec[b] = (
                    decay_b * state.branch_E_rec[b] + (1 - decay_b) * raw_E_rec
                )
                state.branch_I[b] = decay_b * state.branch_I[b] + (1 - decay_b) * raw_I
                state.branch_I_rec[b] = (
                    decay_b * state.branch_I_rec[b] + (1 - decay_b) * raw_I_rec
                )

                # Shunt on 4 integrated traces, then reactivate
                V_b = dbl.shunt_from_currents(
                    trace_E=state.branch_E[b],
                    trace_E_rec=state.branch_E_rec[b],
                    trace_I=state.branch_I[b],
                    trace_I_rec=state.branch_I_rec[b],
                )
                state.branch_V[b] = V_b

            elif compartment == "dendrinet":
                parts = []
                if branch_mod.ff_proj is not None:
                    parts.append(branch_mod.ff_proj(x_ff))
                if branch_mod.rec_proj is not None:
                    parts.append(branch_mod.rec_proj(s_prev))
                combined = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
                V_b = branch_mod.dendrinet(combined)
                state.branch_V[b] = decay_b * state.branch_V[b] + (1 - decay_b) * V_b

            elif compartment == "linear":
                E_drive = torch.zeros(
                    x_ff.shape[0],
                    self.n_exc,
                    device=x_ff.device,
                    dtype=x_ff.dtype,
                )
                if branch_mod.ff_linear is not None:
                    E_drive = E_drive + branch_mod.ff_linear(x_ff)
                if branch_mod.rec_linear is not None:
                    E_drive = E_drive + branch_mod.rec_linear(s_prev)
                raw_E = branch_mod.activation(E_drive)
                state.branch_E[b] = decay_b * state.branch_E[b] + (1 - decay_b) * raw_E

        # Cross-branch somatic normalization
        s, alpha = self._compute_soma(state)

        info: dict = {}
        if self.store_routing:
            info["alpha"] = torch.stack(alpha, dim=-1)
            # Store per-branch trace values for dendritic_branch analysis
            traces = {}
            for b, cfg in enumerate(self.branch_configs):
                if cfg.compartment == "dendritic_branch":
                    traces[f"branch_{b}"] = {
                        "trace_E": state.branch_E[b].detach(),
                        "trace_E_rec": state.branch_E_rec[b].detach(),
                        "trace_I": state.branch_I[b].detach(),
                        "trace_I_rec": state.branch_I_rec[b].detach(),
                        "V": state.branch_V[b].detach(),
                    }
            if traces:
                info["traces"] = traces

        return s, state, info
