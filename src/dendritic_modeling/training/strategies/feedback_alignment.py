"""
Feedback Alignment (FA) and Direct Feedback Alignment (DFA) training strategies.

Implements FA (Lillicrap et al., 2016) and DFA (Nøkland, 2016) for both
standard MLPs and DendriNet architectures. Fixed random feedback matrices
replace the transpose of forward weights during error backpropagation.
"""

import logging

import torch
import torch.nn as nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class FeedbackAlignmentTrainer(Trainer):
    """
    Training strategy using Feedback Alignment (FA) or Direct Feedback
    Alignment (DFA) for error signal propagation.

    FA: errors propagate backward layer-by-layer through fixed random B_l
        instead of W_l^T.
    DFA: each hidden layer receives the output error directly projected
        through its own fixed random matrix B_l.
    """

    def __init__(self, *args, mode: str = "fa", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode.lower()
        if self.mode not in ("fa", "dfa"):
            raise ValueError(f"mode must be 'fa' or 'dfa', got '{self.mode}'")
        self.filename_prefix = f"{self.mode}_"
        self.logger_info_prefix = f"[{self.mode.upper()}] "
        self._feedback_initialized = False
        # Populated lazily on first training step
        self._layer_info: list[tuple[nn.Module, str, int, int]] = []
        self._B_fa: dict[int, torch.Tensor] = {}  # FA: B[l] shape (out_dim_l, in_dim_l)
        self._B_dfa: dict[
            int, torch.Tensor
        ] = {}  # DFA: B[l] shape (out_dim_l, output_dim)

    # ------------------------------------------------------------------
    # Feedback matrix initialisation
    # ------------------------------------------------------------------
    def _initialize_feedback(self, model: BaseModel) -> None:
        """Create fixed random feedback matrices matching layer geometry."""
        if self._feedback_initialized:
            return

        self._layer_info = self._collect_trainable_layers(model)
        n = len(self._layer_info)
        if n == 0:
            logger.warning("No trainable layers found for FA/DFA")
            self._feedback_initialized = True
            return

        output_dim = self._layer_info[-1][2]  # out_dim of last layer

        for idx, (_mod, _pname, out_dim, in_dim) in enumerate(self._layer_info):
            # FA feedback: projects error from out_dim back to in_dim
            # B shape (in_dim, out_dim) so that e_{l-1} = e_l @ B^T has shape [B, in_dim]
            self._B_fa[idx] = torch.randn(in_dim, out_dim, device=self.device) * (
                out_dim**-0.5
            )

            # DFA feedback: projects output error to this layer's out_dim
            # B shape (out_dim, output_dim) so that e_l = e_out @ B^T has shape [B, out_dim]
            if idx < n - 1:
                self._B_dfa[idx] = torch.randn(
                    out_dim, output_dim, device=self.device
                ) * (output_dim**-0.5)

        self._feedback_initialized = True
        logger.info(
            f"Initialized {self.mode.upper()} feedback matrices for "
            f"{n} layers (output_dim={output_dim})"
        )

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_trainable_layers(
        model: BaseModel,
    ) -> list[tuple[nn.Module, str, int, int]]:
        """Return (module, param_name, out_features, in_features) for each
        forward-weight matrix in execution order."""
        layers: list[tuple[nn.Module, str, int, int]] = []
        seen_ids: set[int] = set()

        def _try_add(mod: nn.Module, pname: str) -> None:
            if id(mod) in seen_ids:
                return
            param = getattr(mod, pname, None)
            if param is None:
                return
            seen_ids.add(id(mod))
            # BlockLinear.log_weight has shape (out_features, block_size),
            # but the effective input dimension is in_features = out * block_size
            if isinstance(mod, BlockLinear) and pname == "log_weight":
                out_dim = mod.out_features
                in_dim = mod.in_features  # full input dimension
            else:
                out_dim, in_dim = param.shape[:2]
            layers.append((mod, pname, out_dim, in_dim))

        # Prefer DendriticBranchLayer order if present
        branch_layers = [
            m for m in model.modules() if isinstance(m, DendriticBranchLayer)
        ]
        if branch_layers:
            for bl in branch_layers:
                if getattr(bl, "branch_excitation", None) is not None:
                    _try_add(bl.branch_excitation, "pre_w")
                if getattr(bl, "branches_to_output", None) is not None:
                    _try_add(bl.branches_to_output, "log_weight")
                if getattr(bl, "branch_inhibition", None) is not None:
                    _try_add(bl.branch_inhibition, "pre_w")

        # Always include decoder nn.Linear layers (e.g. the final classifier head)
        decoder = getattr(model, "decoder_network", None)
        if decoder is not None:
            for mod in decoder.modules():
                if isinstance(mod, nn.Linear):
                    _try_add(mod, "weight")

        # Fallback: if no layers found yet, scan all nn.Linear
        if not layers:
            for mod in model.modules():
                if isinstance(mod, nn.Linear):
                    _try_add(mod, "weight")

        return layers

    # ------------------------------------------------------------------
    # Training loop override
    # ------------------------------------------------------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:
        model.train()
        self._initialize_feedback(model)

        total_loss = 0.0
        total_base = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # --- forward with activation caching ---
            activations: dict[int, torch.Tensor] = {}
            handles = []
            for idx, (mod, *_rest) in enumerate(self._layer_info):
                handles.append(
                    mod.register_forward_hook(self._make_hook(activations, idx))
                )
            try:
                y_hat = model(x_batch)
            finally:
                for h in handles:
                    h.remove()

            # --- loss (for logging only) ---
            loss = self.loss_function(model, x_batch, y_batch)
            total_loss += loss.item()
            total_base += loss.item()

            # --- output error ---
            e_output = self._output_error(y_hat, y_batch)

            # --- assign FA/DFA gradients ---
            self.optimizer.zero_grad()
            if self.mode == "fa":
                self._grad_fa(e_output, activations)
            else:
                self._grad_dfa(e_output, activations)

            if self.grad_clip_value:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.grad_clip_value
                )
            self.optimizer.step()

        n = len(train_loader)
        return total_loss / n, total_base / n, 0.0

    # ------------------------------------------------------------------
    # Hook helper
    # ------------------------------------------------------------------
    @staticmethod
    def _make_hook(store: dict, key: int):
        def hook_fn(_module, inp, _out):
            if isinstance(inp, tuple) and len(inp) > 0:
                store[key] = inp[0].detach()
            elif isinstance(inp, torch.Tensor):
                store[key] = inp.detach()

        return hook_fn

    # ------------------------------------------------------------------
    # Output error
    # ------------------------------------------------------------------
    def _output_error(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dL/dy_hat.  Shape [batch, output_dim]."""
        loss_name = getattr(self.loss_function, "_loss_name", "").lower()
        B = y_hat.size(0)

        if any(
            k in loss_name for k in ("cross entropy", "nll", "negative log likelihood")
        ):
            probs = torch.exp(y_hat)
            if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
                tgt = torch.zeros_like(probs)
                tgt.scatter_(1, y.view(-1, 1).long(), 1.0)
            else:
                tgt = y.float()
            return (probs - tgt) / B

        # MSE-like fallback
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return (y_hat - y.float()) / B

    # ------------------------------------------------------------------
    # Gradient assignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_grad(mod: nn.Module, pname: str, grad_dense: torch.Tensor) -> None:
        """Assign a dense gradient to a parameter, handling BlockLinear specially.

        For BlockLinear.log_weight (shape [out, block_size]), extract only the
        block-diagonal entries from the dense gradient (shape [out, in_features]).
        """
        param = getattr(mod, pname)
        if isinstance(mod, BlockLinear) and pname == "log_weight":
            # grad_dense is (out_features, in_features)
            # param is (out_features, block_size)
            # Output i uses inputs [i*bs : (i+1)*bs], so extract those entries
            bs = mod.block_size
            grad_w = grad_dense.view(mod.out_features, mod.out_features, bs)
            # Each row i should use its own block: the [i, i, :] slice
            idx = torch.arange(mod.out_features, device=grad_dense.device)
            grad_w = grad_w[idx, idx, :]  # (out_features, block_size)
        else:
            grad_w = grad_dense

        g = grad_w.to(param.dtype)
        param.grad = g if param.grad is None else param.grad + g

    # ------------------------------------------------------------------
    # FA gradient assignment
    # ------------------------------------------------------------------
    def _grad_fa(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """Layer-by-layer backward propagation through random B."""
        e = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n - 1, -1, -1):
            mod, pname, out_dim, in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None or e.size(-1) != out_dim:
                if idx > 0:
                    e = e @ self._B_fa[idx].t()
                continue

            # grad_W = e^T h_in   — shape (out_dim, in_dim)
            grad_dense = e.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            # bias gradient for nn.Linear
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                gb = e.sum(0)
                mod.bias.grad = (
                    gb.to(mod.bias.dtype)
                    if mod.bias.grad is None
                    else mod.bias.grad + gb.to(mod.bias.dtype)
                )

            # propagate error to previous layer via random B
            if idx > 0:
                e = e @ self._B_fa[idx].t()  # [batch, in_dim]

    # ------------------------------------------------------------------
    # DFA gradient assignment
    # ------------------------------------------------------------------
    def _grad_dfa(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """Direct projection of output error to each layer via random B."""
        e_out = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n):
            mod, pname, out_dim, _in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None:
                continue

            if idx == n - 1:
                # Output layer: use error directly
                e_local = e_out
            else:
                # Hidden layer: project output error → this layer's out_dim
                e_local = e_out @ self._B_dfa[idx].t()  # [batch, out_dim]

            if e_local.size(-1) != out_dim:
                continue

            grad_dense = e_local.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            if isinstance(mod, nn.Linear) and mod.bias is not None:
                gb = e_local.sum(0)
                mod.bias.grad = (
                    gb.to(mod.bias.dtype)
                    if mod.bias.grad is None
                    else mod.bias.grad + gb.to(mod.bias.dtype)
                )
