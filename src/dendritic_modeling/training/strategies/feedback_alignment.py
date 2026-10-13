"""Feedback alignment (FA) and direct feedback alignment (DFA).

FA (Lillicrap et al., 2016) and DFA (Nokland, 2016) replace the transposed
forward weights of backpropagation with fixed random matrices. Training cuts
the forward pass into blocks:

* every dendritic branch layer (one level of a dendritic tree), and
* in MLP cores, encoders and decoders, the computation that ends at the input
  of each linear layer (the previous linear layer and its nonlinearity).

The computation after the last cuts, such as a linear readout and the output
scale, is the output layer and receives the exact gradient of the configured
loss. Every other block receives an error signal at its output and updates its
parameters with the exact gradient of that output, its inputs held fixed. This
local gradient includes the block's nonlinearity, weight transforms and
connectivity masks, and in shunting branch layers the conductance and driving
force of each pathway.

* DFA gives each block the loss gradient at the model output through the
  block's own fixed random matrix. A block whose output feeds the output layer
  directly (the soma of an identity-decoder network) receives the exact error.
* FA passes the error from block to block: the error at each linear layer's
  output is multiplied by a fixed random matrix in place of the transposed
  weights. It needs a sequential chain of linear layers.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn

from dendritic_modeling.config.seeds import derive_seed
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _average_distributed_gradients,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    remove_hook_handles,
    run_with_forward_hooks,
)

logger = logging.getLogger(__name__)

_HEAD = "<output layer>"


@dataclass
class _Cut:
    """A block output and the leaf that replaced it in the forward pass."""

    output: torch.Tensor
    leaf: torch.Tensor


@dataclass
class _Forward:
    """What one forward pass with cuts recorded."""

    cuts: dict[str, _Cut] = field(default_factory=dict)
    linear_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    linear_order: list[str] = field(default_factory=list)
    calls: Counter = field(default_factory=Counter)


@dataclass
class _Plan:
    """Which parameters each block owns, fixed at the first training step."""

    head: list[torch.nn.Parameter]
    blocks: dict[str, list[torch.nn.Parameter]]
    uncovered: list[str]


class FeedbackAlignmentTrainer(ForwardHookRemovalMixin, Trainer):
    """Train with feedback alignment (``mode='fa'``) or direct feedback
    alignment (``mode='dfa'``); see the module docstring."""

    # Options its own step loop does not apply (see STEP_OPTIONS).
    unapplied_options = Trainer.unapplied_options | frozenset(
        {
            "use_amp",
            "enable_nan_checking",
            "profile_dataloader",
        }
    )
    # Gradients come from torch.autograd.grad and are averaged explicitly.
    ddp_reduces_gradients = False

    def __init__(self, *args, mode: str = "fa", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode.lower()
        if self.mode not in ("fa", "dfa"):
            raise ValueError(f"mode must be 'fa' or 'dfa', got '{self.mode}'")
        self.filename_prefix = f"{self.mode}_"
        self.logger_info_prefix = f"[{self.mode.upper()}] "
        self._plan: _Plan | None = None
        # Fixed random feedback matrices by block (DFA) or linear layer (FA).
        self.feedback: dict[str, torch.Tensor] = {}
        # Trainable parameters several blocks read; FA/DFA leaves them fixed.
        self.uncovered_parameters: list[str] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:
        model.train()
        self._reject_unapplied_gradient_scaling(model)

        # (total, base, regularization) float64 sums, read once per epoch.
        loss_sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        n_batches = 0

        for batch in train_loader:
            x_batch, y_batch = self._move_training_batch_to_device(batch, train_loader)

            self.optimizer.zero_grad()
            loss = self.compute_gradients(model, x_batch, y_batch).double()
            regularization = self._add_regularization_gradients(
                model, x_batch, y_batch
            ).double()
            loss_sums += torch.stack([loss + regularization, loss, regularization])
            n_batches += 1
            # Gradients are assigned directly, not by backward(), so DDP cannot
            # reduce them; average them across ranks before clipping.
            _average_distributed_gradients(model)

            clip_gradients(
                model.parameters(), self.grad_clip_value, self.grad_clip_mode
            )
            self.optimizer.step()

        return self._mean_epoch_losses(loss_sums, n_batches)

    def compute_gradients(
        self, model: BaseModel, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        """Run one forward pass and add the FA/DFA gradients to ``.grad``.

        Returns the task loss of that forward pass.
        """
        component = self._unwrap_model(model)
        forward = _Forward()
        y_hat = run_with_forward_hooks(
            attach=partial(self._attach_cut_hooks, component, forward),
            remove=self.remove_forward_hooks,
            body=partial(model, x_batch),
        )
        loss = self._loss_from_predictions(y_hat, y_batch)
        if self._plan is None:
            self._plan = self._build_plan(component, y_hat, forward)
        if self.mode == "fa":
            self._assign_fa_gradients(loss, y_hat, forward)
        else:
            self._assign_dfa_gradients(loss, y_hat, forward)
        return loss.detach()

    def _loss_from_predictions(
        self, y_hat: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        try:
            return self.loss_function.from_predictions(y_hat, y_batch)
        except NotImplementedError as exc:
            raise ValueError(
                f"strategy '{self.mode}' needs a loss computed from the model "
                f"output; {type(self.loss_function).__name__} depends on the "
                "model itself"
            ) from exc

    def _add_regularization_gradients(
        self, model: BaseModel, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate the configured regularization gradients into ``.grad``.

        Returns the detached regularization loss (zero without one).

        L1/L2 and E/I-ratio penalties depend only on the weights, so their
        exact gradients are local; they are added to the FA/DFA gradients
        exactly as the standard trainer adds them to the task gradient.
        ``torch.autograd.grad`` is used instead of ``backward()`` so a DDP
        wrapper's gradient hooks do not fire for this partial gradient.
        """
        if not self.regularization_manager.has_regularization():
            return torch.zeros((), device=x_batch.device)
        reg_loss = self.regularization_manager.compute_regularization_loss(
            model,
            x_batch,
            y_batch,
            param_groups=self.optimizer.param_groups,
        )
        params = [param for param in model.parameters() if param.requires_grad]
        if reg_loss.requires_grad and params:
            grads = torch.autograd.grad(reg_loss, params, allow_unused=True)
            _accumulate(params, grads)
        return reg_loss.detach()

    # ------------------------------------------------------------------
    # Cutting the forward pass into blocks
    # ------------------------------------------------------------------
    def _attach_cut_hooks(
        self, component: nn.Module, forward: _Forward
    ) -> list[torch.utils.hooks.RemovableHandle]:
        branch_names = [
            name
            for name, module in component.named_modules()
            if isinstance(module, DendriticBranchLayer)
        ]
        handles = []
        try:
            for name, module in component.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    handles.append(
                        module.register_forward_hook(
                            partial(self._cut_output, name, forward)
                        )
                    )
                elif isinstance(module, nn.Linear) and not any(
                    name.startswith(f"{branch}.") for branch in branch_names
                ):
                    handles.append(
                        module.register_forward_pre_hook(
                            partial(self._cut_input, name, forward)
                        )
                    )
                    handles.append(
                        module.register_forward_hook(
                            partial(self._record_linear, name, forward)
                        )
                    )
        except Exception:
            remove_hook_handles(handles)
            raise
        return handles

    def _count_call(self, name: str, forward: _Forward) -> None:
        forward.calls[name] += 1
        if forward.calls[name] > 1:
            raise ValueError(
                f"strategy '{self.mode}' needs every branch layer and linear "
                f"layer to run once per forward pass, but {name} ran "
                f"{forward.calls[name]} times; recurrent unrolls are not "
                "supported (use the recurrent or standard strategy)"
            )

    def _cut_output(self, name, forward, _module, _inputs, output):
        self._count_call(name, forward)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"strategy '{self.mode}' expects branch layer {name} to return "
                f"a tensor, got {type(output).__name__}"
            )
        if not output.requires_grad:
            return None
        leaf = output.detach().requires_grad_()
        forward.cuts[name] = _Cut(output, leaf)
        return leaf

    def _cut_input(self, name, forward, _module, args):
        self._count_call(name, forward)
        x = args[0]
        if not (isinstance(x, torch.Tensor) and x.requires_grad):
            return None
        leaf = x.detach().requires_grad_()
        forward.cuts[f"{name}:input"] = _Cut(x, leaf)
        return (leaf, *args[1:])

    @staticmethod
    def _record_linear(name, forward, _module, _inputs, output):
        forward.linear_outputs[name] = output
        forward.linear_order.append(name)

    # ------------------------------------------------------------------
    # Parameter ownership
    # ------------------------------------------------------------------
    def _build_plan(
        self, component: nn.Module, y_hat: torch.Tensor, forward: _Forward
    ) -> _Plan:
        named = [(n, p) for n, p in component.named_parameters() if p.requires_grad]
        params = [p for _, p in named]

        def readers(output: torch.Tensor) -> list[int]:
            grads = torch.autograd.grad(
                output,
                params,
                grad_outputs=torch.ones_like(output),
                allow_unused=True,
                retain_graph=True,
            )
            return [i for i, grad in enumerate(grads) if grad is not None]

        owners: dict[int, list[str]] = defaultdict(list)
        for index in readers(y_hat):
            owners[index].append(_HEAD)
        for name, cut in forward.cuts.items():
            for index in readers(cut.output):
                owners[index].append(name)

        bypassed = sorted(
            {
                name
                for name, module in component.named_modules()
                if isinstance(module, DendriticBranchLayer)
                and not forward.calls[name]
                and any(
                    owners.get(index)
                    for index, (param_name, _) in enumerate(named)
                    if param_name.startswith(f"{name}.")
                )
            }
        )
        if bypassed:
            raise ValueError(
                f"strategy '{self.mode}' cuts the forward pass at branch layer "
                f"outputs, but {bypassed[0]} was evaluated without its forward "
                "method (as recurrent cells do); use the recurrent or standard "
                "strategy"
            )
        head, blocks, uncovered = [], defaultdict(list), []
        for index, owner in sorted(owners.items()):
            if len(owner) > 1:
                uncovered.append(named[index][0])
            elif owner[0] == _HEAD:
                head.append(params[index])
            else:
                blocks[owner[0]].append(params[index])
        if uncovered:
            logger.warning(
                "%sleaves %d trainable parameters without gradient because "
                "several blocks read them: %s",
                self.logger_info_prefix,
                len(uncovered),
                ", ".join(uncovered[:10]) + (" ..." if len(uncovered) > 10 else ""),
            )
        self.uncovered_parameters = uncovered
        logger.info(
            "%s%d blocks and an output layer with %d parameters",
            self.logger_info_prefix,
            len(blocks),
            len(head),
        )
        return _Plan(head=head, blocks=dict(blocks), uncovered=uncovered)

    def _feedback_matrix(
        self, key: str, shape: tuple[int, int], reference: torch.Tensor
    ) -> torch.Tensor:
        """Return the fixed random matrix for ``key``.

        Drawn on the CPU from the trainer seed and the block name, so every
        rank and a resumed run build the same matrix and the training RNG
        stream is not consumed.
        """
        matrix = self.feedback.get(key)
        if matrix is None or tuple(matrix.shape) != shape:
            generator = torch.Generator().manual_seed(
                derive_seed(int(getattr(self, "seed", 0) or 0), f"{self.mode}/{key}")
            )
            matrix = torch.randn(shape, generator=generator) * shape[0] ** -0.5
            self.feedback[key] = matrix
        return matrix.to(device=reference.device, dtype=reference.dtype)

    def _block_cut(self, forward: _Forward, name: str) -> _Cut:
        cut = forward.cuts.get(name)
        if cut is None:
            raise RuntimeError(
                f"block {name} did not run in this forward pass; FA/DFA needs "
                "the same blocks in every step"
            )
        return cut

    # ------------------------------------------------------------------
    # DFA
    # ------------------------------------------------------------------
    def _assign_dfa_gradients(
        self, loss: torch.Tensor, y_hat: torch.Tensor, forward: _Forward
    ) -> None:
        plan = self._plan
        names = list(plan.blocks)
        cuts = [self._block_cut(forward, name) for name in names]
        output_error = torch.autograd.grad(loss, y_hat, retain_graph=True)[0]
        output_error = output_error.reshape(output_error.shape[0], -1)
        exact = torch.autograd.grad(
            loss,
            [*plan.head, *(cut.leaf for cut in cuts)],
            allow_unused=True,
            retain_graph=True,
        )
        _accumulate(plan.head, exact[: len(plan.head)])

        outputs, errors, params = [], [], []
        for name, cut, direct in zip(names, cuts, exact[len(plan.head) :]):
            if direct is None:
                features = cut.output[0].numel()
                feedback = self._feedback_matrix(
                    name, (output_error.shape[1], features), output_error
                )
                direct = (output_error @ feedback).reshape(cut.output.shape)
            outputs.append(cut.output)
            errors.append(direct.to(cut.output.dtype))
            params.extend(plan.blocks[name])
        if outputs:
            grads = torch.autograd.grad(
                outputs, params, grad_outputs=errors, allow_unused=True
            )
            _accumulate(params, grads)

    # ------------------------------------------------------------------
    # FA
    # ------------------------------------------------------------------
    def _assign_fa_gradients(
        self, loss: torch.Tensor, y_hat: torch.Tensor, forward: _Forward
    ) -> None:
        plan = self._plan
        if not forward.linear_order or any(
            not name.endswith(":input") for name in forward.cuts
        ):
            raise ValueError(
                "strategy 'fa' propagates the error layer by layer and needs a "
                "sequential chain of linear layers. Dendritic tree cores are "
                "not such a chain: use 'dfa'."
            )
        order = forward.linear_order
        exact = torch.autograd.grad(
            loss,
            [*plan.head, forward.linear_outputs[order[-1]]],
            allow_unused=True,
            retain_graph=True,
        )
        _accumulate(plan.head, exact[:-1])
        delta = exact[-1]  # loss gradient at the last linear layer's output
        if delta is None:
            raise ValueError(
                "strategy 'fa' needs the model output to follow from the last "
                f"linear layer, {order[-1]}"
            )
        for position in range(len(order) - 1, -1, -1):
            linear = order[position]
            cut = forward.cuts.get(f"{linear}:input")
            if cut is None:
                if position > 0:
                    self._broken_chain(linear, order[position - 1])
                break
            # The error at the layer's input, with a fixed B in place of W^T.
            feedback = self._feedback_matrix(
                linear, (delta.shape[-1], cut.leaf.shape[-1]), delta
            )
            error = (delta @ feedback).reshape(cut.output.shape)
            params = plan.blocks.get(f"{linear}:input", [])
            previous = None
            if position > 0:
                previous = forward.linear_outputs[order[position - 1]]
            inputs = [*params] + ([] if previous is None else [previous])
            if not inputs:
                break
            grads = torch.autograd.grad(
                cut.output,
                inputs,
                grad_outputs=error.to(cut.output.dtype),
                allow_unused=True,
                retain_graph=True,
            )
            _accumulate(params, grads[: len(params)])
            if previous is None:
                break
            delta = grads[-1]
            if delta is None:
                self._broken_chain(linear, order[position - 1])

    @staticmethod
    def _broken_chain(linear: str, previous: str) -> None:
        raise ValueError(
            "strategy 'fa' propagates the error layer by layer and needs a "
            f"sequential chain of linear layers, but the input of {linear} does "
            f"not follow from the output of {previous}"
        )


def _accumulate(params, grads) -> None:
    """Add ``grads`` to the ``.grad`` of ``params``, skipping missing ones."""
    for param, grad in zip(params, grads):
        if grad is None:
            continue
        grad = grad.to(param.dtype)
        param.grad = grad if param.grad is None else param.grad + grad


__all__ = ["FeedbackAlignmentTrainer"]
