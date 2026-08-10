"""Adam with reduced-precision optimizer state (the 1M-gate memory lever).

Standard Adam keeps TWO fp32 shadow tensors per parameter (momentum and
variance): 8 bytes/param of optimizer state on top of the 4-byte weight. At
1M x 5k that state is the single largest sharded term after the weights.
This class stores both moments in ``state_dtype`` (bf16 halves them to
4 bytes/param) while doing every update in fp32:

    upcast state -> textbook Adam math in fp32 -> store back in state_dtype

so the ONLY numerical difference vs ``torch.optim.Adam`` is one bf16 rounding
of each moment per step. With ``state_dtype=torch.float32`` the math is
IDENTICAL to torch's Adam (single-tensor path) -- that is the built-in
self-check used by the equivalence tests.

L2 weight decay follows ``torch.optim.Adam`` semantics (added to the grad).
No amsgrad, no fused/foreach -- deliberately the simplest auditable loop.
Flag-gated via ``optimizer.state_dtype: bf16`` in the training config;
absent flag -> torch.optim.Adam is used, zero behavior change.
"""

from __future__ import annotations

import torch


class BF16StateAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, state_dtype=torch.bfloat16):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state_dtype = state_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, dtype=self.state_dtype, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, dtype=self.state_dtype, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # fp32 working copies (upcast is a no-op when state_dtype=fp32)
                exp_avg = state["exp_avg"].to(torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(torch.float32)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr / bias_c1
                denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # store back at reduced precision (the ONLY lossy operation)
                state["exp_avg"].copy_(exp_avg)
                state["exp_avg_sq"].copy_(exp_avg_sq)
        return loss


__all__ = ["BF16StateAdam"]
