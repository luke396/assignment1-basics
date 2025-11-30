"""For training a transformer LM - Section 4."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch


def cross_entropy(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy loss with class dimension on the last axis.

    Args:
        inputs: Float[Tensor, "... vocab_size"]:
            Unnormalized logits whose final dimension enumerates classes.
        target: Long[Tensor, "..."]:
            Integer class indices matching ``inputs.shape[:-1]`` (values in
            ``[0, vocab_size)``).

    Returns:
        Float[Tensor, ""]: Scalar mean negative log-likelihood across all
        positions.

    """
    inputs = inputs - inputs.amax(dim=-1, keepdim=True)  # for numerical stability
    log_sum = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (log_sum - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    """Minimal AdamW matching PyTorch's algorithm (no AMSGrad/foreach/fused extras)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Initialize the optimizer hyperparameters and parameter groups.

        Args:
            params: Tensors or parameter groups to optimize.
            lr: Base learning rate that scales the parameter updates.
            betas: Exponential decay rates for the moment estimates.
            eps: Term added inside the denominator for numerical stability.
            weight_decay: Decoupled weight decay coefficient applied directly to
                parameters.

        """
        if not isinstance(params, Iterable):
            msg = "AdamW params must be iterable."
            raise TypeError(msg)

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Apply one bias-corrected AdamW update with decoupled weight decay.

        Args:
            closure: Optional function that reevaluates the model and returns a loss.

        Returns:
            Loss value produced by `closure` when provided; otherwise `None`.

        """
        if closure is not None and not isinstance(closure, Callable):
            msg = "closure must be callable returning a loss tensor."
            raise TypeError(msg)

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad  # p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # suffix with _ to indicate in-place operations
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                # Fold bias correction into the step size to avoid extra tensor division
                step_size = lr * (bias_correction2**0.5) / bias_correction1
                denom = v.sqrt().add_(eps)

                p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m, denom, value=-step_size)

                state["m"] = m
                state["v"] = v

        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Compute a learning rate with linear warmup and cosine decay followed by a floor.

    Args:
        it: Current training step.
        max_learning_rate: Maximum learning rate after warmup.
        min_learning_rate: Minimum learning rate at the end of cosine decay.
        warmup_iters: Number of steps to linearly increase the learning rate.
        cosine_cycle_iters: Step index (inclusive) when cosine decay ends;
            rate stays at `min_learning_rate` afterward.

    Returns:
        Learning rate at the current step.

    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cos_decay * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients of the given parameters to have a maximum global L2 norm.

    Args:
        parameters: Iterable of model parameters with gradients to be clipped.
        max_l2_norm: Maximum allowed L2 norm for the gradients. All gradients
            are scaled uniformly in place if the norm exceeds this value.

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # Compute global L2 norm across all gradients
    total_norm_sq = torch.zeros((), device=grads[0].device)
    for g in grads:
        total_norm_sq += torch.sum(g.pow(2))
    total_norm = torch.sqrt(total_norm_sq)

    eps = 1e-6
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
