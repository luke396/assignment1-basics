"""Training utilities for Transformer LMs."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

from cs336_basics.blocks import softmax

if typing.TYPE_CHECKING:
    import os

    from cs336_basics.tokenizer import Tokenizer


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
        if closure is not None and not callable(closure):
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
    if warmup_iters > 0 and it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it <= cosine_cycle_iters and cosine_cycle_iters > warmup_iters:
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


def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and next-token targets."""
    max_start = len(x) - context_length
    if max_start <= 0:
        msg = "Dataset must be longer than the context length."
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    start_indices = rng.integers(0, max_start, size=batch_size)
    inputs = np.stack([x[i : i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in start_indices])

    inputs_tensor = torch.as_tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def generate(  # noqa: PLR0913
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: torch.Generator | None = None,
    context_length: int | None = None,
) -> str:
    """Generate text with temperature and top-p (nucleus) sampling.

    Args:
        model: Language model that maps token ids
            to logits of shape (batch, seq, vocab).
        tokenizer: Tokenizer providing encode/decode and an ``eot_token_id``.
        prompt: Seed text to condition on.
        max_new_tokens: Maximum number of tokens to sample.
        temperature: Softmax temperature (>0); larger flattens, smaller sharpens.
        top_p: Nucleus threshold in (0, 1];
            keep the smallest prefix whose mass ≥ ``top_p``.
        rng: Optional torch.Generator for reproducible sampling.
        context_length: Optional explicit maximum sequence length to enforce;
            if provided, overrides inferring from model buffers.

    Returns:
        The decoded text for the newly generated tokens (prompt excluded).

    """
    if temperature <= 0.0:
        msg = "Temperature must be positive."
        raise ValueError(msg)
    if not (0 < top_p <= 1):
        msg = "top_p must be in the range (0, 1]."
        raise ValueError(msg)

    end_id = tokenizer.eot_token_id
    if end_id is None:
        msg = "Tokenizer must provide an eot_token_id for generation."
        raise ValueError(msg)

    device = next(model.parameters()).device  # same as model parametet's device
    prompt_tokens = tokenizer.encode(prompt)
    input_tokens = prompt_tokens.copy()

    # Enforce the model's maximum sequence length to avoid RoPE index errors.
    max_seq_len: int | None = context_length

    if max_seq_len is not None:
        available_tokens = max_seq_len - len(prompt_tokens)
        if available_tokens <= 0:
            msg = (
                "Prompt length exceeds model context length "
                f"({len(prompt_tokens)} > {max_seq_len})."
            )
            raise ValueError(msg)
        max_new_tokens = min(max_new_tokens, available_tokens)

    rng = rng if rng is not None else torch.Generator(device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(
                torch.as_tensor(
                    [input_tokens],
                    device=device,
                )
            )  # (batch_size, sequence_len, vocab_size)

            # Temperature scaling
            probs = softmax(
                logits[:, -1, :].squeeze(0) / temperature, dim=-1
            )  # (vocab_size,)

            # Top-p nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Clone to avoid overlapping memory between source/dest slices
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            # Fill removed logits with zeros, and re-normalize
            sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
            full_probs = torch.zeros_like(probs)
            full_probs[sorted_indices] = sorted_probs
            full_probs = full_probs / full_probs.sum()

            next_id = torch.multinomial(
                full_probs,
                num_samples=1,
                generator=rng,
            ).item()
            next_id = int(next_id)
            if next_id == end_id:
                break
            input_tokens.append(next_id)

    decoded = tokenizer.decode(input_tokens[len(prompt_tokens) :])
    sentinel = "<|endoftext|>"
    if sentinel in decoded:
        decoded = decoded.split(sentinel, 1)[0]
    return decoded
