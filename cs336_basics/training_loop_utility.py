"""Utility functions for training loops."""

import os
import typing

import numpy as np
import torch


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and next-token targets.

    Args:
        x: 1D numpy array of token ids.
        batch_size: Number of sequences to sample.
        context_length: Length of each sampled sequence.
        device: Torch device string to place the returned tensors on.

    Returns:
        Tuple of (inputs, targets) with shape (batch_size, context_length).

    """
    max_start = len(x) - context_length
    if max_start <= 0:
        msg = "Dataset must be longer than the context length."
        raise ValueError(msg)

    rng = np.random.default_rng()
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
    """Save model and optimizer state to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current training iteration.
        out: Path or file-like object to save the checkpoint to.

    """
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
    """Load model and optimizer state from a checkpoint file.

    Args:
        src: Path or file-like object to load the checkpoint from.
        model: The model to load state into.
        optimizer: The optimizer to load state into.

    Returns:
        int: the previously-serialized number of iterations.

    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
