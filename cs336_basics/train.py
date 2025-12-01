"""Scripts to train models."""

import argparse
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.blocks import RotaryPositionalEmbedding, TransformerLM
from cs336_basics.training_utility import (
    AdamW,
    cross_entropy,
    data_loading,
    gradient_clipping,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    """Configuration for training."""

    train_path: str
    validation_path: str

    batch_size: int = 32
    context_length: int = 8
    vocab_size: int = 50257
    embedding_dim: int = 768
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    n_layers: int = 12

    lr: float = 1e-4
    min_lr: float | None = None
    warmup_iters: int = 500
    cosine_cycle_iters: int | None = None
    weight_decay: float = 0.01
    steps: int = 10000

    max_l2_norm: float = 1.0
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 100
    resume: str | None = None
    device: str = "cpu"
    seed: int = 42
    val_seed: int | None = None
    val_batches: int = 8

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        """Build argument parser for training configuration."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--train-dataset",
            dest="train_path",
            type=str,
            required=True,
            help="Path to training dataset.",
        )
        parser.add_argument(
            "--validation-dataset",
            dest="validation_path",
            type=str,
            required=True,
            help="Path to validation dataset.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=cls.batch_size,
            help="Batch size for training.",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=cls.context_length,
            help="Context length for training.",
        )
        parser.add_argument(
            "--vocab-size",
            type=int,
            default=cls.vocab_size,
            help="Vocabulary size for the model.",
        )
        parser.add_argument(
            "--embedding-dim",
            type=int,
            default=cls.embedding_dim,
            help="Embedding dimension for the model.",
        )
        parser.add_argument(
            "--d-model",
            type=int,
            default=cls.d_model,
            help="Dimension of the model.",
        )
        parser.add_argument(
            "--num-heads",
            type=int,
            default=cls.num_heads,
            help="Number of attention heads.",
        )
        parser.add_argument(
            "--d-ff",
            type=int,
            default=cls.d_ff,
            help="Dimension of the feedforward network.",
        )
        parser.add_argument(
            "--n-layers",
            type=int,
            default=cls.n_layers,
            help="Number of transformer layers.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=cls.lr,
            help="Learning rate for the optimizer.",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=cls.weight_decay,
            help="Weight decay for the optimizer.",
        )
        parser.add_argument(
            "--min-lr",
            type=float,
            default=None,
            help="Minimum learning rate after cosine decay; defaults to 0.1 * lr.",
        )
        parser.add_argument(
            "--warmup-iters",
            type=int,
            default=cls.warmup_iters,
            help="Number of warmup steps before cosine decay.",
        )
        parser.add_argument(
            "--cosine-cycle-iters",
            type=int,
            default=None,
            help="Step index when cosine decay ends; defaults to steps - 1.",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default=None,
            help="Directory to save checkpoints; disabled if not provided.",
        )
        parser.add_argument(
            "--checkpoint-interval",
            type=int,
            default=cls.checkpoint_interval,
            help="Save checkpoint every N steps.",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint to resume from.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=cls.device,
            help="Torch device string, e.g., 'cpu', 'cuda:0', or 'mps'.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=cls.seed,
            help="Global seed for reproducibility.",
        )
        parser.add_argument(
            "--val-seed",
            type=int,
            default=None,
            help="Optional seed for validation sampling; defaults to seed+1.",
        )
        parser.add_argument(
            "--val-batches",
            type=int,
            default=cls.val_batches,
            help="Number of validation batches to average per evaluation.",
        )
        return parser

    @classmethod
    def from_args(cls, args: Sequence[str] | None = None) -> "TrainConfig":
        """Create TrainConfig from command-line arguments."""
        parsed = cls._build_parser().parse_args(args)
        return cls(**vars(parsed))


def batch_generator(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Generate batches of data for training.

    Args:
        data: Numpy array or memmap of the dataset.
        batch_size: Number of sequences per batch.
        context_length: Length of each sequence.
        device: Torch device string to place the returned tensors on.
        rng: Optional numpy random number generator. If None, a new one will be created.

    Yields:
        Tuple of (inputs, targets) tensors with shape (batch_size, context_length).

    """
    # Stateless wrapper to repeatedly sample batches from memmap-backed data.
    while True:
        yield data_loading(data, batch_size, context_length, device, rng)


def train(config: TrainConfig) -> None:
    """Train a Transformer language model.

    Args:
        config: Training configuration.

    """
    # Global seeding for reproducibility.
    np.random.seed(config.seed)  # noqa: NPY002
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_rng = np.random.default_rng(config.seed)
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode="r")
    validation_data = np.memmap(config.validation_path, dtype=np.uint16, mode="r")
    train_loader = batch_generator(
        train_data, config.batch_size, config.context_length, config.device, train_rng
    )

    # Validation RNG is re-seeded each evaluation for stable batches.
    val_seed = config.val_seed if config.val_seed is not None else config.seed + 1

    rope = RotaryPositionalEmbedding(
        100000.0, config.d_model // config.num_heads, config.context_length
    )
    model = TransformerLM(
        config.vocab_size,
        config.embedding_dim,
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.context_length,
        config.n_layers,
        rope,
        dtype=torch.float32,
    )
    model.to(config.device)
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    start_step = 0
    if config.resume:
        start_step = load_checkpoint(config.resume, model, optimizer) + 1

    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_lr = config.lr
    min_lr = config.min_lr if config.min_lr is not None else config.lr * 0.1
    warmup_iters = config.warmup_iters
    cosine_cycle_iters = (
        config.cosine_cycle_iters
        if config.cosine_cycle_iters is not None
        else config.steps - 1
    )

    for step in range(start_step, config.steps):
        lr_now = lr_cosine_schedule(
            step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        model.train()
        inputs, targets = next(train_loader)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=config.max_l2_norm)
        optimizer.step()

        if (step + 1) % 100 == 0:
            model.eval()
            val_rng = np.random.default_rng(val_seed)
            val_losses: list[float] = []
            with torch.no_grad():
                for _ in range(config.val_batches):
                    val_inputs, val_targets = data_loading(
                        validation_data,
                        config.batch_size,
                        config.context_length,
                        config.device,
                        val_rng,
                    )
                    val_logits = model(val_inputs)
                    val_loss = cross_entropy(val_logits, val_targets)
                    val_losses.append(val_loss.item())
            mean_val_loss = float(np.mean(val_losses))
            print(
                f"Step {step + 1}/{config.steps}, "
                f"Train Loss: {loss.item():.4f}, "
                f"Validation Loss: {mean_val_loss:.4f}"
            )

        if checkpoint_dir and ((step + 1) % config.checkpoint_interval == 0):
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)


def main() -> None:
    """Parse arguments and start training."""
    config = TrainConfig.from_args()
    train(config)


if __name__ == "__main__":
    main()
