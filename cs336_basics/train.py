"""Scripts to train models."""

import argparse
import json
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    train_loss_ma_window: int = 100
    log_dir: str | None = None
    run_name: str = "run"
    log_interval: int = 100

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
        parser.add_argument(
            "--train-loss-ma-window",
            type=int,
            default=cls.train_loss_ma_window,
            help="Window size for moving-average training loss logging.",
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default=cls.log_dir,
            help=(
                "Directory to store experiment logs (CSV/metadata). "
                "If omitted or empty, logging is disabled."
            ),
        )
        parser.add_argument(
            "--run-name",
            type=str,
            default=cls.run_name,
            help="Human-readable name for this run; timestamp is appended.",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=cls.log_interval,
            help="How often (in steps) to log and evaluate validation loss.",
        )
        return parser

    @classmethod
    def from_args(cls, args: Sequence[str] | None = None) -> "TrainConfig":
        """Create TrainConfig from command-line arguments."""
        parsed = cls._build_parser().parse_args(args)
        return cls(**vars(parsed))


class BatchLoader:
    """Stateful batch sampler with controllable RNG for reproducibility."""

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str,
        seed: int,
    ) -> None:
        """Initialize batch loader.

        Args:
            data: 1D array of token IDs.
            batch_size: Number of sequences per batch.
            context_length: Length of each sequence.
            device: Torch device string for returned tensors.
            seed: Seed for the internal RNG.

        """
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset RNG to the initial seed (useful for deterministic validation)."""
        self.rng = np.random.default_rng(self.seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the next batch."""
        return data_loading(
            self.data,
            self.batch_size,
            self.context_length,
            self.device,
            self.rng,
        )


class ExperimentLogger:
    """Lightweight logger that tracks metrics over steps and wallclock time.

    Writes metadata to ``meta.json`` and metrics to ``metrics.csv`` inside a
    time-stamped run directory under ``log_dir``.
    """

    def __init__(
        self, log_dir: str | Path | None, run_name: str, config: TrainConfig
    ) -> None:
        """Initialize logger and create run directory/metadata if enabled.

        Args:
            log_dir: Root directory for run logs; disables logging if falsy.
            run_name: Human-readable name for the run (timestamp is appended).
            config: Full training configuration to store in ``meta.json``.

        """
        if not log_dir:
            self.enabled = False
            self.run_dir = None
            self.metrics_path = None
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.run_dir = Path(log_dir) / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.csv"
        meta_path = self.run_dir / "meta.json"

        meta = {
            "run_name": run_name,
            "start_time_utc": timestamp,
            "config": asdict(config),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        header = "step,elapsed_sec,train_loss_ma,val_loss,lr\n"
        self.metrics_path.write_text(header, encoding="utf-8")

    def log_metrics(
        self,
        step: int,
        elapsed_sec: float,
        train_loss_ma: float,
        val_loss: float,
        lr: float,
    ) -> None:
        """Append one metrics row to CSV."""
        if not self.enabled or self.metrics_path is None:
            return

        line = f"{step},{elapsed_sec:.3f},{train_loss_ma:.6f},{val_loss:.6f},{lr:.8f}\n"
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(line)


def set_global_seeds(seed: int) -> None:
    """Seed numpy/torch for reproducibility (CPU and CUDA if available)."""
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config_values(config: TrainConfig) -> None:
    """Validate config fields that must be positive."""
    if config.train_loss_ma_window <= 0:
        msg = "train_loss_ma_window must be positive."
        raise ValueError(msg)
    if config.log_interval <= 0:
        msg = "log_interval must be positive."
        raise ValueError(msg)


def build_data_sources(
    config: TrainConfig, val_seed: int
) -> tuple[BatchLoader, BatchLoader]:
    """Create train and validation batch loaders."""
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode="r")
    validation_data = np.memmap(config.validation_path, dtype=np.uint16, mode="r")
    train_loader = BatchLoader(
        train_data,
        config.batch_size,
        config.context_length,
        config.device,
        seed=config.seed,
    )
    val_loader = BatchLoader(
        validation_data,
        config.batch_size,
        config.context_length,
        config.device,
        seed=val_seed,
    )
    return train_loader, val_loader


def build_model_and_optimizer(
    config: TrainConfig,
) -> tuple[TransformerLM, AdamW]:
    """Instantiate model, positional embedding, and optimizer."""
    rope = RotaryPositionalEmbedding(
        100000.0, config.d_model // config.num_heads, config.context_length
    )
    model = TransformerLM(
        config.vocab_size,
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
    return model, optimizer


def apply_learning_rate(optimizer: AdamW, lr_now: float) -> None:
    """Set the learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now


def run_train_step(
    model: TransformerLM,
    optimizer: AdamW,
    train_loader: BatchLoader,
    config: TrainConfig,
) -> float:
    """Run one training step and return the scalar loss."""
    model.train()
    inputs, targets = train_loader.next_batch()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=config.max_l2_norm)
    optimizer.step()
    return float(loss.item())


def run_validation(
    model: TransformerLM,
    val_loader: BatchLoader,
    config: TrainConfig,
) -> float:
    """Compute mean validation loss using a deterministic loader."""
    model.eval()
    val_loader.reset()
    val_losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.val_batches):
            val_inputs, val_targets = val_loader.next_batch()
            val_logits = model(val_inputs)
            val_loss = cross_entropy(val_logits, val_targets)
            val_losses.append(val_loss.item())
    return float(np.mean(val_losses))


def prepare_checkpoint_dir(path_str: str | None) -> Path | None:
    """Create and return a checkpoint directory if provided."""
    if not path_str:
        return None
    checkpoint_dir = Path(path_str)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def train(config: TrainConfig) -> None:
    """Train a Transformer language model.

    Args:
        config: Training configuration.

    """
    set_global_seeds(config.seed)
    validate_config_values(config)
    val_seed = config.val_seed if config.val_seed is not None else config.seed + 1
    train_loader, val_loader = build_data_sources(config, val_seed)

    model, optimizer = build_model_and_optimizer(config)
    logger = ExperimentLogger(config.log_dir, config.run_name, config)
    if logger.enabled and logger.run_dir is not None:
        print(f"Logging run to {logger.run_dir}")

    start_step = (
        load_checkpoint(config.resume, model, optimizer) + 1 if config.resume else 0
    )
    checkpoint_dir = prepare_checkpoint_dir(config.checkpoint_dir)

    max_lr = config.lr
    min_lr = config.min_lr if config.min_lr is not None else config.lr * 0.1
    warmup_iters = config.warmup_iters
    cosine_cycle_iters = (
        config.cosine_cycle_iters
        if config.cosine_cycle_iters is not None
        else config.steps - 1
    )

    train_losses = deque(maxlen=config.train_loss_ma_window)
    start_time = time.time()
    for step in range(start_step, config.steps):
        lr_now = lr_cosine_schedule(
            step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        apply_learning_rate(optimizer, lr_now)
        step_loss = run_train_step(model, optimizer, train_loader, config)
        train_losses.append(step_loss)

        if (step + 1) % config.log_interval == 0:
            mean_val_loss = run_validation(model, val_loader, config)
            elapsed = time.time() - start_time
            train_ma = float(np.mean(train_losses))
            print(
                f"Step {step + 1}/{config.steps}, "
                f"Train Loss (MA {config.train_loss_ma_window}): {train_ma:.4f}, "
                f"Validation Loss: {mean_val_loss:.4f}"
            )
            logger.log_metrics(
                step=step + 1,
                elapsed_sec=elapsed,
                train_loss_ma=train_ma,
                val_loss=mean_val_loss,
                lr=lr_now,
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
