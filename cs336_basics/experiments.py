"""For section 7."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cs336_basics.tokenizer import Tokenier
from cs336_basics.train import TrainConfig, build_model_and_optimizer
from cs336_basics.training_utility import generate as generate_tokens
from cs336_basics.training_utility import load_checkpoint


def load_run(run_dir: Path):
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.csv"
    if not (meta_path.exists() and metrics_path.exists()):
        return None

    with meta_path.open() as f:
        meta = json.load(f)
    base_lr = meta.get("config", {}).get("lr")
    batch_size = meta.get("config", {}).get("batch_size")
    post_norm = meta.get("config", {}).get("post_norm")
    no_rope = meta.get("config", {}).get("no_rope")
    silu = meta.get("config", {}).get("silu")
    if base_lr is None:
        return None

    steps, train_losses, val_losses, lrs = [], [], [], []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                lrs.append(float(row["lr"]))
                # prefer the moving-average column if present
                train_val = row.get("train_loss_ma") or row.get("train_loss")
                train_losses.append(float(train_val) if train_val is not None else None)
                val_losses.append(float(row["val_loss"]))
            except (KeyError, ValueError):
                continue

    if not steps:
        return None

    return {
        "name": run_dir.name,
        "base_lr": float(base_lr),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lrs": lrs,
        "post_norm": post_norm,
        "no_rope": no_rope,
        "silu": silu,
    }


def load_runs(runs_root: Path):
    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        run = load_run(run_dir)
        if run:
            runs.append(run)
    return sorted(runs, key=lambda r: r["base_lr"])


def _load_runs_from_dirs(run_dirs):
    runs = []
    for run_dir in run_dirs:
        run = load_run(Path(run_dir))
        if run is None:
            print(f"Skipping {run_dir}: missing lr/loss data")
            continue
        runs.append(run)
    return runs


def _plot_lr(ax, runs, label_fn):
    for idx, run in enumerate(runs):
        color = f"C{idx}"
        ax.plot(run["steps"], run["lrs"], color=color, label=label_fn(run))


def _plot_loss(ax, runs, label_fn):
    for idx, run in enumerate(runs):
        color = f"C{idx}"
        base_label = label_fn(run)
        ax.plot(
            run["steps"],
            run["train_losses"],
            color=color,
            linestyle="-",
            label=f"{base_label} train",
        )
        ax.plot(
            run["steps"],
            run["val_losses"],
            color=color,
            linestyle="--",
            label=f"{base_label} val",
        )


def _save_figure(fig, output_path, default_path, message_prefix):
    target_path = Path(output_path or default_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=200)
    print(f"{message_prefix} {target_path}")


def plot_lr_loss(run_dirs=None, output_path=None):
    """Plot LR schedule and loss curves for given runs.

    If `run_dirs` is None, all runs under `output/runs` are used. Otherwise,
    only the specified run directories are loaded.
    """
    if run_dirs:
        runs = _load_runs_from_dirs(run_dirs)
    else:
        runs_root = Path("output/runs")
        runs = load_runs(runs_root)

    if not runs:
        msg = "No runs found with lr and loss data."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    label_fn = lambda run: f"{run['name']} (base lr={run['base_lr']})"
    _plot_lr(ax_lr, runs, label_fn)
    _plot_loss(ax_loss, runs, label_fn)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("LR cosine schedule vs loss over steps")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    _save_figure(fig, output_path, "output/lr_loss_curves.png", "Saved plot to")


def plot_lr_vs_loss_by_batchsize(run_dirs, output_path=None):
    """Compare LR and loss trajectories over steps, grouped by batch size."""
    runs = _load_runs_from_dirs(run_dirs)

    if not runs:
        msg = "No valid runs provided for LR vs loss plotting."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    label_fn = lambda run: f"{run['name']} (bs={run.get('batch_size', '?')})"
    _plot_lr(ax_lr, runs, label_fn)
    _plot_loss(ax_loss, runs, label_fn)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("LR and loss over steps by batch size")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    _save_figure(
        fig,
        output_path,
        "output/lr_loss_by_batchsize.png",
        "Saved LR vs loss plot (batch size comparison) to",
    )


def plot_pre_vs_post_norm_loss(run_dirs=None, output_path=None):
    """Compare pre-norm vs post-norm losses when other settings match."""
    if run_dirs is None:
        msg = "Provide run directories for pre-norm and post-norm comparisons."
        raise SystemExit(msg)
    runs = _load_runs_from_dirs(run_dirs)

    if len(runs) < 2:
        msg = "Need at least two runs (pre-norm and post-norm) to compare losses."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    label_fn = lambda run: (
        f"{'post-norm' if run.get('post_norm') else 'pre-norm'} ({run['name']})"
    )
    _plot_lr(ax_lr, runs, label_fn)
    _plot_loss(ax_loss, runs, label_fn)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("Pre-norm vs post-norm: LR and loss (same hyperparameters)")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    _save_figure(
        fig,
        output_path,
        "output/pre_vs_post_norm_loss.png",
        "Saved pre- vs post-norm LR/loss comparison to",
    )


def plot_rope_vs_no_rope(run_dirs=None, output_path=None):
    """Compare RoPE vs no-RoPE losses when other settings match."""
    if run_dirs is None:
        msg = "Provide run directories for RoPE and no-RoPE comparisons."
        raise SystemExit(msg)
    runs = _load_runs_from_dirs(run_dirs)

    if len(runs) < 2:
        msg = "Need at least two runs (RoPE and no-RoPE) to compare losses."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    label_fn = lambda run: f"{'no RoPE' if run.get('no_rope') else 'RoPE'} ({run['name']})"
    _plot_lr(ax_lr, runs, label_fn)
    _plot_loss(ax_loss, runs, label_fn)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("RoPE vs no-RoPE: LR and loss (same hyperparameters)")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    _save_figure(
        fig,
        output_path,
        "output/rope_vs_no_rope_loss.png",
        "Saved RoPE vs no-RoPE LR/loss comparison to",
    )


def plot_silu_vs_swiglu(run_dirs=None, output_path=None):
    """Compare SiLU vs SwiGLU losses when other settings match."""
    if run_dirs is None:
        msg = "Provide run directories for SiLU and SwiGLU comparisons."
        raise SystemExit(msg)
    runs = _load_runs_from_dirs(run_dirs)

    if len(runs) < 2:
        msg = "Need at least two runs (SiLU and SwiGLU) to compare losses."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    label_fn = lambda run: f"{'SiLU' if run.get('silu') else 'SwiGLU'} ({run['name']})"
    _plot_lr(ax_lr, runs, label_fn)
    _plot_loss(ax_loss, runs, label_fn)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("SiLU vs SwiGLU: LR and loss (same hyperparameters)")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    _save_figure(
        fig,
        output_path,
        "output/silu_vs_swiglu_loss.png",
        "Saved SiLU vs SwiGLU LR/loss comparison to",
    )


def generate_text(
    config: TrainConfig,
    checkpoint_path: str | Path,
    prompt: str,
    tokenizer: Tokenier | None = None,
    tokenizer_vocab_path: str | None = None,
    tokenizer_merges_path: str | None = None,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    """Load a checkpoint, rebuild the model from config, and generate text."""
    if tokenizer is None:
        if tokenizer_vocab_path is None or tokenizer_merges_path is None:
            msg = (
                "Provide a tokenizer instance or both tokenizer_vocab_path and "
                "tokenizer_merges_path."
            )
            raise ValueError(msg)
        tokenizer = Tokenier.from_files(
            tokenizer_vocab_path,
            tokenizer_merges_path,
            special_tokens=["<|endoftext|>"],
        )

    model, optimizer = build_model_and_optimizer(config)
    load_checkpoint(checkpoint_path, model, optimizer)
    model.eval()

    completion = generate_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        context_length=config.context_length,
    )
    return prompt + completion


def generate_tiny(temperature: float = 0.8, top_p: float = 0.9):
    config = TrainConfig(
        train_path="data/ts_train_token_ids.npy",
        validation_path="data/ts_valid_token_ids.npy",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        d_ff=1344,
        n_layers=4,
        num_heads=16,
    )
    checkpoint_path = "output/runs/tiny_128_0.003_1.3461_checkpoint_best.pt"
    tokenizer_vocab_path = "data/tokenizer_vocab_TinyStoriesV2-GPT4-train_10000.json"
    tokenizer_merges_path = "data/tokenizer_merges_TinyStoriesV2-GPT4-train_10000.json"
    prompt = "Once upon a time"
    completion = generate_text(
        config=config,
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        tokenizer_vocab_path=tokenizer_vocab_path,
        tokenizer_merges_path=tokenizer_merges_path,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
    )
    print("Generated text: with temperature =", temperature, "and top_p =", top_p)
    print(completion)


if __name__ == "__main__":
    plot_lr_loss(
        [
            "output/runs/tinystories_base_20251202-045220",
            "output/runs/tinystories_base_20251203-015552",
            "output/runs/tinystories_base_20251203-023410",
            "output/runs/tinystories_base_20251206-042444",
        ]
    )
    plot_lr_vs_loss_by_batchsize(
        [
            "output/runs/tinystories_base_20251203-023410", # normal best model
            "output/runs/tinystories_base_20251206-051150", # batch size 64
        ]
    )
    plot_pre_vs_post_norm_loss(
        [
            "output/runs/tinystories_base_20251203-023410", # pre-norm
            "output/runs/tinystories_post_norm_20251206-105527", # post-norm
        ]
    )
    plot_rope_vs_no_rope(
        [
            "output/runs/tinystories_base_20251203-023410", # RoPE
            "output/runs/tinystories_no_rope_20251206-113333", # no RoPE
        ]
    )
    plot_silu_vs_swiglu(
        [
            "output/runs/tinystories_base_20251203-023410", # SwiGLU base
            "output/runs/tinystories_silu_20251206-121426", # SiLU
        ]
    )
    # generate_tiny(temperature=1, top_p=1)
    # generate_tiny(temperature=0.1, top_p=0.9)
    # generate_tiny(temperature=5, top_p=0.5)
