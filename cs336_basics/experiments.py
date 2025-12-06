#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_run(run_dir: Path):
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.csv"
    if not (meta_path.exists() and metrics_path.exists()):
        return None

    with meta_path.open() as f:
        meta = json.load(f)
    base_lr = meta.get("config", {}).get("lr")
    batch_size = meta.get("config", {}).get("batch_size")
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
    }


def load_runs(runs_root: Path):
    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        run = load_run(run_dir)
        if run:
            runs.append(run)
    return sorted(runs, key=lambda r: r["base_lr"])


def plot_lr_loss(run_dirs=None, output_path=None):
    """Plot LR schedule and loss curves for given runs.

    If `run_dirs` is None, all runs under `output/runs` are used. Otherwise,
    only the specified run directories are loaded.
    """

    if run_dirs:
        runs = []
        for run_dir in run_dirs:
            run = load_run(Path(run_dir))
            if run is None:
                print(f"Skipping {run_dir}: missing lr/loss data")
                continue
            runs.append(run)
    else:
        runs_root = Path("output/runs")
        runs = load_runs(runs_root)

    if not runs:
        msg = "No runs found with lr and loss data."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    for idx, run in enumerate(runs):
        color = f"C{idx}"
        label = f"{run['name']} (base lr={run['base_lr']})"

        ax_lr.plot(run["steps"], run["lrs"], color=color, label=label)

        ax_loss.plot(
            run["steps"],
            run["train_losses"],
            color=color,
            linestyle="-",
            label=f"{label} train",
        )
        ax_loss.plot(
            run["steps"],
            run["val_losses"],
            color=color,
            linestyle="--",
            label=f"{label} val",
        )

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("LR cosine schedule vs loss over steps")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    output_path = Path(output_path or "output/lr_loss_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def plot_lr_vs_loss_by_batchsize(run_dirs, output_path=None):
    """Compare LR and loss trajectories over steps, grouped by batch size."""

    runs = []
    for run_dir in run_dirs:
        run = load_run(Path(run_dir))
        if run is None:
            print(f"Skipping {run_dir}: missing lr/loss data")
            continue
        runs.append(run)

    if not runs:
        msg = "No valid runs provided for LR vs loss plotting."
        raise SystemExit(msg)

    fig, (ax_lr, ax_loss) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    for idx, run in enumerate(runs):
        color = f"C{idx}"
        batch_size = run.get("batch_size", "?")
        label = f"{run['name']} (bs={batch_size})"

        ax_lr.plot(run["steps"], run["lrs"], color=color, label=label)

        ax_loss.plot(
            run["steps"],
            run["train_losses"],
            color=color,
            linestyle="-",
            label=f"{label} train",
        )
        ax_loss.plot(
            run["steps"],
            run["val_losses"],
            color=color,
            linestyle="--",
            label=f"{label} val",
        )

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("LR and loss over steps by batch size")
    ax_lr.grid(True, linestyle=":", linewidth=0.6)
    ax_lr.legend()

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", linewidth=0.6)
    ax_loss.legend()

    output_path = Path(output_path or "output/lr_loss_by_batchsize.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved LR vs loss plot (batch size comparison) to {output_path}")




if __name__ == "__main__":
    plot_lr_loss([
        "output/runs/tinystories_base_20251202-045220",
        "output/runs/tinystories_base_20251203-015552",
        "output/runs/tinystories_base_20251203-023410",
        "output/runs/tinystories_base_20251206-042444"
        ])
    plot_lr_vs_loss_by_batchsize([
      "output/runs/tinystories_base_20251203-023410",
      "output/runs/tinystories_base_20251206-051150",
  ])
