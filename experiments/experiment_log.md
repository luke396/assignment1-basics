# Experiment Log

Keep a record of every training run and evaluation so results are reproducible and comparable.

## How to log automatically
- Training writes metrics/metadata to `<log_dir>/<run_name>_<timestamp>/` only when `--log-dir` is provided. Metrics are in `metrics.csv` with columns `step,elapsed_sec,train_loss_ma,val_loss,lr`. Run metadata is in `meta.json`.
- Suggested command pattern:
  - `python -m cs336_basics.train --train-dataset ... --validation-dataset ... --run-name <short_name> --log-dir logs`

## What to record manually
Fill the table below as you run experiments. Add rows instead of replacing past ones.

| run_name | date_utc | command | key_params | best_val_loss | checkpoint | log_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| example_lr1e-4 | 2024-03-01 | `python -m cs336_basics.train --train-dataset ...` | lr=1e-4, bs=32, steps=10k, seed=42 | 2.345 @ step 5000 | logs/example_lr1e-4_20240301-120000/checkpoint_step_5000.pt | logs/example_lr1e-4_20240301-120000 | Baseline run; stable. |
