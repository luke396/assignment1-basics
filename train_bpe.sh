#!/usr/bin/env bash
set -euo pipefail

# BPE training launcher with support for batch runs.
# Configuration: config/train_bpe.env (override via TRAIN_BPE_ENV_FILE)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TRAIN_BPE_ENV_FILE:-${SCRIPT_DIR}/config/train_bpe.env}"

# Load configuration
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Defaults
TRAIN_MODULE="${BPE_TRAIN_MODULE:-cs336_basics.bpe}"
AUTO_SHUTDOWN="${BPE_AUTO_SHUTDOWN:-false}"

# Single training run
run_training() {
    local dataset="${1:-}"
    local vocab="${2:-}"
    local -a cmd=("python" -m "$TRAIN_MODULE")

    # Add arguments from parameters or environment
    [[ -n "$dataset" ]] && cmd+=("--dataset" "$dataset")
    [[ -n "$vocab" ]] && cmd+=("--vocab-size" "$vocab")
    [[ -n "${BPE_INPUT_PATH:-}" ]] && cmd+=("--input-path" "$BPE_INPUT_PATH")
    [[ -n "${BPE_NUM_WORKERS:-}" ]] && cmd+=("--num-workers" "$BPE_NUM_WORKERS")
    [[ -n "${BPE_OUTPUT_DIR:-}" ]] && cmd+=("--output-dir" "$BPE_OUTPUT_DIR")

    # Auto-generate log file for batch mode (if not explicitly set)
    if [[ -n "$dataset" ]] && [[ -z "${BPE_LOG_FILE:-}" ]] && [[ -n "${BPE_RUNS:-}" ]]; then
        local log_file="logs/bpe_${dataset}_${vocab}_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p "logs"
        cmd+=("--log-file" "$log_file")
    elif [[ -n "${BPE_LOG_FILE:-}" ]]; then
        cmd+=("--log-file" "$BPE_LOG_FILE")
    fi

    [[ "${BPE_SAVE:-true}" == "false" ]] && cmd+=("--no-save")

    echo "[$(date --iso-8601=seconds)] ${cmd[*]}"
    "${cmd[@]}"
}

# Execute training
if [[ -n "${BPE_RUNS:-}" ]]; then
    # Batch mode: BPE_RUNS="dataset1:vocab1,dataset2:vocab2,..."
    IFS=',' read -r -a runs <<< "$BPE_RUNS"
    for entry in "${runs[@]}"; do
        [[ -z "$entry" ]] && continue
        dataset="${entry%%:*}"
        vocab="${entry##*:}"
        run_training "$dataset" "$vocab"
    done
else
    # Single run mode
    run_training "${BPE_DATASET:-}" "${BPE_VOCAB_SIZE:-}"
fi

# Optional auto-shutdown (for cloud instances)
if [[ "$AUTO_SHUTDOWN" == "true" ]]; then
    echo "Auto-shutdown in 60s (Ctrl+C to cancel)..."
    sleep 60
    SHUTDOWN_CMD="$(command -v shutdown || echo /sbin/shutdown)"
    sudo "$SHUTDOWN_CMD" -h now || echo "Warning: shutdown failed"
fi
