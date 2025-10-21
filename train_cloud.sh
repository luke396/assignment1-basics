#!/bin/bash

# Train BPE tokenizer on cloud server
# Automatically shuts down after completion

# DATASET=tinystories_train VOCAB_SIZE=5000 ./train_cloud.sh
# default is owt_train with 32000

# Configuration
DATASET="${DATASET:-owt_train}"  # Default to owt_train
VOCAB_SIZE="${VOCAB_SIZE:-}"     # Empty means use dataset default
NUM_WORKERS="${NUM_WORKERS:-}"   # Empty means use CPU count

# Build command
CMD="python -m cs336_basics.bpe --dataset $DATASET"

if [ -n "$VOCAB_SIZE" ]; then
    CMD="$CMD --vocab-size $VOCAB_SIZE"
fi

if [ -n "$NUM_WORKERS" ]; then
    CMD="$CMD --num-workers $NUM_WORKERS"
fi

echo "Running: $CMD"
$CMD

# Shutdown after completion
echo "Training complete. Shutting down..."
/usr/bin/shutdown   