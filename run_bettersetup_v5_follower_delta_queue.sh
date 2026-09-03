#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/qihan/data/lerobot/data/bettersetup_v5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/train}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/queue_logs"
LOCK_FILE="$OUTPUT_ROOT/.bettersetup_v5_follower_delta_queue.lock"
LOG_FILE="$LOG_DIR/bettersetup_v5_follower_delta_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root not found: $DATASET_ROOT" >&2
    exit 1
fi

for value_name in STEPS BATCH_SIZE NUM_WORKERS; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 1
    fi
done

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 1
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another follower-delta training queue is already running: $LOCK_FILE" >&2
    exit 1
fi

archive_incomplete_output() {
    local output_dir="$1"

    if [[ ! -e "$output_dir" || "$DRY_RUN" == "1" ]]; then
        return
    fi

    local archive_path="${output_dir}.incomplete.${RUN_ID}"
    local suffix=1
    while [[ -e "$archive_path" ]]; do
        archive_path="${output_dir}.incomplete.${RUN_ID}.${suffix}"
        ((suffix += 1))
    done

    echo "Archiving incomplete output: $output_dir -> $archive_path"
    mv "$output_dir" "$archive_path"
}

run_experiment() {
    local job_name="$1"
    local action_target="$2"
    shift 2
    local image_keys=("$@")
    local output_dir="$OUTPUT_ROOT/$job_name"
    local step_dir
    step_dir="$(printf '%06d' "$STEPS")"
    local final_model="$output_dir/checkpoints/$step_dir/pretrained_model/model.safetensors"
    local last_model="$output_dir/checkpoints/last/pretrained_model/model.safetensors"

    echo
    echo "================================================================"
    echo "Starting: $job_name"
    echo "Action target: $action_target"
    echo "Image keys: ${image_keys[*]}"
    echo "Output: $output_dir"
    echo "================================================================"

    if [[ -f "$final_model" || -f "$last_model" ]]; then
        echo "Completed checkpoint already exists; skipping $job_name."
        return
    fi

    archive_incomplete_output "$output_dir"

    local command=(
        "$PYTHON_BIN" mycode/train_lerobot_policy.py
        --policy-type act
        --root "$DATASET_ROOT"
        --repo-id bettersetup_v5
        --image-keys "${image_keys[@]}"
        --state-keys observation.state
        --act-action-target "$action_target"
        --act-follower-state-key observation.state
        --act-action-representation absolute
        --output-dir "$output_dir"
        --job-name "$job_name"
        --steps "$STEPS"
        --seed 1000
        --batch-size "$BATCH_SIZE"
        --chunk-size 60
        --n-action-steps 60
        --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
        --device "$DEVICE"
        --num-workers "$NUM_WORKERS"
        --video-backend pyav
        --rebuild-view
    )

    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}"

    if [[ ! -f "$final_model" && ! -f "$last_model" ]]; then
        echo "Training returned successfully but no final checkpoint was found for $job_name." >&2
        exit 1
    fi

    echo "Finished: $job_name"
}

on_exit() {
    local status=$?
    echo
    if [[ "$status" -eq 0 ]]; then
        echo "Follower-delta queue completed successfully."
    else
        echo "Follower-delta queue stopped with exit code $status." >&2
    fi
    echo "Log: $LOG_FILE"
}
trap on_exit EXIT

echo "Run ID: $RUN_ID"
echo "Dataset: $DATASET_ROOT"
echo "Steps per experiment: $STEPS"
echo "Device: $DEVICE"
echo "Dry run: $DRY_RUN"

# The direct calls below are intentionally serial. A failure stops the queue.
run_experiment \
    FDelta-ACT-v1-F-bettersetup-v5 \
    follower_delta \
    observation.images.front

run_experiment \
    FDelta-ACT-v1-FS-bettersetup-v5 \
    follower_delta \
    observation.images.front \
    observation.images.side

run_experiment \
    FAnchorDelta-ACT-v1-F-bettersetup-v5 \
    follower_anchor_delta \
    observation.images.front

run_experiment \
    FAnchorDelta-ACT-v1-FS-bettersetup-v5 \
    follower_anchor_delta \
    observation.images.front \
    observation.images.side
