#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob

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

FRONT_MODEL="$DATASET_ROOT/models/unet_front_v4_r1/best.pt"
SIDE_MODEL="$DATASET_ROOT/models/unet_side/best.pt"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/queue_logs"
LOG_FILE="$LOG_DIR/bettersetup_v5_unet_sem_delta_${RUN_ID}.log"
LOCK_FILE="$OUTPUT_ROOT/.bettersetup_v5_unet_sem_delta_queue.lock"

FRONT_MASK_KEYS=(
    observation.images.front_occluder
    observation.images.front_object
    observation.images.front_region
    observation.images.front_tool
    observation.images.front_leftarm
    observation.images.front_rightarm
)
SIDE_MASK_KEYS=(
    observation.images.side_occluder
    observation.images.side_object
    observation.images.side_region
    observation.images.side_tool
)

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

for required_path in "$PYTHON_BIN" "$DATASET_ROOT" "$FRONT_MODEL" "$SIDE_MODEL"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path not found: $required_path" >&2
        exit 1
    fi
done
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
    echo "Another bettersetup_v5 UNET semantic delta queue is already running: $LOCK_FILE" >&2
    exit 1
fi

latest_checkpoint() {
    local output_dir="$1"
    local checkpoints=("$output_dir"/checkpoint_step_*/training_state.pt)
    ((${#checkpoints[@]} > 0)) || return 1
    printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
}

run_experiment() {
    local job_name="$1"
    local experiment="$2"
    local action_target="$3"
    local view_mode="$4"
    local output_dir="$OUTPUT_ROOT/$job_name"
    local final_checkpoint="$output_dir/checkpoint_step_$(printf '%06d' "$STEPS")/training_state.pt"
    local -a view_args

    if [[ "$view_mode" == "front" ]]; then
        view_args=(
            --rgb-keys observation.images.front
            --mask-target-keys "${FRONT_MASK_KEYS[@]}"
            --pretrained-segmentation-checkpoints "$FRONT_MODEL"
        )
    elif [[ "$view_mode" == "front-side" ]]; then
        view_args=(
            --rgb-keys observation.images.front observation.images.side
            --mask-target-keys "${FRONT_MASK_KEYS[@]}" "${SIDE_MASK_KEYS[@]}"
            --pretrained-segmentation-checkpoints "$FRONT_MODEL" "$SIDE_MODEL"
        )
    else
        echo "Unknown view mode: $view_mode" >&2
        exit 1
    fi

    local -a command=(
        "$PYTHON_BIN" mycode/train_mask_act_policy.py
        --experiment "$experiment"
        --root "$DATASET_ROOT"
        --repo-id bettersetup_v5
        "${view_args[@]}"
        --state-keys observation.state
        --act-action-target "$action_target"
        --act-follower-state-key observation.state
        --output-dir "$output_dir"
        --steps "$STEPS"
        --seed 1000
        --batch-size "$BATCH_SIZE"
        --chunk-size 60
        --n-action-steps 60
        --action-loss-weight 1.0
        --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
        --device "$DEVICE"
        --num-workers "$NUM_WORKERS"
        --video-backend pyav
        --rebuild-view
    )

    echo
    echo "================================================================"
    echo "Starting: $job_name"
    echo "Experiment: $experiment"
    echo "Action target: $action_target"
    echo "Views: $view_mode"
    echo "Output: $output_dir"
    echo "================================================================"

    if [[ -f "$final_checkpoint" ]]; then
        echo "Completed checkpoint already exists; skipping $job_name."
        return
    fi

    local resume_checkpoint=""
    if resume_checkpoint="$(latest_checkpoint "$output_dir")"; then
        command+=(--resume-checkpoint "$resume_checkpoint")
        echo "Resuming from: $resume_checkpoint"
    elif [[ -d "$output_dir" && "$DRY_RUN" == "0" ]]; then
        local archive_path="${output_dir}.incomplete.${RUN_ID}"
        local suffix=1
        while [[ -e "$archive_path" ]]; do
            archive_path="${output_dir}.incomplete.${RUN_ID}.${suffix}"
            ((suffix += 1))
        done
        echo "Archiving incomplete output: $output_dir -> $archive_path"
        mv "$output_dir" "$archive_path"
    fi

    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}"

    if [[ ! -f "$final_checkpoint" ]]; then
        echo "Training returned successfully but no final checkpoint was found for $job_name." >&2
        exit 1
    fi
    echo "Finished: $job_name"
}

on_exit() {
    local status=$?
    echo
    if [[ "$status" -eq 0 ]]; then
        echo "UNET semantic delta queue completed successfully."
    else
        echo "UNET semantic delta queue stopped with exit code $status." >&2
    fi
    echo "Log: $LOG_FILE"
}
trap on_exit EXIT

echo "Run ID: $RUN_ID"
echo "Dataset: $DATASET_ROOT"
echo "Steps per experiment: $STEPS"
echo "Device: $DEVICE"
echo "Dry run: $DRY_RUN"

# Every target is derived from observation.state (follower); dataset action (leader) is not supervision.
# Single-view experiments run first, followed by their dual-view counterparts.
run_experiment \
    UNET-SEM-V5-FDelta-F-bettersetup-v5 \
    UNET-SEM-V5-F \
    follower_delta \
    front

run_experiment \
    UNET-SEM-V5-FAnchorDelta-F-bettersetup-v5 \
    UNET-SEM-V5-F \
    follower_anchor_delta \
    front

run_experiment \
    UNET-SEM-V5-FDelta-FS-bettersetup-v5 \
    UNET-SEM-V5-FS \
    follower_delta \
    front-side

run_experiment \
    UNET-SEM-V5-FAnchorDelta-FS-bettersetup-v5 \
    UNET-SEM-V5-FS \
    follower_anchor_delta \
    front-side
