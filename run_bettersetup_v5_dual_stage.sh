#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/qihan/data/lerobot/data/bettersetup_v5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/train}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

FRONT_MODEL="$DATASET_ROOT/models/unet_front_v4_r1/best.pt"
SIDE_MODEL="$DATASET_ROOT/models/unet_side/best.pt"
STAGE_SUPERVISION="$DATASET_ROOT/stage_supervision_v5_front.npz"

for required_path in "$PYTHON_BIN" "$FRONT_MODEL" "$SIDE_MODEL" "$STAGE_SUPERVISION"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path not found: $required_path" >&2
        exit 1
    fi
done
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer, got: $STEPS" >&2
    exit 1
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/queue_logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_ROOT/queue_logs/bettersetup_v5_dual_stage_${RUN_ID}.log"
exec 9>"$OUTPUT_ROOT/.bettersetup_v5_dual_stage.lock"
if ! flock -n 9; then
    echo "The bettersetup_v5 dual-stage queue is already running." >&2
    exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1

MASK_KEYS=(
    observation.images.front_occluder observation.images.front_object
    observation.images.front_region observation.images.front_tool
    observation.images.front_leftarm observation.images.front_rightarm
    observation.images.side_occluder observation.images.side_object
    observation.images.side_region observation.images.side_tool
)
COMMON_ARGS=(
    --root "$DATASET_ROOT" --repo-id bettersetup_v5
    --rgb-keys observation.images.front observation.images.side
    --state-keys observation.state --mask-target-keys "${MASK_KEYS[@]}"
    --pretrained-segmentation-checkpoints "$FRONT_MODEL" "$SIDE_MODEL"
    --stage-supervision "$STAGE_SUPERVISION"
    --steps "$STEPS" --seed 1000 --batch-size "$BATCH_SIZE"
    --chunk-size 60 --n-action-steps 60 --action-loss-weight 1.0
    --phase-history-length 16 --phase-history-stride 4 --phase-hidden-dim 128
    --phase-teacher-forcing-steps 10000 --phase-teacher-forcing-ramp-steps 20000
    --stage-conditioning-mode none
    --stage-predicted-input-warmup-steps 0 --stage-predicted-input-ramp-steps 10000
    --stage-phase-loss-weight 0.20 --stage-event-loss-weight 0.10
    --stage-progress-loss-weight 0.10 --stage-transition-loss-weight 0.10
    --stage-relation-loss-weight 0.10 --stage-attention-regularization-weight 0.0
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
    --device "$DEVICE" --num-workers "$NUM_WORKERS" --video-backend pyav
    --rebuild-view
)

latest_checkpoint() {
    local output_dir=$1
    local checkpoints=("$output_dir"/checkpoint_step_*/training_state.pt)
    ((${#checkpoints[@]} > 0)) || return 1
    printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
}

run_experiment() {
    local label=$1
    local experiment=$2
    local output_dir=$3
    local final_checkpoint="$output_dir/checkpoint_step_$(printf '%06d' "$STEPS")/training_state.pt"
    local -a command=(
        "$PYTHON_BIN" mycode/train_mask_act_policy.py
        --experiment "$experiment" --output-dir "$output_dir" "${COMMON_ARGS[@]}"
    )

    if [[ -f "$final_checkpoint" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP $label: final checkpoint exists."
        return
    fi
    local resume=""
    if resume="$(latest_checkpoint "$output_dir")"; then
        command+=(--resume-checkpoint "$resume")
        echo "[$(date --iso-8601=seconds)] RESUME $label from $resume"
    elif [[ "$DRY_RUN" == 0 && -d "$output_dir" ]]; then
        local archived_dir="${output_dir}.incomplete.${RUN_ID}"
        echo "[$(date --iso-8601=seconds)] ARCHIVE incomplete $label output to $archived_dir"
        mv "$output_dir" "$archived_dir"
    fi
    echo "[$(date --iso-8601=seconds)] START $label"
    printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
    [[ "$DRY_RUN" == 1 ]] || "${command[@]}"
}

echo "Queue log: $LOG_FILE"
echo "Device: $DEVICE"

run_experiment \
    STAGE-V5-FS-RGB \
    STAGE-V5-FS-RGB \
    "$OUTPUT_ROOT/STAGE-v5-front-side-RGB-bettersetup-v5"

run_experiment \
    STAGE-V5-FS-UNETSEM \
    STAGE-V5-FS-UNETSEM \
    "$OUTPUT_ROOT/STAGE-v5-front-side-UNETSEM-bettersetup-v5"

echo "[$(date --iso-8601=seconds)] Queue completed."
