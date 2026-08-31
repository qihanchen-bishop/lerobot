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

for required_path in "$PYTHON_BIN" "$FRONT_MODEL" "$SIDE_MODEL"; do
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
LOG_FILE="$OUTPUT_ROOT/queue_logs/bettersetup_v5_sem_front_stage_${RUN_ID}.log"
exec 9>"$OUTPUT_ROOT/.bettersetup_v5_sem_front_stage.lock"
if ! flock -n 9; then
    echo "The bettersetup_v5 semantic/front-stage queue is already running." >&2
    exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1

FRONT_MASK_KEYS=(
    observation.images.front_occluder observation.images.front_object
    observation.images.front_region observation.images.front_tool
    observation.images.front_leftarm observation.images.front_rightarm
)
SIDE_MASK_KEYS=(
    observation.images.side_occluder observation.images.side_object
    observation.images.side_region observation.images.side_tool
)
COMMON_ARGS=(
    --root "$DATASET_ROOT" --repo-id bettersetup_v5
    --state-keys observation.state --steps "$STEPS" --seed 1000
    --batch-size "$BATCH_SIZE" --chunk-size 60 --n-action-steps 60
    --action-loss-weight 1.0
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
    local output_dir=$2
    shift 2
    local final_checkpoint="$output_dir/checkpoint_step_$(printf '%06d' "$STEPS")/training_state.pt"
    local -a command=(
        "$PYTHON_BIN" mycode/train_mask_act_policy.py
        "$@" --output-dir "$output_dir" "${COMMON_ARGS[@]}"
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

run_experiment UNET-SEM-V5-FS "$OUTPUT_ROOT/UNET-SEM-v5-front-side-bettersetup-v5" \
    --experiment UNET-SEM-V5-FS \
    --rgb-keys observation.images.front observation.images.side \
    --mask-target-keys "${FRONT_MASK_KEYS[@]}" "${SIDE_MASK_KEYS[@]}" \
    --pretrained-segmentation-checkpoints "$FRONT_MODEL" "$SIDE_MODEL"

run_experiment STAGE-SIMPLE-V5-F-RGB "$OUTPUT_ROOT/STAGE-SIMPLE-v5-front-RGB-bettersetup-v5" \
    --experiment STAGE-SIMPLE-V5-F-RGB \
    --rgb-keys observation.images.front \
    --mask-target-keys "${FRONT_MASK_KEYS[@]}" \
    --pretrained-segmentation-checkpoints "$FRONT_MODEL"

run_experiment STAGE-SIMPLE-V5-F-UNETSEM "$OUTPUT_ROOT/STAGE-SIMPLE-v5-front-UNETSEM-bettersetup-v5" \
    --experiment STAGE-SIMPLE-V5-F-UNETSEM \
    --rgb-keys observation.images.front \
    --mask-target-keys "${FRONT_MASK_KEYS[@]}" \
    --pretrained-segmentation-checkpoints "$FRONT_MODEL"

echo "[$(date --iso-8601=seconds)] Queue completed."
