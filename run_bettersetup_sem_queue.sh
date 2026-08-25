#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/qihan/lerobot/data/bettersetup}"
QUALITY_DIR="${QUALITY_DIR:-${DATASET_ROOT}/segmentation_quality}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/train}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
PREFLIGHT="${PREFLIGHT:-quick}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset not found: $DATASET_ROOT" >&2
    exit 1
fi
if [[ ! -d "$QUALITY_DIR" ]]; then
    echo "Mask-quality directory not found: $QUALITY_DIR" >&2
    exit 1
fi
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer, got: $STEPS" >&2
    exit 1
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/queue_logs"
QUEUE_STARTED_AT="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="$OUTPUT_ROOT/queue_logs/bettersetup_sem_${QUEUE_STARTED_AT}.log"

exec 9>"$OUTPUT_ROOT/.bettersetup_sem_queue.lock"
if ! flock -n 9; then
    echo "Another bettersetup SEM queue is already running." >&2
    exit 1
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

queue_exit() {
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "[$(date --iso-8601=seconds)] Dry run completed; no training was started."
        else
            echo "[$(date --iso-8601=seconds)] All requested experiments completed."
        fi
    else
        echo "[$(date --iso-8601=seconds)] Queue stopped with exit code $rc."
    fi
}
trap queue_exit EXIT

MASK_KEYS=(
    observation.images.front_occluder
    observation.images.front_object
    observation.images.front_region
    observation.images.front_tool
    observation.images.side_occluder
    observation.images.side_object
    observation.images.side_region
    observation.images.side_tool
)

final_checkpoint() {
    local output_dir=$1
    printf '%s/checkpoint_step_%06d/training_state.pt' "$output_dir" "$STEPS"
}

latest_checkpoint() {
    local output_dir=$1
    local checkpoints=("$output_dir"/checkpoint_step_*/training_state.pt)
    if ((${#checkpoints[@]} == 0)); then
        return 1
    fi
    printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
}

experiment_is_running() {
    local output_dir=$1
    local absolute_output="$PROJECT_ROOT/$output_dir"
    pgrep -f "train_mask_act_policy.py.*--output-dir[= ]+(${output_dir}|${absolute_output})([[:space:]]|$)" \
        >/dev/null
}

wait_for_existing_run() {
    local label=$1
    local output_dir=$2
    if ! experiment_is_running "$output_dir"; then
        return
    fi

    echo "[$(date --iso-8601=seconds)] $label is already running; waiting for it to finish."
    while experiment_is_running "$output_dir"; do
        sleep 60
    done
    echo "[$(date --iso-8601=seconds)] Existing $label process exited."
}

run_preflight() {
    local checker=/home/qihan/.codex/skills/check-lerobot-semantic-training/scripts/check_semantic_training.py
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "Skipping dataset preflight during dry run."
        return
    fi
    case "$PREFLIGHT" in
        skip)
            echo "Skipping dataset preflight (PREFLIGHT=skip)."
            ;;
        quick)
            echo "Running quick semantic-dataset preflight."
            "$PYTHON_BIN" "$checker" \
                --root "$DATASET_ROOT" \
                --repo-id bettersetup \
                --experiment sem-1 \
                --workers "$NUM_WORKERS" \
                --quick
            ;;
        full)
            echo "Running full semantic-dataset preflight."
            "$PYTHON_BIN" "$checker" \
                --root "$DATASET_ROOT" \
                --repo-id bettersetup \
                --experiment sem-1 \
                --workers "$NUM_WORKERS"
            ;;
        *)
            echo "PREFLIGHT must be quick, full, or skip; got: $PREFLIGHT" >&2
            exit 1
            ;;
    esac
}

run_experiment() {
    local label=$1
    local experiment=$2
    local output_dir=$3
    local use_quality=$4

    local final_path
    final_path="$(final_checkpoint "$output_dir")"
    if [[ "$DRY_RUN" == 0 ]]; then
        wait_for_existing_run "$label" "$output_dir"
        if [[ -f "$final_path" ]]; then
            echo "[$(date --iso-8601=seconds)] SKIP $label: final checkpoint already exists."
            return
        fi
    fi

    local -a command=(
        "$PYTHON_BIN" mycode/train_mask_act_policy.py
        --experiment "$experiment"
        --root "$DATASET_ROOT"
        --repo-id bettersetup
        --rgb-keys observation.images.front observation.images.side
        --state-keys observation.state
        --mask-target-keys "${MASK_KEYS[@]}"
        --output-dir "$output_dir"
        --steps "$STEPS"
        --seed 1000
        --batch-size "$BATCH_SIZE"
        --chunk-size 60
        --n-action-steps 60
        --seg-loss-weight 1.0
        --action-loss-weight 1.0
        --dice-loss-weight 1.0
        --semantic-temperature 1.0
        --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
        --device "$DEVICE"
        --num-workers "$NUM_WORKERS"
        --video-backend pyav
        --rebuild-view
    )

    if [[ "$use_quality" == yes ]]; then
        command+=(
            --mask-quality-dir "$QUALITY_DIR"
            --mask-quality-weighting soft
            --mask-quality-full-score 0.95
            --mask-quality-weight-gamma 1.0
        )
    fi

    if [[ "$DRY_RUN" == 0 ]]; then
        local resume_path=""
        if resume_path="$(latest_checkpoint "$output_dir")"; then
            echo "[$(date --iso-8601=seconds)] RESUME $label from $resume_path"
            command+=(--resume-checkpoint "$resume_path")
        elif [[ -d "$output_dir" ]]; then
            local archived_dir="${output_dir}.incomplete.${QUEUE_STARTED_AT}"
            echo "[$(date --iso-8601=seconds)] No checkpoint found; archiving partial output to $archived_dir"
            mv "$output_dir" "$archived_dir"
        fi
    fi

    echo "[$(date --iso-8601=seconds)] START $label"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        return
    fi
    "${command[@]}"

    if [[ ! -f "$final_path" ]]; then
        echo "Training returned successfully but final checkpoint is missing: $final_path" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] DONE $label"
}

echo "Queue log: $QUEUE_LOG"
echo "Project: $PROJECT_ROOT"
echo "Dataset: $DATASET_ROOT"
echo "Device: $DEVICE; steps: $STEPS; batch size: $BATCH_SIZE; workers: $NUM_WORKERS"

CURRENT_SEM_OUTPUT="$OUTPUT_ROOT/SEM-1-bettersetup-front-side-no-quality"
if [[ "$DRY_RUN" == 0 ]] && experiment_is_running "$CURRENT_SEM_OUTPUT"; then
    echo "SEM-1-NoQ is still running. Start this queue after it finishes." >&2
    exit 1
fi

run_preflight

run_experiment \
    SEM-2-NoQ \
    sem-2 \
    "$OUTPUT_ROOT/SEM-2-bettersetup-front-side-no-quality" \
    no

run_experiment \
    SEM-2-Q \
    sem-2 \
    "$OUTPUT_ROOT/SEM-2-bettersetup-front-side-quality" \
    yes
