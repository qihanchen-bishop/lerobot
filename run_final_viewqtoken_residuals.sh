#!/usr/bin/env bash
# Frozen final-ViewQtoken base; two input ablations crossed with MLP / GRU.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
ROOT="${ROOT:-$PWD/data/finaldata_seg}"
BASE="${BASE:-$PWD/outputs/train/semantic/final-ViewQtoken/checkpoint_step_100000}"
RUN="${1:-all}"
case "$RUN" in
  all) GROUPS_TO_RUN=(mlp-rgb mlp-semquery gru-rgb gru-semquery) ;;
  mlp-rgb|mlp-semquery|gru-rgb|gru-semquery) GROUPS_TO_RUN=("$RUN") ;;
  *) printf 'Usage: bash %s [all|mlp-rgb|mlp-semquery|gru-rgb|gru-semquery]\n' "$0" >&2; exit 2 ;;
esac

run_command() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@" 2>&1 | tee -a "$LOG"
  fi
}

for GROUP in "${GROUPS_TO_RUN[@]}"; do
  MODE="${GROUP%%-*}"
  INPUT="${GROUP#*-}"
  if [[ "$INPUT" == rgb ]]; then
    SUFFIX=RGBOnly
  else
    SUFFIX=SemQ
  fi
  NAME="final-ViewQtoken-Residual-${MODE^^}-${SUFFIX}-FS"
  CACHE="$PWD/outputs/cache/final-ViewQtoken-residual-${INPUT}-s6-d3-h9-v1"
  OUT="$PWD/outputs/train/residual/$NAME"
  LOG="$PWD/outputs/logs/$NAME.log"
  COMMON=(--variant "viewqtoken_${INPUT}" --root "$ROOT" --checkpoint "$BASE"
    --cache "$CACHE" --output "$OUT" --mode "$MODE" --device cuda
    --slow-stride 30 --slow-delay 3 --fast-stride 6 --fast-delay 3
    --correction-steps 9 --history 8 --history-gap 0.3 --residual-max-age 0.4
    --hidden 128 --epochs 50 --patience 8 --batch-size 128 --encode-batch 16 --lr 0.0001)
  RESUME_ARGS=()
  if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); fi
  (
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      [[ -f "$BASE/training_state.pt" && -f "$ROOT/meta/info.json" ]] || {
        printf 'Missing base checkpoint or dataset.\n' >&2; exit 1;
      }
      mkdir -p "$OUT" "$CACHE" "$(dirname "$LOG")"
      exec 9>"$OUT/runner.lock"
      flock -n 9 || { printf 'Experiment already running: %s\n' "$NAME" >&2; exit 1; }
      if [[ -f "$OUT/config.json" && "${RESUME:-0}" != 1 ]]; then
        printf 'Output exists: %s. Use RESUME=1 to continue.\n' "$OUT" >&2
        exit 1
      fi
    fi
    (
      if [[ "${DRY_RUN:-0}" != 1 ]]; then
        exec 8>"$CACHE/runner.lock"
        flock 8
      fi
      run_command "$PYTHON" -u -m mycode.train_slow_fast_residual cache "${COMMON[@]}"
    )
    run_command "$PYTHON" -u -m mycode.train_slow_fast_residual train "${COMMON[@]}" "${RESUME_ARGS[@]}"
    run_command "$PYTHON" -u -m mycode.train_slow_fast_residual shadow "${COMMON[@]}"
  )
done
