#!/usr/bin/env bash
# Run in a new tmux. Optional: audit or verify (shadow + audit), without training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${TMUX:?Start this script in a dedicated tmux session}"
: "${PYTHON:=python}"
STAGE="${1:?Usage: bash run_slow_fast_residual_stage.sh A|B|C}"
case "$STAGE" in
  A) VARIANT=act; BASE=outputs/train/act/ACT-bettersetup-front-side/checkpoints/last/pretrained_model ;;
  B) VARIANT=unet; BASE=outputs/train/semantic/UNET-SEM-v5-front-side-bettersetup-v5/checkpoint_step_100000 ;;
  C) VARIANT=qtoken; BASE=outputs/train/semantic/UNET-SEM-v5-front-side-QToken-bettersetup-v5/checkpoint_step_100000 ;;
  *) exit 2 ;;
esac
OUT="outputs/train/residual/SF-v1-${STAGE}-FS"
CACHE="outputs/cache/slow_fast_v1_${STAGE}"
COMMON=(--variant "$VARIANT" --root data/bettersetup_v5 --checkpoint "$BASE" --cache "$CACHE")
if [[ "${PROFILE:-v1}" == v2-s6-d3-h9 ]]; then
  OUT="outputs/train/residual/SF-v2-S6-D3-H9-${STAGE}-FS"
  CACHE="outputs/cache/slow_fast_v2_s6_d3_h9_${STAGE}"
  COMMON=(--variant "$VARIANT" --root data/bettersetup_v5 --checkpoint "$BASE" --cache "$CACHE"
    --fast-stride 6 --fast-delay 3 --correction-steps 9 --history 8
    --history-gap 0.3 --residual-max-age 0.4)
elif [[ "${PROFILE:-v1}" != v1 ]]; then
  echo 'Unknown PROFILE'; exit 2
fi
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This stage already has a runner'; exit 1; }
if [[ "${2:-}" == verify ]]; then
  exec > >(tee -a "$OUT/verify.log") 2>&1
  "$PYTHON" -m pytest -q tests/test_slow_fast_residual.py
  "$PYTHON" -m mycode.train_slow_fast_residual shadow "${COMMON[@]}" --output "$OUT/gru8"
  exec "$PYTHON" -m mycode.train_slow_fast_residual audit "${COMMON[@]}" --output "$OUT"
fi
if [[ "${2:-}" == audit ]]; then
  exec > >(tee -a "$OUT/audit.log") 2>&1
  exec "$PYTHON" -m mycode.train_slow_fast_residual audit "${COMMON[@]}" --output "$OUT"
fi
if [[ -e "$OUT/exit_code" && "${RESUME:-0}" != 1 ]]; then
  echo 'Stage already attempted; inspect results or explicitly RESUME=1'; exit 1
fi
exec > >(tee -a "$OUT/runner.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code"' EXIT
printf 'running\n' > "$OUT/exit_code"
printf 'pid=%s\ntmux=%s\nstart=%s\n' "$$" "$TMUX_PANE" "$(date -Is)" > "$OUT/process.txt"
nvidia-smi
"$PYTHON" -m pytest -q tests/test_slow_fast_residual.py
"$PYTHON" -m mycode.train_slow_fast_residual cache "${COMMON[@]}" --output "$OUT"
RESTART=()
if [[ "${RESUME:-0}" == 1 ]]; then RESTART=(--resume); fi
"$PYTHON" -m mycode.train_slow_fast_residual train "${COMMON[@]}" --output "$OUT/smoke" --smoke --epochs 30 --patience 30 --lr 0.001 "${RESTART[@]}"
# Explicit checkpoint/optimizer resume exercise before full training.
"$PYTHON" -m mycode.train_slow_fast_residual train "${COMMON[@]}" --output "$OUT/smoke" --smoke --epochs 32 --patience 32 --lr 0.001 --resume
"$PYTHON" -m mycode.train_slow_fast_residual train "${COMMON[@]}" --output "$OUT/gru8" --epochs 50 "${RESTART[@]}"
if [[ "$STAGE" == A ]]; then
  "$PYTHON" -m mycode.train_slow_fast_residual train "${COMMON[@]}" --output "$OUT/mlp" --mode mlp --epochs 50 "${RESTART[@]}"
fi
"$PYTHON" -m mycode.train_slow_fast_residual shadow "${COMMON[@]}" --output "$OUT/gru8"
