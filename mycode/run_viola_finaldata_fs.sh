#!/usr/bin/env bash
# Full official-network baseline: proposal caching, then policy training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.cache/viola/venv/bin/python"
CACHE="$ROOT/outputs/cache/VIOLA-finaldata-FS-all/dataset.hdf5"
OUTPUT="$ROOT/outputs/train/baselines/VIOLA-finaldata-FS"
LOG_DIR="$ROOT/outputs/logs/VIOLA-finaldata-FS"
mkdir -p "$LOG_DIR"
exec 9>"$LOG_DIR/run.lock"
flock -n 9 || { printf 'Another VIOLA full run is active.\n' >&2; exit 1; }
if [[ -e "$OUTPUT" ]]; then
    printf 'Refusing to overwrite existing training output: %s\n' "$OUTPUT" >&2
    exit 1
fi
trap 'status=$?; printf "VIOLA pipeline exited with status %s at %s\n" "$status" "$(date -Is)"' EXIT
printf 'VIOLA preparation started at %s; CUDA_VISIBLE_DEVICES=%s\n' "$(date -Is)" "${CUDA_VISIBLE_DEVICES:-default}"
"$PY" -u mycode/viola_baseline.py prepare \
    --root data/finaldata_seg --output "$CACHE" \
    --per-class 0 --valid-per-class 0 2>&1 | tee -a "$LOG_DIR/prepare.log"
printf 'VIOLA policy training started at %s\n' "$(date -Is)"
"$PY" -u mycode/viola_baseline.py train \
    --dataset "$CACHE" --output "$OUTPUT" \
    --epochs 51 --batch-size 16 --workers 4 --seed 1000 \
    --log-every 50 --save-every 5000 2>&1 | tee -a "$LOG_DIR/train.log"
printf 'VIOLA training completed at %s\n' "$(date -Is)"
