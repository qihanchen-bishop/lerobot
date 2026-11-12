#!/usr/bin/env bash
# Independent UR experiments; mode is rgb or rgbd. No SO-ARM entry points are changed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
MODE="${1:?Usage: bash run_ur_semq_v1.sh rgb|rgbd}"
case "$MODE" in rgb|rgbd) ;; *) printf 'Unsupported mode: %s\n' "$MODE" >&2; exit 2 ;; esac
PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
NAME="UR-SemQ-v1-${MODE^^}-realsense-merged4-fullprocess"
CACHE="outputs/cache/ur_semq_v1_merged4"
mkdir -p outputs/logs
exec 9>"outputs/logs/$NAME.lock"
flock -n 9 || { printf 'Experiment already running.\n' >&2; exit 1; }
[[ ! -e "outputs/train/$NAME" ]] || { printf 'Refusing to overwrite outputs/train/%s\n' "$NAME" >&2; exit 1; }
(
  flock 8
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" -u mycode/prepare_ur_semq.py \
    --root data/realsense_rgbd_merged4_seg --output-dir "$CACHE" --workers 12
) 8>outputs/logs/ur-sem-q-cache.lock
"$PYTHON" -u mycode/train_ur_semq_policy.py \
  --root data/realsense_rgbd_merged4_seg --cache "$CACHE" --mode "$MODE" \
  --output-dir "outputs/train/$NAME" --steps 100000 --batch-size 8 --num-workers 16 \
  --seed 1000 --query-weight 0.1 --log-freq 200 --save-freq 20000 --device cuda \
  2>&1 | tee -a "outputs/logs/$NAME.log"
