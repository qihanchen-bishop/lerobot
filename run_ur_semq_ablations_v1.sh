#!/usr/bin/env bash
# One isolated UR ablation per invocation; no changes to SO-ARM experiments.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
VARIANT="${1:?Usage: bash run_ur_semq_ablations_v1.sh semantic-no-query|rgb-query [rgb|rgbd]}"
MODE="${2:-rgb}"
case "$MODE" in rgb|rgbd) ;; *) printf 'Unsupported mode: %s\n' "$MODE" >&2; exit 2 ;; esac
case "$VARIANT" in
  semantic-no-query) NAME="UR-Sem-v1-${MODE^^}-NoQ-realsense-merged4-fullprocess"; WEIGHT=0 ;;
  rgb-query) NAME="UR-Q-v1-${MODE^^}-NoSem-realsense-merged4-fullprocess"; WEIGHT=0.1 ;;
  *) printf 'Unsupported variant: %s\n' "$VARIANT" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
CACHE="outputs/cache/ur_semq_v1_merged4"
mkdir -p outputs/logs
exec 9>"outputs/logs/$NAME.lock"
flock -n 9 || { printf 'Experiment already running.\n' >&2; exit 1; }
[[ ! -e "outputs/train/$NAME" ]] || { printf 'Refusing to overwrite outputs/train/%s\n' "$NAME" >&2; exit 1; }
# Reuse only the previously verified cache; never rebuild shared data in an ablation.
[[ -f "$CACHE/manifest.json" ]] || { printf 'Prepare the UR-SemQ cache first.\n' >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" -u mycode/train_ur_semq_policy.py \
  --root data/realsense_rgbd_merged4_seg --cache "$CACHE" --mode "$MODE" --variant "$VARIANT" \
  --output-dir "outputs/train/$NAME" --steps 100000 --batch-size 8 --num-workers 16 \
  --seed 1000 --query-weight "$WEIGHT" --log-freq 200 --save-freq 20000 --device cuda \
  2>&1 | tee -a "outputs/logs/$NAME.log"
