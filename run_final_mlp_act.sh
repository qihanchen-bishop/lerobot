#!/usr/bin/env bash
# task1: the two MLP residual ablations, then a dataset-normalized RGB-only ACT.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
NAME=ACT-finaldata-seg-FS-DatasetNorm
OUT="outputs/train/act/$NAME"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p outputs/logs
  exec 9>outputs/logs/final-mlp-act.queue.lock
  flock -n 9 || { printf 'MLP/ACT queue is already running.\n' >&2; exit 1; }
  [[ ! -e "$OUT" ]] || { printf 'ACT output already exists: %s\n' "$OUT" >&2; exit 1; }
fi

bash run_final_viewqtoken_residuals.sh mlp-rgb
bash run_final_viewqtoken_residuals.sh mlp-semquery

COMMAND=("$PYTHON" -u mycode/train_lerobot_policy.py
  --policy-type act --root "$PWD/data/finaldata_seg" --repo-id finaldata_seg
  --image-keys observation.images.front observation.images.side --image-normalization dataset
  --state-keys observation.state --output-dir "$OUT" --job-name "$NAME"
  --seed 1000 --chunk-size 60 --n-action-steps 60
  --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
  --batch-size 8 --steps 100000 --device cuda --num-workers 16 --video-backend pyav
  --log-freq 200 --save-freq 20000 --rebuild-view)
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "${COMMAND[@]}" 2>&1 | tee -a "outputs/logs/$NAME.log"
  printf 'MLP/ACT queue completed successfully.\n'
fi
