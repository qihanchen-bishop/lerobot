#!/usr/bin/env bash
# task2: the two GRU residual ablations, then full-frame single-observation DP.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
NAME=DP-1F-Full-finaldata-seg-FS-DatasetNorm
OUT="outputs/train/diffusion/$NAME"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p outputs/logs
  exec 9>outputs/logs/final-gru-dp.queue.lock
  flock -n 9 || { printf 'GRU/DP queue is already running.\n' >&2; exit 1; }
  [[ ! -e "$OUT" ]] || { printf 'DP output already exists: %s\n' "$OUT" >&2; exit 1; }
fi

bash run_final_viewqtoken_residuals.sh gru-rgb
bash run_final_viewqtoken_residuals.sh gru-semquery

COMMAND=("$PYTHON" -u mycode/train_lerobot_policy.py
  --policy-type diffusion --root "$PWD/data/finaldata_seg" --repo-id finaldata_seg
  --image-keys observation.images.front observation.images.side --image-normalization dataset
  --state-keys observation.state --output-dir "$OUT" --job-name "$NAME"
  --seed 1000 --diffusion-n-obs-steps 1 --diffusion-horizon 64 --diffusion-n-action-steps 30
  --diffusion-drop-n-last-frames 29 --diffusion-mask-padding-loss --diffusion-full-frame
  --diffusion-backbone-norm frozen_batch --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
  --diffusion-lr-backbone 1e-5 --diffusion-num-train-timesteps 100 --diffusion-num-inference-steps 50
  --batch-size 8 --steps 100000 --device cuda --num-workers 16 --video-backend pyav
  --log-freq 200 --save-freq 20000 --rebuild-view)
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "${COMMAND[@]}" 2>&1 | tee -a "outputs/logs/$NAME.log"
  printf 'GRU/DP queue completed successfully.\n'
fi
