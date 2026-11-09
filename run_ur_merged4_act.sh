#!/usr/bin/env bash
# ACT RGB, then ACT RGB + depth pseudocolor; preserve the complete recorded process.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-/home/qihan/miniconda3/envs/lerobot/bin/python}"
ROOT="$PWD/data/realsense_rgbd_merged4"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p outputs/logs
  exec 9>outputs/logs/ur-merged4-act.queue.lock
  flock -n 9 || { printf 'ACT queue already running.\n' >&2; exit 1; }
  for mode in RGB RGBD; do
    out="outputs/train/UR-FDeltaSeparateGrip-ACT-${mode}-realsense-merged4-fullprocess"
    [[ ! -e "$out" ]] || { printf 'Refusing to overwrite %s\n' "$out" >&2; exit 1; }
  done
fi
for mode in RGB RGBD; do
  NAME="UR-FDeltaSeparateGrip-ACT-${mode}-realsense-merged4-fullprocess"
  images=(observation.images.front)
  if [[ "$mode" == RGBD ]]; then images+=(observation.images.depth_preview); fi
  COMMAND=("$PYTHON" -u mycode/train_bi_ur3_policy.py
    --root "$ROOT" --repo-id local/realsense_rgbd_merged4
    --image-keys "${images[@]}" --state-keys observation.state
    --act-action-target follower_joint_delta_gripper_absolute --act-follower-state-key observation.state
    --act-gripper-loss-weight 0.2 --act-gripper-positive-weight 2.5
    --output-dir "outputs/train/$NAME" --job-name "$NAME"
    --chunk-size 60 --n-action-steps 60
    --pretrained-backbone-weights ResNet18_Weights.IMAGENET1K_V1
    --batch-size 8 --steps 100000 --seed 1000 --num-workers 16
    --device cuda --video-backend pyav --log-freq 200 --save-freq 20000 --eval-freq 0)
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "${COMMAND[@]}" 2>&1 | tee -a "outputs/logs/$NAME.log"
    printf '%s completed at %s\n' "$NAME" "$(date -Is)"
  fi
done
