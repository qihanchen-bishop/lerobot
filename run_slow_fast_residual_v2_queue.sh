#!/usr/bin/env bash
# Dedicated tmux queue; stops on a failed stage or audit, preserves v1 outputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${TMUX:?Run inside a dedicated tmux session}"
export PROFILE=v2-s6-d3-h9
mkdir -p outputs/train/residual
exec 8>outputs/train/residual/SF-v2-S6-D3-H9.queue.lock
flock -n 8 || exit 1
exec > >(tee -a outputs/train/residual/SF-v2-S6-D3-H9.queue.log) 2>&1
trap 'code=$?; printf "%s\n" "$code" > outputs/train/residual/SF-v2-S6-D3-H9.queue.exit_code' EXIT
for stage in A B C; do
  bash run_slow_fast_residual_stage.sh "$stage"
  bash run_slow_fast_residual_stage.sh "$stage" audit
done
