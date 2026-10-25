#!/usr/bin/env bash
# Isolated compatibility environment; existing ACT/DP packages are not changed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python)}"
mkdir -p .cache/viola
if [[ ! -d .cache/viola/upstream ]]; then
    git clone https://github.com/UT-Austin-RPL/VIOLA.git .cache/viola/upstream
    git -C .cache/viola/upstream checkout 1e8b5ae90e73a1c2a33a369323117cd8d7b0ef36
fi
test "$(git -C .cache/viola/upstream rev-parse HEAD)" = 1e8b5ae90e73a1c2a33a369323117cd8d7b0ef36
git -C .cache/viola/upstream -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
if [[ ! -d .cache/viola/robomimic ]]; then
    git clone https://github.com/ARISE-Initiative/robomimic.git .cache/viola/robomimic
    git -C .cache/viola/robomimic checkout 490dd4d4f758c47357d314d47ffa5bb4aefa8d65
fi
test "$(git -C .cache/viola/robomimic rev-parse HEAD)" = 490dd4d4f758c47357d314d47ffa5bb4aefa8d65
if [[ ! -x .cache/viola/venv/bin/python ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages .cache/viola/venv
fi
PY=.cache/viola/venv/bin/python
"$PY" -m pip install numpy==1.26.4 Pillow==9.5.0 opencv-python==4.10.0.84 \
    h5py==3.16.0 hydra-core==1.3.2 easydict==1.13 kornia==0.6.1 timm==0.4.12 \
    fvcore==0.1.5.post20221221 iopath==0.1.9 pycocotools==2.0.11 lvis==0.5.3 \
    ftfy==6.3.1 yacs==0.1.8 ninja==1.13.2 matplotlib==3.10.9 gym==0.26.2 tensorboard==2.21.0
"$PY" -m pip install --no-deps git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
# This R50 proposal network uses Torch/torchvision CUDA operators. Build only
# Detectron2's CPU extension, avoiding a dependency on a matching system nvcc.
CUDA_VISIBLE_DEVICES="" MAX_JOBS=2 PATH="$ROOT/.cache/viola/venv/bin:$PATH" \
    "$PY" -m pip install --no-build-isolation --no-deps .cache/viola/upstream/third_party/detectron2
"$PY" -m pip install --no-build-isolation --no-deps -e .cache/viola/robomimic
WEIGHT=.cache/viola/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth
if [[ ! -f "$WEIGHT" ]]; then
    curl --fail --location --retry 3 https://dl.fbaipublicfiles.com/detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth -o "$WEIGHT.download"
    mv "$WEIGHT.download" "$WEIGHT"
fi
printf '%s  %s\n' 17fd4f671780ce1fa9389b8f627cb3e86bf82e143871c0c076c990312377a75a "$WEIGHT" | sha256sum --check
"$PY" -m unittest tests.test_viola_baseline
