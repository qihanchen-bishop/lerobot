import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.configs.types import NormalizationMode
from mycode import train_lerobot_policy as runner


@pytest.mark.parametrize("script,policy,residual", [
    ("run_final_mlp_act.sh", "act", "mlp"),
    ("run_final_gru_dp.sh", "diffusion", "gru"),
])
def test_queue_order_and_normalization(script, policy, residual, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", script], cwd=root, env={**os.environ, "DRY_RUN": "1"},
                            capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 7
    assert "viewqtoken_rgb" in lines[0]
    assert "viewqtoken_semquery" in lines[3]
    for line in lines[:6]:
        assert f"--mode {residual}" in line
    cli = shlex.split(lines[-1])
    args_list = cli[cli.index("mycode/train_lerobot_policy.py") + 1:]
    monkeypatch.setattr(sys, "argv", ["train", *args_list])
    args = runner.parse_args()
    assert args.image_normalization == "dataset"
    assert args.policy_type == policy
    assert args.steps == 100000 and args.seed == 1000
    assert args.root == root / "data/finaldata_seg"
    args.output_dir = tmp_path / "training"
    # Exercise config assembly using real local metadata, without creating views or training.
    monkeypatch.setattr(runner, "make_filtered_dataset_view", lambda **kwargs: args.root)
    monkeypatch.setattr(runner, "ensure_device_is_usable", lambda device: None)
    monkeypatch.setattr(runner, "plot_training_curves", lambda *args: None)
    captured = []
    monkeypatch.setattr(runner, "train", captured.append)
    runner.run_training(args, tmp_path / "log")
    cfg = captured[0]
    assert not cfg.dataset.use_imagenet_stats
    assert list(cfg.policy.input_features) == ["observation.state", "observation.images.front", "observation.images.side"]
    expected = NormalizationMode.MEAN_STD if policy == "act" else NormalizationMode.MIN_MAX
    assert cfg.policy.normalization_mapping["STATE"] == expected
    assert cfg.policy.normalization_mapping["ACTION"] == expected
    if policy == "act":
        assert cfg.policy.image_camera_ids is None
        assert cfg.policy.chunk_size == 60
    else:
        assert cfg.policy.crop_shape is None and cfg.policy.use_frozen_batch_norm
        assert cfg.policy.horizon == 64 and cfg.policy.n_obs_steps == 1
        assert cfg.policy.num_train_timesteps == 100 and cfg.policy.num_inference_steps == 50


def test_legacy_rgb_normalization_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    assert runner.parse_args().image_normalization == "imagenet"
