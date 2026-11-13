"""Real-data preflight for both UR-SemQ modes; never operates a robot."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate
from train_ur_semq_policy import VARIANTS, build_data, prepare_batch, train, variant_flags
from ur_semq import FRONT_KEY, SEMANTIC_KEY, query_loss
from ur_semq_inference import URSemQInference

from lerobot.policies.act.modeling_act import METRIC_PRED, ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def check(root, cache, output, device, modes=("rgb", "rgbd"), variants=("full",)):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    reports = []
    for mode, variant in ((mode, variant) for mode in modes for variant in variants):
        args = SimpleNamespace(
            root=root,
            cache=cache,
            output_dir=output / (mode if variant == "full" else f"{mode}-{variant}"),
            mode=mode,
            variant=variant,
            device=device,
            seed=1000,
            steps=3,
            batch_size=8,
            num_workers=2,
            log_freq=1,
            save_freq=3,
            query_weight=0.1,
        )
        dataset, config, stats, view = build_data(args)
        semantic_input, query_supervision = variant_flags(args)
        assert (SEMANTIC_KEY in config.input_features) == semantic_input
        assert (config.metric_mode == "encoder_tokens") == query_supervision
        depth_key = "observation.images.depth_preview"
        expected_images = [FRONT_KEY] + ([depth_key] if mode == "rgbd" else [])
        assert list(config.image_features) == expected_images + ([SEMANTIC_KEY] if semantic_input else [])
        if not semantic_input:
            # RGB/RGBD without semantic inputs must never read mask images.
            def forbidden_labels(index):
                raise AssertionError("No-semantic-input variant read a semantic image.")
            dataset.cache.labels = forbidden_labels
        # Match the previous ACT experiment's action/state normalization exactly.
        baseline = Path(
            f"outputs/train/dataset_views/UR-FDeltaSeparateGrip-ACT-{mode.upper()}-realsense-merged4-fullprocess/meta/stats.json"
        )
        original = json.loads(baseline.read_text())
        for key in ("observation.state", "action"):
            for field in ("mean", "std", "min", "max"):
                np.testing.assert_allclose(stats[key][field], original[key][field], rtol=1e-6, atol=1e-8)
        ids = dataset.cache.arrays
        assert not ids["valid"][ids["frame_index"] < 20, 2].any()
        absent = ids["targets"][:, 0] == 0
        assert ids["valid"][absent, 0].all() and not ids["valid"][absent, 2:].any(axis=0)[[0, 2, 3]].any()
        checked = 0
        for episode in range(0, dataset.dataset.num_episodes, 5):
            row = dataset.dataset.meta.episodes[episode]
            for index in (int(row["dataset_from_index"]), int(row["dataset_to_index"]) - 1):
                sample = dataset[index]
                assert ("semantic_ids" in sample) == semantic_input
                assert ("query_targets" in sample) == query_supervision
                for key in expected_images:
                    assert list(sample[key].shape) == list(config.input_features[key].shape)
                    assert torch.isfinite(sample[key]).all()
                    assert sample[key].min() >= 0 and sample[key].max() <= 1
                if semantic_input:
                    assert sample["semantic_ids"].shape == tuple(config.input_features[FRONT_KEY].shape[1:])
                    assert sample["semantic_ids"].max() < 8
                assert torch.isfinite(sample["action"]).all()
                checked += 1
        policy = ACTPolicy(config, dataset_stats=stats).to(device)
        pre, post = make_act_pre_post_processors(config, stats)
        palette = (
            torch.tensor(dataset.cache.manifest["palette_rgb"], device=device, dtype=torch.float32) / 255
        )
        scales = torch.tensor(dataset.cache.manifest["scales"], device=device)
        raw = default_collate([dataset[len(dataset) // 2 + 100]])
        batch, target, valid, weights = prepare_batch(raw, palette, pre, device)
        action_loss, values = policy(batch)
        if query_supervision:
            query, _ = query_loss(values[METRIC_PRED], target, valid, weights, scales)
        else:
            assert METRIC_PRED not in values
            assert not hasattr(policy.model, "encoder_metric_token_embed")
            assert not hasattr(policy.model, "metric_heads")
            query = action_loss.new_zeros(())
        (action_loss + 0.1 * query).backward()
        if query_supervision:
            assert policy.model.encoder_metric_token_embed.weight.grad.abs().sum() > 0
            for head, columns in zip(policy.model.metric_heads, ((0,), (1,), (2,), (3,), (4, 5)), strict=True):
                if weights[:, list(columns)].sum() > 0:
                    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        policy.eval()
        with torch.no_grad():
            commands = post(policy.predict_action_chunk({k: batch[k] for k in config.input_features}))
        assert commands.shape == (1, 60, 13) and torch.isfinite(commands).all()
        assert ((commands[..., -1].abs() < 1e-5) | ((commands[..., -1] - 1).abs() < 1e-5)).all()
        del policy, commands, action_loss, query, values, batch, pre, post
        gc.collect()
        torch.cuda.empty_cache()
        train(args)
        checkpoint = args.output_dir / "checkpoints/last/pretrained_model"
        adapter = URSemQInference(checkpoint, device=device)
        assert (adapter.segmenter is not None) == semantic_input
        assert (checkpoint / "semantic_model").exists() == semantic_input
        observations = {
            k: v for k, v in raw.items() if k in adapter.policy.config.input_features and k != SEMANTIC_KEY
        }
        commands, descriptors = adapter.predict_action_chunk(observations)
        assert commands.shape == (1, 60, 13) and torch.isfinite(commands).all()
        if query_supervision:
            assert descriptors.shape == (1, 6) and torch.isfinite(descriptors).all()
        else:
            assert descriptors is None
        # Reload the checkpoint's own preprocessing/weights, then compare the cached-mask path.
        first = postprocessed_cached(adapter, raw, palette)
        adapter2 = URSemQInference(checkpoint, device=device)
        second = postprocessed_cached(adapter2, raw, palette)
        torch.testing.assert_close(first, second)
        reports.append(
            {
                "mode": mode,
                "variant": variant,
                "semantic_input": semantic_input,
                "image_keys": list(config.image_features),
                "query_supervision": query_supervision,
                "samples_checked": checked,
                "normalization_matches_baseline": True,
                "forward_backward_variant_isolation": "PASS",
                "training_and_checkpoint": "PASS",
                "online_inference": "PASS (U-Net)" if semantic_input else "PASS (no U-Net)",
                "reload_parity": "PASS",
                "checkpoint": str(checkpoint.resolve()),
            }
        )
        (output / "preflight.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(reports[-1]), flush=True)
        del adapter, adapter2, dataset, commands, first, second
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def postprocessed_cached(adapter, raw, palette):
    observations = {k: v for k, v in raw.items() if k in adapter.policy.config.input_features}
    if SEMANTIC_KEY in adapter.policy.config.input_features:
        observations[SEMANTIC_KEY] = palette[raw["semantic_ids"].to(palette.device).long()].permute(0, 3, 1, 2)
    return adapter.post(adapter.policy.predict_action_chunk(adapter.pre(observations)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/realsense_rgbd_merged4_seg"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache/ur_semq_v1_merged4"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", nargs="+", choices=("rgb", "rgbd"), default=("rgb", "rgbd"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=("full",))
    args = parser.parse_args()
    check(args.root, args.cache, args.output_dir, args.device, args.modes, args.variants)
