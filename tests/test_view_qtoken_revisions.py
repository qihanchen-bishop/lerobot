import json
import os
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from prepare_view_qtoken_revisions import derive_targets
from view_qtoken import (
    CACHE_VERSION, EXPERIMENT, VIEW_V2, VIEW_V3, ViewQueryStore, dataset_signature,
    front_tool_vectors, query_loss, revision_heads, sha256, tool_segment,
)
from train_mask_act_policy import act_image_keys_for_experiment, act_metric_token_output_dims


@pytest.mark.parametrize("experiment,dims", [(VIEW_V2, [2, 2, 2]), (VIEW_V3, [2, 4, 2])])
def test_heads_inputs_and_equal_task_weights(experiment, dims):
    tokens = torch.randn(2, 3, 16, requires_grad=True)
    heads = revision_heads(16, experiment)
    outputs = [head(tokens[:, i]) for i, head in enumerate(heads)]
    assert [p.shape[-1] for p in outputs] == dims
    assert act_metric_token_output_dims(experiment) == dims
    assert act_image_keys_for_experiment(Namespace(experiment=experiment, rgb_keys=["front", "side"])) == [
        "front_semantic", "side_semantic", "front", "side"]
    torch.cat(outputs, -1).sum().backward()
    assert all(p.grad is not None for head in heads for p in head.parameters())
    assert not hasattr(heads[1], "side")
    pred = torch.ones(2, sum(dims), requires_grad=True)
    valid = torch.ones_like(pred, dtype=torch.bool)
    weights = valid.float()
    zero = torch.zeros_like(pred)
    full, logs = query_loss(pred, zero, valid, weights, experiment=experiment)
    half, _ = query_loss(pred, zero, valid, weights * .5, experiment=experiment)
    assert half.item() == pytest.approx(full.item() / 2)
    full.backward()
    assert pred.grad[:, :2].sum().item() == pytest.approx(pred.grad[:, 2:2 + dims[1]].sum().item())
    assert pred.grad[:, :2].sum().item() == pytest.approx(pred.grad[:, -2:].sum().item())
    assert "query_tool_object_side_loss" not in logs
    empty, _ = query_loss(pred, zero, valid & False, weights * 0, experiment=experiment)
    assert empty.item() == 0 and torch.isfinite(empty)


def test_midpoint_is_not_mask_centroid_and_uses_same_pca_segment():
    obj, tool = np.zeros((20, 30), bool), np.zeros((20, 30), bool)
    obj[3, 24] = True
    tool[10, 2:22] = True
    tool[9:12, 2:5] = True  # Asymmetric mass shifts centroid, not segment midpoint.
    target, valid = front_tool_vectors(obj, tool)
    assert valid
    yy, xx = np.nonzero(tool)
    points = np.column_stack((xx / 29, yy / 19))
    endpoints = tool_segment(points)
    assert target[:2] == pytest.approx(np.array([24 / 29, 3 / 19]) - endpoints[0])
    assert target[2:] == pytest.approx(np.array([24 / 29, 3 / 19]) - endpoints.mean(0))
    assert not np.allclose(endpoints.mean(0), points.mean(0))
    assert not front_tool_vectors(obj & False, tool)[1]
    assert not front_tool_vectors(obj, tool & False)[1]
    assert tool_segment(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])) is None


def test_revision_cache_derivation_preserves_baseline_and_quality():
    original = {"targets": np.arange(24, dtype=np.float32).reshape(3, 8),
                "valid": np.ones((3, 8), bool), "weights": np.ones((3, 8), np.float32) * .3}
    original["valid"][1, 2:4] = False
    original["weights"][1, 2:4] = 0
    original["weights"][2, 2:4] = 0  # Geometry valid, but anomaly gating applies to both tool heads.
    original["targets"][1, 2:4] = 0
    vectors = np.zeros((3, 4), np.float32)
    vectors[:, :2] = original["targets"][:, 2:4]
    vectors[[0, 2], 2:] = .17
    v2 = derive_targets(original, VIEW_V2)
    v3 = derive_targets(original, VIEW_V3, vectors, original["valid"][:, 2])
    for key in original:
        np.testing.assert_array_equal(v2[key], original[key][:, [0, 1, 2, 3, 6, 7]])
        np.testing.assert_array_equal(v3[key][:, [0, 1, 2, 3, 6, 7]], v2[key])
    np.testing.assert_array_equal(v3["weights"][:, 4:6], original["weights"][:, 2:4])
    np.testing.assert_array_equal(v3["targets"][:, 4:6], vectors[:, 2:])
    with pytest.raises(AssertionError):
        derive_targets(original, VIEW_V3, vectors + 1, original["valid"][:, 2])


@pytest.mark.parametrize("experiment,width", [(VIEW_V2, 6), (VIEW_V3, 8)])
def test_revision_cache_cannot_be_confused_with_original(tmp_path, experiment, width):
    root, cache = tmp_path / "source", tmp_path / "cache"
    (root / "meta").mkdir(parents=True)
    cache.mkdir()
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 2}))
    (root / "segmentation_manifest.json").write_text("{}")
    path = cache / "targets.npz"
    np.savez(path, targets=np.zeros((2, width), np.float32), valid=np.ones((2, width), bool),
             weights=np.ones((2, width), np.float32))
    metadata = {"version": CACHE_VERSION, "experiment": experiment,
                "source_signature": dataset_signature(root), "targets_sha256": sha256(path)}
    (cache / "manifest.json").write_text(json.dumps(metadata))
    ViewQueryStore(path, root, experiment=experiment)
    for wrong in {EXPERIMENT, VIEW_V2, VIEW_V3} - {experiment}:
        with pytest.raises(ValueError, match="different target"):
            ViewQueryStore(path, root, experiment=wrong)
    metadata["experiment"] = EXPERIMENT
    (cache / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different target"):
        ViewQueryStore(path, root, experiment=experiment)


@pytest.mark.skipif(not os.environ.get("VIEW_QTOKEN_REVISION_SMOKE"), reason="Opt-in real checkpoint and RGB inference")
@pytest.mark.parametrize("version", [2, 3])
def test_real_checkpoint_reload_and_cached_training_equivalence(version):
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mask_act_inference import load_mask_act_for_inference
    from view_qtoken import SemanticLabelStore

    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    root = project / "data/finaldata_seg"
    run = project / f"outputs/temp/final-ViewQtoken-v{version}-smoke"
    policy, pre, post, _ = load_mask_act_for_inference(run, project)
    cfg = json.loads((run / "mask_act_run_config.json").read_text())
    assert cfg["semantic_input_normalization"]["mode"] == "imagenet"
    torch.testing.assert_close(policy.semantic_visual_mean.flatten().cpu(), torch.tensor([.485, .456, .406]))
    torch.testing.assert_close(policy.semantic_visual_std.flatten().cpu(), torch.tensor([.229, .224, .225]))
    assert all(not p.requires_grad for p in policy.pretrained_segmenters.parameters())
    queries = ViewQueryStore(project / f"outputs/cache/final-ViewQtoken-v{version}/targets.npz", root,
                             experiment=VIEW_V2 if version == 2 else VIEW_V3)
    labels = SemanticLabelStore(project / "outputs/cache/final-FrontQtoken-FS-v1", root)
    dataset = LeRobotDataset("finaldata_seg", root=root, video_backend="pyav",
                             delta_timestamps={"action": [i / 30 for i in range(60)]})
    raw = queries.add_batch(default_collate([dataset[500], dataset[31000]]))
    fast = labels.add_batch(raw)
    old_ids = policy.build_semantic_targets(raw, device=torch.device("cpu"))
    new_ids = policy.build_semantic_targets(fast, device=torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(old_ids, new_ids))
    fast = {k: v for k, v in fast.items() if k not in policy.mask_keys}
    policy.train()
    results = []
    for batch in (raw, fast):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(1000)
        loss, logs = policy(pre(batch), batch)
        loss.backward()
        grads = {n: p.grad.detach().cpu().clone() for n, p in policy.named_parameters()
                 if "metric_heads" in n and p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        assert np.isfinite(logs["action_loss"]) and np.isfinite(logs["metric_loss"])
        results.append((loss.item(), grads))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-7)
    for name, grad in results[0][1].items():
        torch.testing.assert_close(grad, results[1][1][name], atol=1e-6, rtol=1e-6)
    policy.eval()
    observation = {key: raw[key] for key in (*policy.rgb_keys, "observation.state")}
    with torch.no_grad():
        action = post(policy.predict_action_chunk(pre(observation)))
    assert action.shape == (2, 60, 10) and torch.isfinite(action).all()
