import json
import os
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from prepare_front_qtoken import legacy_targets
from train_mask_act_policy import (
    MaskACTPolicy, act_image_keys_for_experiment, act_metric_dim,
    act_metric_mode, act_metric_token_output_dims,
)
from view_qtoken import (
    CACHE_VERSION, CLASSES, FRONT_QUERY_EXPERIMENT, FRONT_SYMMETRIC_EXPERIMENT, VIEWS, SemanticLabelStore,
    ViewQueryStore, dataset_signature, quality_weights, query_loss, sha256,
)


@pytest.mark.parametrize("experiment", [FRONT_QUERY_EXPERIMENT, FRONT_SYMMETRIC_EXPERIMENT])
def test_front_queries_keep_both_views_and_old_geometry(experiment):
    assert act_image_keys_for_experiment(Namespace(experiment=experiment, rgb_keys=["front", "side"])) == [
        "front_semantic", "side_semantic", "front", "side"]
    assert act_metric_dim(experiment) == 3
    assert act_metric_mode(experiment) == "encoder_tokens"
    assert act_metric_token_output_dims(experiment) == [1, 1, 2]
    ids = np.zeros((2, 10, 10), np.uint8)
    ids[0, 5, 7] = 2
    ids[0, 7, 1:6] = 4
    ids[0, 4, 6:10] = 3
    target, valid = legacy_targets(ids)
    assert target[0] == pytest.approx([.01, np.hypot(.5, 1) / np.hypot(9, 9), 6 / 9, -2 / 9])
    assert valid[0].all()
    assert valid[1].tolist() == [True, False, False, False]
    # Legacy region distance is centroid-based, not a containment indicator.
    assert target[0, 1] > 0


@pytest.mark.parametrize("experiment", [FRONT_QUERY_EXPERIMENT, FRONT_SYMMETRIC_EXPERIMENT])
def test_front_quality_grouping_and_loss_gradient(experiment):
    valid = np.ones((3, 4), bool)
    scores = np.array([[.8, .6, .7]] * 3)
    anomalies = np.eye(3, dtype=bool)
    weights = quality_weights(valid[:, [0, 2, 3, 1]], scores, anomalies)[:, [0, 3, 1, 2]]
    assert weights == pytest.approx(np.array([[0, 0, 0, 0], [.8, .7, 0, 0], [.8, 0, .6, .6]]))
    pred = torch.ones(2, 4, requires_grad=True)
    valid_tensor = torch.ones_like(pred, dtype=torch.bool)
    full, logs = query_loss(pred, torch.zeros_like(pred), valid_tensor, valid_tensor.float(),
                            experiment=experiment)
    half, _ = query_loss(pred, torch.zeros_like(pred), valid_tensor, valid_tensor.float() * .5,
                         experiment=experiment)
    assert half.item() == pytest.approx(full.item() / 2)
    assert sum(key.endswith("_loss") for key in logs) == 3
    full.backward()
    assert pred.grad[0, 0] == pred.grad[0, 1]
    assert pred.grad[0, 2] * 2 == pred.grad[0, 0]


def test_cached_class_ids_match_binary_mask_path():
    ids = torch.randint(0, 7, (2, 2, 10, 12), dtype=torch.uint8)
    dummy = Namespace(experiment=FRONT_QUERY_EXPERIMENT, rgb_keys=["front", "side"],
                      view_mask_suffixes=[CLASSES, CLASSES],
                      mask_key_map={}, mask_key_local_map={}, _current_visual_frame=lambda x: x)
    raw = {}
    for view in range(2):
        for index in range(6):
            key = f"{view}_{index}"
            dummy.mask_key_map[key] = (view, index)
            dummy.mask_key_local_map[key] = (view, index)
            raw[key] = (ids[:, view] == index + 1).float()[:, None].repeat(1, 3, 1, 1)
    expected = MaskACTPolicy.build_semantic_targets(dummy, raw, torch.device("cpu"))
    actual = MaskACTPolicy.build_semantic_targets(dummy, {"semantic_class_ids": ids}, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    with pytest.raises(ValueError):
        MaskACTPolicy.build_semantic_targets(dummy, {"semantic_class_ids": ids.float()}, torch.device("cpu"))


def test_cache_definitions_integrity_and_random_order(tmp_path):
    root, cache = tmp_path / "source", tmp_path / "cache"
    (root / "meta").mkdir(parents=True)
    cache.mkdir()
    (root / "meta/info.json").write_text(json.dumps({
        "total_frames": 3,
        "features": {f"observation.images.{view}": {"shape": [2, 3, 3]} for view in VIEWS}}))
    (root / "segmentation_manifest.json").write_text("{}")
    ids = (np.arange(36).reshape(3, 2, 2, 3) % 7).astype(np.uint8)
    np.save(cache / "semantic_labels.npy", ids)
    path = cache / "targets.npz"
    np.savez(path, targets=np.arange(12, dtype=np.float32).reshape(3, 4),
             valid=np.ones((3, 4), bool), weights=np.ones((3, 4), np.float32))
    metadata = {
        "version": CACHE_VERSION, "experiment": FRONT_QUERY_EXPERIMENT,
        "source_signature": dataset_signature(root), "targets_sha256": sha256(path),
        "semantic_labels": {"views": list(VIEWS), "classes": list(CLASSES),
                            "file": "semantic_labels.npy", "sha256": sha256(cache / "semantic_labels.npy")}}
    (cache / "manifest.json").write_text(json.dumps(metadata))
    store = ViewQueryStore(path, root, experiment=FRONT_QUERY_EXPERIMENT)
    symmetric_store = ViewQueryStore(path, root, experiment=FRONT_SYMMETRIC_EXPERIMENT)
    for key in store.data:
        np.testing.assert_array_equal(store.data[key], symmetric_store.data[key])
    labels = SemanticLabelStore(cache, root)
    batch = labels.add_batch(store.add_batch({"index": torch.tensor([2, 0, 2])}))
    np.testing.assert_array_equal(batch["semantic_class_ids"].numpy(), ids[[2, 0, 2]])
    assert batch["view_query_targets"][:, 0].tolist() == [8, 0, 8]
    with pytest.raises(ValueError, match="different target"):
        ViewQueryStore(path, root)
    with pytest.raises(IndexError):
        labels.add_batch({"index": torch.tensor([3])})
    np.save(cache / "semantic_labels.npy", np.zeros_like(ids))
    with pytest.raises(ValueError, match="checksum"):
        SemanticLabelStore(cache, root)


@pytest.mark.skipif(not os.environ.get("FRONT_QTOKEN_SMOKE_RUN"), reason="Opt-in real checkpoint/cache test")
def test_real_checkpoint_cache_equivalence_and_inference():
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mask_act_inference import load_mask_act_for_inference

    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    root = project / "data/finaldata_seg"
    cache = project / "outputs/cache/final-FrontQtoken-FS-v1"
    queries = ViewQueryStore(cache / "targets.npz", root, experiment=FRONT_QUERY_EXPERIMENT)
    labels = SemanticLabelStore(cache, root)
    run = project / os.environ["FRONT_QTOKEN_SMOKE_RUN"]
    policy, pre, post, _ = load_mask_act_for_inference(run, project)
    assert [head.out_features for head in policy.act_policy.model.metric_heads] == [1, 1, 2]
    if policy.experiment == FRONT_SYMMETRIC_EXPERIMENT:
        assert torch.all(policy.semantic_visual_mean == .5)
        assert torch.all(policy.semantic_visual_std == .5)
        image = torch.tensor([0., .5, 1.], device=policy.semantic_visual_mean.device)[None, :, None, None]
        torch.testing.assert_close(policy.normalize_semantic_map(image), 2 * image - 1, rtol=0, atol=0)
        config = json.loads((run / "mask_act_run_config.json").read_text())
        assert config["semantic_input_normalization"]["mode"] == "minus_one_to_one"
    dataset = LeRobotDataset("finaldata_seg", root=root, video_backend="pyav",
                             delta_timestamps={"action": [i / 30 for i in range(60)]})
    indices = np.linspace(0, len(dataset) - 1, 8, dtype=int)
    for start in range(0, len(indices), 2):
        raw = queries.add_batch(default_collate([dataset[int(i)] for i in indices[start:start + 2]]))
        fast = labels.add_batch(raw)
        old_ids = policy.build_semantic_targets(raw, device=torch.device("cpu"))
        new_ids = policy.build_semantic_targets(fast, device=torch.device("cpu"))
        assert all(torch.equal(a, b) for a, b in zip(old_ids, new_ids))
        target, valid = legacy_targets(new_ids[0].numpy())
        np.testing.assert_allclose(target, raw["view_query_targets"].numpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(valid, raw["view_query_valid"].numpy())
    fast = {k: v for k, v in fast.items() if k not in policy.mask_keys}
    policy.train()
    results = []
    for batch in (raw, fast):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(1000)
        loss, logs = policy(pre(batch), batch)
        loss.backward()
        gradients = {name: p.grad.detach().cpu().clone() for name, p in policy.named_parameters()
                     if p.grad is not None and ("metric_heads" in name or "input_proj" in name)}
        assert gradients
        results.append((loss.item(), gradients))
        assert np.isfinite(logs["action_loss"]) and np.isfinite(logs["metric_loss"])
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-7)
    for name, grad in results[0][1].items():
        torch.testing.assert_close(grad, results[1][1][name], atol=1e-6, rtol=1e-6)
    policy.eval()
    observation = {key: raw[key] for key in (*policy.rgb_keys, "observation.state")}
    with torch.no_grad():
        action = post(policy.predict_action_chunk(pre(observation)))
    assert action.shape == (2, 60, 10) and torch.isfinite(action).all()


@pytest.mark.skipif(not os.environ.get("FRONT_QTOKEN_SMOKE_RUN"), reason="Opt-in full model comparison")
def test_normalization_ablation_only_changes_two_buffers():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from train_mask_act_policy import make_policy

    project = Path(__file__).resolve().parents[1]
    run = project / os.environ["FRONT_QTOKEN_SMOKE_RUN"]
    config = json.loads((run / "mask_act_run_config.json").read_text())
    args = Namespace(**config)
    args.device = "cpu"
    args.pretrained_backbone_weights = None
    meta = LeRobotDatasetMetadata("finaldata_seg", root=project / "data/finaldata_seg")
    states = []
    for experiment in (FRONT_QUERY_EXPERIMENT, FRONT_SYMMETRIC_EXPERIMENT):
        args.experiment = experiment
        torch.manual_seed(1000)
        model = make_policy(args, meta, meta.stats)
        states.append(model.state_dict())
    assert states[0].keys() == states[1].keys()
    changed = {name for name in states[0] if not torch.equal(states[0][name], states[1][name])}
    assert changed == {"semantic_visual_mean", "semantic_visual_std"}
