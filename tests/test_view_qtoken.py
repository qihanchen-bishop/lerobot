import json
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from view_qtoken import (
    CACHE_VERSION, FrozenExportedUNet, ViewHeads, ViewQueryStore, dataset_signature,
    filter_class_ids, forward_training, sha256,
    quality_weights, query_loss, region_distance_map, view_geometry,
)
from train_mask_act_policy import (
    act_metric_dim, act_metric_mode, act_metric_token_output_dims, act_image_keys_for_experiment,
)


def test_shared_queries_independent_view_heads():
    tokens = torch.randn(2, 3, 16, requires_grad=True)
    heads = nn.ModuleList(ViewHeads(16, width) for width in (1, 2, 1))
    result = torch.cat([head(tokens[:, i]) for i, head in enumerate(heads)], -1)
    assert result.shape == (2, 8)
    result.sum().backward()
    assert all(h.front.weight.grad is not None and h.side.weight.grad is not None for h in heads)
    assert act_metric_dim("FINAL-VIEWQTOKEN") == 3
    assert act_metric_token_output_dims("FINAL-VIEWQTOKEN") == [2, 4, 2]
    assert act_image_keys_for_experiment(Namespace(experiment="FINAL-VIEWQTOKEN", rgb_keys=["front", "side"])) == [
        "front_semantic", "side_semantic", "front", "side"]


def test_geometry_endpoints_containment_and_absence():
    obj = np.zeros((10, 10), bool)
    tool, region = obj.copy(), obj.copy()
    obj[5, 7] = True
    tool[7, 1:6] = True
    region[4, 6:10] = True
    distance = region_distance_map(obj.shape, [[6, 4], [9, 4], [9, 6], [6, 6]])
    front, valid = view_geometry(obj, tool, region, distance, "front")
    side, _ = view_geometry(obj, tool, region, distance, "side")
    assert valid.all()
    assert front == pytest.approx([.01, 6 / 9, -2 / 9, 0])
    assert side[1] == pytest.approx(4 / 9)
    obj[5, 5] = True
    assert view_geometry(obj, tool, region, distance, "front")[0][3] > 0
    obj[:] = False
    target, valid = view_geometry(obj, tool, region, distance, "front")
    assert target.tolist() == [0] * 4
    assert valid.tolist() == [True, False, False, False]


def test_class_anomalies_only_gate_related_targets():
    valid = np.ones((3, 4), bool)
    scores = np.array([[.8, .6, .7]] * 3)
    anomalies = np.eye(3, dtype=bool)
    weights = quality_weights(valid, scores, anomalies)
    assert weights == pytest.approx(np.array([[0, 0, 0, 0], [.8, 0, 0, .7], [.8, .6, .6, 0]]))


def test_quality_scales_loss_without_renormalizing_away():
    pred = torch.ones(2, 8, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    full, _ = query_loss(pred, target, valid, valid.float())
    half, _ = query_loss(pred, target, valid, valid.float() * .5)
    assert half.item() == pytest.approx(full.item() / 2)
    zero, _ = query_loss(pred, target, valid & False, valid.float() * 0)
    zero.backward()
    assert zero.item() == 0 and torch.isfinite(pred.grad).all()


def test_filter_removes_small_and_excess_components():
    mask = np.zeros((10, 10), np.uint8)
    mask[0:3, 0:3] = 1
    mask[5:7, 0:2] = 1
    mask[9, 9] = 1
    policy = {"min_area": 2, "component_limits": {"object": 1}}
    filtered = filter_class_ids(mask, ["background", "object"], policy)
    assert (filtered == 1).sum() == 9
    assert (mask == 1).sum() == 14
    with pytest.raises(ValueError):
        filter_class_ids(mask, ["background", "object"], dict(policy, fill_holes=True))


def test_torchscript_frozen_hard_output(tmp_path):
    net = nn.Conv2d(3, 2, 1)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.copy_(torch.tensor([0., 1.]))
    torch.jit.trace(net, torch.zeros(1, 3, 8, 8)).save(str(tmp_path / "model.ts"))
    (tmp_path / "model_config.json").write_text(json.dumps({
        "labels": ["background", "object"], "torchscript": "model.ts",
        "palette_rgb": [[0, 0, 0], [255, 0, 0]], "input_size_wh": [8, 8],
        "normalization_mean": [0, 0, 0], "normalization_std": [1, 1, 1]}))
    (tmp_path / "filter_policy.json").write_text(json.dumps({"policy": {
        "min_area": 2, "component_limits": {"object": 1}}}))
    model = FrozenExportedUNet(tmp_path / "model.ts", expected_labels=("background", "object"))
    model.train()
    assert not model.training and not model.network.training
    assert not any(p.requires_grad for p in model.parameters())
    output = model(torch.rand(2, 3, 10, 12))
    assert output.shape == (2, 2, 10, 12)
    assert torch.all(output[:, 1] == 1) and torch.all(output[:, 0] == 0)


def test_training_uses_dataset_masks_not_segmenter():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.experiment = "FINAL-VIEWQTOKEN"
            self.pred = nn.Parameter(torch.ones(1, 8))
            self.rgb_keys = ["front", "side"]
            self._semantic_palette_buffer_names = ["palette", "palette"]
            self.register_buffer("palette", torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
            self.action_loss_weight = self.metric_loss_weight = 1

        def build_semantic_targets(self, raw, device):
            return [torch.ones(1, 2, 2, dtype=torch.long, device=device)] * 2

        def normalize_semantic_map(self, image):
            return image

        def act_policy(self, batch):
            assert torch.all(batch["front_semantic"][:, 0] == 1)
            assert torch.equal(batch["side_semantic"], batch["front_semantic"])
            return self.pred.sum() * 0, {"metric_pred": self.pred}

        def predict_view_logits_and_latent_from_rgbs(self, *args):
            raise AssertionError("Training must not call segmentation network")

    model = Dummy()
    raw = {"view_query_targets": torch.zeros(1, 8), "view_query_valid": torch.ones(1, 8, dtype=torch.bool),
           "view_query_weights": torch.ones(1, 8)}
    loss, logs = forward_training(model, {}, raw)
    loss.backward()
    assert model.pred.grad.abs().sum() > 0
    assert logs["seg_loss"] == 0


def test_cache_integrity_and_index_alignment(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 2}))
    (root / "segmentation_manifest.json").write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "targets.npz"
    np.savez(path, targets=np.arange(16, dtype=np.float32).reshape(2, 8),
             valid=np.ones((2, 8), bool), weights=np.ones((2, 8), np.float32))
    (cache / "manifest.json").write_text(json.dumps({
        "version": CACHE_VERSION, "source_signature": dataset_signature(root), "targets_sha256": sha256(path)}))
    store = ViewQueryStore(path, root)
    batch = store.add_batch({"index": torch.tensor([1, 0])})
    assert batch["view_query_targets"][0, 0].item() == 8
    with pytest.raises(IndexError):
        store.add_batch({"index": torch.tensor([-1])})
    (root / "segmentation_manifest.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="stale"):
        ViewQueryStore(path, root)


@pytest.mark.parametrize("experiment,semantic,queries", [
    ("FINAL-SEM-FS-NOQ", True, False),
    ("FINAL-RGB-FS-QTOKEN", False, True),
])
def test_final_ablation_inputs_and_losses(experiment, semantic, queries):
    keys = act_image_keys_for_experiment(Namespace(experiment=experiment, rgb_keys=["front", "side"]))
    assert keys == (["front_semantic", "side_semantic", "front", "side"] if semantic else ["front", "side"])
    assert act_metric_mode(experiment) == ("encoder_tokens" if queries else None)
    assert act_metric_token_output_dims(experiment) == ([2, 4, 2] if queries else None)

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.experiment = experiment
            self.pred = nn.Parameter(torch.ones(1, 8))
            self.rgb_keys = ["front", "side"]
            self._semantic_palette_buffer_names = ["palette", "palette"]
            self.register_buffer("palette", torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
            self.action_loss_weight = 1
            self.metric_loss_weight = int(queries)

        def build_semantic_targets(self, raw, device):
            assert semantic, "RGB-only model must not access masks"
            return [torch.ones(1, 2, 2, dtype=torch.long, device=device)] * 2

        def normalize_semantic_map(self, image):
            return image

        def act_policy(self, batch):
            assert ("front_semantic" in batch) == semantic
            assert ("side_semantic" in batch) == semantic
            return self.pred.sum(), ({"metric_pred": self.pred} if queries else {})

    raw = {"view_query_targets": torch.zeros(1, 8), "view_query_valid": torch.ones(1, 8, dtype=torch.bool),
           "view_query_weights": torch.ones(1, 8)} if queries else {}
    model = Dummy()
    loss, logs = forward_training(model, {}, raw)
    loss.backward()
    assert torch.isfinite(model.pred.grad).all()
    assert (logs["metric_loss"] > 0) == queries
    assert "metric_pred" not in logs
