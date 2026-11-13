import sys
from pathlib import Path

import numpy as np
import torch
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from ur_semq import descriptor_weights, descriptors, query_loss, reference_region


def test_zero_area_supervised_but_no_fake_position():
    target, valid = descriptors(np.zeros((10, 20), np.uint8), np.ones((10, 20), bool))
    assert np.array_equal(target, np.zeros(6))
    assert valid.tolist() == [True, True, False, True, False, False]


def test_distance_is_centroid_not_inside_test():
    ids = np.zeros((10, 20), np.uint8)
    ids[2:4, 3:5] = 2
    region = np.ones_like(ids, bool)
    target, valid = descriptors(ids, region)
    assert valid.all()
    np.testing.assert_allclose(target[4:], [3.5 / 19, 2.5 / 9])
    assert target[2] > 0
    np.testing.assert_allclose(target[2], np.hypot(6, 2) / np.hypot(19, 9))


def test_reference_requires_ten_reliable_consistent_frames():
    masks = np.zeros((20, 20, 20), bool)
    masks[:, :10, :10] = True
    quality = np.ones(20, np.float32)
    uncertain = np.zeros(20, bool)
    ref, score, count = reference_region(masks, quality, uncertain)
    assert count == 20 and score == 1 and np.array_equal(ref, masks[0])
    uncertain[:11] = True
    assert reference_region(masks, quality, uncertain)[0] is None
    # A displaced reference must not be silently pooled with this episode.
    masks[10:] = np.roll(masks[10:], 10, axis=2)
    assert reference_region(masks, quality, np.zeros(20, bool))[0] is None


def test_quality_only_gates_dependent_heads():
    valid = np.ones((2, 6), bool)
    scores = {k: np.array([0.8, 0.4], np.float32) for k in ("object", "organ", "occluder")}
    bad = {k: np.zeros(2, bool) for k in scores}
    bad["object"][0] = True
    weights = descriptor_weights(valid, scores, bad, 0.7)
    np.testing.assert_allclose(weights[0], [0, 0.8, 0, 0.8, 0, 0])
    np.testing.assert_allclose(weights[1], [0.4] * 6)


def test_quality_weight_is_not_cancelled_by_normalization():
    prediction = torch.ones(2, 6, requires_grad=True)
    target, valid, weights, scales = (
        torch.zeros(2, 6),
        torch.ones(2, 6, dtype=torch.bool),
        torch.ones(2, 6),
        torch.ones(6),
    )
    full, _ = query_loss(prediction, target, valid, weights, scales)
    low, _ = query_loss(prediction, target, valid, weights * 0.2, scales)
    torch.testing.assert_close(low, full * 0.2)
    none, _ = query_loss(prediction, target, valid * False, weights * 0, scales)
    assert none.item() == 0
    none.backward()
    assert torch.isfinite(prediction.grad).all()


def test_ur_import_does_not_change_soarm_layout():
    import view_qtoken

    from lerobot.policies.act.configuration_act import ACTConfig

    assert view_qtoken.CLASSES == ("occluder", "object", "region", "tool", "leftarm", "rightarm")
    assert view_qtoken.OUTPUT_DIMS == (2, 4, 2)
    assert ACTConfig().metric_mode is None


@pytest.mark.parametrize("variant,semantic,query", [
    ("full", True, True), ("semantic-no-query", True, False), ("rgb-query", False, True),
])
def test_variant_data_paths(variant, semantic, query):
    from train_ur_semq_policy import URQueryDataset, prepare_batch, variant_flags

    assert variant_flags(SimpleNamespace(variant=variant)) == (semantic, query)
    sample = {
        "episode_index": 0, "frame_index": 0,
        "observation.images.front": torch.zeros(3, 4, 5),
        "observation.state": torch.zeros(13), "action": torch.zeros(60, 13),
        "action_is_pad": torch.zeros(60, dtype=torch.bool),
    }
    calls = []
    def labels(index):
        assert semantic, "RGB-only must not read masks"
        calls.append(index)
        return np.zeros((4, 5), np.uint8)
    cache = SimpleNamespace(
        arrays={"episode_index": [0], "frame_index": [0],
                "targets": np.zeros((1, 6), np.float32),
                "valid": np.ones((1, 6), bool), "weights": np.ones((1, 6), np.float32)},
        labels=labels,
    )
    data = URQueryDataset([sample], cache, ["observation.images.front"], semantic, query)
    raw = torch.utils.data.default_collate([data[0]])
    assert ("semantic_ids" in raw) == semantic
    assert ("query_targets" in raw) == query
    batch, target, valid, weights = prepare_batch(raw, torch.zeros(8, 3), lambda x: x, "cpu")
    assert ("observation.images.ur_semantic" in batch) == semantic
    assert not any(k.startswith("query_") for k in batch)
    assert (target is not None) == query
    assert (valid is not None) == query and (weights is not None) == query
    assert bool(calls) == semantic


def test_variant_default_preserves_existing_experiments():
    from train_ur_semq_policy import variant_flags

    assert variant_flags(SimpleNamespace()) == (True, True)
    with pytest.raises(ValueError, match="Unknown UR variant"):
        variant_flags(SimpleNamespace(variant="typo"))
