from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from mycode.precompute_semantic_states import FEATURES_PER_VIEW, SEMANTIC_CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import SemanticStateStore  # noqa: E402


def write_state_cache(path: Path) -> np.ndarray:
    frame_count = 10
    states = np.arange(frame_count * 2 * 14, dtype=np.float32).reshape(frame_count, 2, 14)
    states /= states.max()
    quality = np.ones((frame_count, 2, 4), dtype=np.float32)
    uncertain = np.zeros_like(quality, dtype=bool)
    uncertain[8, 1, 3] = True
    episode_end = np.asarray([5] * 5 + [10] * 5, dtype=np.int64)
    np.savez_compressed(
        path,
        semantic_states=states,
        quality_score=quality,
        uncertain=uncertain,
        episode_end_index=episode_end,
        processed=np.ones(frame_count, dtype=bool),
        rgb_keys=np.asarray(["observation.images.front", "observation.images.side"]),
        class_names=np.asarray(SEMANTIC_CLASSES),
        feature_names=np.asarray(FEATURES_PER_VIEW),
    )
    return states


def test_store_builds_future_targets_without_crossing_episode(tmp_path: Path):
    path = tmp_path / "states.npz"
    states = write_state_cache(path)
    store = SemanticStateStore(
        path,
        total_frames=10,
        rgb_keys=["observation.images.front", "observation.images.side"],
        minimum_quality_score=0.6,
    )

    batch = store.add_batch_semantics(
        {"index": torch.tensor([3, 5])},
        prediction_offsets=(1, 3),
    )

    assert batch["future_semantic_states"].shape == (2, 2, 28)
    np.testing.assert_allclose(batch["future_semantic_states"][0, 0], states[4].reshape(-1))
    assert batch["future_semantic_valid"].tolist() == [[True, False], [True, False]]
    assert batch["mask_quality_current_valid"].shape == (2, 2, 4)


def test_store_rejects_different_view_order(tmp_path: Path):
    path = tmp_path / "states.npz"
    write_state_cache(path)

    with pytest.raises(ValueError, match="exact ordered view list"):
        SemanticStateStore(
            path,
            total_frames=10,
            rgb_keys=["observation.images.side", "observation.images.front"],
            minimum_quality_score=0.6,
        )
