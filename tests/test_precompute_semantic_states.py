import numpy as np
import pytest
import torch

from mycode.precompute_semantic_states import FEATURES_PER_VIEW, semantic_state_from_masks
from mycode.semantic_servo import SoftSemanticStateExtractor


@pytest.mark.parametrize("height,width", [(10, 20), (21, 17), (36, 64)])
def test_precomputed_hard_state_matches_online_extractor(height: int, width: int):
    masks = np.zeros((4, height, width), dtype=bool)
    masks[0, : max(height // 3, 1)] = True
    masks[1, height // 2 : height // 2 + 2, width // 2 : width // 2 + 3] = True
    masks[2, height // 2 : height // 2 + 3, width // 2 + 3 : width // 2 + 6] = True
    masks[3, height // 3 : height // 3 + 2, 1:4] = True
    occupied = np.zeros((height, width), dtype=bool)
    for class_idx in range(4):
        masks[class_idx] &= ~occupied
        occupied |= masks[class_idx]

    offline = semantic_state_from_masks(masks)
    online = SoftSemanticStateExtractor(include_confidence=False)(
        torch.from_numpy(masks).float().unsqueeze(0).unsqueeze(0)
    )[0].detach().numpy()

    assert offline.shape == (len(FEATURES_PER_VIEW),)
    np.testing.assert_allclose(offline, online, rtol=1e-5, atol=1e-6)


def test_precomputed_state_is_finite_for_missing_classes():
    masks = np.zeros((4, 12, 18), dtype=bool)
    masks[0, :4] = True

    state = semantic_state_from_masks(masks)

    assert np.isfinite(state).all()
    assert (state[4:] == 0.0).all()
