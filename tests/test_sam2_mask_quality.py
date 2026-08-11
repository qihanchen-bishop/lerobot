import numpy as np

from mycode.sam2_mask_quality import (
    MaskQualityThresholds,
    binary_iou,
    score_mask_diagnostics,
    thresholds_for_mask_key,
)


def test_binary_iou_handles_empty_and_disjoint_masks():
    empty = np.zeros((4, 4), dtype=bool)
    first = empty.copy()
    second = empty.copy()
    first[0, 0] = True
    second[3, 3] = True
    assert binary_iou(empty, empty) == 1.0
    assert binary_iou(first, second) == 0.0


def test_quality_score_combines_temporal_and_geometric_anomalies():
    diagnostics = {
        "temporal_iou": np.array([0.95, 0.0], dtype=np.float32),
        "area_log_change": np.array([0.0, 3.0], dtype=np.float32),
        "centroid_jump": np.array([0.0, 0.5], dtype=np.float32),
        "components": np.array([1, 8], dtype=np.int16),
    }
    score, uncertain, failures = score_mask_diagnostics(diagnostics, MaskQualityThresholds())
    assert score[0] > score[1]
    assert not uncertain[0]
    assert uncertain[1]
    assert failures["low_temporal_iou"].tolist() == [False, True]
    assert failures["area_jump"].tolist() == [False, True]


def test_tool_and_cloth_profiles_allow_expected_motion_and_components():
    base = MaskQualityThresholds()
    cloth = thresholds_for_mask_key(base, "observation.images.front_occluder")
    tool = thresholds_for_mask_key(base, "observation.images.front_tool")
    assert cloth.max_components >= 8
    assert tool.max_components >= 6
    assert tool.temporal_iou_min <= 0.4
    assert tool.centroid_jump_max >= 0.12
