from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import (  # noqa: E402
    build_mask_layout,
    semantic_palette_for_experiment,
    validate_semantic_class_layout,
)


SEM1_N_CLASSES = ["occluder", "object", "region", "tool", "leftarm", "rightarm"]


def test_sem1_n_accepts_dataset_defined_arm_classes():
    validate_semantic_class_layout("SEM-1-N", SEM1_N_CLASSES)

    mask_keys = [f"observation.images.front_{name}" for name in SEM1_N_CLASSES]
    suffixes, key_map = build_mask_layout(["observation.images.front"], mask_keys)

    assert suffixes == SEM1_N_CLASSES
    assert key_map["observation.images.front_leftarm"] == (0, 4)
    assert key_map["observation.images.front_rightarm"] == (0, 5)
    palette = semantic_palette_for_experiment("SEM-1-N")
    assert tuple(round(channel * 255) for channel in palette["occluder"]) == (64, 160, 255)
    assert tuple(round(channel * 255) for channel in palette["leftarm"]) == (234, 146, 199)


def test_legacy_sem1_keeps_its_four_class_contract():
    with pytest.raises(ValueError, match="Use SEM-1-N"):
        validate_semantic_class_layout("SEM-1", SEM1_N_CLASSES)


def test_sem1_n_rejects_classes_without_palette_and_loss_weights():
    with pytest.raises(ValueError, match="without palette/loss definitions"):
        validate_semantic_class_layout("SEM-1-N", [*SEM1_N_CLASSES, "unknown"])
