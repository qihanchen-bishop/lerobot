from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import (  # noqa: E402
    act_image_keys_for_experiment,
    build_mask_layout_for_experiment,
    semantic_suffixes_by_view,
)


FRONT = "observation.images.front"
SIDE = "observation.images.side"
FRONT_CLASSES = ["occluder", "object", "region", "tool", "leftarm", "rightarm"]
SIDE_CLASSES = ["occluder", "object", "region", "tool"]


def v5_mask_keys() -> list[str]:
    return [*(f"{FRONT}_{name}" for name in FRONT_CLASSES), *(f"{SIDE}_{name}" for name in SIDE_CLASSES)]


def test_unet_sem_v5_accepts_asymmetric_view_classes() -> None:
    suffixes, key_map = build_mask_layout_for_experiment(
        "UNET-SEM-V5", [FRONT, SIDE], v5_mask_keys()
    )

    assert suffixes == FRONT_CLASSES
    assert semantic_suffixes_by_view([FRONT, SIDE], v5_mask_keys()) == [
        FRONT_CLASSES,
        SIDE_CLASSES,
    ]
    assert key_map[f"{FRONT}_rightarm"] == (0, 5)
    assert key_map[f"{SIDE}_tool"] == (1, 3)


def test_view_named_unet_sem_v5_alias_uses_same_visual_inputs() -> None:
    original = Namespace(experiment="UNET-SEM-V5", rgb_keys=[FRONT, SIDE])
    view_named = Namespace(experiment="UNET-SEM-V5-FS", rgb_keys=[FRONT, SIDE])

    assert act_image_keys_for_experiment(view_named) == act_image_keys_for_experiment(original)


def test_legacy_multiview_semantic_layout_still_rejects_missing_classes() -> None:
    with pytest.raises(ValueError, match="same semantic mask suffixes"):
        build_mask_layout_for_experiment("SEM-1-N", [FRONT, SIDE], v5_mask_keys())


def test_legacy_single_view_independent_mask_names_remain_supported() -> None:
    rgb = ["observation.images.camera"]
    masks = ["observation.images.object", "observation.images.region"]
    assert semantic_suffixes_by_view(rgb, masks) == [masks]


@pytest.mark.parametrize(
    ("experiment", "rgb_keys", "expected"),
    [
        ("STAGE-V5-F-RGB", [FRONT], [FRONT]),
        ("STAGE-V5-FS-RGB", [FRONT, SIDE], [FRONT, SIDE]),
        (
            "STAGE-V5-F-UNETSEM",
            [FRONT],
            [f"{FRONT}_semantic", FRONT],
        ),
        (
            "STAGE-V5-FS-UNETSEM",
            [FRONT, SIDE],
            [f"{FRONT}_semantic", f"{SIDE}_semantic", FRONT, SIDE],
        ),
        (
            "UNET-SEM-V5",
            [FRONT, SIDE],
            [f"{FRONT}_semantic", f"{SIDE}_semantic", FRONT, SIDE],
        ),
    ],
)
def test_stage_v5_visual_ablation_inputs(
    experiment: str,
    rgb_keys: list[str],
    expected: list[str],
) -> None:
    args = Namespace(experiment=experiment, rgb_keys=rgb_keys)
    assert act_image_keys_for_experiment(args) == expected
