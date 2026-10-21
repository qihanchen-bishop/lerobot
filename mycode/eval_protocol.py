"""Finaldata evaluation protocol, independent of Tk and robot hardware."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "eval_configs" / "finaldata"
CALIBRATION_PATH = CONFIG_DIR / "calibration.json"


def load_calibration():
    if not CALIBRATION_PATH.exists():
        warnings.warn(
            f"Local evaluation calibration not found: {CALIBRATION_PATH}. "
            "Using finaldata defaults (40 s, coverage 0.32).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "source": "builtin_finaldata_defaults",
            "recommended_episode_time_s": 40,
            "recommended_occluder_complete_ratio": 0.32,
        }
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


STAGES = (
    ("unexposed", "未暴露"),
    ("exposed_not_separated", "暴露未分离"),
    ("separated_not_in_region", "分离未入区"),
    ("in_region_not_restored", "入区未恢复"),
    ("complete", "完成"),
)


def stage_from_metrics(metrics):
    for index, field in enumerate(("final_object_exposed", "final_object_separated",
                                   "final_object_reached_region", "final_task_success")):
        if not metrics.get(field):
            return STAGES[index]
    return STAGES[-1]


# Codes are stable for cross-run aggregation. Examples come from historical eval tags.
DEVIATIONS = (
    ("cloth_interference", "布料干扰", (
        ("object_trapped", "物体卡布里／被布套住"),
        ("tool_trapped", "tool 卡布里／被布套住"),
        ("object_pushed_under_cloth", "目标被推回布里／过早盖布"),
        ("cloth_not_restored", "入区后布没拨好／恢复不到位"),
    )),
    ("contact_failure", "接触失效", (
        ("air_sweep", "tool 拨空气／未接触目标"),
        ("tool_too_high", "在物体上面拨／tool 太高"),
        ("contact_lost", "拨一半断开接触／跟踪丢失"),
    )),
    ("transport_error", "输送偏差", (
        ("pushed_away", "推远／拨远了拨不回来"),
        ("near_region", "差点进／停在区域边缘"),
        ("overshoot", "推过头／弹射／滑出目标区"),
        ("wrong_direction", "推偏／方向错误"),
    )),
    ("policy_stall", "策略停滞", (
        ("hesitation", "启动或中途犹豫／动作选择迟疑"),
        ("oscillation", "反复横跳／小范围摆动／抽搐"),
        ("no_progress", "原地不动／反复拨但无进展"),
        ("post_entry_stall", "进区后犹豫／迟迟不恢复布料"),
    )),
)


def deviation_records(selected):
    """Serialize selected code -> recovery choice; require an explicit yes/no."""
    records = []
    for category, category_label, choices in DEVIATIONS:
        for code, label in choices:
            if code not in selected:
                continue
            recovery = selected[code]
            if recovery not in ("已恢复", "未恢复"):
                raise ValueError(f"请选择「{label}」是否已从该偏差恢复。")
            records.append({"category": category, "category_label": category_label,
                            "code": code, "label": label, "recovered": recovery == "已恢复"})
    unknown = set(selected) - {item["code"] for item in records}
    if unknown:
        raise ValueError(f"Unknown deviation codes: {sorted(unknown)}")
    return records
