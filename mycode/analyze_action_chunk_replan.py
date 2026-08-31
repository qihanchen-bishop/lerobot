"""Analyze action-chunk motion scales for replanning decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("/home/qihan/data/lerobot/data/bettersetup_v5")
DEFAULT_K_STEPS = (1, 4, 8, 16, 24, 60)
DEFAULT_ENDPOINT_BUDGETS = (8.0, 12.0, 16.0, 20.0, 30.0, 40.0)
DEFAULT_PATH_BUDGETS = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
DEFAULT_SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


JOINT_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
)


def quantiles(values: np.ndarray | list[float], qs: tuple[float, ...] = (0, 10, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"p{q:g}": float("nan") for q in qs}
    return {f"p{q:g}": float(np.percentile(array, q)) for q in qs}


def summarize_delta(delta: np.ndarray) -> dict[str, Any]:
    abs_delta = np.abs(delta)
    return {
        "per_joint_abs_all_values": quantiles(abs_delta.reshape(-1), (50, 75, 90, 95, 99, 99.5, 99.9, 100)),
        "max_joint_abs_per_step": quantiles(abs_delta.max(axis=1), (50, 75, 90, 95, 99, 99.5, 99.9, 100)),
        "l2_per_step": quantiles(np.linalg.norm(delta, axis=1), (50, 75, 90, 95, 99, 99.5, 99.9, 100)),
        "mean_joint_abs_per_step": quantiles(abs_delta.mean(axis=1), (50, 90, 95, 99, 100)),
    }


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / axis_norm
    c, s = np.cos(angle), np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(part) for part in value.split()], dtype=np.float64)


def rotation_angle_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    relative = r0.T @ r1
    cos_angle = (np.trace(relative) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


class SimpleUrdfFk:
    """Small URDF FK helper for serial fixed/revolute chains."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        joint_names: tuple[str, ...] = DEFAULT_SO101_JOINTS,
        base_link: str = "base_link",
        target_link: str = "gripper_frame_link",
    ) -> None:
        self.joint_values = {name: 0.0 for name in joint_names}
        root = ET.parse(urdf_path).getroot()
        joints = []
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            parent = joint.find("parent")
            child = joint.find("child")
            axis = joint.find("axis")
            if parent is None or child is None:
                continue
            joints.append(
                {
                    "name": joint.attrib["name"],
                    "type": joint.attrib.get("type", "fixed"),
                    "parent": parent.attrib["link"],
                    "child": child.attrib["link"],
                    "origin": make_transform(
                        parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                        parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                    ),
                    "axis": parse_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                }
            )

        chain = []
        current = target_link
        while current != base_link:
            parent_joint = next((joint for joint in joints if joint["child"] == current), None)
            if parent_joint is None:
                raise ValueError(f"Could not find a URDF chain from {base_link} to {target_link}.")
            chain.append(parent_joint)
            current = parent_joint["parent"]
        self.chain = list(reversed(chain))

    def forward(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        for name, value in zip(self.joint_values, joint_pos_deg, strict=False):
            self.joint_values[name] = float(np.deg2rad(value))

        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            transform = transform @ joint["origin"]
            if joint["type"] in {"revolute", "continuous"}:
                joint_transform = np.eye(4, dtype=np.float64)
                joint_transform[:3, :3] = axis_angle_matrix(joint["axis"], self.joint_values.get(joint["name"], 0.0))
                transform = transform @ joint_transform
        return transform


def load_actions(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((dataset_root / "data/chunk-000").glob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data/chunk-000'}")

    episode_arrays: list[np.ndarray] = []
    frame_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    state_arrays: list[np.ndarray] = []
    for file in files:
        table = pq.read_table(file, columns=["episode_index", "frame_index", "action", "observation.state"])
        episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        order = np.lexsort((frame, episode))
        episode_arrays.append(episode[order])
        frame_arrays.append(frame[order])
        action_arrays.append(action[order])
        state_arrays.append(state[order])

    episodes = np.concatenate(episode_arrays)
    frames = np.concatenate(frame_arrays)
    actions = np.concatenate(action_arrays)
    states = np.concatenate(state_arrays)
    order = np.lexsort((frames, episodes))
    return episodes[order], frames[order], actions[order], states[order]


def first_budget_hit(displacements: np.ndarray, budget: float) -> int:
    hit = np.flatnonzero(displacements >= budget)
    return int(hit[0] + 1) if hit.size else int(displacements.shape[0])


def analyze_joint_space(
    episodes: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    *,
    k_steps: tuple[int, ...],
    endpoint_budgets: tuple[float, ...],
    path_budgets: tuple[float, ...],
    fps: float,
) -> dict[str, Any]:
    unique_episodes = np.unique(episodes)
    episode_lengths: list[int] = []
    consecutive_deltas: list[np.ndarray] = []
    second_deltas: list[np.ndarray] = []
    tracking_errors: list[np.ndarray] = []
    displacement_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_steps}
    path_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_steps if k > 1}
    max_step_inside_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_steps if k > 1}
    endpoint_budget_k: dict[float, list[int]] = {budget: [] for budget in endpoint_budgets}
    path_budget_k: dict[float, list[int]] = {budget: [] for budget in path_budgets}

    max_k = max(k_steps)
    for episode_index in unique_episodes:
        episode_actions = actions[episodes == episode_index]
        episode_lengths.append(int(len(episode_actions)))
        tracking_errors.append(episode_actions - states[episodes == episode_index])
        if len(episode_actions) >= 2:
            delta = np.diff(episode_actions, axis=0)
            consecutive_deltas.append(delta)
            abs_delta = np.abs(delta)
            l2_delta = np.linalg.norm(delta, axis=1)
        else:
            abs_delta = np.empty((0, actions.shape[1]), dtype=np.float64)
            l2_delta = np.empty((0,), dtype=np.float64)
        if len(episode_actions) >= 3:
            second_deltas.append(np.diff(episode_actions, n=2, axis=0))

        for k in k_steps:
            if len(episode_actions) > k:
                displacement_by_k[k].append(episode_actions[k:] - episode_actions[:-k])
            if k > 1 and len(l2_delta) >= k:
                path_by_k[k].append(
                    np.asarray([l2_delta[i : i + k].sum() for i in range(len(l2_delta) - k + 1)])
                )
                max_step_inside_by_k[k].append(
                    np.asarray([abs_delta[i : i + k].max() for i in range(len(abs_delta) - k + 1)])
                )

        if len(episode_actions) > max_k:
            for start in range(len(episode_actions) - max_k):
                future = episode_actions[start + 1 : start + max_k + 1]
                endpoint_motion = np.abs(future - episode_actions[start]).max(axis=1)
                path_motion = np.cumsum(l2_delta[start : start + max_k])
                for budget in endpoint_budgets:
                    endpoint_budget_k[budget].append(first_budget_hit(endpoint_motion, budget))
                for budget in path_budgets:
                    path_budget_k[budget].append(first_budget_hit(path_motion, budget))

    step_delta = np.concatenate(consecutive_deltas)
    tracking = np.concatenate(tracking_errors)
    result: dict[str, Any] = {
        "frames": int(len(actions)),
        "episodes": int(len(unique_episodes)),
        "fps_assumption": float(fps),
        "episode_length_frames": quantiles(episode_lengths),
        "episode_duration_s": quantiles(np.asarray(episode_lengths, dtype=np.float64) / fps),
        "consecutive_action_delta_deg": summarize_delta(step_delta),
        "consecutive_action_velocity_deg_s": summarize_delta(step_delta * fps),
        "action_state_tracking_error_deg": summarize_delta(tracking),
        "displacement_over_k_steps_deg": {},
        "chunk_motion_windows_deg": {},
        "motion_budget_first_k_frames": {
            "endpoint_max_joint_deg": {},
            "path_l2_deg": {},
        },
        "per_joint_consecutive_delta_deg": [],
    }
    if second_deltas:
        result["second_difference_deg"] = summarize_delta(np.concatenate(second_deltas))

    for k in k_steps:
        if displacement_by_k[k]:
            result["displacement_over_k_steps_deg"][str(k)] = summarize_delta(np.concatenate(displacement_by_k[k]))
        if k > 1 and path_by_k[k]:
            result["chunk_motion_windows_deg"][str(k)] = {
                "cumulative_l2_path_deg": quantiles(np.concatenate(path_by_k[k])),
                "max_single_joint_step_inside_window_deg": quantiles(np.concatenate(max_step_inside_by_k[k])),
            }

    for budget, values in endpoint_budget_k.items():
        array = np.asarray(values, dtype=np.float64)
        result["motion_budget_first_k_frames"]["endpoint_max_joint_deg"][str(budget)] = {
            **quantiles(array),
            "mean": float(np.mean(array)),
            "at_60_pct": float(np.mean(array == max_k) * 100.0),
            "at_or_before_4_pct": float(np.mean(array <= 4) * 100.0),
        }
    for budget, values in path_budget_k.items():
        array = np.asarray(values, dtype=np.float64)
        result["motion_budget_first_k_frames"]["path_l2_deg"][str(budget)] = {
            **quantiles(array),
            "mean": float(np.mean(array)),
            "at_60_pct": float(np.mean(array == max_k) * 100.0),
            "at_or_before_4_pct": float(np.mean(array <= 4) * 100.0),
        }

    abs_step = np.abs(step_delta)
    for joint_index, joint_name in enumerate(JOINT_NAMES[: actions.shape[1]]):
        result["per_joint_consecutive_delta_deg"].append(
            {
                "joint": joint_name,
                "p95_abs_step_deg": float(np.percentile(abs_step[:, joint_index], 95)),
                "p99_abs_step_deg": float(np.percentile(abs_step[:, joint_index], 99)),
                "p99_9_abs_step_deg": float(np.percentile(abs_step[:, joint_index], 99.9)),
                "max_abs_step_deg": float(abs_step[:, joint_index].max()),
            }
        )
    return result


def analyze_end_effector_space(
    episodes: np.ndarray,
    actions: np.ndarray,
    *,
    urdf_path: Path,
    k_steps: tuple[int, ...],
    fps: float,
) -> dict[str, Any]:
    fk = SimpleUrdfFk(urdf_path)
    result: dict[str, Any] = {
        "urdf_path": str(urdf_path),
        "note": "Left and right arms use the same SO101 kinematic chain. Base-to-world transforms are not needed for motion magnitudes.",
        "consecutive_ee_delta": {},
        "displacement_over_k_steps": {},
    }
    positions: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    rotations: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    for side, slice_ in {"left": slice(0, 5), "right": slice(5, 10)}.items():
        side_pos = []
        side_rot = []
        for action in actions:
            transform = fk.forward(action[slice_])
            side_pos.append(transform[:3, 3].copy())
            side_rot.append(transform[:3, :3].copy())
        positions[side] = np.asarray(side_pos, dtype=np.float64)
        rotations[side] = np.asarray(side_rot, dtype=np.float64)

    for side in ("left", "right"):
        consecutive_pos: list[np.ndarray] = []
        consecutive_rot: list[float] = []
        disp_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_steps}
        rot_by_k: dict[int, list[float]] = {k: [] for k in k_steps}
        for episode_index in np.unique(episodes):
            mask = episodes == episode_index
            pos = positions[side][mask]
            rot = rotations[side][mask]
            if len(pos) >= 2:
                consecutive_pos.append(np.diff(pos, axis=0))
                consecutive_rot.extend(rotation_angle_deg(rot[i], rot[i + 1]) for i in range(len(rot) - 1))
            for k in k_steps:
                if len(pos) > k:
                    disp_by_k[k].append(pos[k:] - pos[:-k])
                    rot_by_k[k].extend(rotation_angle_deg(rot[i], rot[i + k]) for i in range(len(rot) - k))

        pos_delta = np.concatenate(consecutive_pos)
        result["consecutive_ee_delta"][side] = {
            "translation_mm": quantiles(np.linalg.norm(pos_delta, axis=1) * 1000.0, (50, 75, 90, 95, 99, 100)),
            "translation_velocity_mm_s": quantiles(
                np.linalg.norm(pos_delta, axis=1) * 1000.0 * fps,
                (50, 75, 90, 95, 99, 100),
            ),
            "rotation_deg": quantiles(consecutive_rot, (50, 75, 90, 95, 99, 100)),
        }
        result["displacement_over_k_steps"][side] = {}
        for k in k_steps:
            if not disp_by_k[k]:
                continue
            pos_disp = np.concatenate(disp_by_k[k])
            result["displacement_over_k_steps"][side][str(k)] = {
                "translation_mm": quantiles(np.linalg.norm(pos_disp, axis=1) * 1000.0, (50, 75, 90, 95, 99, 100)),
                "rotation_deg": quantiles(rot_by_k[k], (50, 75, 90, 95, 99, 100)),
            }
    return result


def write_markdown(result: dict[str, Any], output_path: Path) -> None:
    disp = result["displacement_over_k_steps_deg"]
    windows = result["chunk_motion_windows_deg"]
    lines = [
        "# bettersetup_v5 action chunk replan 分析",
        "",
        "## 技术摘要",
        "",
        "- `bettersetup_v5` 有 90 条 episode、53,723 帧；按 30 Hz 估算，中位 episode 时长约 "
        f"{result['episode_duration_s']['p50']:.2f} s。",
        "- 固定 60 action 并不等价于固定任务进度：60 步终点相对当前 action 的单关节变化中位数为 "
        f"{disp['60']['max_joint_abs_per_step']['p50']:.2f} deg，p95 为 "
        f"{disp['60']['max_joint_abs_per_step']['p95']:.2f} deg。",
        "- 数据支持把 replan 条件从“固定执行 K 步”扩展为“固定步数 + 运动预算 + 跟踪误差 + 语义阶段切换”。",
        "- SO101 末端空间统计使用公开 URDF 的串联链 FK；图像里的 `tool` 只作为视觉语义，不参与 FK。",
        "",
        "## 不同步长下的关节变化",
        "",
        "| K steps | final max-joint p50 | final max-joint p95 | final max-joint p99 | cumulative L2 path p50 | cumulative L2 path p95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in ("1", "4", "8", "16", "24", "60"):
        d = result["consecutive_action_delta_deg"] if k == "1" else disp[k]
        p = windows.get(k, {}).get("cumulative_l2_path_deg")
        path_p50 = "" if p is None else f"{p['p50']:.2f}"
        path_p95 = "" if p is None else f"{p['p95']:.2f}"
        lines.append(
            f"| {k} | {d['max_joint_abs_per_step']['p50']:.2f} | "
            f"{d['max_joint_abs_per_step']['p95']:.2f} | "
            f"{d['max_joint_abs_per_step']['p99']:.2f} | "
            f"{path_p50} | "
            f"{path_p95} |"
        )
    lines += [
        "",
        "## Motion budget 触发会怎样",
        "",
        "下面的 K 表示从当前 action 开始，累计运动首次超过阈值时已经执行了多少帧；如果 60 帧内都没超过，则记为 60。",
        "",
        "| Final max-joint budget | median first K | p75 first K | p90 first K | <=4 steps | never within 60 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in ("8.0", "12.0", "16.0", "20.0", "30.0", "40.0"):
        row = result["motion_budget_first_k_frames"]["endpoint_max_joint_deg"][budget]
        lines.append(
            f"| {float(budget):.0f} deg | {row['p50']:.0f} | {row['p75']:.0f} | {row['p90']:.0f} | "
            f"{row['at_or_before_4_pct']:.1f}% | {row['at_60_pct']:.1f}% |"
        )
    lines += [
        "",
        "| Cumulative L2 path budget | median first K | p75 first K | p90 first K | <=4 steps | never within 60 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in ("10.0", "15.0", "20.0", "30.0", "45.0", "60.0"):
        row = result["motion_budget_first_k_frames"]["path_l2_deg"][budget]
        lines.append(
            f"| {float(budget):.0f} deg | {row['p50']:.0f} | {row['p75']:.0f} | {row['p90']:.0f} | "
            f"{row['at_or_before_4_pct']:.1f}% | {row['at_60_pct']:.1f}% |"
        )
    lines += [
        "",
        "## 末端空间变化",
        "",
    ]
    ee = result.get("end_effector_space")
    if ee is None:
        lines += [
            "本次没有提供 URDF，因此没有计算末端空间。运行脚本时加入 `--urdf /path/to/so101_new_calib.urdf` 即可补充。",
            "",
        ]
    else:
        lines += [
            "| Arm | 1-step translation p50 | 1-step translation p95 | 60-step translation p50 | 60-step translation p95 | 60-step rotation p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for side in ("left", "right"):
            one = ee["consecutive_ee_delta"][side]
            sixty = ee["displacement_over_k_steps"][side]["60"]
            lines.append(
                f"| {side} | {one['translation_mm']['p50']:.1f} mm | "
                f"{one['translation_mm']['p95']:.1f} mm | "
                f"{sixty['translation_mm']['p50']:.1f} mm | "
                f"{sixty['translation_mm']['p95']:.1f} mm | "
                f"{sixty['rotation_deg']['p95']:.1f} deg |"
            )
        lines.append("")
    lines += [
        "",
        "## 判断",
        "",
        "只要机械臂运动幅度过大就重新规划，这个想法是合理的，但它应该作为 motion budget 或安全触发条件，而不是替代闭环判断。"
        "大动作可能是正确任务动作，例如快速 uncover 或 transport；此时 replan 未必解决问题，反而可能造成策略抖动。",
        "",
        "更合适的在线逻辑是：固定最小 replan 间隔仍保留；当已执行 action 的累计关节变化或末端位移超过预算时提前 replan；"
        "当当前关节状态和旧 chunk 计划位置偏差过大时提前 replan；当语义阶段变化时立刻 replan。安装新 chunk 时继续使用 overlap fusion，避免硬切换带来的动作跳变。",
        "",
        "可以先试的保守阈值是：关节空间使用 `max_joint_final_displacement >= 16-20 deg` 或 `cumulative_joint_L2_path >= 20-30 deg`；"
        "末端空间使用左臂 `20-30 mm`、右臂 `40-60 mm` 的累计平移预算。"
        "这些阈值来自示范分布的中位到 p75 附近，目标是让快动作比固定 24/60 步更早重规划，同时避免 4 步内频繁抖动。",
        "",
        "SO101 的在线版本应把左右 5 维关节 action 分别做 FK，统计末端平移、旋转和速度。图像里的 `tool` 是视觉标注对象，"
        "可以用于语义阶段和接触关系判断，但不应替代真实末端运动学。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json-output", type=Path, default=Path("/tmp/bettersetup_v5_action_chunk_stats.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/draftv2/action_chunk_replan_bettersetup_v5.md"))
    parser.add_argument("--urdf", type=Path, default=None, help="Optional SO101 URDF for end-effector FK statistics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes, _frames, actions, states = load_actions(args.dataset_root)
    result = analyze_joint_space(
        episodes,
        actions,
        states,
        k_steps=DEFAULT_K_STEPS,
        endpoint_budgets=DEFAULT_ENDPOINT_BUDGETS,
        path_budgets=DEFAULT_PATH_BUDGETS,
        fps=args.fps,
    )
    result["dataset"] = str(args.dataset_root)
    if args.urdf is not None:
        result["end_effector_space"] = analyze_end_effector_space(
            episodes,
            actions,
            urdf_path=args.urdf,
            k_steps=DEFAULT_K_STEPS,
            fps=args.fps,
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.markdown_output)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
