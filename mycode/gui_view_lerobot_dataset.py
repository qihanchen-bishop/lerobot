#!/usr/bin/env python

"""GUI browser and cleaner for local LeRobot v3 datasets.

Run from the LeRobot conda environment:
    conda run -n lerobot python mycode/gui_view_lerobot_dataset.py --root mydata
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import write_stats


DEFAULT_DATASET_ROOT = Path(
    "/home/romilab/.cache/huggingface/lerobot/seeedstudio123/test_20260506_153720"
)
PREVIEW_SIZE = (640, 360)


@dataclass(frozen=True)
class EpisodeRef:
    episode_index: int
    row: pd.Series


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def read_nested_parquets(
    root: Path, *, skip_bad: bool = False, warnings: list[str] | None = None
) -> pd.DataFrame:
    paths = sorted(root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            if not skip_bad:
                raise
            if warnings is not None:
                warnings.append(f"Skipped unreadable parquet {path}: {exc}")
    if not frames:
        raise FileNotFoundError(f"No readable parquet files found under {root}")
    return pd.concat(frames, ignore_index=True)


def as_py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def clean_scalar(value: Any) -> str:
    value = as_py(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("ffmpeg is required to rewrite episode video files.") from exc


def extract_video_segment(src: Path, dst: Path, start_s: float, end_s: float, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-y",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=max(60, int(duration * 8)))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while extracting {src} [{start_s:.2f}, {end_s:.2f}]") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while extracting {src} [{start_s:.2f}, {end_s:.2f}]") from exc
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced an empty file: {dst}")


def clear_rewritten_dirs(root: Path) -> None:
    for rel in ("data", "videos", "meta/episodes"):
        path = root / rel
        if path.exists():
            shutil.rmtree(path)


def copy_static_metadata(src: Path, dst: Path) -> None:
    meta_src = src / "meta"
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "tasks.parquet", "subtasks.parquet", "stats.json"):
        in_path = meta_src / name
        if in_path.exists():
            shutil.copy2(in_path, meta_dst / name)


def copy_sidecar_episode_files(src: Path, dst: Path, old_ep: int, new_ep: int) -> None:
    sidecar_root = src / "sidecar_depth"
    if not sidecar_root.exists():
        return
    for old_path in sidecar_root.glob(f"**/episode_{old_ep:06d}.*"):
        rel = old_path.relative_to(sidecar_root)
        new_name = f"episode_{new_ep:06d}{old_path.suffix}"
        new_rel = rel.with_name(new_name)
        new_path = dst / "sidecar_depth" / new_rel
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)


def unwrap_nested_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return unwrap_nested_arrays(value.tolist()) if value.dtype == object else value.tolist()
    if isinstance(value, (list, tuple)):
        return [unwrap_nested_arrays(item) for item in value]
    return value


def stats_from_episode_row(row: pd.Series, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _prefix, feature, stat_name = key.split("/", 2)
        dtype = np.int64 if stat_name == "count" else np.float64
        array = np.asarray(unwrap_nested_arrays(value), dtype=dtype)
        if stat_name == "count":
            array = np.asarray([int(array.reshape(-1)[0])], dtype=np.int64)
        if stat_name != "count" and features.get(feature, {}).get("dtype") in {"image", "video"} and array.shape == (3,):
            array = array.reshape(3, 1, 1)
        stats.setdefault(feature, {})[stat_name] = array
    return stats


def put_progress(progress: queue.Queue[str], current: int, total: int, message: str) -> None:
    print(f"[dataset viewer] {current}/{total} {message}", flush=True)
    progress.put(f"PROGRESS|{current}|{total}|{message}")


def rewrite_dataset(root: Path, keep_episode_indices: list[int], progress: queue.Queue[str]) -> Path:
    root = root.resolve()
    check_ffmpeg()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = int(info["fps"])
    video_keys = video_keys_from_info(info)
    chunks_size = int(info.get("chunks_size", 1000))

    episodes_df = read_nested_parquets(root / "meta" / "episodes").sort_values("episode_index")
    read_warnings: list[str] = []
    frames_df = read_nested_parquets(root / "data", skip_bad=True, warnings=read_warnings)
    for warning in read_warnings:
        print(f"[dataset viewer] {warning}", flush=True)
        progress.put(warning)
    keep_set = set(keep_episode_indices)
    kept_rows = [row for _, row in episodes_df.iterrows() if int(row["episode_index"]) in keep_set]
    if not kept_rows:
        raise ValueError("Refusing to delete every episode. Keep at least one episode.")
    total_steps = len(kept_rows) * (len(video_keys) + 3) + 3
    step = 0

    tmp_root = root.parent / f".{root.name}.rewrite.tmp"
    put_progress(progress, step, total_steps, "Preparing temporary rewritten dataset...")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    copy_static_metadata(root, tmp_root)
    clear_rewritten_dirs(tmp_root)

    new_episode_rows: list[pd.Series] = []
    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    global_index = 0

    for new_ep, old_row in enumerate(kept_rows):
        old_ep = int(old_row["episode_index"])
        put_progress(progress, step, total_steps, f"Episode {old_ep} -> {new_ep}: loading frame rows...")

        ep_frames = frames_df[frames_df["episode_index"] == old_ep].copy().reset_index(drop=True)
        ep_len = len(ep_frames)
        if ep_len == 0:
            raise RuntimeError(f"Episode {old_ep} has no frame rows.")

        ep_frames["episode_index"] = new_ep
        ep_frames["frame_index"] = np.arange(ep_len, dtype=np.int64)
        ep_frames["index"] = np.arange(global_index, global_index + ep_len, dtype=np.int64)
        if "timestamp" in ep_frames:
            ep_frames["timestamp"] = ep_frames["frame_index"].astype(np.float32) / fps

        chunk_idx = new_ep // chunks_size
        file_idx = new_ep % chunks_size
        data_rel = Path(info["data_path"].format(chunk_index=chunk_idx, file_index=file_idx))
        data_path = tmp_root / data_rel
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_frames.to_parquet(data_path, index=False)
        step += 1
        put_progress(progress, step, total_steps, f"Episode {old_ep} -> {new_ep}: wrote {ep_len} frame rows.")

        new_row = old_row.copy()
        new_row["episode_index"] = new_ep
        new_row["length"] = ep_len
        new_row["data/chunk_index"] = chunk_idx
        new_row["data/file_index"] = file_idx
        new_row["dataset_from_index"] = global_index
        new_row["dataset_to_index"] = global_index + ep_len
        new_row["meta/episodes/chunk_index"] = chunk_idx
        new_row["meta/episodes/file_index"] = file_idx

        for video_key in video_keys:
            src = root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(old_row[f"videos/{video_key}/chunk_index"]),
                file_index=int(old_row[f"videos/{video_key}/file_index"]),
            )
            dst = tmp_root / info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            start_s = float(old_row[f"videos/{video_key}/from_timestamp"])
            end_s = float(old_row[f"videos/{video_key}/to_timestamp"])
            put_progress(
                progress,
                step,
                total_steps,
                f"Episode {old_ep} -> {new_ep}: extracting {video_key} ({end_s - start_s:.1f}s)...",
            )
            extract_video_segment(src, dst, start_s, end_s, fps)
            step += 1
            put_progress(progress, step, total_steps, f"Episode {old_ep} -> {new_ep}: wrote {video_key}.")
            new_row[f"videos/{video_key}/chunk_index"] = chunk_idx
            new_row[f"videos/{video_key}/file_index"] = file_idx
            new_row[f"videos/{video_key}/from_timestamp"] = 0.0
            new_row[f"videos/{video_key}/to_timestamp"] = ep_len / fps

        put_progress(progress, step, total_steps, f"Episode {old_ep} -> {new_ep}: copying sidecar files...")
        copy_sidecar_episode_files(root, tmp_root, old_ep, new_ep)
        step += 1

        episode_rel = Path(f"meta/episodes/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet")
        episode_path = tmp_root / episode_rel
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([new_row]).to_parquet(episode_path, index=False)
        new_episode_rows.append(new_row)
        all_stats.append(stats_from_episode_row(new_row, info["features"]))
        global_index += ep_len
        step += 1
        put_progress(progress, step, total_steps, f"Episode {old_ep} -> {new_ep}: metadata complete.")

    put_progress(progress, step, total_steps, "Writing repaired info.json...")
    info["total_episodes"] = len(new_episode_rows)
    info["total_frames"] = int(global_index)
    info["splits"] = {"train": f"0:{len(new_episode_rows)}"}
    (tmp_root / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    step += 1
    if all_stats:
        put_progress(progress, step, total_steps, "Aggregating dataset statistics...")
        write_stats(aggregate_stats(all_stats), tmp_root)
    step += 1

    backup_root = root.parent / f"{root.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    put_progress(progress, step, total_steps, f"Creating backup: {backup_root.name}")
    root.rename(backup_root)
    tmp_root.rename(root)
    put_progress(progress, total_steps, total_steps, "Rewrite complete.")
    return backup_root


class DatasetViewerApp:
    def __init__(self, root_window: tk.Tk, dataset_root: Path) -> None:
        self.root_window = root_window
        self.root_window.title("LeRobot Dataset Viewer")
        self.root_window.geometry("1320x820")
        self.root_window.protocol("WM_DELETE_WINDOW", self.close)

        self.dataset_root = dataset_root
        self.info: dict[str, Any] = {}
        self.episodes_df = pd.DataFrame()
        self.video_keys: list[str] = []
        self.episodes: list[EpisodeRef] = []
        self.deleted: set[int] = set()
        self.current_episode: EpisodeRef | None = None

        self.captures: dict[str, cv2.VideoCapture] = {}
        self.current_images: dict[str, ImageTk.PhotoImage] = {}
        self.playing = False
        self.play_started_at = 0.0
        self.play_offset_s = 0.0
        self.progress_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.is_rewriting = False

        self.status_text = tk.StringVar(value="")
        self.dataset_text = tk.StringVar(value=str(self.dataset_root))
        self.detail_text = tk.StringVar(value="")
        self.play_text = tk.StringVar(value="Play")
        self.progress_text = tk.StringVar(value="Idle")
        self.progress_value = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.load_dataset(self.dataset_root)
        self.root_window.after(30, self._tick)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root_window, padding=(10, 10, 10, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Dataset").pack(side=tk.LEFT)
        self.dataset_entry = ttk.Entry(top, textvariable=self.dataset_text, width=86)
        self.dataset_entry.pack(side=tk.LEFT, padx=(8, 6), fill=tk.X, expand=True)
        self.browse_button = ttk.Button(top, text="Browse", command=self.choose_dataset)
        self.browse_button.pack(side=tk.LEFT)
        self.reload_button = ttk.Button(top, text="Reload", command=lambda: self.load_dataset(Path(self.dataset_text.get())))
        self.reload_button.pack(side=tk.LEFT, padx=(6, 0))

        main = ttk.PanedWindow(self.root_window, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(main, width=360)
        main.add(left, weight=1)
        table_frame = ttk.Frame(left)
        table_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("episode", "status", "frames", "task")
        self.episode_tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        for col, width in (("episode", 72), ("status", 70), ("frames", 82), ("task", 210)):
            self.episode_tree.heading(col, text=col)
            self.episode_tree.column(col, width=width, anchor=tk.W)
        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.episode_tree.yview)
        self.episode_tree.configure(yscrollcommand=yscroll.set)
        self.episode_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.episode_tree.bind("<<TreeviewSelect>>", self.on_episode_selected)

        left_buttons = ttk.Frame(left, padding=(0, 8, 0, 0))
        left_buttons.pack(fill=tk.X)
        self.mark_button = ttk.Button(left_buttons, text="Mark Delete", command=self.toggle_delete_selected)
        self.mark_button.pack(side=tk.LEFT)
        self.apply_button = ttk.Button(left_buttons, text="Apply Delete + Reindex", command=self.apply_delete)
        self.apply_button.pack(side=tk.RIGHT)

        right = ttk.Frame(main)
        main.add(right, weight=3)
        self.video_frame = ttk.Frame(right)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(right, padding=(0, 8, 0, 0))
        controls.pack(fill=tk.X)
        self.play_button = ttk.Button(controls, textvariable=self.play_text, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        self.first_frame_button = ttk.Button(controls, text="First Frame", command=lambda: self.seek_current(0.0))
        self.first_frame_button.pack(side=tk.LEFT, padx=(6, 0))
        self.detail_label = ttk.Label(controls, textvariable=self.detail_text, anchor=tk.W)
        self.detail_label.pack(side=tk.LEFT, padx=(14, 0), fill=tk.X, expand=True)

        progress_frame = ttk.Frame(self.root_window, padding=(10, 0, 10, 6))
        progress_frame.pack(fill=tk.X)
        ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=100).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Label(progress_frame, textvariable=self.progress_text, width=44, anchor=tk.W).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        status = ttk.Label(self.root_window, textvariable=self.status_text, anchor=tk.W, padding=(10, 0, 10, 10))
        status.pack(fill=tk.X)

    def _set_rewrite_controls(self, rewriting: bool) -> None:
        self.is_rewriting = rewriting
        state = tk.DISABLED if rewriting else tk.NORMAL
        for widget in (
            self.dataset_entry,
            self.browse_button,
            self.reload_button,
            self.mark_button,
            self.apply_button,
            self.play_button,
            self.first_frame_button,
        ):
            widget.configure(state=state)
        self.episode_tree.configure(selectmode="none" if rewriting else "browse")
        self.progress_text.set("Rewriting..." if rewriting else "Idle")
        if not rewriting:
            self.progress_value.set(0.0)

    def choose_dataset(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.dataset_root), title="Select LeRobot dataset root")
        if selected:
            self.dataset_text.set(selected)
            self.load_dataset(Path(selected))

    def load_dataset(self, dataset_root: Path) -> None:
        self.stop_playback()
        self.close_captures()
        self.dataset_root = dataset_root.resolve()
        self.dataset_text.set(str(self.dataset_root))
        try:
            self.info = json.loads((self.dataset_root / "meta" / "info.json").read_text())
            self.episodes_df = read_nested_parquets(self.dataset_root / "meta" / "episodes")
            self.episodes_df = self.episodes_df.sort_values("episode_index").reset_index(drop=True)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.video_keys = video_keys_from_info(self.info)
        self.episodes = [
            EpisodeRef(int(row["episode_index"]), row) for _, row in self.episodes_df.iterrows()
        ]
        self.deleted.clear()
        self._rebuild_episode_list()
        self._rebuild_video_grid()
        if self.episodes:
            self.episode_tree.selection_set(str(self.episodes[0].episode_index))
            self.select_episode(self.episodes[0])
        mismatch = ""
        if int(self.info.get("total_episodes", len(self.episodes))) != len(self.episodes):
            mismatch = f" Warning: info.json says {self.info.get('total_episodes')} episodes."
        self.status_text.set(
            f"Loaded {len(self.episodes)} metadata episode(s), {self.info.get('total_frames', '?')} frames, "
            f"{len(self.video_keys)} video stream(s).{mismatch}"
        )

    def _rebuild_episode_list(self) -> None:
        self.episode_tree.delete(*self.episode_tree.get_children())
        for ep in self.episodes:
            tasks = ep.row.get("tasks", [])
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            task_text = ", ".join(tasks) if isinstance(tasks, list) else str(tasks)
            self.episode_tree.insert(
                "",
                tk.END,
                iid=str(ep.episode_index),
                values=(ep.episode_index, "delete" if ep.episode_index in self.deleted else "keep", int(ep.row["length"]), task_text),
            )

    def _rebuild_video_grid(self) -> None:
        for child in self.video_frame.winfo_children():
            child.destroy()
        for i, key in enumerate(self.video_keys):
            box = ttk.Frame(self.video_frame, padding=4)
            box.grid(row=i // 2, column=i % 2, sticky="nsew")
            ttk.Label(box, text=key).pack(anchor=tk.W)
            label = ttk.Label(box, anchor=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            label.configure(image="")
            setattr(self, f"video_label_{i}", label)
        for col in range(2):
            self.video_frame.columnconfigure(col, weight=1)
        for row in range(max(1, (len(self.video_keys) + 1) // 2)):
            self.video_frame.rowconfigure(row, weight=1)

    def on_episode_selected(self, _event: tk.Event) -> None:
        selection = self.episode_tree.selection()
        if not selection:
            return
        ep_idx = int(selection[0])
        for ep in self.episodes:
            if ep.episode_index == ep_idx:
                self.select_episode(ep)
                break

    def select_episode(self, episode: EpisodeRef) -> None:
        self.stop_playback()
        self.close_captures()
        self.current_episode = episode
        row = episode.row
        details = [
            f"episode {episode.episode_index}",
            f"frames {int(row['length'])}",
            f"{float(row['length']) / int(self.info['fps']):.1f}s",
            f"dataset index {int(row['dataset_from_index'])}:{int(row['dataset_to_index'])}",
        ]
        self.detail_text.set(" | ".join(details))
        self.seek_current(0.0)

    def video_path_for(self, row: pd.Series, video_key: str) -> Path:
        return self.dataset_root / self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
            file_index=int(row[f"videos/{video_key}/file_index"]),
        )

    def open_capture(self, video_key: str) -> cv2.VideoCapture | None:
        if self.current_episode is None:
            return None
        if video_key in self.captures:
            return self.captures[video_key]
        path = self.video_path_for(self.current_episode.row, video_key)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.status_text.set(f"Failed to open video: {path}")
            return None
        self.captures[video_key] = cap
        return cap

    def seek_current(self, rel_s: float) -> None:
        if self.current_episode is None:
            return
        self.play_offset_s = max(0.0, rel_s)
        row = self.current_episode.row
        for i, video_key in enumerate(self.video_keys):
            cap = self.open_capture(video_key)
            if cap is None:
                continue
            start_s = float(row[f"videos/{video_key}/from_timestamp"])
            end_s = float(row[f"videos/{video_key}/to_timestamp"])
            target_s = min(start_s + self.play_offset_s, max(start_s, end_s - 1.0 / int(self.info["fps"])))
            cap.set(cv2.CAP_PROP_POS_MSEC, target_s * 1000.0)
            ok, frame = cap.read()
            if ok:
                self.show_frame(i, frame)

    def show_frame(self, index: int, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail(PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(image)
        self.current_images[str(index)] = photo
        label = getattr(self, f"video_label_{index}", None)
        if label is not None:
            label.configure(image=photo)

    def toggle_play(self) -> None:
        if self.playing:
            self.stop_playback()
        else:
            self.playing = True
            self.play_started_at = time.monotonic() - self.play_offset_s
            self.play_text.set("Pause")

    def stop_playback(self) -> None:
        self.playing = False
        self.play_text.set("Play")

    def toggle_delete_selected(self) -> None:
        selection = self.episode_tree.selection()
        if not selection:
            return
        ep_idx = int(selection[0])
        if ep_idx in self.deleted:
            self.deleted.remove(ep_idx)
        else:
            self.deleted.add(ep_idx)
        self._rebuild_episode_list()
        self.episode_tree.selection_set(str(ep_idx))
        self.status_text.set(f"Marked {len(self.deleted)} episode(s) for deletion.")

    def apply_delete(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        keep = [ep.episode_index for ep in self.episodes if ep.episode_index not in self.deleted]
        if not keep:
            messagebox.showerror("Refused", "At least one episode must be kept.")
            return
        if self.deleted:
            msg = (
                f"Delete {len(self.deleted)} episode(s) and rewrite the dataset with {len(keep)} kept episode(s)?\n\n"
                "A timestamped backup folder will be created next to the dataset."
            )
        else:
            msg = (
                f"Rewrite and reindex all {len(keep)} complete metadata episode(s)?\n\n"
                "This can repair stale info.json counts and skip unreadable unfinished parquet files. "
                "A timestamped backup folder will be created next to the dataset."
            )
        if not messagebox.askyesno("Confirm rewrite", msg):
            return

        self.stop_playback()
        self.close_captures()
        self.status_text.set("Starting rewrite...")
        self.progress_value.set(0.0)
        self.progress_text.set("Starting rewrite...")
        self._set_rewrite_controls(True)

        def worker() -> None:
            try:
                print("[dataset viewer] rewrite worker started", flush=True)
                backup = rewrite_dataset(self.dataset_root, keep, self.progress_queue)
                print(f"[dataset viewer] rewrite worker done, backup={backup}", flush=True)
                self.progress_queue.put(f"DONE|{backup}")
            except Exception as exc:
                traceback.print_exc()
                self.progress_queue.put(f"ERROR|{exc}")

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def close_captures(self) -> None:
        for cap in self.captures.values():
            cap.release()
        self.captures.clear()

    def _tick(self) -> None:
        while True:
            try:
                msg = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            if msg.startswith("DONE|"):
                backup = msg.split("|", 1)[1]
                self._set_rewrite_controls(False)
                self.status_text.set(f"Rewrite complete. Backup: {backup}")
                self.progress_value.set(100.0)
                self.progress_text.set("Complete")
                messagebox.showinfo("Rewrite complete", f"Dataset rewritten and reindexed.\nBackup: {backup}")
                self.load_dataset(self.dataset_root)
            elif msg.startswith("ERROR|"):
                self._set_rewrite_controls(False)
                self.status_text.set(msg.split("|", 1)[1])
                self.progress_text.set("Failed")
                messagebox.showerror("Rewrite failed", msg.split("|", 1)[1])
            elif msg.startswith("PROGRESS|"):
                _tag, current, total, detail = msg.split("|", 3)
                total_i = max(1, int(total))
                current_i = max(0, min(int(current), total_i))
                self.progress_value.set(current_i * 100.0 / total_i)
                self.progress_text.set(f"{current_i}/{total_i} {detail}")
                self.status_text.set(detail)
            else:
                self.status_text.set(msg)

        if self.playing and self.current_episode is not None:
            length_s = float(self.current_episode.row["length"]) / int(self.info["fps"])
            rel_s = time.monotonic() - self.play_started_at
            if rel_s >= length_s:
                self.stop_playback()
                rel_s = 0.0
            self.seek_current(rel_s)
        self.root_window.after(33, self._tick)

    def close(self) -> None:
        if self.is_rewriting or (self.worker and self.worker.is_alive()):
            force_close = messagebox.askyesno(
                "Rewrite in progress",
                "Dataset rewrite is still running.\n\n"
                "Waiting is recommended. Force closing may leave a temporary rewrite folder, "
                "but the original dataset is not replaced until the final backup step.\n\n"
                "Force close now?",
                default=messagebox.NO,
            )
            if not force_close:
                return
            print("[dataset viewer] force closing while rewrite is in progress", flush=True)
        self.stop_playback()
        self.close_captures()
        self.root_window.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse, delete, and reindex a local LeRobot dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT, help="LeRobot dataset root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    DatasetViewerApp(root, args.root)
    root.mainloop()


if __name__ == "__main__":
    main()
