"""Position provenance and dataset cleanup checks; no hardware is connected."""

import csv
import json
import queue
from types import SimpleNamespace

import pytest

from mycode import gui_record_so101_bimanual as recorder_module
from mycode.episode_positions import (
    POSITIONS_PATH,
    export_position_counts,
    grid_position,
    load_episode_positions,
    position_counts,
    reindex_episode_positions,
    save_episode_position,
)


def label(index, cell="(0,0)", task="cube", rows=3, cols=4):
    return {**grid_position(rows, cols, cell), "episode_index": index, "task": task}


def test_old_datasets_are_unlabeled_and_invalid_selections_are_rejected(tmp_path):
    assert load_episode_positions(tmp_path) == {}
    for rows, cols, cell in [(3, 4, ""), (3, 4, "(4,0)"), (3, 4, "(0,3)"), (0, 4, "(0,0)")]:
        with pytest.raises(ValueError):
            grid_position(rows, cols, cell)


def test_reindex_and_export_keep_task_and_grid_shapes_separate(tmp_path):
    root = tmp_path / "dataset"
    destination = tmp_path / "rewritten"
    save_episode_position(root, label(0))
    save_episode_position(root, label(1, "(1,2)"))
    save_episode_position(root, label(2, "(1,2)", rows=4))
    save_episode_position(root, label(3, "(1,2)", task="screw"))
    reindex_episode_positions(root, destination, {1: 0, 2: 1, 3: 2})
    records = load_episode_positions(destination)
    assert records[0]["grid_cell"] == "(1,2)"
    assert len(load_episode_positions(root)) == 4  # Source/backup is unchanged.
    assert len(position_counts(records)) == 3
    csv_path = tmp_path / "counts.csv"
    export_position_counts(csv_path, records)
    with csv_path.open(encoding="utf-8-sig") as stream:
        counts = list(csv.DictReader(stream))
    assert sum(int(row["episodes"]) for row in counts) == 3
    reindex_episode_positions(destination, destination, {1: 0})
    assert load_episode_positions(destination)[0]["grid_rows"] == 4


def test_corrupt_or_duplicate_labels_fail_instead_of_silent_miscount(tmp_path):
    save_episode_position(tmp_path, label(0))
    path = tmp_path / POSITIONS_PATH
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_episode_positions(tmp_path)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    instance = recorder_module.BimanualRecorder(
        SimpleNamespace(task="cube", capture_depth_sidecar=False), queue.Queue(), queue.Queue(), queue.Queue()
    )
    instance.ready = True
    instance.dataset = SimpleNamespace(root=tmp_path, num_episodes=0, episode_buffer={"size": 2})
    instance.dataset.save_episode = lambda: setattr(
        instance.dataset, "num_episodes", instance.dataset.num_episodes + 1
    )
    instance.dataset.clear_episode_buffer = lambda **kwargs: None
    monkeypatch.setattr(instance, "_delete_temporary_episode_images", lambda index: None)
    monkeypatch.setattr(instance, "_validate_temporary_video_frames", lambda index, count: None)
    return instance


def test_position_is_frozen_at_record_and_preserved_across_background_save(recorder, monkeypatch):
    selected = grid_position(3, 4, "(2,1)")
    recorder.request_record("cube", selected)
    selected.update(grid_position(3, 4, "(0,0)"))
    recorder._process_commands()
    assert recorder.current_episode_position["grid_cell"] == "(2,1)"

    class DeferredThread:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            pass

    monkeypatch.setattr(recorder_module.threading, "Thread", DeferredThread)
    recorder.recording = False
    recorder._begin_episode_save()
    recorder.current_episode_position = grid_position(3, 4, "(3,2)")
    recorder.current_episode_task = "screw"
    recorder.save_thread.target(*recorder.save_thread.args)
    record = load_episode_positions(recorder.dataset.root)[0]
    assert record["grid_cell"] == "(2,1)"
    assert record["task"] == "cube"
    assert record["frame_count"] == 2
    assert not recorder.failed


def test_discard_and_failed_save_do_not_write_labels(recorder):
    recorder.request_record("cube", grid_position(3, 4, "(2,1)"))
    recorder._process_commands()
    recorder._discard_episode_if_needed()
    assert load_episode_positions(recorder.dataset.root) == {}

    def fail_save():
        raise OSError("save failed")

    recorder.dataset.save_episode = fail_save
    recorder._save_episode_worker(0, 2, [], {}, 0, [], label(0))
    assert recorder.failed
    assert load_episode_positions(recorder.dataset.root) == {}


def test_full_rewrite_fallback_reindexes_position_labels(tmp_path, monkeypatch):
    import pandas as pd

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"features": {}}))
    save_episode_position(root, label(0))
    save_episode_position(root, label(1, "(3,2)"))
    monkeypatch.setattr(
        recorder_module, "read_nested_parquets", lambda path: pd.DataFrame({"episode_index": [0, 1]})
    )
    monkeypatch.setattr(recorder_module, "_can_fast_reindex", lambda *args: (False, "shared files"))

    def rewrite(source, keep, progress):
        backup = tmp_path / "backup"
        source.rename(backup)
        source.mkdir()
        return backup

    monkeypatch.setattr(recorder_module, "rewrite_dataset", rewrite)
    recorder_module.fast_delete_or_rewrite_dataset(root, [1], queue.Queue())
    assert load_episode_positions(root)[0]["grid_cell"] == "(3,2)"
    assert len(load_episode_positions(tmp_path / "backup")) == 2


def test_fast_delete_reindexes_labels_without_altering_backup(tmp_path):
    import pandas as pd

    root = tmp_path / "dataset"
    info = {
        "fps": 30,
        "total_episodes": 3,
        "total_frames": 6,
        "features": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    for index in range(3):
        (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"episode_index": [index] * 2, "frame_index": [0, 1], "index": [index * 2, index * 2 + 1]}
        ).to_parquet(root / f"data/chunk-000/file-{index:03d}.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "episode_index": index,
                    "length": 2,
                    "data/chunk_index": 0,
                    "data/file_index": index,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": index,
                    "dataset_from_index": index * 2,
                    "dataset_to_index": index * 2 + 2,
                }
            ]
        ).to_parquet(root / f"meta/episodes/chunk-000/file-{index:03d}.parquet", index=False)
        save_episode_position(root, label(index, f"({index},0)"))
    (root / "meta/info.json").write_text(json.dumps(info))
    outcome = recorder_module.fast_delete_or_rewrite_dataset(root, [0, 2], queue.Queue())
    assert outcome.mode == "fast suffix reindex"
    assert load_episode_positions(root)[1]["grid_cell"] == "(2,0)"
    assert set(load_episode_positions(root)) == {0, 1}
    assert set(load_episode_positions(outcome.backup_root)) == {0, 1, 2}
