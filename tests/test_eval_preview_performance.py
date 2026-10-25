from collections import deque
from pathlib import Path
import os
import json
import queue
import sys
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mycode"))
from eval_preview import AsyncSegmentationPreview
from gui_eval_lerobot_policy import EvalPolicyApp
from train_mask_act_policy import MaskACTPolicy


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class EvalPreviewPerformanceTest(unittest.TestCase):
    def app(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {key: Variable(value) for key, value in {
            "task": "cube", "segmentation_model_path": "front", "segmentation_side_model_path": "side",
            "segmentation_device": "auto",
        }.items()}
        app.enable_segmentation_preview = Variable(False)
        app.eval_running = app.eval_finishing = app.saving_result = False
        app.started_at = None
        app.segmentation_worker = Mock()
        app.segmentation_loaded_logged = set()
        app.segmentation_error_logged = False
        app.segmentation_eval_metrics = app._new_segmentation_eval_metrics()
        app.model_mask_signature = None
        app.model_mask_cache = ({}, {})
        for field in ("preview_labels", "current_images", "current_segmentation_images",
                      "segmentation_preview_labels", "segmentation_summary_labels",
                      "current_model_mask_images", "model_mask_preview_labels", "model_mask_summary_labels"):
            setattr(app, field, {})
        app.log_text = Mock()
        app._append_log = Mock()
        return app

    def test_feed_only_while_evaluating_or_manually_previewing(self):
        app = self.app()
        for running, preview, saving, finishing, expected in (
            (False, False, False, False, 0), (False, True, False, False, 1),
            (True, False, False, False, 1), (True, True, True, False, 0),
            (True, True, False, True, 0),
        ):
            app.eval_running, app.saving_result, app.eval_finishing = running, saving, finishing
            app.enable_segmentation_preview.set(preview)
            app.segmentation_worker.submit.reset_mock()
            app._update_preview({"front": np.zeros((2, 2, 3), np.uint8)}, {}, 30., "", {})
            self.assertEqual(app.segmentation_worker.submit.call_count, expected)

    def test_continuous_camera_producer_cannot_trap_gui_in_preview_loop(self):
        app = self.app()
        app.preview_queue = queue.Queue(maxsize=1)
        item = ({}, {}, 30., "", {})
        app.preview_queue.put(item)
        def render(*_args):
            app.preview_queue.put(item)  # a producer replaces the frame during rendering
        app._update_preview = Mock(side_effect=render)
        app._consume_preview()
        self.assertEqual(app._update_preview.call_count, 1)
        self.assertEqual(app.preview_queue.qsize(), 1)

    def test_live_segmentation_receives_the_recorded_rgb_source_identity(self):
        app = self.app()
        app.eval_running = True
        app.started_at = 7.
        app.semantic_recorder = object()
        source = {"run_token": 7., "frame_index": 12, "timestamp_s": .4}
        images = {"front": np.zeros((2, 2, 3), np.uint8), "_semantic_source": source}
        app._update_preview(images, {}, 30., "", {})
        self.assertIs(app.segmentation_worker.submit.call_args.kwargs["source"], source)
        self.assertIs(app.segmentation_worker.submit.call_args.kwargs["recorder"], app.semantic_recorder)

    def test_worker_is_bounded_and_runs_outside_the_gui_thread(self):
        entered, release = threading.Event(), threading.Event()
        threads = []
        class Model:
            loaded_device = "cpu"
            def __init__(self, path, device):
                pass
            def predict(self, images):
                threads.append(threading.get_ident())
                entered.set()
                release.wait(2)
                return {}, {}, {"front": {"valid": True}}
        worker = AsyncSegmentationPreview(Model, interval_s=0)
        try:
            self.assertTrue(worker.submit({"front": object()}, {"front": "path"}, "cpu", 1.))
            self.assertTrue(entered.wait(2))
            self.assertNotEqual(threads[0], threading.get_ident())
            self.assertFalse(worker.submit({"front": object()}, {"front": "path"}, "cpu", 1.))
            self.assertIsNone(worker.poll())
            release.set()
            worker.future.result(timeout=2)
            self.assertIsNotNone(worker.poll())
        finally:
            release.set()
            worker.close()

    def test_stale_configuration_results_are_dropped_and_models_stay_loaded(self):
        created = []
        class Model:
            loaded_device = "cpu"
            def __init__(self, path, device):
                created.append(path)
            def predict(self, images):
                return {}, {}, {}
        worker = AsyncSegmentationPreview(Model, interval_s=0)
        try:
            worker.submit({"front": 1}, {"front": "path"}, "cpu", 1.)
            worker.future.result(timeout=2)
            worker.reset()
            self.assertIsNone(worker.poll())
            worker.submit({"front": 2}, {"front": "path"}, "cpu", 2.)
            worker.future.result(timeout=2)
            self.assertIsNotNone(worker.poll())
            self.assertEqual(created, ["path"])
        finally:
            worker.close()

    def test_measurements_from_an_earlier_trial_cannot_change_current_stage(self):
        app = self.app()
        app.eval_running, app.started_at = True, 2.
        app._update_segmentation_eval_metrics = Mock()
        result = ({}, {}, {"front": {"valid": True}}, {})
        app.segmentation_worker.poll.return_value = ((0, 1., 1.5), result)
        app._poll_segmentation_preview()
        app._update_segmentation_eval_metrics.assert_not_called()
        app.segmentation_worker.poll.return_value = ((0, 2., 2.5), result)
        app._poll_segmentation_preview()
        app._update_segmentation_eval_metrics.assert_called_once_with(result[2], sampled_at=2.5)

    def test_cached_mask_act_action_does_not_repeat_segmentation_or_forward(self):
        policy = object.__new__(MaskACTPolicy)
        torch.nn.Module.__init__(policy)
        policy.experiment = "FINAL-VIEWQTOKEN"
        first, second = torch.tensor([1.]), torch.tensor([2.])
        policy.act_policy = SimpleNamespace(_action_queue=deque([first, second]),
                                            config=SimpleNamespace(temporal_ensemble_coeff=None))
        policy._prepare_inference_batch = Mock(side_effect=AssertionError("unexpected segmentation"))
        self.assertIs(policy.select_action({}), first)
        self.assertIs(policy.select_action({}), second)
        policy._prepare_inference_batch.assert_not_called()

    def test_confirmed_video_encoding_preserves_all_streams_frames_and_metadata(self):
        import av
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        parent_priority = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None
        with TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            keys = [f"observation.images.{view}" for view in ("front", "side", "front_policy_input", "side_policy_input")]
            features = {"action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
                        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}}
            features.update({key: {"dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channels"]} for key in keys})
            dataset = LeRobotDataset.create("local/encoding-test", 30, root=root, features=features,
                                           use_videos=True, batch_encoding_size=1000000, image_writer_threads=2)
            try:
                for index in range(5):
                    image = np.full((24, 32, 3), index * 30, np.uint8)
                    dataset.add_frame({"action": np.zeros(2, np.float32), "observation.state": np.zeros(2, np.float32),
                                       **{key: image for key in keys}, "task": "test"})
                dataset.save_episode()
            finally:
                EvalPolicyApp._finalize_eval_dataset(dataset)
            app = self.app()
            app.log_queue = queue.Queue()
            record = {"save_video": True, "dataset_root": str(root), "dataset_repo_id": "local/encoding-test"}
            result_path = Path(directory) / "eval_results.jsonl"
            app._commit_result_save(record, result_path)
            self.assertTrue(json.loads(result_path.read_text())["save_video"])
            meta = LeRobotDatasetMetadata("local/encoding-test", root=root)
            for key in keys:
                path = root / meta.get_video_file_path(0, key)
                with av.open(str(path)) as video:
                    frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 5)
                    self.assertEqual((frames[0].height, frames[0].width), (24, 32))
                    for index, frame in enumerate(frames):
                        self.assertAlmostEqual(float(frame.pts * frame.time_base), index / 30, places=5)
                        self.assertLess(abs(float(frame.to_ndarray(format="rgb24").mean()) - index * 30), 5)
            self.assertFalse(list(root.rglob("frame-*.png")))
            self.assertEqual(record["video_encoding"]["parallel_streams"], 1)
            self.assertTrue(record["video_encoding"]["process_isolation"])
            self.assertEqual(set(record["video_encoding"]["stream_seconds"]), set(keys))
            # An already committed episode must be safe to confirm a second time.
            app._encode_confirmed_run_videos(record)
        if parent_priority is not None:
            self.assertEqual(os.getpriority(os.PRIO_PROCESS, 0), parent_priority)

    def test_encoder_subprocess_failure_keeps_source_frames_and_reports_error(self):
        from eval_video import encode_eval_video_in_subprocess

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame-000000.png"
            source.write_bytes(b"invalid PNG")
            with self.assertRaisesRegex(RuntimeError, "Video encoding failed"):
                encode_eval_video_in_subprocess(root, root / "video.mp4", 30)
            self.assertEqual(source.read_bytes(), b"invalid PNG")
            self.assertFalse((root / "video.mp4").exists())

    def test_post_run_image_choice_preserves_portable_images_and_labels_without_encoding(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from eval_video import save_pending_episode_as_images

        with TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            keys = [f"observation.images.{view}" for view in ("front", "side", "front_semantic", "front_policy_input")]
            features = {"action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}}
            features.update({key: {"dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channels"]} for key in keys})
            dataset = LeRobotDataset.create("local/image-choice", 30, root=root, features=features,
                                           use_videos=True, batch_encoding_size=1000000, image_writer_threads=2)
            try:
                for index in range(17):
                    dataset.add_frame({"action": np.array([index, -index], np.float32),
                                       **{key: np.full((24, 32, 3), index * 10, np.uint8) for key in keys}, "task": "test"})
                dataset.save_episode()
            finally:
                EvalPolicyApp._finalize_eval_dataset(dataset)
            record = {"save_video": False, "dataset_root": str(root), "dataset_repo_id": "local/image-choice",
                      "result": "success", "full_semantic_recording_format": "pending"}
            (root / "evaluation_metadata.json").write_text(json.dumps({"save_video": None}))
            app = self.app()
            app.log_queue = queue.Queue()
            result_path = Path(directory) / "eval_results.jsonl"
            with patch("eval_video.encode_eval_video_in_subprocess", side_effect=AssertionError("MP4 encoder called")):
                app._commit_result_save(record, result_path)
                save_pending_episode_as_images(root, "local/image-choice")  # retry is harmless
            saved = json.loads(result_path.read_text())
            self.assertFalse(saved["save_video"])
            self.assertEqual(saved["result"], "success")
            self.assertEqual(saved["full_semantic_recording_format"], "image_frames")
            self.assertFalse(json.loads((root / "evaluation_metadata.json").read_text())["save_video"])
            self.assertFalse(list(root.rglob("*.mp4")))
            self.assertFalse(list(root.rglob("frame-*.png")))  # PNG bytes now live inside parquet
            moved = Path(directory) / "relocated"
            root.rename(moved)
            meta = LeRobotDatasetMetadata("local/image-choice", root=moved)
            self.assertEqual(meta.video_keys, [])
            self.assertEqual(set(meta.image_keys), set(keys))
            loaded = LeRobotDataset("local/image-choice", root=moved)
            self.assertEqual(len(loaded), 17)
            for index in range(17):
                frame = loaded[index]
                np.testing.assert_array_equal(frame["action"], [index, -index])
                self.assertAlmostEqual(frame["timestamp"].item(), index / 30, places=6)
                for key in keys:
                    self.assertEqual(tuple(frame[key].shape), (3, 24, 32))
                    np.testing.assert_allclose(frame[key], index * 10 / 255, atol=1e-7)

    def test_image_save_failure_retains_original_recording_for_retry(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from eval_video import save_pending_episode_as_images

        with TemporaryDirectory() as directory:
            root = Path(directory) / "episode"
            key = "observation.images.front"
            dataset = LeRobotDataset.create("local/image-retry", 30, root=root,
                features={key: {"dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channels"]}},
                use_videos=True, batch_encoding_size=1000000)
            dataset.add_frame({key: np.zeros((24, 32, 3), np.uint8), "task": "test"})
            dataset.save_episode()
            EvalPolicyApp._finalize_eval_dataset(dataset)
            data_path = next((root / "data").rglob("*.parquet"))
            original_data, original_info = data_path.read_bytes(), (root / "meta/info.json").read_bytes()
            with patch.object(Path, "read_bytes", side_effect=OSError("simulated I/O error")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    save_pending_episode_as_images(root, "local/image-retry")
            self.assertEqual(data_path.read_bytes(), original_data)
            self.assertEqual((root / "meta/info.json").read_bytes(), original_info)
            self.assertEqual(len(list(root.rglob("frame-*.png"))), 1)
            save_pending_episode_as_images(root, "local/image-retry")

    def test_each_new_result_defaults_to_no_video_and_save_uses_current_choice(self):
        app = self.app()
        app.save_video = Variable(True)
        app.deviation_selected = {}
        for key in ("result", "object_exposed", "object_pushed_out", "object_push_success", "result_tags"):
            app.vars[key] = Variable("")
        app._reset_result_fields()
        self.assertFalse(app.save_video.get())
        with TemporaryDirectory() as directory:
            app.vars.update({"policy_type": Variable("act"), "dataset_root": Variable(directory)})
            app.last_run = {"policy_type": "act", "save_parent": directory, "task": "cube",
                            "grid_cell": "0,0", "grid_rows": 3, "grid_cols": 4, "save_video": None}
            for choice in (False, True, False):
                app.save_video.set(choice)
                record, _ = app._prepare_result_save()
                self.assertEqual(record["save_video"], choice)

    def test_repeated_trials_release_image_workers_and_flush_pending_frames(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image

        baseline = set(threading.enumerate())
        with TemporaryDirectory() as directory:
            for trial in range(31):
                dataset = LeRobotDataset.create(
                    "local/worker-cleanup-test", 30, root=Path(directory) / str(trial),
                    features={"action": {"dtype": "float32", "shape": (1,), "names": ["a"]}},
                    image_writer_threads=4,
                )
                writer = dataset.image_writer
                target = dataset.root / "pending.png"
                writer.save_image(np.full((24, 32, 3), trial, np.uint8), target)
                try:
                    EvalPolicyApp._finalize_eval_dataset(dataset)
                    self.assertTrue(all(not thread.is_alive() for thread in writer.threads))
                    self.assertIsNone(dataset.image_writer)
                    with Image.open(target) as saved:
                        np.testing.assert_array_equal(np.asarray(saved), np.full((24, 32, 3), trial, np.uint8))
                    self.assertEqual(set(threading.enumerate()) - baseline, set())
                finally:
                    writer.stop()

    def test_dataset_files_are_closed_even_if_image_writer_stop_fails(self):
        dataset = Mock()
        dataset.stop_image_writer.side_effect = RuntimeError("writer failed")
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            EvalPolicyApp._finalize_eval_dataset(dataset)
        dataset.finalize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
