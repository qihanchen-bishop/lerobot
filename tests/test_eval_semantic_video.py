import io
import queue
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mycode'))
from eval_preview import AsyncSegmentationPreview  # noqa: E402
from eval_semantic_video import LiveSemanticRecorder, full_semantic_features, semantic_source_features  # noqa: E402
from gui_eval_lerobot_policy import EvalPolicyApp  # noqa: E402


class LiveSemanticRecordingTest(unittest.TestCase):
    def test_existing_live_results_are_saved_without_extra_inference(self):
        import av
        import pandas as pd

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        for storage in ('video', 'image'):
            with self.subTest(storage=storage), TemporaryDirectory() as directory:
                calls = []
                palette = [[0, 0, 0], [64, 160, 255]]

                class Model:
                    loaded_device = 'cpu'
                    def __init__(self, path, device, colors=palette):
                        self.checkpoint_path = path
                        self.labels = ['background', 'occluder']
                        self.model = SimpleNamespace(bundle_config={'palette_rgb': colors},
                                                     filter_policy={'min_area': 20}, model_sha256='test')
                        self.latest_class_ids = {}
                    def predict(self, images, counter=calls):
                        view, image = next(iter(images.items()))
                        counter.append(view)
                        self.latest_class_ids[view] = (image[..., 0] > 127).astype(np.uint8)
                        # Preview annotations must never enter saved semantic maps.
                        return {view: np.full_like(image, 200)}, {}, {}

                root = Path(directory) / 'episode'
                cameras = {f'observation.images.{v}': {'dtype': storage, 'shape': (24, 32, 3),
                           'names': ['height', 'width', 'channels']} for v in ('front', 'side')}
                full = full_semantic_features(cameras, list(cameras))
                held = {f'{key}_policy_segmentation': dict(value) for key, value in cameras.items()}
                features = {'action': {'dtype': 'float32', 'shape': (1,), 'names': ['a']},
                            'observation.state': {'dtype': 'float32', 'shape': (1,), 'names': ['a']},
                            **cameras, **full, **held, **semantic_source_features(full)}
                recorder = LiveSemanticRecorder(full, 1.)
                worker = AsyncSegmentationPreview(Model, interval_s=0)
                dataset = LeRobotDataset.create('local/live-semantic-test', 30, root=root, features=features,
                                               use_videos=storage == 'video', batch_encoding_size=1000000,
                                               image_writer_threads=2)
                expected = {}
                try:
                    for index in range(5):
                        images = {}
                        observation = {}
                        for view in ('front', 'side'):
                            image = np.zeros((24, 32, 3), np.uint8)
                            image[3:9, 2 + index:8 + index, 0] = 255
                            images[view] = image
                            observation[f'observation.images.{view}'] = image
                        semantic = recorder.frame(observation)
                        for key in full:
                            expected[key, index] = semantic[key].copy()
                        dataset.add_frame({'action': np.zeros(1, np.float32),
                                           'observation.state': np.zeros(1, np.float32), 'task': 'test',
                                           **observation, **semantic,
                                           **{key: np.zeros((24, 32, 3), np.uint8) for key in held}})
                        if index in (0, 2):
                            worker.submit(images, {'front': 'front', 'side': 'side'}, 'cpu', 1.,
                                          recorder=recorder,
                                          source={'run_token': 1., 'frame_index': index, 'timestamp_s': index / 30})
                            worker.future.result(timeout=3)
                            worker.poll()
                    report = recorder.finish()
                    self.assertEqual(len(calls), 4)  # two updates x two views; no offline calls
                    self.assertEqual(report['additional_segmentation_calls'], 0)
                    self.assertEqual(report['samples_by_view'], {'front': 2, 'side': 2})
                    dataset.save_episode()
                finally:
                    worker.close()
                    dataset.finalize()
                meta = LeRobotDatasetMetadata('local/live-semantic-test', root=root)
                self.assertEqual(meta.total_frames, 5)
                data = pd.read_parquet(next((root / 'data').rglob('*.parquet')))
                for view in ('front', 'side'):
                    self.assertEqual(data[f'observation.semantic.{view}.source_frame_index'].astype(int).tolist(),
                                     [-1, 0, 0, 2, 2])
                    self.assertEqual(data[f'observation.semantic.{view}.valid'].astype(bool).tolist(),
                                     [False, True, True, True, True])
                    key = f'observation.images.{view}_semantic'
                    np.testing.assert_array_equal(expected[key, 1], expected[key, 2])
                    self.assertFalse(np.array_equal(expected[key, 2], expected[key, 3]))
                    self.assertGreater(float(meta.stats[key]['mean'].sum()), 0)
                    if storage == 'image':
                        with Image.open(io.BytesIO(data.iloc[4][key]['bytes'])) as image:
                            np.testing.assert_array_equal(image, expected[key, 4])
                if storage == 'video':
                    app = object.__new__(EvalPolicyApp)
                    app.log_queue = queue.Queue()
                    app._encode_confirmed_run_videos({'save_video': True, 'dataset_root': str(root),
                                                     'dataset_repo_id': 'local/live-semantic-test'})
                    meta = LeRobotDatasetMetadata('local/live-semantic-test', root=root)
                    for key in (*cameras, *full, *held):
                        with av.open(str(root / meta.get_video_file_path(0, key))) as video:
                            self.assertEqual(len(list(video.decode(video=0))), 5)
                    self.assertEqual(len(calls), 4)  # saving/encoding never invokes segmentation

    def test_previous_run_and_late_results_cannot_change_recording(self):
        full = {'observation.images.front_semantic': {}}
        recorder = LiveSemanticRecorder(full, 2.)
        observation = {'observation.images.front': np.zeros((2, 2, 3), np.uint8)}
        model = SimpleNamespace(checkpoint_path='front', labels=['background', 'object'], model=object())
        ids = np.ones((2, 2), np.uint8)
        source = {'run_token': 1., 'frame_index': 9, 'timestamp_s': .3}
        recorder.update('front', ids, [[0, 0, 0], [255, 0, 0]], source, model)
        self.assertFalse(recorder.frame(observation)['observation.semantic.front.valid'][0])
        recorder.finish()
        recorder.update('front', ids, [[0, 0, 0], [255, 0, 0]], {**source, 'run_token': 2.}, model)
        self.assertFalse(recorder.frame(observation)['observation.semantic.front.valid'][0])


if __name__ == '__main__':
    unittest.main()
