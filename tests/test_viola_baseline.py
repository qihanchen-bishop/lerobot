"""Run using the isolated VIOLA environment: python -m unittest tests.test_viola_baseline."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import h5py
import pandas as pd
import torch

from mycode.viola_baseline import (
    BOX_KEY,
    build_model,
    denormalize_action,
    normalizer,
    normalize,
    proposals,
    resize_rgb,
    seed_all,
    select_episodes,
    sequence_dataset,
)


class AdapterTests(unittest.TestCase):
    def test_normalization_roundtrip(self):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        stats = normalizer(data, data * 2 - 50)
        actions = data * 2 - 50
        normalized = normalize(actions, stats, "action")
        self.assertAlmostEqual(float(normalized.min()), -1.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)
        np.testing.assert_allclose(denormalize_action(normalized, stats), actions, atol=1e-5)
        np.testing.assert_allclose(normalize(data, stats, "state").mean(0), 0, atol=1e-6)
        # Out-of-training-range targets are not clipped to conceal validation error.
        self.assertGreater(normalize(actions + 1000, stats, "action").max(), 1)

    def test_constant_joint_is_finite(self):
        data = np.ones((20, 10), np.float32)
        stats = normalizer(data, data)
        np.testing.assert_array_equal(normalize(data, stats, "state"), np.zeros_like(data))
        np.testing.assert_array_equal(denormalize_action(normalize(data, stats, "action"), stats), data)

    def test_episode_split_no_overlap(self):
        rows = pd.DataFrame({"episode_index": range(18), "task": ["a"] * 6 + ["b"] * 6 + ["c"] * 6})
        selected, valid = select_episodes(rows, 4, 1)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(valid), 3)
        train = set(selected) - set(valid)
        self.assertFalse(train & set(valid))
        self.assertEqual(len(train), 9)
        with self.assertRaises(ValueError):
            select_episodes(rows, 1, 1)

    def test_rgb_and_full_view_resize(self):
        image = np.zeros((360, 640, 3), np.uint8)
        image[:] = (10, 20, 200)
        result = resize_rgb(image)
        self.assertEqual(result.shape, (128, 128, 3))
        np.testing.assert_array_equal(result[64, 64], [200, 20, 10])

    def test_no_fabricated_box_padding(self):
        from detectron2.structures import Boxes

        box = Boxes(torch.tensor([[10.0, 10.0, 20.0, 20.0]]))

        class Output:
            proposal_boxes = box

        with self.assertRaisesRegex(RuntimeError, "refusing"):
            proposals(lambda image: {"proposals": Output()}, np.zeros((128, 128, 3), np.uint8))

    def test_original_sequence_loader_padding_and_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.hdf5"
            with h5py.File(path, "w") as f:
                for episode in range(2):
                    group = f.create_group(f"data/demo_{episode}")
                    group.attrs["num_samples"] = 3
                    group.create_dataset("actions", data=np.full((3, 10), episode, np.float32))
                    obs = group.create_group("obs")
                    for key in ("agentview_rgb", "eye_in_hand_rgb"):
                        obs.create_dataset(key, data=np.full((3, 128, 128, 3), 255, np.uint8))
                    obs.create_dataset("joint_states", data=np.full((3, 10), episode, np.float32))
                    obs.create_dataset(
                        BOX_KEY, data=np.tile(np.array([1, 1, 20, 20], np.float32), (3, 15, 1))
                    )
                f.create_dataset("mask/train", data=np.array([b"demo_0", b"demo_1"]))
            dataset = sequence_dataset(path, "train")
            last_first, first_next = dataset[2], dataset[3]
            np.testing.assert_array_equal(last_first["actions"], np.zeros((10, 10)))
            np.testing.assert_array_equal(first_next["actions"], np.ones((10, 10)))
            self.assertEqual(last_first["obs"]["agentview_rgb"].shape, (10, 3, 128, 128))
            np.testing.assert_array_equal(last_first["obs"]["agentview_rgb"], 1.0)
            dataset.close_and_delete_hdf5_handle()


class OfficialNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(1000)
        torch.set_num_threads(2)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model, cls.config = build_model(cls.device)

    def data(self, steps=4):
        return {
            "obs": {
                "agentview_rgb": torch.rand(1, steps, 3, 128, 128, device=self.device),
                "eye_in_hand_rgb": torch.rand(1, steps, 3, 128, 128, device=self.device),
                "joint_states": torch.randn(1, steps, 10, device=self.device),
                BOX_KEY: torch.tensor([20.0, 20.0, 60.0, 60.0], device=self.device).repeat(1, steps, 15, 1),
            }
        }

    def test_official_network_backward(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        dist = self.model(self.data())
        self.assertEqual(dist.mean.shape, (1, 4, 10))
        loss = -dist.log_prob(torch.zeros(1, 4, 10, device=self.device)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_causal_attention(self):
        self.model.eval()
        original = self.data()
        changed = copy.deepcopy(original)
        for value in changed["obs"].values():
            value[:, 2:] = 0
        changed["obs"][BOX_KEY][:, 2:] = torch.tensor([1.0, 1.0, 10.0, 10.0], device=self.device)
        with torch.no_grad():
            a, b = self.model(original).mean, self.model(changed).mean
        torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-5)

    def test_streaming_matches_batched_history(self):
        self.model.eval()
        self.model.reset()
        data = self.data()
        with torch.no_grad():
            expected = self.model(copy.deepcopy(data)).mean[0, -1].cpu().numpy()
        # Only replace stochastic sampling with its mean for this parity assertion.
        with patch.object(torch.distributions.MixtureSameFamily, "sample", lambda dist: dist.mean):
            for index in range(4):
                obs = {key: value[0, index].clone() for key, value in data["obs"].items()}
                actual = self.model.get_action({"obs": obs})
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
        self.model.reset()
        self.assertEqual(self.model.queue, [])


if __name__ == "__main__":
    unittest.main()
