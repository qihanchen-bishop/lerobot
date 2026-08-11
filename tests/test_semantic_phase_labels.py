import numpy as np

from mycode.semantic_phase_labels import _segment_episode, _soft_labels


def test_ordered_segmentation_recovers_five_piecewise_constant_phases():
    phase_length = 20
    centroids = np.arange(5, dtype=np.float32)[:, None]
    features = np.repeat(centroids, phase_length, axis=0)
    confidence = np.ones(features.shape[0], dtype=np.float32)

    boundaries, cost = _segment_episode(
        features,
        confidence,
        centroids,
        min_duration=5,
        time_prior_weight=0.1,
    )

    assert np.array_equal(boundaries, np.array([20, 40, 60, 80]))
    assert cost < 0.01


def test_soft_labels_blend_only_adjacent_phases_and_sum_to_one():
    probabilities = _soft_labels(25, np.array([5, 10, 15, 20]), width=2)

    assert probabilities.shape == (25, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[5, 0] == probabilities[5, 1] == 0.5
    assert np.count_nonzero(probabilities[5]) == 2
