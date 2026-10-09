import numpy as np
import pytest

from utils.evaluate import _aggregate_by_clip


def _frames(clip, n=3):
    return [f"/x/test/l_set/{clip}_{100 + i}.jpg" for i in range(n)]


def test_mean_aggregation_one_row_per_clip():
    paths = _frames("1_10") + _frames("1_20")
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array(
        [[.6, .4], [.6, .4], [.0, 1.], [.2, .8], [.2, .8], [.9, .1]]
    )
    t, p, s = _aggregate_by_clip(paths, y_true, scores, method="mean")
    assert t.tolist() == [0, 1]
    assert p.tolist() == [1, 1]  # clip 1: mean [.4,.6] -> class 1
    assert s.shape == (2, 2)


def test_vote_uses_majority():
    paths = _frames("1_10")
    scores = np.array([[.9, .1], [.8, .2], [.1, .9]])
    _, p, s = _aggregate_by_clip(paths, np.zeros(3, int), scores, method="vote")
    assert p.tolist() == [0]
    assert s[0].tolist() == pytest.approx([2 / 3, 1 / 3])


def test_mixed_labels_raise():
    with pytest.raises(ValueError):
        _aggregate_by_clip(_frames("1_10", 2), np.array([0, 1]), np.ones((2, 2)))


def test_bad_method_raises():
    with pytest.raises(ValueError):
        _aggregate_by_clip(_frames("1_10", 1), np.zeros(1, int), np.ones((1, 2)), method="x")
