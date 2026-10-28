import numpy as np
import pytest

from detrseg.quality import binary_iou, pairwise_overlap, perturb_box, summarize_values


def test_stability_does_not_confuse_empty_masks_with_accuracy():
    first = np.zeros((8, 8), np.uint8)
    assert binary_iou(first, first) == 1
    second = first.copy()
    second[2:4, 2:4] = 1
    assert binary_iou(first, second) == 0
    assert summarize_values([]) is None


def test_overlap_flags_shared_pixels_without_deleting_instances():
    first = np.zeros((8, 8), np.uint8)
    first[1:5, 1:5] = 1
    second = np.zeros_like(first)
    second[3:7, 3:7] = 1
    original = first.copy()
    largest, pairs = pairwise_overlap([first, second])
    assert largest.tolist() == [0.25, 0.25]
    assert pairs[0]["intersection_px"] == 4
    np.testing.assert_array_equal(first, original)


def test_box_perturbation_uses_original_axes():
    assert perturb_box([10, 20, 20, 10], scale=1.1) == pytest.approx([9, 19.5, 22, 11])
    assert perturb_box([10, 20, 20, 10], dx=0.05, dy=-0.05) == pytest.approx([11, 19.5, 20, 10])
