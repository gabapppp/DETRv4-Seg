import numpy as np
import pytest

from scripts.export_onnx_bundle import compare_outputs


def test_detector_parity_accepts_joint_query_permutation():
    logits = np.array([[[1.0], [2.0], [3.0]]], np.float32)
    boxes = np.array(
        [[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]], np.float32
    )
    expected = [logits, boxes]
    actual = [value[:, [2, 0, 1]].copy() for value in expected]
    assert compare_outputs(expected, actual, ["logits", "pred_boxes"]) == {
        "logits": 0.0,
        "pred_boxes": 0.0,
    }
    # Matching must preserve the box/class pairing; a real class error cannot pass.
    actual[0][0, 0, 0] += 0.5
    with pytest.raises(AssertionError, match="parity failed for logits"):
        compare_outputs(expected, actual, ["logits", "pred_boxes"])


def test_mask_parity_does_not_permute_pixels():
    before = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    assert compare_outputs([before], [before.copy()], ["mask_logits"]) == {"mask_logits": 0.0}
    with pytest.raises(AssertionError, match="parity failed"):
        compare_outputs([before], [before[..., ::-1].copy()], ["mask_logits"])
    with pytest.raises(AssertionError, match="shape mismatch"):
        compare_outputs([before], [before.reshape(1, 1, 4)], ["mask_logits"])
