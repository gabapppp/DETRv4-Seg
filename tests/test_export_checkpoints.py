import json
from types import SimpleNamespace

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import torch

from detrseg import pipeline
from scripts.export_onnx_bundle import export_checkpoints, main


@pytest.mark.parametrize("native,dynamic", [(False, False), (True, False), (True, True)])
def test_checkpoint_export_creates_bundle_and_verified_onnx(tmp_path, monkeypatch, native, dynamic):
    detector = tmp_path / "det"
    detector.mkdir()
    (detector / "config.yml").write_text("num_classes: 1\n")
    (detector / "label_map.json").write_text(json.dumps({"0": 7}))
    (detector / "train.json").write_text(json.dumps({"categories": [{"id": 7, "name": "ball"}]}))
    torch.save({"label_map": {0: 7}}, detector / "best.pt")
    mask = tmp_path / "mask.pt"
    torch.save({"format_version": 1, "padding": 0.25}, mask)
    repo = tmp_path / "repo"
    repo.mkdir()

    class Detector(torch.nn.Module):
        def forward(self, images):
            score = images.mean((1, 2, 3))[:, None, None]
            return {
                "pred_logits": torch.cat((score, score + 1), dim=1),
                "pred_boxes": score.expand(-1, 2, 4).sigmoid(),
            }

    def predictor(*args):
        from detrseg.model import make_mask_head

        return SimpleNamespace(
            detector=SimpleNamespace(model=Detector(), size=64, mapping={0: 7}),
            mask_head=SimpleNamespace(
                model=make_mask_head("tiny_unet", width=4) if native else torch.nn.Conv2d(4, 1, 1),
                size=64 if native else 32,
                padding=0.25,
                mask_overflow=0.25,
                fill_mode="holes",
                roi_resize_mode="native" if native else "resize",
            ),
            mask_threshold=0.5,
        )

    monkeypatch.setattr(pipeline, "InstancePredictor", predictor)
    output = tmp_path / "export"
    result = export_checkpoints(detector, mask, output, repo, mask_dynamic_batch=dynamic)
    manifest = json.loads((output / "onnx/manifest.json").read_text())
    assert manifest["height"] == 64 and manifest["id2label"] == {"0": "ball"}
    assert manifest["fill_mode"] == "holes"
    assert manifest["roi_resize_mode"] == ("native" if native else "resize")
    assert manifest["mask_dynamic_batch"] == dynamic
    assert len(manifest["detector_sha256"]) == len(manifest["mask_sha256"]) == 64
    assert manifest["verification"]["mask_max_abs_error"]["mask_logits"] < 0.0005
    assert (output / "onnx/detector.onnx").is_file() and (output / "onnx/mask.onnx").is_file()
    state, paths = pipeline.bundle_paths(result["bundle"])
    assert state["categories"] == [{"id": 7, "name": "ball"}]
    assert paths["mask_checkpoint"] == str(mask)
    with pytest.raises(ValueError, match="new or empty"):
        export_checkpoints(detector, mask, output, repo)


def test_export_cli_requires_mask_with_detector_directory(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--detector-dir", "not_opened", "--output", str(tmp_path / "export")])
    assert error.value.code == 2
