import numpy as np
import pytest
import torch

from detrseg.detector_train import update_ema
from detrseg.infer import MaskPredictor
from detrseg.model import make_mask_head, mask_loss
from detrseg.pyramid import p2_config, transfer_p2


def test_p2_stage_transfer_does_not_shift_p5_semantics():
    old_offset = torch.arange(8 * 12 * 2 * 2).reshape(8 * 12 * 2, 2).float()
    old = {
        "encoder.input_proj.0.conv.weight": torch.ones(4, 8, 1, 1),
        "encoder.downsample_convs.0.weight": torch.ones(4),
        "encoder.lateral_convs.0.weight": torch.ones(4) * 2,
        "decoder.decoder.layers.0.cross_attn.sampling_offsets.weight": old_offset,
        "decoder.anchors": torch.ones(100, 4),
    }
    current = {
        "encoder.input_proj.1.conv.weight": torch.zeros(4, 8, 1, 1),
        "encoder.downsample_convs.1.weight": torch.zeros(4),
        "encoder.lateral_convs.0.weight": torch.zeros(4),
        "decoder.decoder.layers.0.cross_attn.sampling_offsets.weight": torch.zeros(8 * 15 * 2, 2),
        "decoder.anchors": torch.zeros(400, 4),
    }
    transferred = transfer_p2(old, current)
    assert transferred["encoder.input_proj.1.conv.weight"].sum() > 0
    assert torch.equal(transferred["decoder.anchors"], current["decoder.anchors"])
    new = transferred["decoder.decoder.layers.0.cross_attn.sampling_offsets.weight"].reshape(
        8, 15, 2, 2
    )
    assert torch.equal(new[:, 3:], old_offset.reshape(8, 12, 2, 2))
    assert p2_config()["DFINETransformer"]["feat_strides"] == [4, 8, 16, 32]


def test_resnet_mask_head_gradient_and_offline_checkpoint(tmp_path):
    pytest.importorskip("torchvision")
    torch.set_num_threads(1)
    model = make_mask_head("resnet18_unet", pretrained=False).train()
    assert not model.encoder.bn1.training
    inputs = torch.rand(2, 4, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 16:48, 16:48] = 1
    outputs = model(inputs)
    assert outputs.shape == target.shape
    loss = mask_loss(outputs, target)
    loss.backward()
    assert model.out.weight.grad is not None
    assert torch.isfinite(model.encoder.conv1.weight.grad).all()
    path = tmp_path / "resnet.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": "resnet18_unet",
            "model": model.state_dict(),
            "width": 16,
            "roi_size": 64,
            "padding": 0.25,
            "mask_overflow": 0.25,
        },
        path,
    )
    predictor = MaskPredictor(path, "cpu")
    masks = predictor.predict(np.full((48, 48, 3), 100, np.uint8), [[12, 12, 20, 20]])
    assert masks[0].shape == (48, 48)


def test_ema_keeps_infinite_geometry_buffers_valid():
    import copy

    student = torch.nn.BatchNorm1d(2)
    student.register_buffer("anchors", torch.tensor([float("inf"), 1.0]))
    student.register_parameter(
        "reg_scale", torch.nn.Parameter(torch.tensor(4), requires_grad=False)
    )
    average = copy.deepcopy(student)
    with torch.no_grad():
        student.weight.fill_(3.0)
        student.running_mean.fill_(4.0)
    update_ema(average, student, 0.5)
    torch.testing.assert_close(average.weight, torch.full((2,), 2.0))
    torch.testing.assert_close(average.running_mean, torch.full((2,), 2.0))
    assert torch.isinf(average.anchors[0]) and not torch.isnan(average.anchors).any()
    assert average.reg_scale.dtype == torch.int64 and average.reg_scale.item() == 4
