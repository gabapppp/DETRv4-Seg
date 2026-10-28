"""Small class-agnostic U-Net accepting RGB plus a bounding-box prior."""

import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs),
        nn.SiLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs),
        nn.SiLU(inplace=True),
    )


class TinyMaskHead(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.enc1 = block(4, width)
        self.enc2 = block(width, width * 2)
        self.center = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        center = self.center(F.avg_pool2d(e2, 2))
        d2 = self.dec2(
            torch.cat(
                (
                    F.interpolate(center, size=e2.shape[-2:], mode="bilinear", align_corners=False),
                    e2,
                ),
                dim=1,
            )
        )
        d1 = self.dec1(
            torch.cat(
                (F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1),
                dim=1,
            )
        )
        return self.out(d1)


class ResNet18MaskHead(nn.Module):
    """Pretrained four-channel encoder with P1–P5 skips and a full-resolution detail path."""

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        original = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3].copy_(original.weight)
            self.encoder.conv1.weight[:, 3:].zero_()
        self.encoder.fc = nn.Identity()
        self.detail = block(4, 16)
        self.dec4 = block(512 + 256, 128)
        self.dec3 = block(128 + 128, 64)
        self.dec2 = block(64 + 64, 48)
        self.dec1 = block(48 + 64, 32)
        self.dec0 = block(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained BN statistics stable on small microscopy ROI batches.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        normalized = torch.cat(((x[:, :3] - self.rgb_mean) / self.rgb_std, x[:, 3:]), 1)
        detail = self.detail(x)
        e1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(normalized)))
        e2 = self.encoder.layer1(self.encoder.maxpool(e1))
        e3 = self.encoder.layer2(e2)
        e4 = self.encoder.layer3(e3)
        e5 = self.encoder.layer4(e4)

        def decode(module, deep, skip):
            return module(
                torch.cat(
                    (
                        F.interpolate(
                            deep, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ),
                    1,
                )
            )

        d4 = decode(self.dec4, e5, e4)
        d3 = decode(self.dec3, d4, e3)
        d2 = decode(self.dec2, d3, e2)
        d1 = decode(self.dec1, d2, e1)
        return self.out(decode(self.dec0, d1, detail))


def make_mask_head(architecture="tiny", width=16, pretrained=False):
    if architecture == "tiny":
        return TinyMaskHead(width)
    if architecture == "resnet18_unet":
        return ResNet18MaskHead(pretrained)
    raise ValueError(f"Unsupported mask architecture: {architecture}")


def mask_loss(logits, target):
    # Float32 reductions keep Dice stable under FP16 autocast.
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (probabilities * target).sum(axes) + 1) / (
        probabilities.sum(axes) + target.sum(axes) + 1
    )
    return bce + 1 - dice.mean()
