"""Custom image-space reconstruction loss for 3D volumes.

    Loss = l1_weight * L1  +  ssim_weight * (1 - SSIM3D)  +  perceptual_weight * Perc3D

All three terms operate on a (pred, target) pair of image-space tensors shaped
(B, C, D, H, W); the loss is use-case agnostic as long as the data is 3D.

The perceptual term uses a frozen 3D ResNet backbone (Tencent MedicalNet,
Med3D) as the feature extractor. See README note at the bottom for how to fetch
the pretrained weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D SSIM
# ---------------------------------------------------------------------------
def _gaussian_window_3d(window_size: int, sigma: float, channels: int,
                        device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return w.expand(channels, 1, window_size, window_size, window_size).contiguous()


def ssim3d(x: torch.Tensor, y: torch.Tensor, window_size: int = 7,
           sigma: float = 1.5, data_range: float = 2.0) -> torch.Tensor:
    """Mean 3D SSIM. data_range=2.0 matches tanh outputs in [-1, 1]."""
    c = x.shape[1]
    w = _gaussian_window_3d(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv3d(x, w, padding=pad, groups=c)
    mu_y = F.conv3d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x = F.conv3d(x * x, w, padding=pad, groups=c) - mu_x2
    sig_y = F.conv3d(y * y, w, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv3d(x * y, w, padding=pad, groups=c) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# MedicalNet 3D ResNet (feature extractor for the perceptual term)
# Architecture matches Tencent/MedicalNet so official .pth weights load directly.
# ---------------------------------------------------------------------------
def _conv3x3x3(in_p, out_p, stride=1, dilation=1):
    return nn.Conv3d(in_p, out_p, 3, stride=stride, padding=dilation,
                     dilation=dilation, bias=False)


def _downsample_A(x, out_ch, stride):
    """Parameter-free shortcut (MedicalNet shortcut type 'A')."""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    pad = torch.zeros(out.size(0), out_ch - out.size(1), out.size(2),
                      out.size(3), out.size(4), device=out.device, dtype=out.dtype)
    return torch.cat([out, pad], dim=1)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


_LAYERS = {10: (_BasicBlock, [1, 1, 1, 1]), 18: (_BasicBlock, [2, 2, 2, 2]),
           34: (_BasicBlock, [3, 4, 6, 3]), 50: (_Bottleneck, [3, 4, 6, 3])}


class MedicalNetResNet3D(nn.Module):
    """Encoder portion of MedicalNet ResNet; returns intermediate feature maps."""

    def __init__(self, depth: int = 18, shortcut_type: str = "A"):
        super().__init__()
        block, layers = _LAYERS[depth]
        self.shortcut_type = shortcut_type
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        out_ch = planes * block.expansion
        if stride != 1 or self.inplanes != out_ch:
            if self.shortcut_type == "A":
                downsample = lambda x: _downsample_A(x, out_ch, stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_ch))
        layers = [block(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {1: f1, 2: f2, 3: f3, 4: f4}


def load_medicalnet(depth: int, shortcut_type: str, weights_path: str | None):
    net = MedicalNetResNet3D(depth, shortcut_type)
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # conv_seg.* keys are the segmentation head and are expected to be unexpected
        leftover = [m for m in missing if not m.startswith("conv_seg")]
        if leftover:
            print(f"[MedicalNet] warning: missing keys not loaded: {leftover[:6]} ...")
    return net


class MedicalNetPerceptual(nn.Module):
    def __init__(self, depth=18, shortcut_type="A", weights_path=None,
                 feature_layers=(1, 2, 3)):
        super().__init__()
        self.net = load_medicalnet(depth, shortcut_type, weights_path)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.feature_layers = feature_layers

    @staticmethod
    def _prep(x):
        # collapse channels to MedicalNet's single-channel input, then standardize
        b, c = x.shape[:2]
        x = x.reshape(b * c, 1, *x.shape[2:])
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-5
        return (x - mean) / std

    def forward(self, pred, target):
        f_pred = self.net(self._prep(pred))
        with torch.no_grad():
            f_tgt = self.net(self._prep(target))
        loss = pred.new_zeros(())
        for l in self.feature_layers:
            loss = loss + F.l1_loss(f_pred[l], f_tgt[l])
        return loss / len(self.feature_layers)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CustomLoss(nn.Module):
    """Weighted perceptual + SSIM + L1 over 3D (pred, target) volumes.

    Returns (total_loss, components_dict). Designed to be used directly as the
    Conditional GAN generator reconstruction term and as the VAE recon term.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        perceptual_depth: int = 18,
        perceptual_shortcut: str = "A",
        medicalnet_weights: str | None = None,
        feature_layers=(1, 2, 3),
        ssim_window: int = 7,
        ssim_sigma: float = 1.5,
        data_range: float = 2.0,
    ):
        super().__init__()
        self.l1_w = l1_weight
        self.ssim_w = ssim_weight
        self.perc_w = perceptual_weight
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma
        self.data_range = data_range
        self.perceptual = (
            MedicalNetPerceptual(perceptual_depth, perceptual_shortcut,
                                 medicalnet_weights, feature_layers)
            if perceptual_weight > 0 else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(pred, target)
        ssim_l = 1.0 - ssim3d(pred, target, self.ssim_window, self.ssim_sigma,
                              self.data_range)
        perc = (self.perceptual(pred, target)
                if self.perceptual is not None else pred.new_zeros(()))
        total = self.l1_w * l1 + self.ssim_w * ssim_l + self.perc_w * perc
        return total, {"l1": l1.item(), "ssim": ssim_l.item(),
                       "perceptual": float(perc.detach())}
