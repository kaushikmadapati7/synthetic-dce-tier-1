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
           sigma: float = 1.5, data_range: float = 2.0,
           return_map: bool = False) -> torch.Tensor:
    """3D SSIM. data_range=2.0 matches tanh outputs in [-1, 1].

    Returns the scalar mean SSIM, or the per-voxel SSIM map (same spatial size
    as the input) when ``return_map`` — used to average SSIM inside an ROI mask.
    """
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
    return ssim_map if return_map else ssim_map.mean()


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
        # weights_only=False: these are trusted local MedicalNet (Med3D) checkpoints,
        # and torch>=2.6 defaults weights_only=True which rejects their 2019 metadata.
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
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
# Clinical / radiomics-aware terms (3D ports of the ClinDCE losses)
#
# A pixel-fidelity objective (L1/SSIM) flattens the focal-enhancement signal that
# is the whole clinical point of DCE. These terms target it directly, inside the
# prostate ROI. Intensities are mapped [-1,1] -> [0,1] first so the enhancement-
# ratio feature (which assumes non-negative signal) stays well-behaved. Both are
# gated behind weights that default to 0 (byte-for-byte no-op when off).
# ---------------------------------------------------------------------------
def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


def regional_radiomics_loss3d(pred, target, mask, patch_size: int = 7):
    """Regional Radiomics Consistency Loss (ClinDCE), 3D. Local avg-pool feature
    MAPS (mean / variance / upper-mean / enhancement-ratio) matched within the
    mask, so *where* enhancement differs is penalized -- not just global stats."""
    p, t, m = _to01(pred), _to01(target), mask
    pad = patch_size // 2
    pool = lambda z: F.avg_pool3d(z, patch_size, stride=1, padding=pad)
    count = pool(m).clamp(min=1e-6)

    p_mean = pool(p * m) / count
    t_mean = pool(t * m) / count
    p_var = pool((p - p_mean) ** 2 * m) / count
    t_var = pool((t - t_mean) ** 2 * m) / count

    up_p = F.relu(p - p_mean) * m            # above-local-mean ~ upper distribution
    up_t = F.relu(t - t_mean) * m
    up_count = pool((up_t > 0).float() * m).clamp(min=1e-6)
    p_upper = pool(up_p) / up_count
    t_upper = pool(up_t) / up_count

    p_ratio = p_upper / (p_mean.abs() + 1e-6)   # enhancement ratio ~ Ktrans proxy
    t_ratio = t_upper / (t_mean.abs() + 1e-6)

    return (F.l1_loss(p_mean * m, t_mean * m) + F.l1_loss(p_var * m, t_var * m)
            + F.l1_loss(p_upper * m, t_upper * m) + F.l1_loss(p_ratio * m, t_ratio * m))


def focal_enhancement_loss3d(pred, target, mask, patch_size: int = 7,
                             threshold_std: float = 1.5):
    """Match intensity in focal-enhancement regions (PI-RADS positivity) AND
    penalize over-enhancement of non-focal tissue. The symmetric over-enhancement
    term is the fix ClinDCE needed to stop the model brightening everything to
    satisfy the focal-matching term alone."""
    p, t, m = _to01(pred), _to01(target), mask
    pad = patch_size // 2
    pool = lambda z: F.avg_pool3d(z, patch_size, stride=1, padding=pad)
    tm = t * m
    local_mean = pool(tm)
    local_std = torch.sqrt(F.relu(pool((tm - local_mean) ** 2)) + 1e-8)
    focal = ((tm - local_mean) > threshold_std * local_std).float() * m
    if focal.sum() < 10:
        return pred.new_zeros(())
    loss_focal = F.l1_loss(p * focal, t * focal)
    non_focal = m * (1.0 - focal)
    if non_focal.sum() > 10:
        over = F.relu(p * non_focal - t * non_focal)
        loss_over = over.sum() / non_focal.sum().clamp(min=1)
    else:
        loss_over = pred.new_zeros(())
    return loss_focal + 0.5 * loss_over


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CustomLoss(nn.Module):
    """Weighted perceptual + SSIM + L1 over 3D (pred, target) volumes.

    Returns (total_loss, components_dict). Designed to be used directly as the
    Conditional GAN generator reconstruction term and as the VAE recon term.

    ROI emphasis: the prostate is ~1% of the volume, so an unweighted loss is
    dominated by background. When a ``mask`` is passed to ``forward`` and
    ``roi_weight > 1``, the L1 term is reweighted so ROI voxels count
    ``roi_weight``x more, and an extra ROI-SSIM term is added. With no mask (or
    an empty one) the behaviour is identical to the unweighted loss.

    Clinical terms (ClinDCE-style): when a mask is present and ``radio_weight``
    / ``focal_weight`` > 0, regional-radiomics and focal-enhancement losses are
    added to directly preserve the focal-enhancement signal a pixel-fidelity
    objective flattens. Both default to 0 (off). Because this criterion is reused
    as the GAN recon term, the VAE recon term, AND the flow's trajectory-anchor
    criterion, enabling them propagates to all three models at once.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        roi_weight: float = 1.0,
        radio_weight: float = 0.0,
        focal_weight: float = 0.0,
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
        self.roi_w = roi_weight
        self.radio_w = radio_weight
        self.focal_w = focal_weight
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma
        self.data_range = data_range
        self.perceptual = (
            MedicalNetPerceptual(perceptual_depth, perceptual_shortcut,
                                 medicalnet_weights, feature_layers)
            if perceptual_weight > 0 else None
        )

    def _has_roi(self, mask):
        return mask is not None and self.roi_w > 1.0 and mask.sum() > 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None):
        use_roi = self._has_roi(mask)
        has_mask = mask is not None and mask.sum() > 0

        # L1: reweighted so ROI voxels dominate the gradient (also lifts ROI PSNR)
        if use_roi:
            w = 1.0 + (self.roi_w - 1.0) * mask
            l1 = (w * (pred - target).abs()).sum() / w.sum()
        else:
            l1 = F.l1_loss(pred, target)

        ssim_l = 1.0 - ssim3d(pred, target, self.ssim_window, self.ssim_sigma,
                              self.data_range)
        # extra structural term inside the ROI (averaged over mask voxels)
        if use_roi:
            smap = ssim3d(pred, target, self.ssim_window, self.ssim_sigma,
                          self.data_range, return_map=True)
            m = mask > 0.5
            ssim_roi_l = 1.0 - smap[m].mean()
        else:
            ssim_roi_l = pred.new_zeros(())

        perc = (self.perceptual(pred, target)
                if self.perceptual is not None else pred.new_zeros(()))

        # Clinical terms (ClinDCE): preserve focal-enhancement / radiomic signal.
        radio = (regional_radiomics_loss3d(pred, target, mask)
                 if self.radio_w > 0 and has_mask else pred.new_zeros(()))
        focal = (focal_enhancement_loss3d(pred, target, mask)
                 if self.focal_w > 0 and has_mask else pred.new_zeros(()))

        total = (self.l1_w * l1 + self.ssim_w * ssim_l
                 + self.ssim_w * ssim_roi_l + self.perc_w * perc
                 + self.radio_w * radio + self.focal_w * focal)
        return total, {"l1": l1.item(), "ssim": ssim_l.item(),
                       "ssim_roi": float(ssim_roi_l.detach() if use_roi else 0.0),
                       "perceptual": float(perc.detach()),
                       "radio": float(radio.detach() if self.radio_w > 0 else 0.0),
                       "focal": float(focal.detach() if self.focal_w > 0 else 0.0)}
