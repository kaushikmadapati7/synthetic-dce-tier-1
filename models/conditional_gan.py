"""3D Conditional GAN.

Generator maps (noise z, condition c) -> volume.
Discriminator scores (volume, condition c) as real/fake (projection discriminator).

Conditioning is flexible:
  - `cond_channels`: a conditioning volume concatenated channel-wise (e.g. a
    pre-contrast image / mask) at the same spatial size as the output.
  - `num_classes`: optional discrete class label embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, ch), ch)


class Generator3D(nn.Module):
    def __init__(
        self,
        z_dim: int = 128,
        out_channels: int = 1,
        cond_channels: int = 0,
        num_classes: int = 0,
        base_ch: int = 64,
        init_size: int = 4,
        n_upsamples: int = 4,
    ):
        super().__init__()
        self.z_dim = z_dim
        # init_size may be an int (cubic) or a (D, H, W) tuple (anisotropic targets)
        self.init_size = (init_size,) * 3 if isinstance(init_size, int) else tuple(init_size)
        self.cond_channels = cond_channels
        self.num_classes = num_classes

        cond_dim = 0
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, z_dim)
            cond_dim += z_dim

        chans = [base_ch * (2 ** i) for i in reversed(range(n_upsamples + 1))]
        init_numel = self.init_size[0] * self.init_size[1] * self.init_size[2]
        self.fc = nn.Linear(z_dim + cond_dim, chans[0] * init_numel)
        self.chans0 = chans[0]

        # if a conditioning volume is provided, encode it and fuse at the bottleneck
        if cond_channels > 0:
            self.cond_enc = nn.Sequential(
                nn.Conv3d(cond_channels, chans[0], 3, padding=1),
                _norm(chans[0]), nn.SiLU(),
            )
            self.cond_pool = nn.AdaptiveAvgPool3d(self.init_size)

        blocks = []
        in_ch = chans[0]
        for out_ch in chans[1:]:
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                _norm(out_ch), nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                _norm(out_ch), nn.SiLU(),
            ))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.to_out = nn.Sequential(nn.Conv3d(in_ch, out_channels, 3, padding=1), nn.Tanh())

    def forward(self, z, cond_vol=None, labels=None):
        ctx = [z]
        if self.num_classes > 0 and labels is not None:
            ctx.append(self.class_emb(labels))
        h = self.fc(torch.cat(ctx, dim=1))
        h = h.view(-1, self.chans0, *self.init_size)

        if self.cond_channels > 0 and cond_vol is not None:
            h = h + self.cond_pool(self.cond_enc(cond_vol))

        for block in self.blocks:
            h = block(h)
        return self.to_out(h)


class Discriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 0,
        num_classes: int = 0,
        base_ch: int = 64,
        n_downsamples: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        chans = [base_ch * (2 ** i) for i in range(n_downsamples + 1)]

        layers = []
        in_ch = in_channels + cond_channels
        for out_ch in chans:
            layers.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
                _norm(out_ch), nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch
        self.features = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(in_ch, 1)
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, in_ch)  # projection discriminator

    def forward(self, x, cond_vol=None, labels=None):
        if cond_vol is not None:
            x = torch.cat([x, cond_vol], dim=1)
        for layer in self.features:
            x = layer(x)
        feat = self.pool(x).flatten(1)
        out = self.head(feat)
        if self.num_classes > 0 and labels is not None:
            out = out + (feat * self.class_emb(labels)).sum(dim=1, keepdim=True)
        return out


# ---- losses (hinge GAN, a stable default) -------------------------------------
def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def g_total_loss(fake_logits, fake, real, criterion=None, adv_weight=1.0, mask=None,
                 zone_weight=None):
    """Generator objective: adversarial term + image-space reconstruction term.

    criterion: optional callable(pred, target, mask=None, zone_weight=None) ->
    (loss, components), e.g. loss.CustomLoss (perceptual + SSIM + L1, with optional
    ROI / zone weighting). When None, returns the plain adversarial (hinge) loss.
    """
    adv = g_hinge_loss(fake_logits)
    parts = {"adv": adv.item()}
    rec = fake.new_zeros(())
    if criterion is not None:
        rec, rec_parts = criterion(fake, real, mask, zone_weight=zone_weight)
        parts.update(rec_parts)
    total = adv_weight * adv + rec
    return total, parts


class ConditionalGAN3D(nn.Module):
    """Convenience wrapper bundling G and D plus the sampling helper."""

    def __init__(self, **kwargs):
        super().__init__()
        g_keys = {"z_dim", "out_channels", "cond_channels", "num_classes",
                  "base_ch", "init_size", "n_upsamples"}
        d_keys = {"in_channels", "cond_channels", "num_classes", "base_ch", "n_downsamples"}
        self.generator = Generator3D(**{k: v for k, v in kwargs.items() if k in g_keys})
        self.discriminator = Discriminator3D(**{k: v for k, v in kwargs.items() if k in d_keys})
        self.z_dim = self.generator.z_dim

    @torch.no_grad()
    def sample(self, n, device, cond_vol=None, labels=None):
        z = torch.randn(n, self.z_dim, device=device)
        return self.generator(z, cond_vol=cond_vol, labels=labels)
