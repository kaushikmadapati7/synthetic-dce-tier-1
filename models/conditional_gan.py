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


def _stride_schedule(spatial_size, n_levels, min_dim=2):
    """Per-level (D,H,W) strides that never shrink an axis below `min_dim`.

    Inputs here are strongly anisotropic (e.g. 32x192x192): striding every axis by 2
    at every level would drive depth to 1 long before the in-plane axes are reduced.
    Each axis is halved only while it can afford it, so depth keeps >=2 slices.
    """
    cur = list(spatial_size)
    sched = []
    for _ in range(n_levels):
        s = tuple(2 if c // 2 >= min_dim else 1 for c in cur)
        cur = [c // st for c, st in zip(cur, s)]
        sched.append(s)
    return sched


def _dn3(ic, oc, stride, norm=True):
    k = tuple(4 if s == 2 else 3 for s in stride)
    layers = [nn.Conv3d(ic, oc, k, stride, 1, bias=not norm)]
    if norm:
        layers.append(_norm(oc))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _up3(ic, oc, stride, drop=False):
    """Nearest upsample + 3x3x3 conv, NOT ConvTranspose3d.

    Transposed convs stamp a period-2 checkerboard into the output. Measured on
    runs/v2_3d_gan: lag-1 autocorrelation of the first difference was -0.60 in the
    prediction (more sign-alternating than white noise, -0.50) against +0.81 for
    the real DCE target, with 5x the target's high-frequency power. k=4/stride=2
    gives uniform overlap in theory, but stacking these under an adversarial loss
    reintroduces it anyway. common.Upsample3D (used by the VAE/UNet3D path) already
    does it this way; the GAN decoder just wasn't.

    Output shape is unchanged: ConvTranspose3d(k=4,s=2,p=1) and
    ConvTranspose3d(k=3,s=1,p=1) both map n -> n*s, as does upsample(s) + conv3(p=1).
    """
    layers = [nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
              nn.Conv3d(ic, oc, 3, 1, 1, bias=False), _norm(oc)]
    if drop:
        layers.append(nn.Dropout3d(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class UNetGenerator3D(nn.Module):
    """pix2pix-style 3D U-Net: cond volume -> DCE, tanh output.

    Replaces Generator3D for image-to-image translation. Generator3D projects a noise
    vector to a tiny grid and fuses the condition through AdaptiveAvgPool3d(init_size)
    -- at 32x192x192 with n_upsamples=4 that pools the entire 3-channel anatomy into a
    2x12x12 summary with NO skip connections, so the generator cannot see fine
    structure and can only emit a smooth blob at roughly the right brightness. That is
    the 3D counterpart of our 2D pix2pix generator, which has full skips and produced
    crisp output.

    Deterministic by construction (no z): stochasticity comes from decoder dropout, as
    in pix2pix, which also makes evaluation reproducible.
    """

    def __init__(self, in_channels=3, out_channels=1, base_ch=32, n_levels=5,
                 spatial_size=(32, 192, 192)):
        super().__init__()
        self.strides = _stride_schedule(spatial_size, n_levels)
        chs = [base_ch * min(2 ** i, 8) for i in range(n_levels)]      # cap growth at 8x
        self.downs = nn.ModuleList()
        ic = in_channels
        for i, oc in enumerate(chs):
            self.downs.append(_dn3(ic, oc, self.strides[i], norm=(i > 0)))
            ic = oc
        self.ups = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            skip = chs[i - 1]
            first = i == n_levels - 1
            self.ups.append(_up3(chs[i] if first else chs[i] + chs[i], skip,
                                 self.strides[i], drop=(i >= n_levels - 2)))
        # Final layer writes straight to the image, so it is the worst place for a
        # transposed conv -- its checkerboard lands in the output with nothing after
        # it to smooth it. Same upsample+conv treatment as _up3.
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=tuple(self.strides[0]), mode="nearest"),
            nn.Conv3d(chs[0] * 2, out_channels, 3, 1, 1),
            nn.Tanh())

    def forward(self, cond_vol):
        feats = []
        h = cond_vol
        for d in self.downs:
            h = d(h); feats.append(h)
        h = feats[-1]
        for j, u in enumerate(self.ups):
            h = u(h if j == 0 else torch.cat([h, feats[-1 - j]], 1))
        return self.out(torch.cat([h, feats[0]], 1))


class ConditionalGAN3D(nn.Module):
    """Convenience wrapper bundling G and D plus the sampling helper.

    generator='unet' (default) uses the pix2pix-style UNetGenerator3D; 'resnet' keeps
    the original noise-vector Generator3D (kept for reproducing older runs).
    """

    def __init__(self, generator: str = "unet", **kwargs):
        super().__init__()
        self.generator_type = generator
        d_keys = {"in_channels", "cond_channels", "num_classes", "base_ch", "n_downsamples"}
        if generator == "unet":
            self.generator = UNetGenerator3D(
                in_channels=kwargs.get("cond_channels", 3),
                out_channels=kwargs.get("out_channels", 1),
                base_ch=kwargs.get("base_ch", 32),
                spatial_size=kwargs.get("spatial_size", (32, 192, 192)))
            self.z_dim = 0
        else:
            g_keys = {"z_dim", "out_channels", "cond_channels", "num_classes",
                      "base_ch", "init_size", "n_upsamples"}
            self.generator = Generator3D(**{k: v for k, v in kwargs.items() if k in g_keys})
            self.z_dim = self.generator.z_dim
        self.discriminator = Discriminator3D(**{k: v for k, v in kwargs.items() if k in d_keys})

    def generate(self, cond_vol, labels=None):
        """Forward pass used for both training and eval (deterministic for 'unet')."""
        if self.generator_type == "unet":
            return self.generator(cond_vol)
        z = torch.randn(cond_vol.size(0), self.z_dim, device=cond_vol.device)
        return self.generator(z, cond_vol=cond_vol, labels=labels)

    @torch.no_grad()
    def sample(self, n, device, cond_vol=None, labels=None):
        if self.generator_type == "unet":
            return self.generator(cond_vol)
        z = torch.randn(n, self.z_dim, device=device)
        return self.generator(z, cond_vol=cond_vol, labels=labels)
