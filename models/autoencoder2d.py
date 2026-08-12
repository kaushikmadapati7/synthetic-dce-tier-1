"""2D KL-autoencoder: the trained-on-our-data first stage for the 2D latent flow.

Completes the first-stage ablation. Until now the 2D track could only run
pixel-space CFM or MedVAE-latent CFM, so "does an encoder bottleneck hurt, and does
it matter whether the encoder saw prostate DCE?" was unanswerable in 2D --
`--first-stage vae` silently fell through to pixel space.

Architecture mirrors AutoencoderKL3D exactly (same ResBlock/Attention/Down/Up
structure, same DiagonalGaussian posterior, same KL) with Conv2d in place of Conv3d,
so a 2D-vs-3D comparison isolates dimensionality rather than confounding it with a
different autoencoder design.

Exposes the first-stage interface the latent flows expect:
``encode -> object with .sample()/.kl()``, ``decode``, ``decoder``,
``latent_channels``, ``scaling_factor``, ``latent_shift``, ``loss``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder3d import DiagonalGaussian


def _norm(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.ch = ch

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / (c ** 0.5), dim=-1)
        out = (v @ att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample2D(nn.Module):
    """Nearest upsample + conv, never ConvTranspose -- see gan2d._up for the measured
    period-2 checkerboard transposed convs produce here."""

    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class Encoder2D(nn.Module):
    def __init__(self, in_ch, base_ch, ch_mults, latent_ch, num_res, attn_at):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chans = [base_ch * m for m in ch_mults]
        blocks, ch = [], base_ch
        for i, out_ch in enumerate(chans):
            for _ in range(num_res):
                blocks.append(ResBlock2D(ch, out_ch)); ch = out_ch
                if i in attn_at:
                    blocks.append(AttentionBlock2D(ch))
            if i < len(chans) - 1:
                blocks.append(Downsample2D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.mid = nn.ModuleList([ResBlock2D(ch, ch), AttentionBlock2D(ch), ResBlock2D(ch, ch)])
        self.norm_out = _norm(ch)
        self.conv_out = nn.Conv2d(ch, 2 * latent_ch, 3, padding=1)      # mean + logvar

    def forward(self, x):
        h = self.conv_in(x)
        for b in self.blocks:
            h = b(h)
        for b in self.mid:
            h = b(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder2D(nn.Module):
    def __init__(self, out_ch, base_ch, ch_mults, latent_ch, num_res, attn_at):
        super().__init__()
        chans = [base_ch * m for m in ch_mults]
        ch = chans[-1]
        self.conv_in = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.mid = nn.ModuleList([ResBlock2D(ch, ch), AttentionBlock2D(ch), ResBlock2D(ch, ch)])
        blocks, n = [], len(chans)
        for i, out_c in enumerate(reversed(chans)):
            for _ in range(num_res):
                blocks.append(ResBlock2D(ch, out_c)); ch = out_c
                if (n - 1 - i) in attn_at:
                    blocks.append(AttentionBlock2D(ch))
            if i < n - 1:
                blocks.append(Upsample2D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = _norm(ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        for b in self.mid:
            h = b(h)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.conv_out(F.silu(self.norm_out(h))))


class AutoencoderKL2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 latent_channels: int = 4, base_ch: int = 32,
                 ch_mults: tuple = (1, 2, 4), num_res_blocks: int = 2,
                 attn_resolutions: tuple = (2,), scaling_factor: float = 1.0):
        super().__init__()
        self.encoder = Encoder2D(in_channels, base_ch, ch_mults, latent_channels,
                                 num_res_blocks, attn_resolutions)
        self.decoder_net = Decoder2D(out_channels, base_ch, ch_mults, latent_channels,
                                     num_res_blocks, attn_resolutions)
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.latent_shift = 0.0

    # the latent flows call fs.decoder(z) for the raw decode and fs.decode(z) for the
    # normalization-inverting one, matching MedVAEFirstStage
    def decoder(self, z):
        return self.decoder_net(z)

    def encode(self, x) -> DiagonalGaussian:
        return DiagonalGaussian(self.encoder(x))

    def decode(self, z):
        return self.decoder_net(z / self.scaling_factor + self.latent_shift)

    def forward(self, x):
        posterior = self.encode(x)
        return self.decoder_net(posterior.sample()), posterior

    def loss(self, x, kl_weight: float = 1e-6, criterion=None, mask=None):
        """Reconstruction + KL. ``mask`` drives ROI emphasis in ``criterion``; for a
        latent flow this recon term is where in-gland fidelity is actually set, since
        the flow objective itself lives in latent space."""
        recon, posterior = self(x)
        if criterion is not None:
            rec, parts = criterion(recon, x, mask)
        else:
            rec = F.l1_loss(recon, x)
            parts = {"recon": rec.item()}
        kl = posterior.kl()
        return rec + kl_weight * kl, {**parts, "kl": kl.item()}
