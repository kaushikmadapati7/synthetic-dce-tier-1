"""2D pixel-space conditional flow matching -- the data-scale test for the flow.

The flow is the *data-hungry* model, so slice-scale data (~20x samples) is where
it should benefit most. Pixel-space (no VAE) 2D flow mirrors the collaborator's
CFM: velocity field v(x_t, t, cond) transporting a source (noise, or the T2w for
an image-to-image path) to the DCE along a straight OT path. Same eval/metrics as
the 2D GAN, so results drop into the same table.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._cfg import roi_weighted_mse


def _sin_time(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    a = t.float()[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock2D(nn.Module):
    def __init__(self, ic, oc, emb):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ic), ic); self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
        self.emb = nn.Linear(emb, oc)
        self.norm2 = nn.GroupNorm(min(8, oc), oc); self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(e))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet2D(nn.Module):
    """3-scale conditional U-Net velocity predictor. Input HxW divisible by 8."""

    def __init__(self, in_ch=1, cond_ch=3, base=64):
        super().__init__()
        emb = base * 4
        self.emb_dim = base
        self.temb = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cin = nn.Conv2d(in_ch + cond_ch, base, 3, 1, 1)
        self.d1 = ResBlock2D(base, base, emb); self.dn1 = nn.Conv2d(base, base, 4, 2, 1)
        self.d2 = ResBlock2D(base, base * 2, emb); self.dn2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
        self.d3 = ResBlock2D(base * 2, base * 4, emb); self.dn3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)
        self.mid = ResBlock2D(base * 4, base * 4, emb)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1); self.u3 = ResBlock2D(base * 8, base * 2, emb)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1); self.u2 = ResBlock2D(base * 4, base, emb)
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1); self.u1 = ResBlock2D(base * 2, base, emb)
        self.out = nn.Sequential(nn.GroupNorm(min(8, base), base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, 1, 1))

    def forward(self, x, t, cond):
        e = self.temb(_sin_time(t, self.emb_dim))
        h = self.cin(torch.cat([x, cond], dim=1))
        h1 = self.d1(h, e); h = self.dn1(h1)
        h2 = self.d2(h, e); h = self.dn2(h2)
        h3 = self.d3(h, e); h = self.dn3(h3)
        h = self.mid(h, e)
        h = self.u3(torch.cat([self.up3(h), h3], 1), e)
        h = self.u2(torch.cat([self.up2(h), h2], 1), e)
        h = self.u1(torch.cat([self.up1(h), h1], 1), e)
        return self.out(h)


class FlowMatching2D(nn.Module):
    """Pixel-space CFM. t=0 -> data (DCE), t=1 -> source (noise or T2w)."""

    def __init__(self, cond_ch=3, base=64, source="noise", time_scale=1000.0):
        super().__init__()
        self.unet = UNet2D(in_ch=1, cond_ch=cond_ch, base=base)
        self.source = source
        self.time_scale = time_scale

    def _src(self, x1, cond):
        return cond[:, 0:1] if self.source == "t2w" else torch.randn_like(x1)

    def loss(self, x1, cond, mask=None, roi_weight=1.0):
        b = x1.shape[0]
        t = torch.rand(b, device=x1.device)
        src = self._src(x1, cond)
        tb = t.view(b, 1, 1, 1)
        xt = (1.0 - tb) * x1 + tb * src
        v_target = src - x1
        v_pred = self.unet(xt, t * self.time_scale, cond)
        return roi_weighted_mse(v_pred, v_target, mask, roi_weight)

    @torch.no_grad()
    def sample(self, cond, steps=50):
        b = cond.size(0)
        z = (cond[:, 0:1] if self.source == "t2w"
             else torch.randn(b, 1, *cond.shape[2:], device=cond.device))
        ts = torch.linspace(1.0, 0.0, steps + 1, device=cond.device)
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            tb = torch.full((b,), float(t), device=cond.device)
            v = self.unet(z, tb * self.time_scale, cond)
            z = z + (tn - t) * v
        return z


class LatentFlowMatching2D(nn.Module):
    """2D conditional flow matching in a frozen first-stage latent (e.g. MedVAE-2D).

    Mirrors the 3D LDM flow but per-slice: encode the DCE slice -> CFM on the latent
    with the bpMRI condition downsampled to the latent grid -> decode. This gives the
    crispness of 2D (high in-plane res, no depth compression) plus the realism of a
    data-rich foundation VAE. An optional image-space anchor (decode the predicted
    clean latent, ROI recon loss through the exact/grad decoder) adds the same direct
    faithfulness supervision as in 3D. Reuses UNet2D as the velocity net.

    ``first_stage`` exposes the AutoencoderKL3D-style interface (encode->.sample(),
    decode, decoder, scaling_factor, latent_shift, latent_channels)."""

    def __init__(self, first_stage, cond_ch=3, base=64, time_scale=1000.0):
        super().__init__()
        self.first_stage = first_stage
        self.latent_channels = first_stage.latent_channels
        self.unet = UNet2D(in_ch=self.latent_channels, cond_ch=cond_ch, base=base)
        self.time_scale = time_scale

    @torch.no_grad()
    def encode(self, x):
        fs = self.first_stage
        z = fs.encode(x).sample()
        return (z - fs.latent_shift) * fs.scaling_factor

    def decode(self, z):
        return self.first_stage.decode(z)

    @staticmethod
    def _to2d(m):
        return m.squeeze(2) if (m is not None and m.dim() == 5) else m   # accept (B,1,1,H,W) or (B,1,H,W)

    @staticmethod
    def _ds(x, hw):
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    def loss(self, x1_img, cond, mask=None, roi_weight=1.0,
             anchor_image=None, anchor_mask=None, anchor_zone_weight=None,
             anchor_criterion=None, anchor_weight=0.0, anchor_t_max=1.0):
        with torch.no_grad():
            z1 = self.encode(x1_img)
        lat_hw = z1.shape[2:]
        cond_ds = self._ds(cond, lat_hw)
        mask2d = self._to2d(mask)
        mask_ds = self._ds(mask2d, lat_hw) if mask2d is not None else None
        b = z1.shape[0]
        t = torch.rand(b, device=z1.device)
        src = torch.randn_like(z1)
        tb = t.view(b, 1, 1, 1)
        zt = (1.0 - tb) * z1 + tb * src
        v_target = src - z1
        v_pred = self.unet(zt, t * self.time_scale, cond_ds)
        loss = roi_weighted_mse(v_pred, v_target, mask_ds, roi_weight)

        if anchor_weight > 0 and anchor_image is not None:
            lo = t < anchor_t_max                          # anchor only low-noise steps (sharp z0_hat)
            if lo.any():
                fs = self.first_stage
                z0_hat = zt[lo] - tb[lo] * v_pred[lo]      # predicted clean latent
                img = fs.decoder(z0_hat / fs.scaling_factor + fs.latent_shift)   # grad-enabled decode
                w5 = lambda x: x.unsqueeze(2) if x is not None else None          # 2D -> (B,1,1,H,W) for the 3D criterion
                ai = self._to2d(anchor_image)[lo]
                am = self._to2d(anchor_mask)[lo] if anchor_mask is not None else None
                azw = self._to2d(anchor_zone_weight)[lo] if anchor_zone_weight is not None else None
                loss = loss + anchor_weight * anchor_criterion(
                    w5(img), w5(ai), w5(am), zone_weight=w5(azw))[0]
        return loss

    @torch.no_grad()
    def sample(self, cond, steps=50):
        z = torch.randn_like(self.encode(cond[:, 0:1]))    # probe latent shape, then start from noise
        cond_ds = self._ds(cond, z.shape[2:])
        ts = torch.linspace(1.0, 0.0, steps + 1, device=cond.device)
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            tb = torch.full((z.shape[0],), float(t), device=cond.device)
            z = z + (tn - t) * self.unet(z, tb * self.time_scale, cond_ds)
        return self.decode(z)
