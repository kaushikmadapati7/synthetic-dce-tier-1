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
