"""Shared 3D building blocks used across the GAN and LDM models."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal timestep/continuous-time embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.dim))


class ResBlock3D(nn.Module):
    """GroupNorm + SiLU residual block with optional time/conditioning injection."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch) if emb_dim is not None else None
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.emb_proj is not None and emb is not None:
            h = h + self.emb_proj(F.silu(emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock3D(nn.Module):
    """Self-attention over flattened spatial volume."""

    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.heads, c // self.heads, d * h * w).unbind(1)
        # (b, heads, head_dim, N) -> (b, heads, N, head_dim); use the flash/memory-
        # efficient SDPA kernel so the (N x N) attention matrix is never materialized
        # (3D grids make N=D*H*W large -> O(N^2) memory OOMs). Same scaling (1/sqrt(dh)).
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)        # (b, heads, N, head_dim)
        out = out.transpose(-1, -2).reshape(b, c, d, h, w)
        return x + self.proj(out)


class CondEncoder3D(nn.Module):
    """Encode the raw bpMRI conditioning volume into a compact token grid
    (downsampled to ~the U-Net bottleneck resolution) for cross-attention to
    attend to. Keeps the key/value token count small so cross-attn is cheap."""

    def __init__(self, cond_channels: int, cond_dim: int, n_down: int):
        super().__init__()
        layers = [nn.Conv3d(cond_channels, cond_dim, 3, padding=1), nn.SiLU()]
        for _ in range(n_down):
            layers += [nn.Conv3d(cond_dim, cond_dim, 3, stride=2, padding=1), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.res = ResBlock3D(cond_dim, cond_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.res(self.net(cond))


class CrossAttentionBlock3D(nn.Module):
    """U-Net spatial features (queries) attend to encoded conditioning tokens
    (keys/values). Injects the bpMRI conditioning at every attention scale rather
    than only concatenating it once at the input -- aimed at the localization
    ceiling. Query and key/value token counts need not match."""

    def __init__(self, ch: int, cond_dim: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.to_q = nn.Conv3d(ch, ch, 1)
        self.to_kv = nn.Conv3d(cond_dim, ch * 2, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)  # start as identity

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor | None) -> torch.Tensor:
        if cond_feat is None:
            return x
        b, c, d, h, w = x.shape
        dh = c // self.heads
        q = self.to_q(self.norm(x)).reshape(b, self.heads, dh, d * h * w)
        kv = self.to_kv(cond_feat)
        nkv = kv.shape[2] * kv.shape[3] * kv.shape[4]
        k, v = kv.reshape(b, 2, self.heads, dh, nkv).unbind(1)
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))  # (b,heads,N,dh)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, d, h, w)
        return x + self.proj(out)


class Downsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)
