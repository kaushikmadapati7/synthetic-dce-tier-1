"""Conditional 3D U-Net denoiser.

Shared backbone for both LDMs. It predicts an output the same shape as the
latent input given a continuous/discrete time `t`. The interpretation of the
output (noise eps vs. velocity v) is decided by the LDM that wraps it.

Conditioning:
  - `t`: timestep / continuous time, added via sinusoidal time embedding.
  - `cond_channels`: latent-space conditioning concatenated to the input.
  - `num_classes`: optional class label embedding added to the time embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (ResBlock3D, AttentionBlock3D, CrossAttentionBlock3D,
                     CondEncoder3D, Downsample3D, Upsample3D, TimeEmbedding)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        cond_channels: int = 0,
        num_classes: int = 0,
        base_ch: int = 64,
        ch_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (1, 2),
        cond_dim: int = 0,
    ):
        super().__init__()
        emb_dim = base_ch * 4
        self.time_emb = TimeEmbedding(base_ch)
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.num_classes = num_classes

        # cross-attention conditioning: encode cond to coarse tokens that the
        # attention blocks attend to (in ADDITION to the input concat). cond_dim>0 on.
        self.cond_encoder = (CondEncoder3D(cond_channels, cond_dim, n_down=len(ch_mults) - 1)
                             if cond_dim > 0 and cond_channels > 0 else None)
        self.conv_in = nn.Conv3d(in_channels + cond_channels, base_ch, 3, padding=1)

        def _attn(c):
            blocks = [AttentionBlock3D(c)]
            if self.cond_encoder is not None:
                blocks.append(CrossAttentionBlock3D(c, cond_dim))
            return blocks

        chans = [base_ch * m for m in ch_mults]
        # ---- encoder ----
        self.down = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        for i, out_ch in enumerate(chans):
            for _ in range(num_res_blocks):
                stage = nn.ModuleList([ResBlock3D(ch, out_ch, emb_dim)])
                ch = out_ch
                if i in attn_resolutions:
                    stage.extend(_attn(ch))
                self.down.append(stage)
                skip_chs.append(ch)
            if i < len(chans) - 1:
                self.down.append(nn.ModuleList([Downsample3D(ch)]))
                skip_chs.append(ch)

        # ---- middle ----
        self.mid = nn.ModuleList([ResBlock3D(ch, ch, emb_dim), *_attn(ch),
                                  ResBlock3D(ch, ch, emb_dim)])

        # ---- decoder ----
        self.up = nn.ModuleList()
        for i, out_ch in reversed(list(enumerate(chans))):
            for _ in range(num_res_blocks + 1):
                stage = nn.ModuleList([ResBlock3D(ch + skip_chs.pop(), out_ch, emb_dim)])
                ch = out_ch
                if i in attn_resolutions:
                    stage.extend(_attn(ch))
                self.up.append(stage)
            if i > 0:
                self.up.append(nn.ModuleList([Upsample3D(ch)]))

        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, cond=None, labels=None):
        emb = self.time_emb(t)
        if self.num_classes > 0 and labels is not None:
            emb = emb + self.class_emb(labels)

        # encode cond tokens for cross-attention BEFORE it gets concatenated in
        cond_feat = self.cond_encoder(cond) if (self.cond_encoder is not None and cond is not None) else None
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        h = self.conv_in(x)

        def run(layer, h):
            if isinstance(layer, ResBlock3D):
                return layer(h, emb)
            if isinstance(layer, CrossAttentionBlock3D):
                return layer(h, cond_feat)
            return layer(h)

        skips = [h]
        for stage in self.down:
            for layer in stage:
                h = run(layer, h)
            skips.append(h)

        for layer in self.mid:
            h = run(layer, h)

        for stage in self.up:
            if isinstance(stage[0], ResBlock3D):
                h = torch.cat([h, skips.pop()], dim=1)
            for layer in stage:
                h = run(layer, h)

        return self.conv_out(F.silu(self.norm_out(h)))
