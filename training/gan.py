"""Training loop for the 3D conditional GAN (pix2pix-style, single-stage)."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch

from ..models import ConditionalGAN3D, d_hinge_loss, g_total_loss
from .utils import log_epoch, save_ckpt

log = logging.getLogger("tier1")


def _build_gan(args, device):
    n_up = args.n_upsamples
    factor = 2 ** n_up
    if any(s % factor for s in args.spatial_size):
        raise ValueError(f"spatial_size {args.spatial_size} must be divisible by 2**n_upsamples ({factor})")
    init = [s // factor for s in args.spatial_size]
    # discriminator depth is capped by the smallest dim (kernel-4 stride-2 needs dim>=2 each step):
    # layers = n_down+1 must satisfy 2**layers <= min_dim
    n_down = min(n_up, max(1, int(math.log2(min(args.spatial_size))) - 1))
    return ConditionalGAN3D(z_dim=args.z_dim, out_channels=1, cond_channels=3,
                            base_ch=args.base_ch, init_size=init, n_upsamples=n_up,
                            in_channels=1, n_downsamples=n_down).to(device)


def _gan_gen(gan, args, device):
    return lambda cond: gan.sample(cond.size(0), device, cond_vol=cond)


def load_gan(args, train_loader, test_loader, device):
    """Rebuild the GAN and load its checkpoint for eval-only (no training)."""
    gan = _build_gan(args, device)
    ckpt = Path(args.output_dir) / "checkpoints" / "gan_last.pt"
    gan.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    gan.eval()
    log.info(f"loaded GAN checkpoint {ckpt}")
    return gan, _gan_gen(gan, args, device)


def train_gan(args, train_loader, test_loader, criterion, device):
    gan = _build_gan(args, device)
    log.info(f"GAN params: G={sum(p.numel() for p in gan.generator.parameters())/1e6:.1f}M "
             f"D={sum(p.numel() for p in gan.discriminator.parameters())/1e6:.1f}M")
    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        gan.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device); real = batch["target"].to(device)
            mask = batch["mask"].to(device)
            z = torch.randn(cond.size(0), args.z_dim, device=device)

            fake = gan.generator(z, cond_vol=cond)
            d_loss = d_hinge_loss(gan.discriminator(real, cond_vol=cond),
                                  gan.discriminator(fake.detach(), cond_vol=cond))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            g_loss, parts = g_total_loss(gan.discriminator(fake, cond_vol=cond),
                                         fake, real, criterion, adv_weight=args.adv_weight,
                                         mask=mask)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            for k, v in {"d": float(d_loss.detach()), "g": float(g_loss.detach()), **parts}.items():
                agg[k] = agg.get(k, 0.0) + v
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, "gan", gan, epoch, args.epochs)

    return gan, _gan_gen(gan, args, device)
