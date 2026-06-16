"""Shared two-stage latent-diffusion training (VAE first stage + denoiser).

Both the DDPM and flow-matching trainers wrap `train_ldm` with a different
`flow` flag; the only differences are the model class and the sampler.
"""
from __future__ import annotations

import logging
import time

import torch

from ..models import AutoencoderKL3D, LDM_DDPM, LDM_FlowMatching
from .utils import log_epoch, save_ckpt, downsample_cond

log = logging.getLogger("tier1")


def train_vae(args, train_loader, criterion, device):
    vae = AutoencoderKL3D(in_channels=1, out_channels=1,
                          latent_channels=args.latent_channels, base_ch=args.base_ch,
                          ch_mults=tuple(args.ch_mults)).to(device)
    log.info(f"VAE params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")
    if args.vae_ckpt:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        log.info(f"loaded VAE checkpoint {args.vae_ckpt}")
    else:
        opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
        for epoch in range(args.vae_epochs):
            vae.train(); t0 = time.time(); agg = {}
            for batch in train_loader:
                x = batch["target"].to(device)
                mask = batch["mask"].to(device)
                loss, parts = vae.loss(x, kl_weight=args.kl_weight, criterion=criterion, mask=mask)
                opt.zero_grad(); loss.backward(); opt.step()
                for k, v in {"vae": loss.item(), **parts}.items():
                    agg[k] = agg.get(k, 0.0) + v
            log_epoch(epoch, args.vae_epochs, agg, len(train_loader), time.time() - t0, "VAE")
            save_ckpt(args, "vae", vae, epoch, args.vae_epochs, state_dict=True)

    # scaling factor: normalize latent std to ~1 (standard LDM practice)
    vae.eval()
    with torch.no_grad():
        zs = []
        for i, batch in enumerate(train_loader):
            zs.append(vae.encode(batch["target"].to(device)).sample())
            if i >= 4:
                break
        std = torch.cat(zs).std().item()
    vae.scaling_factor = 1.0 / (std + 1e-8)
    log.info(f"latent std={std:.4f} -> scaling_factor={vae.scaling_factor:.4f}")
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def train_ldm(args, train_loader, test_loader, criterion, device, flow: bool):
    vae = train_vae(args, train_loader, criterion, device)

    # infer latent spatial size from one encode
    with torch.no_grad():
        z0 = vae.encode(next(iter(train_loader))["target"].to(device)).sample()
    lat_spatial = tuple(z0.shape[2:])
    log.info(f"latent grid: {z0.shape[1]}x{lat_spatial}")

    unet_kwargs = dict(in_channels=args.latent_channels, out_channels=args.latent_channels,
                       cond_channels=3, base_ch=args.base_ch, ch_mults=tuple(args.unet_ch_mults))
    name = "ldm_flow" if flow else "ldm_ddpm"
    if flow:
        ldm = LDM_FlowMatching(autoencoder=vae, unet_kwargs=unet_kwargs).to(device)
    else:
        ldm = LDM_DDPM(autoencoder=vae, timesteps=args.timesteps, unet_kwargs=unet_kwargs).to(device)
    log.info(f"UNet params: {sum(p.numel() for p in ldm.unet.parameters())/1e6:.1f}M")

    opt = torch.optim.Adam(ldm.unet.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        ldm.unet.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = batch["cond"].to(device)
            with torch.no_grad():
                z0 = ldm.encode(batch["target"].to(device))
            cond_ds = downsample_cond(cond, z0.shape[2:])
            loss = ldm.loss(z0, cond=cond_ds)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["diff"] = agg.get("diff", 0.0) + loss.item()
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, name, ldm, epoch, args.epochs, state_dict=True)

    def gen(cond):
        cond_ds = downsample_cond(cond, lat_spatial)
        shape = (cond.size(0), args.latent_channels, *lat_spatial)
        if flow:
            return ldm.sample(shape, device, steps=args.sample_steps, cond=cond_ds)
        return ldm.ddim_sample(shape, device, steps=args.sample_steps, cond=cond_ds)

    return ldm, gen
