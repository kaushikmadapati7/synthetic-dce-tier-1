"""Shared two-stage latent-diffusion training (VAE first stage + denoiser).

Both the DDPM and flow-matching trainers wrap `train_ldm` with a different
`flow` flag; the only differences are the model class and the sampler.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import (AutoencoderKL3D, LDM_DDPM, LDM_FlowMatching,
                      PatchDiscriminator3D, d_hinge_loss)
from .utils import (log_epoch, save_ckpt, downsample_cond, is_ckpt_epoch,
                    val_score, save_best, best_or_last_ckpt, prep_cond)

log = logging.getLogger("tier1")


def _cond_channels(args):
    """3 input sequences, doubled to 6 (image + availability mask) under Layer-1
    modality dropout so the U-Net can tell a missing sequence from a dark voxel."""
    return 6 if getattr(args, "modality_dropout", False) else 3


def _build_ldm(args, device, flow: bool):
    """Construct the VAE + (flow|ddpm) LDM with the configured architecture."""
    vae = AutoencoderKL3D(in_channels=1, out_channels=1,
                          latent_channels=args.latent_channels, base_ch=args.base_ch,
                          ch_mults=tuple(args.ch_mults)).to(device)
    unet_kwargs = dict(in_channels=args.latent_channels, out_channels=args.latent_channels,
                       cond_channels=_cond_channels(args), base_ch=args.base_ch,
                       ch_mults=tuple(args.unet_ch_mults))
    cfg_dropout = getattr(args, "cfg_dropout", 0.0)
    if flow:
        ldm = LDM_FlowMatching(autoencoder=vae, unet_kwargs=unet_kwargs,
                               cfg_dropout=cfg_dropout).to(device)
    else:
        ldm = LDM_DDPM(autoencoder=vae, timesteps=args.timesteps,
                       beta_schedule=getattr(args, "beta_schedule", "linear"),
                       unet_kwargs=unet_kwargs, cfg_dropout=cfg_dropout).to(device)
    return vae, ldm


def _set_scaling_factor(vae, train_loader, device, center=False):
    """Latent std -> scaling_factor (~1 std) and, if ``center``, latent mean ->
    latent_shift (zero-mean latents, so a DDPM's N(0,1) prior matches). Neither is
    stored in the checkpoint, so both are (re)computed for training and eval-only."""
    vae.eval()
    with torch.no_grad():
        zs = []
        for i, batch in enumerate(train_loader):
            zs.append(vae.encode(batch["target"].to(device)).sample())
            if i >= 4:
                break
        z = torch.cat(zs)
        std = z.std().item()
        mean = z.mean().item() if center else 0.0
    vae.scaling_factor = 1.0 / (std + 1e-8)
    vae.latent_shift = mean
    log.info(f"latent std={std:.4f} mean={mean:.4f} -> scaling_factor={vae.scaling_factor:.4f} "
             f"latent_shift={vae.latent_shift:.4f}")


def _ldm_gen(ldm, args, lat_spatial, device, flow: bool):
    w = getattr(args, "guidance_scale", 1.0)
    def gen(cond):
        cond = prep_cond(cond, args, training=False)   # Layer-1: fixed --eval-modalities subset
        cond_ds = downsample_cond(cond, lat_spatial)
        shape = (cond.size(0), args.latent_channels, *lat_spatial)
        if flow:
            return ldm.sample(shape, device, steps=args.sample_steps, cond=cond_ds,
                              guidance_scale=w)
        return ldm.ddim_sample(shape, device, steps=args.sample_steps, cond=cond_ds,
                               x0_clamp=getattr(args, "x0_clamp", 3.0), guidance_scale=w)
    return gen


def load_ldm(args, train_loader, test_loader, device, flow: bool):
    """Rebuild the LDM, load its checkpoint, restore scaling_factor, return gen (eval-only)."""
    vae, ldm = _build_ldm(args, device, flow)
    name = "ldm_flow" if flow else "ldm_ddpm"
    ckpt = best_or_last_ckpt(args.output_dir, name)
    ldm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    ldm.eval()
    log.info(f"loaded {name} checkpoint {ckpt}")
    # scaling_factor + latent_shift aren't in the state_dict; recompute (same flags as training)
    _set_scaling_factor(vae, train_loader, device, center=getattr(args, "latent_center", False))
    with torch.no_grad():
        z0 = ldm.encode(next(iter(train_loader))["target"].to(device))
    lat_spatial = tuple(z0.shape[2:])
    return ldm, _ldm_gen(ldm, args, lat_spatial, device, flow)


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
        # optional adversarial term: a patch discriminator sharpens reconstructions
        # the L1/SSIM/perceptual recon loss blurs (standard LDM autoencoder recipe).
        adv_w = getattr(args, "vae_adv_weight", 0.0)
        warmup = getattr(args, "vae_adv_warmup", 5)
        disc = opt_d = None
        if adv_w > 0:
            disc = PatchDiscriminator3D(in_channels=1, base_ch=args.base_ch).to(device)
            opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
            log.info(f"VAE adversarial: patch-disc on, weight={adv_w}, warmup={warmup} epochs")
        for epoch in range(args.vae_epochs):
            vae.train(); t0 = time.time(); agg = {}
            adv_on = disc is not None and epoch >= warmup
            for batch in train_loader:
                x = batch["target"].to(device)
                mask = batch["mask"].to(device)
                recon, posterior = vae(x)
                rec, parts = criterion(recon, x, mask)
                loss = rec + args.kl_weight * posterior.kl()
                parts = {**parts, "kl": posterior.kl().item()}
                if adv_on:
                    # discriminator step (real DCE vs detached recon)
                    d_loss = d_hinge_loss(disc(x), disc(recon.detach()))
                    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                    g_adv = -disc(recon).mean()             # hinge generator term
                    loss = loss + adv_w * g_adv
                    parts = {**parts, "d": float(d_loss.detach()), "g_adv": float(g_adv.detach())}
                opt.zero_grad(); loss.backward(); opt.step()
                for k, v in {"vae": loss.item(), **parts}.items():
                    agg[k] = agg.get(k, 0.0) + v
            log_epoch(epoch, args.vae_epochs, agg, len(train_loader), time.time() - t0, "VAE")
            save_ckpt(args, "vae", vae, epoch, args.vae_epochs, state_dict=True)

    # scaling factor (+ optional centering) for the latent diffusion stage
    _set_scaling_factor(vae, train_loader, device, center=getattr(args, "latent_center", False))
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def train_ldm(args, train_loader, val_loader, test_loader, criterion, device, flow: bool):
    vae = train_vae(args, train_loader, criterion, device)

    # infer latent spatial size from one encode
    with torch.no_grad():
        z0 = vae.encode(next(iter(train_loader))["target"].to(device)).sample()
    lat_spatial = tuple(z0.shape[2:])
    log.info(f"latent grid: {z0.shape[1]}x{lat_spatial}")

    unet_kwargs = dict(in_channels=args.latent_channels, out_channels=args.latent_channels,
                       cond_channels=_cond_channels(args), base_ch=args.base_ch,
                       ch_mults=tuple(args.unet_ch_mults))
    name = "ldm_flow" if flow else "ldm_ddpm"
    cfg_dropout = getattr(args, "cfg_dropout", 0.0)
    if flow:
        ldm = LDM_FlowMatching(autoencoder=vae, unet_kwargs=unet_kwargs,
                               cfg_dropout=cfg_dropout).to(device)
    else:
        ldm = LDM_DDPM(autoencoder=vae, timesteps=args.timesteps,
                       beta_schedule=getattr(args, "beta_schedule", "linear"),
                       unet_kwargs=unet_kwargs, cfg_dropout=cfg_dropout).to(device)
    log.info(f"UNet params: {sum(p.numel() for p in ldm.unet.parameters())/1e6:.1f}M "
             f"(cfg_dropout={cfg_dropout}, guidance_scale={getattr(args, 'guidance_scale', 1.0)})")

    opt = torch.optim.Adam(ldm.unet.parameters(), lr=args.lr)
    gen = _ldm_gen(ldm, args, lat_spatial, device, flow)
    val_every = args.val_every or args.ckpt_every
    best = float("-inf")
    for epoch in range(args.epochs):
        ldm.unet.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            cond = prep_cond(batch["cond"].to(device), args, training=True)  # Layer-1 dropout
            mask = batch["mask"].to(device)
            with torch.no_grad():
                z0 = ldm.encode(batch["target"].to(device))
            cond_ds = downsample_cond(cond, z0.shape[2:])
            mask_ds = downsample_cond(mask, z0.shape[2:])          # prostate mask -> latent grid
            loss = ldm.loss(z0, cond=cond_ds, mask=mask_ds, roi_weight=args.roi_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["diff"] = agg.get("diff", 0.0) + loss.item()
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, name, ldm, epoch, args.epochs, state_dict=True)
        # best-checkpoint selection scores via full sampling, so honor val_every
        # (raise it for LDMs if sampling the val set every interval is too slow)
        if val_loader is not None and is_ckpt_epoch(epoch, args.epochs, val_every):
            ldm.unet.eval()
            best = save_best(args, name, ldm, val_score(gen, val_loader, device), best)
            ldm.unet.train()

    return ldm, gen
