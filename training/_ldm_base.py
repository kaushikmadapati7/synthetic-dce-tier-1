"""Shared two-stage latent-diffusion training (VAE first stage + denoiser).

Both the DDPM and flow-matching trainers wrap `train_ldm` with a different
`flow` flag; the only differences are the model class and the sampler.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ..models import (AutoencoderKL3D, WaveletFirstStage3D, LDM_DDPM, LDM_FlowMatching,
                      PatchDiscriminator3D, CondPatchDiscriminator3D,
                      d_hinge_loss, feature_matching_loss)
from .utils import (log_epoch, save_ckpt, downsample_cond, is_ckpt_epoch,
                    val_score, save_best, best_or_last_ckpt, prep_cond, EMA)

log = logging.getLogger("tier1")


def _cond_channels(args):
    """3 input sequences, doubled to 6 (image + availability mask) under Layer-1
    modality dropout so the U-Net can tell a missing sequence from a dark voxel."""
    return 6 if getattr(args, "modality_dropout", False) else 3


def _new_first_stage(args, device):
    """The (untrained) first stage: a learned VAE (default) or a fixed invertible
    3D Haar wavelet transform (``--first-stage wavelet``). Both expose the same
    encode/decode/decoder/scaling_factor/latent_shift/latent_channels interface so
    the LDM classes are agnostic. Returns (module, latent_channels)."""
    if getattr(args, "first_stage", "vae") == "wavelet":
        fs = WaveletFirstStage3D(levels=getattr(args, "wavelet_levels", 1)).to(device)
        return fs, fs.latent_channels
    fs = AutoencoderKL3D(in_channels=1, out_channels=1,
                         latent_channels=args.latent_channels, base_ch=args.base_ch,
                         ch_mults=tuple(args.ch_mults)).to(device)
    return fs, fs.latent_channels


def _wavelet_channel_weight(args, first_stage, device):
    """Per-subband loss weight (C,) for the wavelet first stage: variance**gamma,
    normalized to mean 1 (so overall loss scale is unchanged). gamma=1 recovers
    the natural image-space L2 balance (structural low-freq band dominates); the
    near-flat, unit-normalized high-freq subbands are down-weighted to their real
    energy. Returns None for the VAE stage or --wavelet-loss uniform (uniform MSE).
    Uses coeff_std (a saved buffer), so it's consistent train vs eval."""
    if getattr(args, "first_stage", "vae") != "wavelet" or getattr(args, "wavelet_loss", "energy") != "energy":
        return None
    var = (first_stage.coeff_std.to(device) ** 2)
    w = var ** getattr(args, "wavelet_loss_gamma", 1.0)
    return w * (w.numel() / w.sum())


def _build_ldm(args, device, flow: bool):
    """Construct the first stage + (flow|ddpm) LDM with the configured architecture."""
    vae, lat_ch = _new_first_stage(args, device)
    unet_kwargs = dict(in_channels=lat_ch, out_channels=lat_ch,
                       cond_channels=_cond_channels(args), base_ch=args.base_ch,
                       ch_mults=tuple(args.unet_ch_mults),
                       cond_dim=getattr(args, "cond_dim", 0))
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
    # the sampler's noise tensor must match the first stage's latent channels, NOT
    # args.latent_channels (they diverge for --first-stage wavelet: 8**levels)
    lat_ch = getattr(ldm.autoencoder, "latent_channels", args.latent_channels)
    src_t2w = flow and getattr(args, "flow_source", "noise") == "t2w"
    def gen(cond):
        # image-to-image flow: start the ODE from the encoded T2w (cond channel 0)
        source = ldm.encode(cond[:, 0:1]) if src_t2w else None
        cond = prep_cond(cond, args, training=False)   # Layer-1: fixed --eval-modalities subset
        cond_ds = downsample_cond(cond, lat_spatial)
        shape = (cond.size(0), lat_ch, *lat_spatial)
        if flow:
            return ldm.sample(shape, device, steps=args.sample_steps, cond=cond_ds,
                              guidance_scale=w, source=source)
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


def build_first_stage(args, train_loader, criterion, device):
    """Return a ready-to-use, frozen first stage. Wavelet: fit per-subband stats
    (no training, it's lossless). VAE: train it (or load --vae-ckpt)."""
    if getattr(args, "first_stage", "vae") == "wavelet":
        fs = WaveletFirstStage3D(levels=getattr(args, "wavelet_levels", 1)).to(device)
        fs.fit(train_loader, device)
        log.info(f"wavelet first stage: levels={fs.levels}, latent_channels={fs.latent_channels} "
                 f"(lossless, no training); subband std range "
                 f"[{fs.coeff_std.min().item():.3f}, {fs.coeff_std.max().item():.3f}]")
        _set_scaling_factor(fs, train_loader, device, center=getattr(args, "latent_center", False))
        return fs
    return train_vae(args, train_loader, criterion, device)


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
        joint = getattr(args, "vae_joint", False)   # train on DCE + T2w so encode(T2w)
        if joint:                                    # is well-defined for an i2i flow source
            log.info("VAE joint mode: reconstructing DCE + T2w (sound encode(T2w) for --flow-source t2w)")
        for epoch in range(args.vae_epochs):
            vae.train(); t0 = time.time(); agg = {}
            adv_on = disc is not None and epoch >= warmup
            for batch in train_loader:
                x = batch["target"].to(device)
                mask = batch["mask"].to(device)
                zw = batch["zone_weight"].to(device) if "zone_weight" in batch else None
                if joint:                            # stack T2w (cond ch0) as extra samples
                    t2w = batch["cond"][:, 0:1].to(device)
                    x = torch.cat([x, t2w], dim=0)
                    mask = torch.cat([mask, mask], dim=0)
                    if zw is not None:
                        zw = torch.cat([zw, zw], dim=0)
                recon, posterior = vae(x)
                rec, parts = criterion(recon, x, mask, zone_weight=zw)
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
    vae = build_first_stage(args, train_loader, criterion, device)
    lat_ch = getattr(vae, "latent_channels", args.latent_channels)

    # infer latent spatial size from one encode
    with torch.no_grad():
        z0 = vae.encode(next(iter(train_loader))["target"].to(device)).sample()
    lat_spatial = tuple(z0.shape[2:])
    log.info(f"latent grid: {z0.shape[1]}x{lat_spatial}")

    unet_kwargs = dict(in_channels=lat_ch, out_channels=lat_ch,
                       cond_channels=_cond_channels(args), base_ch=args.base_ch,
                       ch_mults=tuple(args.unet_ch_mults),
                       cond_dim=getattr(args, "cond_dim", 0))
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
    ema = EMA(ldm.unet, args.ema_decay) if getattr(args, "ema_decay", 0.0) > 0 else None

    # optional adversarial head on the flow's decoded one-shot prediction: a
    # conditional patch-discriminator + feature matching forces the generated
    # texture to be FAITHFUL (well-localized) rather than plausible-but-hallucinated
    # -- the ClinDCE flow+GAN ingredient our pure-velocity flow lacks.
    flow_adv = flow and getattr(args, "flow_adv_weight", 0.0) > 0
    disc = opt_d = None
    if flow_adv:
        disc = CondPatchDiscriminator3D(in_channels=1 + _cond_channels(args),
                                        base_ch=args.base_ch).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
        log.info(f"flow adversarial: cond patch-disc on, adv_w={args.flow_adv_weight} "
                 f"feat_w={getattr(args, 'flow_feat_weight', 1.0)} "
                 f"warmup={getattr(args, 'flow_adv_warmup', 10)} t={getattr(args, 'flow_adv_t', 0.5)}")

    # wavelet-subband energy weighting: reweight the velocity loss per channel by
    # natural subband variance so the structural (low-freq) band dominates instead
    # of the unit-normalized near-flat HF subbands. Fixed after the fit, so compute
    # once. None (uniform) for the VAE first stage or --wavelet-loss uniform.
    channel_weight = _wavelet_channel_weight(args, vae, device)
    if channel_weight is not None:
        log.info(f"wavelet energy loss (gamma={args.wavelet_loss_gamma}): channel weight "
                 f"range [{channel_weight.min().item():.3f}, {channel_weight.max().item():.3f}], "
                 f"top-band share {channel_weight.max().item()/channel_weight.numel():.2%}")

    gen = _ldm_gen(ldm, args, lat_spatial, device, flow)
    val_every = args.val_every or args.ckpt_every
    best = float("-inf")
    for epoch in range(args.epochs):
        ldm.unet.train(); t0 = time.time(); agg = {}
        adv_on = flow_adv and epoch >= getattr(args, "flow_adv_warmup", 10)
        src_t2w = flow and getattr(args, "flow_source", "noise") == "t2w"
        for batch in train_loader:
            cond_raw = batch["cond"].to(device)
            cond = prep_cond(cond_raw, args, training=True)       # Layer-1 dropout
            mask = batch["mask"].to(device)
            target_img = batch["target"].to(device)
            with torch.no_grad():
                z0 = ldm.encode(target_img)
                # image-to-image flow: source endpoint = encoded T2w (cond channel 0)
                source = ldm.encode(cond_raw[:, 0:1]) if src_t2w else None
            zone_weight = batch["zone_weight"].to(device) if "zone_weight" in batch else None
            cond_ds = downsample_cond(cond, z0.shape[2:])
            mask_ds = downsample_cond(mask, z0.shape[2:])          # prostate mask -> latent grid
            anchor_kw = {}
            if flow and getattr(args, "anchor_weight", 0.0) > 0:  # FlowMI-style image-space anchoring
                anchor_kw = dict(anchor_image=target_img, anchor_mask=mask,
                                 anchor_zone_weight=zone_weight,
                                 anchor_criterion=criterion, anchor_weight=args.anchor_weight,
                                 anchor_t_max=getattr(args, "anchor_t_max", 1.0))
            loss = ldm.loss(z0, cond=cond_ds, mask=mask_ds, roi_weight=args.roi_weight,
                            source=source, channel_weight=channel_weight, **anchor_kw)

            if adv_on:
                fake_img = ldm.predict_image(z0, cond=cond_ds, t_val=getattr(args, "flow_adv_t", 0.5))
                # discriminator step: real DCE vs detached fake, conditioned on bpMRI
                d_loss = d_hinge_loss(disc(target_img, cond)[0], disc(fake_img.detach(), cond)[0])
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                # generator adversarial + feature matching (real feats are fixed targets)
                fake_logits, fake_feats = disc(fake_img, cond)
                with torch.no_grad():
                    _, real_feats = disc(target_img, cond)
                g_adv = -fake_logits.mean()
                feat = feature_matching_loss(real_feats, fake_feats)
                loss = loss + args.flow_adv_weight * g_adv \
                    + getattr(args, "flow_feat_weight", 1.0) * feat
                for k, v in {"d": float(d_loss.detach()), "g_adv": float(g_adv.detach()),
                             "feat": float(feat.detach())}.items():
                    agg[k] = agg.get(k, 0.0) + v

            opt.zero_grad(); loss.backward(); opt.step()
            if ema: ema.update(ldm.unet)
            agg["diff"] = agg.get("diff", 0.0) + loss.item()
        log_epoch(epoch, args.epochs, agg, len(train_loader), time.time() - t0)
        save_ckpt(args, name, ldm, epoch, args.epochs, state_dict=True)
        # best-checkpoint selection scores via full sampling, so honor val_every
        # (raise it for LDMs if sampling the val set every interval is too slow)
        if val_loader is not None and is_ckpt_epoch(epoch, args.epochs, val_every):
            ldm.unet.eval()
            if ema: ema.apply_to(ldm.unet)        # score + save the EMA weights
            best = save_best(args, name, ldm, val_score(gen, val_loader, device), best)
            if ema: ema.restore(ldm.unet)
            ldm.unet.train()

    if ema:                                       # bake EMA into the returned gen + final ckpt
        ema.apply_to(ldm.unet)
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ldm.state_dict(), ckpt_dir / f"{name}_last.pt")
    return ldm, gen
