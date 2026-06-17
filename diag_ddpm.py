"""Diagnostic for the broken ldm_ddpm sampling.

Isolates each stage so we can see WHERE the garbage comes from:
  - VAE round-trip (decode(encode(target)))  -> is the VAE/decoder fine?
  - encoded latent stats                      -> is scaling_factor sane (std~1)?
  - DDIM-sampled latent (clamped & unclamped) -> does the sampler blow up?
  - ancestral (full 1000-step) sampled latent -> is it DDIM-specific?

Run from the project root on a GPU node:
    python -m tier1_static.diag_ddpm runs/cmp_ldm_ddpm
"""
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")

from .main import build_data
from .training import LOADERS
from .training.utils import downsample_cond


def stats(name, t):
    t = t.float()
    print(f"  {name:42s} shape={tuple(t.shape)} mean={t.mean():+.3f} "
          f"std={t.std():.3f} min={t.min():+.3f} max={t.max():+.3f}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/cmp_ldm_ddpm"
    cfg = json.loads(Path(out, "config.json").read_text())
    cfg.update(eval_only=True, limit=8, num_workers=2, compute_fid=False)
    args = SimpleNamespace(**cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model} output_dir={out} device={device}")

    train_loader, test_loader = build_data(args)
    model, _ = LOADERS[args.model](args, train_loader, test_loader, device)
    model.eval()

    batch = next(iter(test_loader))
    cond = batch["cond"].to(device)
    target = batch["target"].to(device)
    print(f"\nscaling_factor = {model.autoencoder.scaling_factor:.4f}")

    with torch.no_grad():
        stats("target (image)", target)

        z0 = model.encode(target)                      # scaled latent
        stats("encoded z0 (scaled latent)", z0)

        recon = model.decode(z0)                       # VAE round-trip
        stats("VAE recon decode(encode(target))", recon)
        print(f"  VAE round-trip L1 vs target = {(recon - target).abs().mean():.4f}")

        cond_ds = downsample_cond(cond, z0.shape[2:])
        shape = (cond.size(0), args.latent_channels, *z0.shape[2:])

        zt = model.ddim_sample(shape, device, steps=args.sample_steps, cond=cond_ds, decode=False)
        stats("DDIM latent (clamp=5)", zt)
        stats("DDIM image (clamp=5)", model.decode(zt))

        zt_nc = model.ddim_sample(shape, device, steps=args.sample_steps, cond=cond_ds,
                                  decode=False, x0_clamp=0)
        stats("DDIM latent (no clamp)", zt_nc)
        stats("DDIM image (no clamp)", model.decode(zt_nc))

        za = model.sample(shape, device, cond=cond_ds, decode=False)  # full ancestral
        stats("ancestral latent (1000 steps)", za)
        stats("ancestral image (1000 steps)", model.decode(za))


if __name__ == "__main__":
    main()
