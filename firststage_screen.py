"""Reconstruction-ceiling screen for an LDM first stage (Cajas et al. 2604.12152).

The first stage's encode->decode fidelity on real DCE is an ABSOLUTE UPPER BOUND on
any LDM built on it, and predicts final quality WITHOUT training the diffusion model
(they report r=0.82). So screen a first stage in minutes before committing GPU:

    # MedVAE foundation VAE
    python -m tier1_static.firststage_screen --first-stage medvae --medvae-model medvae_4_1_3d \
        --data-root .../Bao_DCE --output-dir runs/gan_run --reader-cases 20
    # wavelet (lossless -> ceiling should be ~perfect) and a trained VAE (needs --vae-ckpt)
    python -m tier1_static.firststage_screen --first-stage wavelet --wavelet-levels 2 ...
    python -m tier1_static.firststage_screen --first-stage vae --vae-ckpt runs/x/checkpoints/vae_last.pt ...

Reports whole-volume + ROI reconstruction fidelity. For the realism goal the ROI
numbers matter most: roi_grad_ratio/roi_var_ratio ~1 means the first stage can
represent prostate texture (so an LDM on it *can* look realistic); a low ceiling
means it can't, no matter how good the diffusion model is.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .main import parse_args, build_data, set_seed, setup_logging
from .training._ldm_base import build_first_stage
from .metrics import eval_metrics, aggregate

log = logging.getLogger("tier1")


@torch.no_grad()
def main():
    args = parse_args()
    args.eval_only = True                       # load saved harmonizer; no re-fit / no training
    set_seed(args.seed)
    setup_logging(Path(args.output_dir))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train, val, test = build_data(args)
    fs = build_first_stage(args, train, None, device)   # criterion unused for wavelet/medvae/vae-ckpt
    loader = {"val": val, "test": test, "train": train}[getattr(args, "reader_split", "val")]
    if loader is None:
        log.error(f"no '{args.reader_split}' loader; nothing to screen"); return

    per, n = [], 0
    limit = getattr(args, "reader_cases", 20)
    for batch in loader:
        x = batch["target"].to(device)
        mask = batch["mask"].to(device)
        z = fs.encode(x).sample()
        xr = fs.decoder(z).clamp(-1, 1)          # decoder = raw encode->decode round-trip (the ceiling)
        if xr.shape != x.shape:
            import torch.nn.functional as F
            xr = F.interpolate(xr, size=x.shape[2:], mode="trilinear", align_corners=False)
        per.append(eval_metrics(xr, x, mask))
        n += x.shape[0]
        if n >= limit:
            break
    agg = aggregate(per)
    keys = ["psnr", "ssim", "mae", "psnr_roi", "ssim_roi", "mae_roi",
            "roi_pearson", "roi_var_ratio", "roi_grad_ratio", "roi_w1"]
    stage = getattr(args, "first_stage", "vae")
    detail = getattr(args, "medvae_model", "") if stage == "medvae" else (
        f"levels={getattr(args, 'wavelet_levels', 1)}" if stage == "wavelet" else "")
    log.info(f"FIRST-STAGE CEILING [{stage} {detail}] encode->decode over {n} cases: "
             f"{json.dumps({k: round(agg[k], 4) for k in keys if k in agg})}")


if __name__ == "__main__":
    main()
