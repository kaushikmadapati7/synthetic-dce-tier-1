"""End-to-end Tier-1 pipeline entrypoint: data -> harmonization -> model ->
train -> eval.

Task: predict the peak-contrast DCE volume from (T2w, DWI, ADC).

Per-model training loops live in tier1_static/training/ (one file per model);
evaluation lives in tier1_static/eval.py.

Run from the project root:
    python -m tier1_static.main --model ldm_flow --data-root /path/to/Bao_DCE \
        --output-dir runs/exp1 --epochs 100

Outputs written to --output-dir:
    config.json, harmonizer.json, train.log,
    checkpoints/, metrics.json, samples/{*.nii.gz, montage.png}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PreprocessConfig, Harmonizer, build_tier1_datasets,
                   CanonicalDCEDataset, fit_harmonizer_from_dataset,
                   CANONICAL_HOSPITALS, TIER1_TEST_HOSPITALS)
from .loss import CustomLoss
from .training import TRAINERS
from .eval import evaluate

log = logging.getLogger("tier1")


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train.log"); fh.setFormatter(fmt); log.addHandler(fh)


def resolve_medicalnet_weights(args):
    """Default --medicalnet-weights to the depth-matched MedicalNet file in
    pretrain/, and warn loudly if the perceptual term is on but no real weights
    are available (otherwise it silently runs on a randomly-initialized backbone).
    """
    if args.perceptual <= 0:
        return
    if args.medicalnet_weights is None:
        cand = Path(f"pretrain/resnet_{args.perceptual_depth}.pth")
        if cand.exists():
            args.medicalnet_weights = str(cand)
            log.info(f"perceptual: using MedicalNet weights {cand}")
            return
    elif Path(args.medicalnet_weights).exists():
        log.info(f"perceptual: using MedicalNet weights {args.medicalnet_weights}")
        return
    log.warning(
        f"perceptual_weight={args.perceptual} but no MedicalNet weights found "
        f"(--medicalnet-weights={args.medicalnet_weights}, expected e.g. "
        f"pretrain/resnet_{args.perceptual_depth}.pth). The perceptual term will "
        f"run on a RANDOMLY-INITIALIZED backbone — pass valid weights or set "
        f"--perceptual 0 to disable it."
    )


def make_criterion(args, device) -> CustomLoss:
    return CustomLoss(
        l1_weight=args.l1, ssim_weight=args.ssim, perceptual_weight=args.perceptual,
        roi_weight=args.roi_weight,
        perceptual_depth=args.perceptual_depth, perceptual_shortcut=args.perceptual_shortcut,
        medicalnet_weights=args.medicalnet_weights,
    ).to(device)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_data(args):
    cfg = PreprocessConfig(reference=args.reference, spatial_size=tuple(args.spatial_size))
    out = Path(args.output_dir)

    harmonizer = None
    if args.harmonize:
        harmonizer = Harmonizer()
        fit_hospitals = [h for h in CANONICAL_HOSPITALS if h not in TIER1_TEST_HOSPITALS]
        fit_ds = CanonicalDCEDataset(args.data_root, fit_hospitals, cfg)
        if len(fit_ds) == 0:
            log.warning("no canonical exams found to fit harmonizer; disabling harmonization")
            harmonizer = None
        else:
            log.info(f"fitting harmonizer (Nyul) on {min(len(fit_ds), args.harmonize_max)} cases ...")
            fit_harmonizer_from_dataset(harmonizer, fit_ds, max_cases=args.harmonize_max)
            harmonizer.save(out / "harmonizer.json")

    train = build_tier1_datasets(args.data_root, cfg, "train", harmonizer)
    test = build_tier1_datasets(args.data_root, cfg, "test", harmonizer)
    if args.limit:
        train = torch.utils.data.Subset(train, range(min(args.limit, len(train))))
        test = torch.utils.data.Subset(test, range(min(max(1, args.limit // 4), len(test))))
    log.info(f"train cases: {len(train)}  test cases: {len(test)}")

    dl = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size, shuffle=shuf,
                                     num_workers=args.num_workers, drop_last=shuf,
                                     pin_memory=torch.cuda.is_available())
    return dl(train, True), (dl(test, False) if len(test) else None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    resolve_medicalnet_weights(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.info(f"device={device}  model={args.model}")
    log.info(f"config: {json.dumps(vars(args))}")

    train_loader, test_loader = build_data(args)
    criterion = make_criterion(args, device)

    t0 = time.time()
    trainer = TRAINERS[args.model]
    _, gen = trainer(args, train_loader, test_loader, criterion, device)
    log.info(f"training done in {(time.time() - t0) / 60:.1f} min")

    metrics = evaluate(args, gen, test_loader, device)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"artifacts written to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Tier-1 synthetic DCE pipeline")
    p.add_argument("--model", choices=list(TRAINERS), default="ldm_flow")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="runs/exp")
    # data / geometry
    p.add_argument("--spatial-size", type=int, nargs=3, default=[32, 192, 192])
    p.add_argument("--reference", choices=["t2w", "dce", "iso"], default="dce")
    p.add_argument("--harmonize", action="store_true", default=True)
    p.add_argument("--no-harmonize", dest="harmonize", action="store_false")
    p.add_argument("--harmonize-max", type=int, default=200)
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0, help="cap #cases (smoke testing)")
    # model size
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--n-upsamples", type=int, default=4, help="GAN up/down sampling depth")
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 4], help="VAE")
    p.add_argument("--unet-ch-mults", type=int, nargs="+", default=[1, 2, 4], help="diffusion UNet")
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--vae-ckpt", default="")
    # loss
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--ssim", type=float, default=1.0)
    p.add_argument("--perceptual", type=float, default=0.1)
    p.add_argument("--perceptual-depth", type=int, default=18)
    p.add_argument("--perceptual-shortcut", default="A")
    p.add_argument("--medicalnet-weights", default="")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--roi-weight", type=float, default=10.0,
                   help="how much more ROI voxels count in the recon loss (1.0 = off; "
                        "needs prostate masks in the data)")
    # evaluation
    p.add_argument("--compute-fid", action="store_true", default=True)
    p.add_argument("--no-fid", dest="compute_fid", action="store_false")
    p.add_argument("--fid-slices", type=int, default=8,
                   help="axial slices per volume for FID (2D Inception)")
    p.add_argument("--fid-batch-size", type=int, default=32)
    args = p.parse_args()
    if not args.medicalnet_weights:
        args.medicalnet_weights = None
    return args


if __name__ == "__main__":
    main()
