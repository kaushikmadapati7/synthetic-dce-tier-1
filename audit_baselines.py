"""Sanity audit: does the trained model actually beat trivial baselines?

Aggregate metrics can look "reasonable" while a model has learned almost nothing --
especially here, where the DCE target inside the prostate is fairly homogeneous, so
a predictor that just guesses the right overall brightness scores well on MAE/PSNR/
SSIM without localizing anything. This script makes that explicit by scoring the
model against baselines that require no learning at all:

    const   cohort-wide ROI mean, identical for every case      (zero information)
    level   THIS case's own ROI mean -- an ORACLE that knows the
            correct brightness but has zero internal structure
    t2w     the T2w input channel copied straight through        (identity baseline)
    model   the trained generator

How to read it:
  * model must clearly beat `const`, or it has learned nothing useful.
  * `level` is the score obtainable with perfect brightness and NO structure. If the
    model is near or worse than `level`, then essentially all of its apparent skill
    is brightness matching, and the within-gland structure is not being predicted.
  * `level` also decomposes the error: MAE(const) - MAE(level) is how much of the
    signal is per-case brightness, and MAE(level) is what is left for structure.

Also reports how well case brightness is tracked (pearson of per-case ROI means)
and the signed bias, since a systematic offset inflates every ROI metric.

    python -m tier1_static.audit_baselines --model gan --ucsf-main-root <staged> \
        --output-dir runs/ucsf_gan --t2w-norm percentile --dce-norm robust \
        --dce-robust-k 1 --spatial-size 32 192 192 --audit-split test

The flags MUST match the training run (framing, normalization, cohort filters) or
the loaded checkpoint is scored on a different distribution than it was trained on.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .main import parse_args, build_data, set_seed, setup_logging
from .training import LOADERS
from .metrics import ssim3d, pearson

log = logging.getLogger("tier1")


def _roi_stats(vol, mask):
    m = mask > 0.5
    if m.sum() < 16:
        return None
    v = vol[m].float()
    return float(v.mean()), float(v.std())


def _roi_pearson(p: torch.Tensor, t: torch.Tensor):
    """Within-gland co-localization for ONE case (never pool across the batch --
    cross-case brightness differences masquerade as within-gland correlation)."""
    pc, tc = p - p.mean(), t - t.mean()
    den = torch.sqrt((pc ** 2).sum() * (tc ** 2).sum())
    return float((pc * tc).sum() / den) if float(den) > 0 else None


@torch.no_grad()
def main(argv=None):
    if argv is not None:                        # main.parse_args() takes no argv
        sys.argv = [sys.argv[0]] + list(argv)
    args = parse_args()
    args.eval_only = True                       # reuse the saved harmonizer, no re-fit
    split = getattr(args, "audit_split", "val")
    set_seed(args.seed)
    setup_logging(Path(args.output_dir))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader = build_data(args)
    loader = {"val": val_loader, "test": test_loader, "train": train_loader}[split]
    if loader is None:
        log.error(f"split '{split}' is empty"); return 1

    _, gen = LOADERS[args.model](args, train_loader, test_loader, device)
    log.info(f"auditing {args.model} on {split} ({len(loader.dataset)} cases)")

    rows, roi_vals = [], []
    for batch in loader:
        cond = batch["cond"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = gen(cond).clamp(-1, 1)
        for i in range(pred.size(0)):
            st = _roi_stats(target[i], mask[i])
            if st is None:
                continue
            t_mean, t_std = st
            m = mask[i] > 0.5
            t, p = target[i][m].float(), pred[i][m].float()
            # cond[i] is (3,D,H,W) and m is (1,D,H,W): index channel 0 as a SLICE so
            # the ranks match. cond[i][0][m] is (D,H,W) indexed by a 4-d mask -> IndexError.
            t2w = cond[i][0:1][m].float()
            roi_vals.append(t.cpu().numpy().astype(np.float16))
            rp_m, rp_t = _roi_pearson(p, t), _roi_pearson(t2w, t)
            rows.append({
                "t_mean": t_mean, "t_std": t_std,
                "p_mean": float(p.mean()), "p_std": float(p.std()),
                "mae_model": float((p - t).abs().mean()),
                "mae_level": float((t - t_mean).abs().mean()),     # oracle brightness
                "mae_t2w": float((t2w - t).abs().mean()),
                # the headline question: does the model localize better than copying T2w?
                "rp_model": rp_m if rp_m is not None else 0.0,
                "rp_t2w": rp_t if rp_t is not None else 0.0,
            })
    if not rows:
        log.error("no usable cases (empty masks?)"); return 1

    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)
    cohort_mean = A["t_mean"].mean()
    # exact const baseline: mean |t - cohort_mean| per case, over the cached ROI voxels
    mae_const = float(np.mean([np.abs(v.astype(np.float32) - cohort_mean).mean()
                               for v in roi_vals]))

    print(f"\n=== {args.model} · {split} · n={n} ===")
    print(f"{'predictor':10s} {'MAE_roi':>9s}   {'roi_r':>7s}   what it knows")
    print("-" * 72)
    print(f"{'const':10s} {mae_const:9.4f}        --      nothing (one number for every case)")
    print(f"{'t2w copy':10s} {A['mae_t2w'].mean():9.4f}   {A['rp_t2w'].mean():+7.3f}   identity baseline")
    print(f"{'MODEL':10s} {A['mae_model'].mean():9.4f}   {A['rp_model'].mean():+7.3f}   <-- trained generator")
    print(f"{'level':10s} {A['mae_level'].mean():9.4f}        --      ORACLE brightness, no structure")
    print()
    print(f"localization headroom over the identity baseline: "
          f"{A['rp_model'].mean() - A['rp_t2w'].mean():+.3f} roi_pearson")
    print()
    print(f"target ROI: mean {A['t_mean'].mean():+.3f} (between-case sd {A['t_mean'].std():.3f})"
          f"   within-case sd {A['t_std'].mean():.3f}")
    print(f"pred   ROI: mean {A['p_mean'].mean():+.3f} (between-case sd {A['p_mean'].std():.3f})"
          f"   within-case sd {A['p_std'].mean():.3f}")
    r = pearson(list(A["t_mean"]), list(A["p_mean"]))
    bias = (A["p_mean"] - A["t_mean"])
    print(f"brightness tracking: pearson(t_mean,p_mean) = {r if r is None else round(r,3)}"
          f"   signed bias {bias.mean():+.3f} +/- {bias.std():.3f}")

    beats_const = mae_const - A["mae_model"].mean()
    gap_to_level = A["mae_model"].mean() - A["mae_level"].mean()
    print()
    print(f"beats const by {beats_const:+.4f}  (must be clearly > 0 to have learned anything)")
    print(f"gap to oracle-brightness {gap_to_level:+.4f}  "
          f"({'WORSE than knowing only brightness' if gap_to_level > 0 else 'better than brightness alone'})")
    # how much of the signal is brightness vs structure
    frac = A["t_mean"].std() ** 2 / (A["t_mean"].std() ** 2 + (A["t_std"].mean() ** 2) + 1e-9)
    print(f"between-case brightness accounts for {100*frac:.0f}% of target ROI variance "
          f"(the rest is within-gland structure)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / f"audit_{split}.json").write_text(json.dumps(
        {"n": n, "mae_const": float(mae_const),
         "mae_t2w": float(A["mae_t2w"].mean()),
         "mae_model": float(A["mae_model"].mean()),
         "mae_level_oracle": float(A["mae_level"].mean()),
         "roi_pearson_model": float(A["rp_model"].mean()),
         "roi_pearson_t2w_baseline": float(A["rp_t2w"].mean()),
         "brightness_pearson": r, "bias_mean": float(bias.mean()),
         "bias_sd": float(bias.std()),
         "target_between_case_sd": float(A["t_mean"].std()),
         "target_within_case_sd": float(A["t_std"].mean())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
