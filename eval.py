"""Evaluation: run a trained model's generator on the test set, report metrics,
and save qualitative samples."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import (eval_metrics, aggregate, compute_fid, roi_p75, pearson,
                      REALISM_KEYS, realism_score)

log = logging.getLogger("tier1")


def _log_realism(metrics: dict, tag: str):
    """Log the label-free realism panel (how real it looks to a reader) separately
    from the faithfulness metrics, per the clinical-realism objective. FID is the
    primary realism number; roi_var_ratio/roi_grad_ratio (->1) and roi_w1 (->0)
    are the texture/detail/intensity-distribution companions."""
    panel = {k: round(metrics[k], 4) for k in REALISM_KEYS if k in metrics}
    if not panel:
        return
    rs = realism_score(metrics)
    extra = f"  realism_score={rs:.4f}" if rs is not None else ""
    log.info(f"{tag} REALISM (real-looking? var_ratio/grad_ratio->1, w1->0, fid lower): "
             f"{json.dumps(panel)}{extra}")


@torch.no_grad()
def evaluate(args, gen, test_loader, device):
    """gen: callable(cond) -> predicted DCE volume in [-1, 1]."""
    if test_loader is None:
        log.warning("no test set available; skipping evaluation")
        return {}
    per_batch, first = [], None
    all_preds, all_targets = [], []
    p75_real, p75_pred = [], []   # per-case ROI enhancement level -> cross-case scatter
    compute_fid_flag = getattr(args, "compute_fid", True)
    for batch in test_loader:
        cond = batch["cond"].to(device); target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        zones = batch["zones"].to(device) if "zones" in batch else None
        pred = gen(cond).clamp(-1, 1)
        if pred.shape != target.shape:  # guard against any model/target size mismatch
            pred = F.interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
        per_batch.append(eval_metrics(pred, target, mask, zones))
        for i in range(pred.size(0)):
            rr = roi_p75(target[i:i+1], mask[i:i+1]); pp = roi_p75(pred[i:i+1], mask[i:i+1])
            if rr is not None and pp is not None:
                p75_real.append(rr); p75_pred.append(pp)
        if compute_fid_flag:
            for i in range(pred.size(0)):
                all_preds.append(pred[i].cpu())
                all_targets.append(target[i].cpu())
        if first is None:
            first = (cond.cpu(), target.cpu(), pred.cpu(), mask.cpu(), batch["id"][0])
    metrics = aggregate(per_batch)
    # cross-case "does synthetic enhancement track real" (their Fig. scatter r)
    p75_corr = pearson(p75_real, p75_pred)
    if p75_corr is not None:
        metrics["p75_corr"] = p75_corr
    if compute_fid_flag and all_preds:
        fid = compute_fid(all_preds, all_targets, device,
                          slices_per_volume=getattr(args, "fid_slices", 8),
                          batch_size=getattr(args, "fid_batch_size", 32))
        if fid is not None:
            metrics["fid"] = fid
    log.info(f"TEST metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    _log_realism(metrics, "TEST")
    if first is not None:
        save_samples(Path(args.output_dir) / "samples", *first)
    return metrics


@torch.no_grad()
def save_indist_sample(args, gen, loader, device, montage_name="montage_indist"):
    """IN-DISTRIBUTION (val) evaluation + qualitative sample. The held-out TEST
    center (jiulong) has near-flat DCE targets (std ~0.03), which both undersells
    the model and makes the ROI radiomic metrics meaningless -- ``roi_var_ratio``
    explodes against ~zero target variance, and ``roi_pearson``/``p75_corr`` have
    no enhancement gradient to track. The val split is the only place the clinical-
    fidelity numbers are interpretable, so we aggregate them here and log a
    ``VAL metrics:`` line, plus the montage -> samples/montage_indist.png."""
    if loader is None:
        log.info("no val loader; skipping in-distribution eval")
        return
    per_batch, p75_real, p75_pred, first = [], [], [], None
    all_preds, all_targets = [], []
    compute_fid_flag = getattr(args, "compute_fid", True)
    for batch in loader:
        cond = batch["cond"].to(device); target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        zones = batch["zones"].to(device) if "zones" in batch else None
        pred = gen(cond).clamp(-1, 1)
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
        per_batch.append(eval_metrics(pred, target, mask, zones))
        for i in range(pred.size(0)):
            rr = roi_p75(target[i:i+1], mask[i:i+1]); pp = roi_p75(pred[i:i+1], mask[i:i+1])
            if rr is not None and pp is not None:
                p75_real.append(rr); p75_pred.append(pp)
            if compute_fid_flag:
                all_preds.append(pred[i].cpu()); all_targets.append(target[i].cpu())
        if first is None:
            first = (cond.cpu(), target.cpu(), pred.cpu(), mask.cpu(), batch["id"][0])
    metrics = aggregate(per_batch)
    pc = pearson(p75_real, p75_pred)
    if pc is not None:
        metrics["p75_corr"] = pc
    if compute_fid_flag and all_preds:   # VAL is the reliable in-distribution set -> FID here too
        fid = compute_fid(all_preds, all_targets, device,
                          slices_per_volume=getattr(args, "fid_slices", 8),
                          batch_size=getattr(args, "fid_batch_size", 32))
        if fid is not None:
            metrics["fid"] = fid
    log.info(f"VAL metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    _log_realism(metrics, "VAL")
    if first is not None:
        cond, target, pred, mask, cid = first
        log.info(f"in-distribution sample [{cid}] (val)")
        save_samples(Path(args.output_dir) / "samples", cond, target, pred, mask,
                     "indist_" + cid, montage_name=montage_name)
    return metrics


def save_samples(out_dir: Path, cond, target, pred, mask, case_id, montage_name="montage"):
    """Qualitative dump for one case. Unlike a naive mid-slice montage, this picks
    the slice with the most prostate-mask voxels, windows the DCE consistently
    (shared robust percentile range, so a single hot vessel voxel can't crush the
    contrast to black), draws a pred-target error map + mask outline, and logs the
    target's ROI intensity stats so we can tell a model failure from a near-empty
    target. ``mask`` is (1,1,D,H,W); may be all-zero (then falls back to mid-slice)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt, prd, m = target[0, 0], pred[0, 0], mask[0, 0]            # (D,H,W) each
    D = tgt.shape[0]
    has_mask = bool(m.sum() > 0)
    d = int(m.sum(dim=(1, 2)).argmax()) if has_mask else D // 2   # most-ROI slice

    try:
        import SimpleITK as sitk
        for name, vol in [("target", tgt), ("pred", prd)]:
            sitk.WriteImage(sitk.GetImageFromArray(vol.numpy()),
                            str(out_dir / f"{case_id.replace('/', '_')}_{name}.nii.gz"))
    except Exception as e:  # noqa
        log.warning(f"NIfTI save skipped: {e}")

    # ROI signal check: is there real enhancement in the prostate to predict?
    if has_mask:
        mb = m > 0.5
        tr, pr = tgt[mb], prd[mb]
        q = torch.tensor([0.01, 0.5, 0.99])
        log.info(f"ROI signal [{case_id}] slice {d}/{D} nvox={int(mb.sum())} | "
                 f"target mean={tr.mean():+.3f} std={tr.std():.3f} "
                 f"p01/50/99={[round(float(x), 3) for x in tr.quantile(q)]} | "
                 f"pred mean={pr.mean():+.3f} std={pr.std():.3f} "
                 f"p01/50/99={[round(float(x), 3) for x in pr.quantile(q)]}")

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # shared DCE window from the target's ROI (or whole slice) 1-99 percentile
        ref = tgt[m > 0.5] if has_mask else tgt
        lo, hi = (float(x) for x in np.percentile(ref.numpy(), [1, 99]))
        if hi <= lo:
            lo, hi = float(tgt.min()), float(tgt.max())
        err = (prd[d] - tgt[d]).numpy()
        emax = max(abs(err.min()), abs(err.max()), 1e-6)
        msl = (m[d] > 0.5).numpy()
        panels = [("T2w", cond[0, 0, d].numpy(), None, "gray"),
                  ("DWI", cond[0, 1, d].numpy(), None, "gray"),
                  ("ADC", cond[0, 2, d].numpy(), None, "gray"),
                  ("DCE target", tgt[d].numpy(), (lo, hi), "gray"),
                  ("DCE pred", prd[d].numpy(), (lo, hi), "gray"),
                  ("pred - target", err, (-emax, emax), "seismic")]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, img, rng, cmap) in zip(axes, panels):
            kw = {"cmap": cmap}
            if rng is not None:
                kw["vmin"], kw["vmax"] = rng
            ax.imshow(img, **kw); ax.set_title(title, fontsize=8); ax.axis("off")
            if msl.any() and title.startswith("DCE"):
                ax.contour(msl, levels=[0.5], colors="lime", linewidths=0.6)
        fig.suptitle(f"{case_id}  (slice {d}/{D}, DCE window [{lo:.2f}, {hi:.2f}])", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{montage_name}.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa
        log.warning(f"montage save skipped: {e}")
