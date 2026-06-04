"""Evaluation: run a trained model's generator on the test set, report metrics,
and save qualitative samples."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import eval_metrics, aggregate, compute_fid

log = logging.getLogger("tier1")


@torch.no_grad()
def evaluate(args, gen, test_loader, device):
    """gen: callable(cond) -> predicted DCE volume in [-1, 1]."""
    if test_loader is None:
        log.warning("no test set available; skipping evaluation")
        return {}
    per_batch, first = [], None
    all_preds, all_targets = [], []
    compute_fid_flag = getattr(args, "compute_fid", True)
    for batch in test_loader:
        cond = batch["cond"].to(device); target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = gen(cond).clamp(-1, 1)
        if pred.shape != target.shape:  # guard against any model/target size mismatch
            pred = F.interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
        per_batch.append(eval_metrics(pred, target, mask))
        if compute_fid_flag:
            for i in range(pred.size(0)):
                all_preds.append(pred[i].cpu())
                all_targets.append(target[i].cpu())
        if first is None:
            first = (cond.cpu(), target.cpu(), pred.cpu(), batch["id"][0])
    metrics = aggregate(per_batch)
    if compute_fid_flag and all_preds:
        fid = compute_fid(all_preds, all_targets, device,
                          slices_per_volume=getattr(args, "fid_slices", 8),
                          batch_size=getattr(args, "fid_batch_size", 32))
        if fid is not None:
            metrics["fid"] = fid
    log.info(f"TEST metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    if first is not None:
        save_samples(Path(args.output_dir) / "samples", *first)
    return metrics


def save_samples(out_dir: Path, cond, target, pred, case_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import SimpleITK as sitk
        for name, vol in [("target", target[0, 0]), ("pred", pred[0, 0])]:
            sitk.WriteImage(sitk.GetImageFromArray(vol.numpy()),
                            str(out_dir / f"{case_id.replace('/', '_')}_{name}.nii.gz"))
    except Exception as e:  # noqa
        log.warning(f"NIfTI save skipped: {e}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = target.shape[2] // 2
        panels = [("T2w", cond[0, 0, d]), ("DWI", cond[0, 1, d]), ("ADC", cond[0, 2, d]),
                  ("DCE target", target[0, 0, d]), ("DCE pred", pred[0, 0, d])]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img.numpy(), cmap="gray"); ax.set_title(title); ax.axis("off")
        fig.suptitle(case_id); fig.tight_layout()
        fig.savefig(out_dir / "montage.png", dpi=120); plt.close(fig)
    except Exception as e:  # noqa
        log.warning(f"montage save skipped: {e}")
