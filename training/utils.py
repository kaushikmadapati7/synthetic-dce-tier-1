"""Shared helpers for the per-model training loops."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger("tier1")


def log_epoch(epoch, total, agg, n_batches, dt, tag=""):
    msg = "  ".join(f"{k}={v / max(1, n_batches):.4f}" for k, v in agg.items())
    log.info(f"[{tag or 'epoch'} {epoch + 1}/{total}] {msg}  ({dt:.1f}s)")


def is_ckpt_epoch(epoch, total_epochs, every):
    """True on the configured interval and always on the final epoch."""
    return (epoch + 1) % max(1, every) == 0 or (epoch + 1) == total_epochs


def save_ckpt(args, name, model, epoch, total_epochs, state_dict=False):
    """Save on the configured interval and always on the final epoch."""
    if not is_ckpt_epoch(epoch, total_epochs, args.ckpt_every):
        return
    d = Path(args.output_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / f"{name}_last.pt")


@torch.no_grad()
def val_score(gen, val_loader, device):
    """Checkpoint-selection score: mean ROI-SSIM of ``gen`` over the val set
    (falls back to global SSIM when no mask is present). Returns -inf when there
    is no val set, so best-checkpoint tracking is a silent no-op in that case."""
    if val_loader is None:
        return float("-inf")
    from ..metrics import eval_metrics, aggregate
    per = []
    for batch in val_loader:
        cond = batch["cond"].to(device); target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = gen(cond).clamp(-1, 1)
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
        per.append(eval_metrics(pred, target, mask))
    agg = aggregate(per)
    return agg.get("ssim_roi", agg.get("ssim", float("-inf")))


def save_best(args, name, model, score, best):
    """Write ``{name}_best.pt`` when ``score`` beats the running ``best``; return
    the (possibly updated) best. Lets long runs keep their peak checkpoint even as
    later epochs degrade (e.g. GAN adversarial over-training)."""
    if score <= best:
        return best
    d = Path(args.output_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / f"{name}_best.pt")
    prev = "-inf" if best == float("-inf") else f"{best:.4f}"
    log.info(f"  ** new best {name}: val_ssim_roi={score:.4f} (was {prev}) -> {name}_best.pt")
    return score


def best_or_last_ckpt(output_dir, name):
    """Prefer ``{name}_best.pt`` (peak val score) over ``{name}_last.pt`` (final
    epoch) for eval-only loading; fall back to last when no best was saved."""
    d = Path(output_dir) / "checkpoints"
    best = d / f"{name}_best.pt"
    return best if best.exists() else d / f"{name}_last.pt"


def downsample_cond(cond: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(cond, size=tuple(size), mode="trilinear", align_corners=False)
