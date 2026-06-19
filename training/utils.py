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


# ---------------------------------------------------------------------------
# Modality dropout (Layer 1: missing-sequence robustness)
# ---------------------------------------------------------------------------
# The conditioning is a fixed (B, 3, ...) stack [T2w, DWI, ADC]. To make one model
# robust to missing sequences we (a) randomly drop a subset of those channels each
# training step and (b) append a per-modality availability mask so the network can
# distinguish "missing" from "dark voxel" -> cond_channels 3 -> 6. Sampling/eval
# uses a *fixed* availability set (--eval-modalities) so we can trace the full
# degradation curve (111 -> 101 -> 100 ...) from one trained model.

def sample_modality_keep(batch_size, n_mod=3, full_prob=0.3, device=None):
    """Random per-sample availability mask for training. With prob ``full_prob``
    keep all sequences (preserves full-modality quality); otherwise drop a random
    subset but always keep >=1 (an all-empty condition is meaningless here)."""
    keep = torch.rand(batch_size, n_mod, device=device) > 0.5
    keep[torch.rand(batch_size, device=device) < full_prob] = True
    empty = ~keep.any(dim=1)
    if empty.any():
        rows = empty.nonzero(as_tuple=True)[0]
        keep[rows, torch.randint(0, n_mod, (rows.numel(),), device=device)] = True
    return keep


def parse_modality_keep(spec, batch_size, n_mod=3, device=None):
    """Fixed availability mask from a bit string like '111'/'101'/'100' (T2w,DWI,ADC)."""
    bits = [c == "1" for c in str(spec)]
    bits = (bits + [True] * n_mod)[:n_mod]
    keep = torch.tensor(bits, device=device, dtype=torch.bool)
    return keep.unsqueeze(0).expand(batch_size, -1)


def apply_modality_availability(cond, keep):
    """``cond`` (B,C,...) + ``keep`` (B,C bool) -> (B,2C,...): dropped channels
    zeroed, concatenated with C availability-mask channels (1 present / 0 missing).
    The mask is spatially uniform, so a later trilinear downsample to the latent
    grid preserves it exactly."""
    b, c = cond.shape[:2]
    k = keep.view(b, c, *([1] * (cond.dim() - 2))).to(cond.dtype)
    return torch.cat([cond * k, k.expand_as(cond)], dim=1)


def prep_cond(cond, args, training):
    """Apply Layer-1 modality dropout/availability to ``cond`` when enabled; a
    no-op (returns ``cond`` unchanged) otherwise. Train: random keep-set. Eval:
    the fixed --eval-modalities subset."""
    if not getattr(args, "modality_dropout", False):
        return cond
    if training:
        keep = sample_modality_keep(cond.size(0), device=cond.device,
                                    full_prob=getattr(args, "modality_full_prob", 0.3))
    else:
        keep = parse_modality_keep(getattr(args, "eval_modalities", "111"),
                                   cond.size(0), device=cond.device)
    return apply_modality_availability(cond, keep)


def downsample_cond(cond: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(cond, size=tuple(size), mode="trilinear", align_corners=False)
