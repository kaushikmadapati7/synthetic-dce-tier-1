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


def save_ckpt(args, name, model, epoch, total_epochs, state_dict=False):
    """Save on the configured interval and always on the final epoch."""
    if (epoch + 1) % args.ckpt_every and (epoch + 1) != total_epochs:
        return
    d = Path(args.output_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    obj = model.state_dict() if state_dict else model.state_dict()
    torch.save(obj, d / f"{name}_last.pt")


def downsample_cond(cond: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(cond, size=tuple(size), mode="trilinear", align_corners=False)
