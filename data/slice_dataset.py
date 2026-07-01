"""2D axial-slice view over the 3D Tier-1 datasets, for the pixel-space validation
lever. Each 3D case is unrolled into its prostate-bearing axial slices, so
training sees ~one sample per slice (the ~20x data multiplication 2D buys). All
3D preprocessing/harmonization is reused verbatim -- we only slice its output.

The slices are precomputed into memory ONCE at init as stacked float16 tensors:
this avoids reloading a full 3D volume per random slice under a shuffled loader
(which would read thousands of NIfTIs per epoch), and stacked tensors are
fork-shareable across DataLoader workers without per-object refcount churn.
Slices are returned as (C, H, W); the trainer wraps them to (C, 1, H, W) to reuse
the 3D loss/metrics.
"""
from __future__ import annotations

import logging

import torch
from torch.utils.data import Dataset

log = logging.getLogger("tier1")


class SliceDCEDataset(Dataset):
    def __init__(self, dataset3d, depth: int, prostate_only: bool = True,
                 min_area: int = 50, log_every: int = 100):
        conds, targets, masks, ids = [], [], [], []
        n = len(dataset3d)
        for c in range(n):
            s = dataset3d[c]
            for z in range(depth):
                m = s["mask"][:, z]
                if prostate_only and float(m.sum()) < min_area:
                    continue
                conds.append(s["cond"][:, z].half())
                targets.append(s["target"][:, z].half())
                masks.append(m.half())
                ids.append(f"{s['id']}_z{z:02d}")
            if log_every and (c + 1) % log_every == 0:
                log.info(f"[slice-prep] {c + 1}/{n} cases -> {len(ids)} slices")
        self.cond = torch.stack(conds) if conds else torch.empty(0)
        self.target = torch.stack(targets) if targets else torch.empty(0)
        self.mask = torch.stack(masks) if masks else torch.empty(0)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "cond": self.cond[i].float(),
            "target": self.target[i].float(),
            "mask": self.mask[i].float(),
            "id": self.ids[i],
        }
