"""2D axial-slice view over the 3D Tier-1 datasets, for the pixel-space validation
lever. Each 3D case (cond 3xDxHxW, target 1xDxHxW, mask, zones) is unrolled into
its D axial slices, so training sees ~D samples per case (the ~20x data
multiplication that 2D buys). All 3D preprocessing/harmonization is reused
verbatim -- we only slice its output.

Index is (case, z) sorted by case; a 1-item cache holds the last processed 3D
case, so a worker streaming consecutive slices of a case loads it once. Slices
are returned as (C, H, W); the trainer wraps them to (C, 1, H, W) to reuse the 3D
loss/metrics (SSIM/pooling degrade gracefully at D=1).
"""
from __future__ import annotations

from torch.utils.data import Dataset


class SliceDCEDataset(Dataset):
    def __init__(self, dataset3d, depth: int, prostate_only: bool = False):
        self.ds = dataset3d
        self.depth = depth
        self.prostate_only = prostate_only
        # (case_idx, z) for every slice; sorted by case for cache locality.
        self.index = [(c, z) for c in range(len(dataset3d)) for z in range(depth)]
        self._cache_idx = None
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _case(self, c):
        if c != self._cache_idx:
            self._cache = self.ds[c]
            self._cache_idx = c
        return self._cache

    def __getitem__(self, i):
        c, z = self.index[i]
        s = self._case(c)
        out = {
            "cond": s["cond"][:, z],       # (3, H, W)
            "target": s["target"][:, z],   # (1, H, W)
            "mask": s["mask"][:, z],       # (1, H, W)
            "id": f"{s['id']}_z{z:02d}",
        }
        for k in ("zone_weight", "zones"):
            if k in s:
                out[k] = s[k][:, z]
        return out
