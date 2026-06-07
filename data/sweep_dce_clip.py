"""Sweep the DCE Nyul upper-bound percentile and report how much prostate peak
signal survives at each ceiling.

ceiling = that percentile of the foreground (voxels > mean), mirroring
data/harmonization.py. percentile=100 means "clamp at the volume max", i.e. the
option-2 fix where np.interp interpolates the whole tail and nothing is flattened.

Reuses the loaders from measure_dce_clipping.py.
"""
from __future__ import annotations

import numpy as np

from measure_dce_clipping import IMG, load, foreground_mask, find_dce, find_mask

UPPER_BOUNDS = [99.0, 99.5, 99.9, 99.99, 100.0]  # 100.0 == max == no clamp (option-2)


def collect():
    exams = []
    for center_dir in sorted(IMG.iterdir()):
        if not center_dir.is_dir():
            continue
        for subject_dir in sorted(center_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            mask = find_mask(center_dir.name, subject_dir.name) if False else find_mask(subject_dir.name, center_dir.name)
            dce, _ = find_dce(subject_dir, mask)
            if dce is None or mask is None or not mask.any():
                continue
            exams.append((dce, mask))
    return exams


def main():
    exams = collect()
    print(f"\n{'='*70}\nDCE upper-bound sweep over {len(exams)} masked exams\n{'='*70}")
    print(f"{'upper %ile':>11}{'mean %ROI clipped':>20}{'max %ROI clipped':>19}{'exams w/ clip':>15}")
    print("-" * 65)
    for pc in UPPER_BOUNDS:
        roi_clipped = []
        for dce, mask in exams:
            fg = dce[foreground_mask(dce)]
            ceil = np.percentile(fg, pc)  # pc=100 -> fg.max()
            roi = dce[mask]
            roi_clipped.append((roi > ceil).mean() * 100)
        arr = np.array(roi_clipped)
        n_aff = int((arr > 0).sum())
        label = f"{pc:g}" + (" (max)" if pc == 100.0 else "")
        print(f"{label:>11}{arr.mean():>19.3f}%{arr.max():>18.3f}%{n_aff:>11}/{len(exams)}")

    print(f"\n{'='*70}\nPer-exam detail at each ceiling (only exams that ever clip)\n{'='*70}")
    # build matrix: rows = exams that clip at pc=99, cols = bounds
    base_ceil = []
    rows = []
    for idx, (dce, mask) in enumerate(exams):
        fg = dce[foreground_mask(dce)]
        roi = dce[mask]
        vals = []
        for pc in UPPER_BOUNDS:
            ceil = np.percentile(fg, pc)
            vals.append((roi > ceil).mean() * 100)
        if vals[0] > 0:  # clipped at 99
            rows.append((idx, vals))
    hdr = f"{'exam#':>6}" + "".join(f"{('p'+format(pc,'g')):>10}" for pc in UPPER_BOUNDS)
    print(hdr)
    print("-" * len(hdr))
    for idx, vals in rows:
        print(f"{idx:>6}" + "".join(f"{v:>9.3f}%" for v in vals))
    print("\n(p100 = no clamp; the option-2 fix. Compare its column to p99.)")


if __name__ == "__main__":
    main()
