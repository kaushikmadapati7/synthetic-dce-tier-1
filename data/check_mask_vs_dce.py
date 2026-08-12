"""Is the prostate mask actually on the prostate, in the 4D DCE's own space?

Written because gland-mean enhancement curves rose monotonically for ~5 minutes and
never washed out, which is not prostate behaviour -- it is what the BLADDER does,
since contrast is renally excreted and the bladder fills continuously. Two failure
modes produce exactly that curve:

  1. mask and 4D share array SHAPE but not physical space (origin/spacing/direction),
     so the mask samples the wrong anatomy. Shape equality is a weak check and it is
     the only one plot_enh_curves does.
  2. the mask is correct but the segmentation leaks into the bladder.

This checks both: prints geometry for mask vs 4D, mask volume in cc (a prostate is
~20-100 cc), and renders the mask outline over an early and a late DCE phase so the
anatomy is visible rather than inferred. It also splits the curve by mask sub-region
(superior/inferior halves) -- bladder contamination sits superior, so a superior half
that rises while the inferior half plateaus is diagnostic.

    python -m tier1_static.data.check_mask_vs_dce --pids 8iiT8ff2iif 8iiT8ijjfio
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import SimpleITK as sitk

DS2 = "/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/UCSF_data/registered"
DS3 = "/mnt/fac/CX000018_DS3/yanglab/Prostate_data_all/UCSF_data/registered"


def geom(img):
    return dict(size=tuple(img.GetSize()[:3]),
                spacing=tuple(round(float(x), 4) for x in img.GetSpacing()[:3]),
                origin=tuple(round(float(x), 2) for x in img.GetOrigin()[:3]),
                direction=tuple(round(float(x), 3) for x in img.GetDirection()[:9]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default=DS2)
    ap.add_argument("--dce-root", default=DS3)
    ap.add_argument("--pids", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4, help="if --pids not given, sample this many")
    ap.add_argument("--out", default="mask_vs_dce.png")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = a.pids
    if not pids:
        pids = [os.path.basename(os.path.dirname(os.path.dirname(f)))
                for f in sorted(glob.glob(os.path.join(a.dce_root, "*", "DCE",
                                                       "dce_times.json")))][:a.n]

    fig, axes = plt.subplots(len(pids), 3, figsize=(13, 4 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        try:
            d = os.path.join(a.dce_root, pid, "DCE")
            img4d = sitk.ReadImage(os.path.join(d, "DCE_4D_to_T2W.nii.gz"))
            mimg = sitk.ReadImage(os.path.join(a.main_root, pid, "prostate_mask.nii.gz"))
            n = img4d.GetSize()[3]
            first, last = img4d[:, :, :, 0], img4d[:, :, :, n - 1]

            g4, gm = geom(first), geom(mimg)
            same = all(g4[k] == gm[k] for k in ("size", "spacing", "origin", "direction"))
            print(f"\n=== {pid} ===")
            print(f"  4D phase : {g4}")
            print(f"  mask     : {gm}")
            print(f"  SAME PHYSICAL SPACE: {same}"
                  + ("" if same else "   <<< MASK IS NOT IN THE 4D's SPACE"))

            m = sitk.GetArrayFromImage(mimg) > 0
            sp = mimg.GetSpacing()
            cc = m.sum() * sp[0] * sp[1] * sp[2] / 1000.0
            print(f"  mask: {m.sum()} voxels = {cc:.1f} cc "
                  f"({'plausible prostate' if 15 <= cc <= 120 else '*** IMPLAUSIBLE ***'})")

            a0 = sitk.GetArrayFromImage(first)
            aL = sitk.GetArrayFromImage(last)
            if a0.shape != m.shape:
                print(f"  shape mismatch {a0.shape} vs {m.shape}; skipping render")
                continue

            # superior vs inferior half of the mask, along z
            zs = np.where(m.any(axis=(1, 2)))[0]
            zmid = int(zs.mean())
            sup = m.copy(); sup[:zmid] = False
            inf = m.copy(); inf[zmid:] = False
            for nm, sub in (("superior", sup), ("inferior", inf)):
                if sub.sum():
                    print(f"  {nm:9} half: phase0={a0[sub].mean():8.1f} "
                          f"last={aL[sub].mean():8.1f} "
                          f"delta={aL[sub].mean() - a0[sub].mean():8.1f}")

            z = int(np.argmax(m.sum(axis=(1, 2))))     # slice with most mask
            for c, (arr, ttl) in enumerate(((a0[z], "phase 0"), (aL[z], f"phase {n-1}"))):
                ax = axes[r][c]
                ax.imshow(arr, cmap="gray")
                ax.contour(m[z], levels=[0.5], colors="r", linewidths=0.8)
                ax.set_title(f"{pid}  {ttl}  z={z}", fontsize=8)
                ax.axis("off")
            ax = axes[r][2]
            ax.imshow(aL[z] - a0[z], cmap="hot")
            ax.contour(m[z], levels=[0.5], colors="c", linewidths=0.8)
            ax.set_title(f"{pid}  last - phase0", fontsize=8)
            ax.axis("off")
        except Exception as e:
            print(f"  {pid}: {type(e).__name__}: {str(e)[:80]}")
            for c in range(3):
                axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
