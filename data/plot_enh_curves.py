"""Plot raw prostate enhancement curves from the UCSF 4D DCE series.

Diagnostic, deliberately assumption-free: for each case it plots the mask-mean
signal MINUS the first phase, against real elapsed time, for every phase in the
series. No target time, no thresholds, no phase selection, no interleave handling
-- whatever the data does is what you see.

The point is to settle empirically whether these curves PEAK or PLATEAU, which
determines whether a target phase should be chosen by `argmax` of enhancement or
by a fixed post-injection time. That question had been answered from five curves
and never revisited; see ASSUMPTIONS.md section A.

Cases whose timestamps are unusable (missing, non-monotonic, absurd) are SKIPPED
rather than redrawn against phase index. Index spacing is uniform while real
cadence is not (1.7-13 s across this cohort), so an index axis compresses sparsely
sampled regions and can make a plateau look like a peak -- the exact confusion this
plot exists to avoid.

    python -m tier1_static.data.plot_enh_curves --n 9 --out enh_curves.png

Printed per case: argmax index and time, peak height, final height, and end/peak.
`end/peak` is the summary that answers the question -- near 1.0 means the curve is
still rising or flat when the scan ends (plateau, so argmax lands on noise), well
below 1.0 means it genuinely peaks and washes out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import SimpleITK as sitk

DS2 = "/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/UCSF_data/registered"
DS3 = "/mnt/fac/CX000018_DS3/yanglab/Prostate_data_all/UCSF_data/registered"


def case_curve(pid, main_root, dce_root):
    """(times, enhancement, n_phases, series) for one case, or None if unusable.

    enhancement = mask-mean(phase k) - mask-mean(phase 0), in raw scanner units.
    """
    d = os.path.join(dce_root, pid, "DCE")
    meta = json.load(open(os.path.join(d, "dce_times.json")))
    img = sitk.ReadImage(os.path.join(d, "DCE_4D_to_T2W.nii.gz"))
    mask_p = os.path.join(main_root, pid, "prostate_mask.nii.gz")
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_p)) > 0
    if not mask.any():
        return None

    n = img.GetSize()[3] if img.GetDimension() > 3 else 1
    means = []
    for k in range(n):
        arr = sitk.GetArrayFromImage(img[:, :, :, k])
        if arr.shape != mask.shape:
            return None
        means.append(float(arr[mask].mean()))
    means = np.asarray(means, float)

    times = np.asarray([np.nan if p.get("rel_time_s") is None else p["rel_time_s"]
                        for p in meta["phases"]], float)
    if len(times) != n:
        return None
    return times, means - means[0], n, str(meta.get("series", "?"))


def usable_times(t):
    """Monotonic, finite, physiologically plausible."""
    return bool(np.isfinite(t).all() and np.all(np.diff(t) >= 0) and t.max() < 3600)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default=DS2)
    ap.add_argument("--dce-root", default=DS3)
    ap.add_argument("--n", type=int, default=9, help="cases to plot")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", default="enh_curves.png")
    ap.add_argument("--csv", default="", help="also dump every curve here")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = [os.path.basename(os.path.dirname(os.path.dirname(f)))
            for f in sorted(glob.glob(os.path.join(a.dce_root, "*", "DCE",
                                                   "dce_times.json")))]
    random.Random(a.seed).shuffle(pids)
    print(f"  {len(pids)} candidate cases; plotting {a.n}")

    rows = int(np.ceil(a.n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows), squeeze=False)
    axes = axes.ravel()

    done = skipped = 0
    rec = []
    for pid in pids:
        if done >= a.n:
            break
        try:
            got = case_curve(pid, a.main_root, a.dce_root)
        except Exception as e:
            print(f"  {pid}: {type(e).__name__}: {str(e)[:60]}")
            continue
        if got is None:
            continue
        t, enh, n, series = got
        if not usable_times(t):
            print(f"  {pid}: SKIPPED (timestamps unusable)")
            skipped += 1
            continue

        gaps = np.diff(t)
        k = int(np.argmax(enh))
        ratio = enh[-1] / enh[k] if enh[k] else np.nan
        print(f"  {pid} n={n:3d} argmax={k:3d} t={t[k]:6.1f}s "
              f"peak={enh[k]:8.1f} end={enh[-1]:8.1f} end/peak={ratio:5.2f} "
              f"span={t.max():5.0f}s cadence={np.median(gaps):4.1f}s")
        rec.append((pid, n, k, t[k], enh[k], enh[-1], ratio, t.max()))

        ax = axes[done]
        ax.plot(t, enh, "o-", ms=3.5, lw=1.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{pid}  n={n}  {series[:22]}\n"
                     f"cadence med {np.median(gaps):.1f}s "
                     f"({gaps.min():.1f}-{gaps.max():.1f})", fontsize=8)
        ax.set_xlabel("seconds since phase 0")
        ax.set_ylabel("mask-mean - phase0")
        ax.grid(alpha=0.3)
        done += 1

    for ax in axes[done:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print(f"\n  wrote {a.out}  ({done} plotted, {skipped} skipped)")

    if rec:
        r = np.array([x[6] for x in rec], float)
        r = r[np.isfinite(r)]
        if len(r):
            print(f"  end/peak: median {np.median(r):.2f}  "
                  f"min {r.min():.2f}  max {r.max():.2f}")
            print("  -> near 1.0 = still at max when the scan ends (plateau)")
            print("  -> well below 1.0 = genuine peak then washout")
    if a.csv and rec:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pid", "n_phases", "argmax_idx", "argmax_t_s",
                        "peak", "end", "end_over_peak", "span_s"])
            w.writerows(rec)
        print(f"  wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
