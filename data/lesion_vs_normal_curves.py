"""Do lesion voxels behave differently from normal gland over the DCE series?

This is a GO/NO-GO measurement for the whole physically-structured-ODE direction.

`plot_enh_curves` showed gland-mean curves that rise and plateau with no washout
(end/peak 0.93-1.00 on nine cases). If that is genuinely all the dynamics there is,
then every voxel is a saturating exponential, a free-form velocity field fits it as
easily as a pharmacokinetic one, and structuring the ODE buys nothing -- which a
reviewer will say before we do. But a gland mean averages 23-64 cc of mostly normal
tissue, so it is exactly the statistic that would hide focal washout. Now that
report-aligned lesion masks exist for 842 cases with DCE, we can look directly.

What this measures, per case, PAIRED WITHIN PATIENT so that injection dose, cardiac
output, and scan timing cancel out:

  * mean enhancement curve for lesion voxels vs normal gland vs TZ vs PZ
  * per-voxel `end/peak` distributions inside lesion vs normal tissue -- the width
    of these is the curve-shape DIVERSITY that a learned velocity field would have
    to explain, and a tight histogram is the no-go outcome
  * time-to-peak as a FRACTION of scan duration, never in absolute seconds, since
    span varies 166-329 s across this cohort

Deliberately assumption-free about time: curves are plotted against real elapsed
seconds, no target time, no phase index, no washout window. The one genuinely
`CHOSEN` constant is --lesion-dilate, which keeps partial-volume lesion rim out of
the "normal" comparison group; --lesion-thresh is nominal because the report-aligned
maps are already effectively binary (nonzero == above 0.5 to within a voxel or two).

Also reports QC that the manifest audit cannot: what fraction of each "lesion" falls
OUTSIDE the prostate mask. The manifest says components are "not clipped to the
report sector", so a lesion largely outside the gland is a mislocalized teacher blob
rather than a tumour, and would poison the comparison.

    python -m tier1_static.data.lesion_vs_normal_curves --n 60
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


def usable_times(t):
    """Monotonic, finite, physiologically plausible. Same rule as plot_enh_curves."""
    return bool(np.isfinite(t).all() and np.all(np.diff(t) >= 0) and t.max() < 3600)


def end_over_peak(enh):
    """enh is (n_phases,) or (n_phases, n_voxels). NaN where nothing enhanced."""
    peak = enh.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(peak > 0, enh[-1] / peak, np.nan)
    return r


def load_case(pid, a):
    """Everything needed for one case, or None if unusable. Raises on real errors."""
    d = os.path.join(a.dce_root, pid, "DCE")
    meta = json.load(open(os.path.join(d, "dce_times.json")))
    p4d = os.path.join(d, "DCE_4D_to_T2W.nii.gz")

    # Header first: GetSize() without decompressing the volume, so an oversized
    # series is skipped rather than thrashing a shared login node for minutes.
    r = sitk.ImageFileReader()
    r.SetFileName(p4d)
    r.ReadImageInformation()
    dims = tuple(r.GetSize())
    gb = float(np.prod(dims, dtype=np.float64)) * 4 / 1e9
    print(f"  {pid} dims={dims} ~{gb:.1f}GB in RAM", flush=True)
    if gb > a.max_gb:
        print(f"    skipped: exceeds --max-gb {a.max_gb}", flush=True)
        return None

    img = sitk.ReadImage(p4d)
    n = img.GetSize()[3] if img.GetDimension() > 3 else 1
    t = np.asarray([np.nan if p.get("rel_time_s") is None else p["rel_time_s"]
                    for p in meta["phases"]], float)
    if len(t) != n or not usable_times(t) or n < 4:
        return None

    gland = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(a.main_root, pid, "prostate_mask.nii.gz"))) > 0
    lesp = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(a.main_root, pid, a.lesion_file)))
    zpath = os.path.join(a.main_root, pid, "prostate_zones.nii.gz")
    zones = sitk.GetArrayFromImage(sitk.ReadImage(zpath)) if os.path.exists(zpath) else None
    if gland.shape != lesp.shape or not gland.any():
        return None

    lesion = lesp >= a.lesion_thresh
    if not lesion.any():
        return None
    frac_outside = float((lesion & ~gland).sum()) / float(lesion.sum())

    # keep partial-volume lesion rim out of the "normal" group
    grown = lesion
    if a.lesion_dilate > 0:
        try:
            from scipy.ndimage import binary_dilation
            grown = binary_dilation(lesion, iterations=a.lesion_dilate)
        except ImportError:
            print("    (scipy unavailable; --lesion-dilate ignored)")

    sel = {"lesion": (lesion & gland),
           "normal": (gland & ~grown)}
    if zones is not None and zones.shape == gland.shape:
        sel["TZ"] = gland & ~grown & (zones == a.tz_label)
        sel["PZ"] = gland & ~grown & (zones == a.pz_label)
    if sel["lesion"].sum() < a.min_lesion_voxels or sel["normal"].sum() < a.min_lesion_voxels:
        return None

    # pull every phase once, restricted to the gland-or-lesion support
    sup = gland | lesion
    vals = np.stack([sitk.GetArrayFromImage(img[:, :, :, k])[sup].astype(np.float32)
                     for k in range(n)])                       # (n_phases, n_sup)
    del img                                    # release the 4D before the maths
    enh = vals - vals[0:1]
    idx = {k: v[sup] for k, v in sel.items()}

    curves = {k: enh[:, m].mean(axis=1) for k, m in idx.items() if m.any()}
    vox_ratio = {k: end_over_peak(enh[:, m]) for k, m in idx.items() if m.any()}
    return dict(pid=pid, t=t, n=n, curves=curves, vox_ratio=vox_ratio,
                nvox={k: int(m.sum()) for k, m in idx.items()},
                frac_outside=frac_outside, series=str(meta.get("series", "?")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default=DS2)
    ap.add_argument("--dce-root", default=DS3)
    ap.add_argument("--lesion-file", default="lesion_probability_report_aligned.nii.gz",
                    help="swap to lesion_probability_picai_stage2.nii.gz to compare")
    ap.add_argument("--lesion-thresh", type=float, default=0.5,
                    help="nominal: report-aligned maps are already ~binary")
    ap.add_argument("--lesion-dilate", type=int, default=2,
                    help="CHOSEN: voxels of lesion dilation excluded from 'normal'")
    ap.add_argument("--min-lesion-voxels", type=int, default=20,
                    help="CHOSEN: floor for a stable per-region mean")
    ap.add_argument("--tz-label", type=int, default=1, help="manifest says 1=TZ")
    ap.add_argument("--pz-label", type=int, default=2, help="manifest says 2=PZ")
    ap.add_argument("--max-gb", type=float, default=3.0,
                    help="skip a 4D series whose in-RAM size exceeds this")
    ap.add_argument("--n", type=int, default=60, help="cases to analyse")
    ap.add_argument("--plot", type=int, default=9, help="cases to draw individually")
    ap.add_argument("--out", default="lesion_vs_normal.png")
    ap.add_argument("--out-summary", default="lesion_vs_normal_summary.png")
    ap.add_argument("--csv", default="lesion_vs_normal.csv")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = sorted({os.path.relpath(f, a.dce_root).split(os.sep)[0]
                   for f in glob.glob(os.path.join(a.dce_root, "*", "DCE",
                                                   "DCE_4D_to_T2W.nii.gz"))})
    have = {os.path.relpath(f, a.main_root).split(os.sep)[0]
            for f in glob.glob(os.path.join(a.main_root, "*", a.lesion_file))}
    pids = [p for p in pids if p in have]
    print(f"  {len(pids)} cases with both DCE and {a.lesion_file}; analysing {a.n}")

    cases, skipped = [], 0
    for pid in pids:
        if len(cases) >= a.n:
            break
        try:
            c = load_case(pid, a)
        except Exception as e:
            print(f"  {pid}: {type(e).__name__}: {str(e)[:60]}")
            skipped += 1
            continue
        if c is None:
            skipped += 1
            continue
        cases.append(c)
        lr = end_over_peak(c["curves"]["lesion"])
        nr = end_over_peak(c["curves"]["normal"])
        print(f"    -> n={c['n']:3d} span={c['t'][-1]:5.0f}s  "
              f"lesion {c['nvox']['lesion']:6d}vx end/peak={lr:5.2f}  "
              f"normal end/peak={nr:5.2f}  diff={lr-nr:+5.2f}  "
              f"outside gland={c['frac_outside']:.0%}", flush=True)

    if not cases:
        print("  no usable cases")
        return 1
    print(f"\n  {len(cases)} usable, {skipped} skipped")

    # ---- per-case figure -------------------------------------------------
    show = cases[:a.plot]
    rows = int(np.ceil(len(show) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.8 * rows), squeeze=False)
    axes = axes.ravel()
    for i, c in enumerate(show):
        ax = axes[i]
        for k, style in (("lesion", "r-o"), ("normal", "k-o"),
                         ("TZ", "b--"), ("PZ", "g--")):
            if k in c["curves"]:
                ax.plot(c["t"], c["curves"][k], style, ms=3, lw=1.2, label=k)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{c['pid']}  n={c['n']}  lesion {c['nvox']['lesion']}vx\n"
                     f"{c['frac_outside']:.0%} of lesion outside gland", fontsize=8)
        ax.set_xlabel("seconds since phase 0")
        ax.set_ylabel("mean enhancement")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes[len(show):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print(f"  wrote {a.out}")

    # ---- cohort summary --------------------------------------------------
    lr = np.array([end_over_peak(c["curves"]["lesion"]) for c in cases], float)
    nr = np.array([end_over_peak(c["curves"]["normal"]) for c in cases], float)
    ok = np.isfinite(lr) & np.isfinite(nr)
    lr, nr = lr[ok], nr[ok]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) cohort mean curve on a common grid, amplitude normalised WITHIN patient
    #     by that patient's normal-tissue peak, so lesion/normal ratio survives.
    tmax = max(c["t"][-1] for c in cases)
    grid = np.linspace(0, tmax, 60)
    stacks = {"lesion": [], "normal": []}
    for c in cases:
        ref = c["curves"]["normal"].max()
        if not np.isfinite(ref) or ref <= 0:
            continue
        for k in stacks:
            y = np.interp(grid, c["t"], c["curves"][k] / ref)
            stacks[k].append(np.where(grid <= c["t"][-1], y, np.nan))
    for k, col in (("lesion", "r"), ("normal", "k")):
        A = np.asarray(stacks[k], float)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(A, axis=0)
            s = np.nanstd(A, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(A), axis=0), 1))
        ax[0].plot(grid, m, col, lw=1.8, label=k)
        ax[0].fill_between(grid, m - s, m + s, color=col, alpha=0.2)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title(f"cohort mean, normalised by each patient's\nnormal-tissue peak "
                    f"(n={len(stacks['normal'])})", fontsize=9)
    ax[0].set_xlabel("seconds since phase 0"); ax[0].legend(); ax[0].grid(alpha=0.3)

    # (b) paired end/peak
    ax[1].scatter(nr, lr, s=14, alpha=0.6)
    lim = [min(nr.min(), lr.min()) - 0.05, max(nr.max(), lr.max()) + 0.05]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlabel("normal end/peak"); ax[1].set_ylabel("lesion end/peak")
    ax[1].set_title("paired within patient\nbelow the line = lesion washes out more",
                    fontsize=9)
    ax[1].grid(alpha=0.3)

    # (c) per-voxel shape diversity, pooled
    for k, col in (("lesion", "r"), ("normal", "k")):
        v = np.concatenate([c["vox_ratio"][k] for c in cases if k in c["vox_ratio"]])
        v = v[np.isfinite(v)]
        ax[2].hist(np.clip(v, -0.5, 1.5), bins=60, histtype="step", color=col,
                   density=True, label=f"{k} (n={len(v)})")
    ax[2].set_xlabel("per-voxel end/peak")
    ax[2].set_title("curve-shape diversity\ntight = nothing for an ODE to learn",
                    fontsize=9)
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(a.out_summary, dpi=110)
    print(f"  wrote {a.out_summary}")

    # ---- the verdict -----------------------------------------------------
    d = lr - nr
    print(f"\n  === PAIRED end/peak, n={len(d)} ===")
    print(f"    lesion  {lr.mean():.3f} +/- {lr.std():.3f}")
    print(f"    normal  {nr.mean():.3f} +/- {nr.std():.3f}")
    print(f"    diff    {d.mean():+.3f} +/- {d.std():.3f}   "
          f"lesion lower in {int((d < 0).sum())}/{len(d)} ({100*(d<0).mean():.0f}%)")
    try:
        from scipy.stats import wilcoxon
        print(f"    wilcoxon p = {wilcoxon(lr, nr).pvalue:.3g}")
    except Exception:
        pass

    for k in ("lesion", "normal"):
        v = np.concatenate([c["vox_ratio"][k] for c in cases if k in c["vox_ratio"]])
        v = v[np.isfinite(v)]
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"    per-voxel end/peak {k:6s}: p5={q[0]:.2f} p25={q[1]:.2f} "
              f"med={q[2]:.2f} p75={q[3]:.2f} p95={q[4]:.2f}  "
              f"IQR={q[3]-q[1]:.2f}  frac<0.9: {np.mean(v < 0.9):.1%}")

    fo = np.array([c["frac_outside"] for c in cases])
    print(f"\n    QC lesion voxels outside prostate mask: med {np.median(fo):.0%}, "
          f">50% outside in {int((fo > 0.5).sum())}/{len(fo)} cases")
    print("\n    GO if lesions wash out relative to normal (diff < 0) and the")
    print("    per-voxel IQR is wide. NO-GO if both regions are the same flat plateau.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pid", "n_phases", "span_s", "lesion_vox", "normal_vox",
                        "lesion_end_peak", "normal_end_peak", "frac_outside_gland"])
            for c in cases:
                w.writerow([c["pid"], c["n"], f"{c['t'][-1]:.1f}",
                            c["nvox"].get("lesion", 0), c["nvox"].get("normal", 0),
                            f"{end_over_peak(c['curves']['lesion']):.4f}",
                            f"{end_over_peak(c['curves']['normal']):.4f}",
                            f"{c['frac_outside']:.4f}"])
        print(f"  wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
