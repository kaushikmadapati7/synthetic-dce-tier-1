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


def end_over_peak(enh, topk=1):
    """enh is (n_phases,) or (n_phases, n_voxels). NaN where nothing enhanced.

    `topk`>1 averages the top-k phases for the peak and the last-k for the end.
    A plain max() latches onto the largest positive noise excursion, which
    inflates the peak and so DEFLATES end/peak -- and the bias grows as the ROI
    shrinks, which is exactly the difference between a lesion and whole-gland
    normal tissue. topk is the cheap half of guarding against that; the
    size-matched null ROIs below are the real control.
    """
    k = max(1, min(int(topk), enh.shape[0]))
    peak = enh.max(axis=0) if k == 1 else np.sort(enh, axis=0)[-k:].mean(axis=0)
    end = enh[-1] if k == 1 else enh[-k:].mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(peak > 0, end / peak, np.nan)


def matched_null(enh, coords, normal_idx, n_vox, spacing_zyx, n_draw, rng, topk):
    """end/peak for spatially CONTIGUOUS normal ROIs of the same voxel count.

    Contiguous, not scattered: DCE noise is spatially correlated, so a scattered
    sample of normal voxels averages down more effectively than a compact lesion
    of the same size and would understate the null.
    """
    out = []
    pool = coords[normal_idx] * spacing_zyx          # physical mm
    for _ in range(n_draw):
        s = rng.integers(len(normal_idx))
        d = ((pool - pool[s]) ** 2).sum(axis=1)
        take = normal_idx[np.argpartition(d, min(n_vox, len(d) - 1))[:n_vox]]
        out.append(float(end_over_peak(enh[:, take].mean(axis=1), topk)))
    return np.asarray(out, float)


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
    if n < 4:
        print(f"    skipped: only {n} phase(s); not a dynamic series", flush=True)
        return None
    if len(t) != n or not usable_times(t):
        print("    skipped: timestamps unusable", flush=True)
        return None

    gimg = sitk.ReadImage(os.path.join(a.main_root, pid, "prostate_mask.nii.gz"))
    gland = sitk.GetArrayFromImage(gimg) > 0
    spacing_zyx = np.asarray(gimg.GetSpacing(), float)[::-1]     # (z, y, x) mm
    lesp = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(a.main_root, pid, a.lesion_file)))
    zpath = os.path.join(a.main_root, pid, "prostate_zones.nii.gz")
    zones = sitk.GetArrayFromImage(sitk.ReadImage(zpath)) if os.path.exists(zpath) else None
    if gland.shape != lesp.shape or not gland.any():
        print("    skipped: gland/lesion shape mismatch or empty gland", flush=True)
        return None

    lesion = lesp >= a.lesion_thresh
    if not lesion.any():
        print(f"    skipped: no lesion voxel >= {a.lesion_thresh}", flush=True)
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
        print(f"    skipped: lesion {int(sel['lesion'].sum())}vx / normal "
              f"{int(sel['normal'].sum())}vx below --min-lesion-voxels "
              f"{a.min_lesion_voxels}", flush=True)
        return None

    # pull every phase once, restricted to the gland-or-lesion support
    sup = gland | lesion
    vals = np.stack([sitk.GetArrayFromImage(img[:, :, :, k])[sup].astype(np.float32)
                     for k in range(n)])                       # (n_phases, n_sup)
    del img                                    # release the 4D before the maths
    enh = vals - vals[0:1]
    idx = {k: v[sup] for k, v in sel.items()}

    curves = {k: enh[:, m].mean(axis=1) for k, m in idx.items() if m.any()}
    vox_ratio = {k: end_over_peak(enh[:, m], a.peak_topk) for k, m in idx.items() if m.any()}

    # THE CONTROL: same voxel count, same contiguity, normal tissue.
    coords = np.argwhere(sup)
    null = matched_null(enh, coords, np.where(idx["normal"])[0],
                        int(idx["lesion"].sum()), spacing_zyx, a.n_null,
                        np.random.default_rng(a.seed), a.peak_topk)

    return dict(pid=pid, t=t, n=n, curves=curves, vox_ratio=vox_ratio, null=null,
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
    ap.add_argument("--n-null", type=int, default=20,
                    help="size-matched contiguous normal ROIs drawn per case")
    ap.add_argument("--peak-topk", type=int, default=3,
                    help="average top-k phases for peak / last-k for end, to blunt "
                         "the max() noise bias that scales with ROI size")
    ap.add_argument("--seed", type=int, default=0)
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
        lr = end_over_peak(c["curves"]["lesion"], a.peak_topk)
        nr = end_over_peak(c["curves"]["normal"], a.peak_topk)
        nul = c["null"][np.isfinite(c["null"])]
        nm = float(np.median(nul)) if len(nul) else np.nan
        pct = float((nul <= lr).mean() * 100) if len(nul) else np.nan
        print(f"    -> n={c['n']:3d} span={c['t'][-1]:5.0f}s  "
              f"lesion {c['nvox']['lesion']:6d}vx end/peak={lr:5.2f}  "
              f"normal={nr:5.2f}  size-matched null={nm:5.2f} "
              f"(lesion at {pct:3.0f}th pct)  diff_vs_null={lr-nm:+5.2f}  "
              f"outside={c['frac_outside']:.0%}", flush=True)

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
    lr = np.array([end_over_peak(c["curves"]["lesion"], a.peak_topk) for c in cases], float)
    nr = np.array([end_over_peak(c["curves"]["normal"], a.peak_topk) for c in cases], float)
    nu = np.array([np.nanmedian(c["null"]) if np.isfinite(c["null"]).any() else np.nan
                   for c in cases], float)
    ok = np.isfinite(lr) & np.isfinite(nr) & np.isfinite(nu)
    lr, nr, nu = lr[ok], nr[ok], nu[ok]

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

    # (b) lesion against the SIZE-MATCHED null, which is the comparison that
    #     survives the ROI-size noise artifact
    ax[1].scatter(nu, lr, s=16, alpha=0.65, label="vs size-matched null")
    ax[1].scatter(nr, lr, s=10, alpha=0.35, marker="x", color="grey",
                  label="vs whole-gland normal")
    lim = [min(nu.min(), nr.min(), lr.min()) - 0.05,
           max(nu.max(), nr.max(), lr.max()) + 0.05]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlabel("normal-tissue end/peak"); ax[1].set_ylabel("lesion end/peak")
    ax[1].set_title("paired within patient\nbelow the line = lesion washes out more",
                    fontsize=9)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

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
    print(f"\n  === PAIRED end/peak, n={len(lr)} (peak_topk={a.peak_topk}) ===")
    print(f"    lesion            {lr.mean():.3f} +/- {lr.std():.3f}")
    print(f"    whole-gland normal{nr.mean():.3f} +/- {nr.std():.3f}   "
          f"(~100k voxels: almost noise-free, NOT a fair comparison)")
    print(f"    size-matched null {nu.mean():.3f} +/- {nu.std():.3f}   "
          f"(same voxel count and contiguity, normal tissue)")
    for nm, ref in (("vs whole-gland normal", nr), ("vs SIZE-MATCHED null", nu)):
        d = lr - ref
        line = (f"    {nm:22s} diff {d.mean():+.3f} +/- {d.std():.3f}  "
                f"lower in {int((d < 0).sum())}/{len(d)} ({100*(d<0).mean():.0f}%)")
        try:
            from scipy.stats import wilcoxon
            line += f"  p={wilcoxon(lr, ref).pvalue:.3g}"
        except Exception:
            pass
        print(line)

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
    print("\n    GO if lesion end/peak sits below the SIZE-MATCHED null. The")
    print("    whole-gland comparison is confounded: it averages ~100k voxels so")
    print("    its max() is nearly unbiased, while a few-hundred-voxel lesion's is")
    print("    not, which manufactures washout out of noise alone. Ignore the")
    print("    per-voxel IQR entirely -- single-voxel curves are noise-dominated in")
    print("    both regions, so it cannot separate biology from noise either way.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pid", "n_phases", "span_s", "lesion_vox", "normal_vox",
                        "lesion_end_peak", "normal_end_peak", "null_end_peak_med",
                        "null_end_peak_p05", "lesion_pct_in_null",
                        "frac_outside_gland"])
            for c in cases:
                lv = float(end_over_peak(c["curves"]["lesion"], a.peak_topk))
                nul = c["null"][np.isfinite(c["null"])]
                w.writerow([c["pid"], c["n"], f"{c['t'][-1]:.1f}",
                            c["nvox"].get("lesion", 0), c["nvox"].get("normal", 0),
                            f"{lv:.4f}",
                            f"{float(end_over_peak(c['curves']['normal'], a.peak_topk)):.4f}",
                            f"{np.median(nul):.4f}" if len(nul) else "",
                            f"{np.percentile(nul, 5):.4f}" if len(nul) else "",
                            f"{(nul <= lv).mean()*100:.1f}" if len(nul) else "",
                            f"{c['frac_outside']:.4f}"])
        print(f"  wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
