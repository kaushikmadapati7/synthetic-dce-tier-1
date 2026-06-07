"""Measure how much DCE peak signal the preprocessing clips.

The Tier-1 harmonized path runs DCE through Nyul standardization. Two things
clip the peak:
  1. Nyul landmarks top out at the 99th percentile of the *foreground*
     (foreground = voxels above the volume mean, per harmonization._foreground).
  2. np.interp clamps everything above that top landmark to a single ceiling,
     and _to_unit re-clips to [1, 100].
The non-harmonized fallback clips DCE at the 99.5th percentile instead.

Clinically, peak/wash-in enhancement inside the prostate is the signal of
interest. This script asks: is that peak landing above the clip ceiling?

It reads Bao_intern_package/processed_registered_samples directly (images and
masks already in T2 space, so no resampling needed), mirrors the peak-phase
selection for multi-phase zhongyiyuan, and reports per-modality-of-concern (DCE)
how much is lost — globally and *inside the prostate mask*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent / "Bao_intern_package" / "processed_registered_samples"
IMG = ROOT / "Image_volumes"
MASK = ROOT / "Prostate_masks"

# Nyul foreground/landmark config copied verbatim from data/harmonization.py
PC_HIGH = 99.0          # NyulConfig.pc_high  -> the harmonized clip ceiling
FALLBACK_HIGH = 99.5    # preprocessing.normalize default upper percentile


def load(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p), sitk.sitkFloat32))


def foreground_mask(arr: np.ndarray) -> np.ndarray:
    """Nyul background exclusion: voxels above the volume mean."""
    return arr > arr.mean()


def find_dce(subject_dir: Path, mask: np.ndarray | None):
    """Return the DCE array. For multi-phase exams, pick the peak phase by
    argmax mean intensity inside the prostate mask (matches peak_phase_index)."""
    single = list(subject_dir.glob("DCE_to_T2W*.nii.gz"))
    if single:
        return load(single[0]), "single-phase"
    phases = sorted(subject_dir.glob("DCE_ph*_to_T2W*.nii.gz"))
    if not phases:
        return None, None
    arrs = [load(p) for p in phases]
    if mask is not None and mask.any():
        means = [a[mask].mean() for a in arrs]
    else:
        means = [a.mean() for a in arrs]
    peak = int(np.argmax(means))
    return arrs[peak], f"peak={phases[peak].name.split('_')[1]} of {len(phases)}"


def find_mask(subject: str, center: str) -> np.ndarray | None:
    p = MASK / center / subject / "prostate_mask.nii.gz"
    if not p.exists():
        return None
    return load(p) > 0.5


def measure_subject(center: str, subject_dir: Path):
    subject = subject_dir.name
    mask = find_mask(subject, center)
    dce, phase_note = find_dce(subject_dir, mask)
    if dce is None:
        return None

    fg = foreground_mask(dce)
    fg_vals = dce[fg]
    ceil99 = np.percentile(fg_vals, PC_HIGH)        # harmonized clip ceiling
    ceil995 = np.percentile(fg_vals, FALLBACK_HIGH)  # fallback clip ceiling
    vmax = float(dce.max())

    # how much of the *intensity range* lives above the ceiling (gets flattened)
    range_above = (vmax - ceil99) / (vmax - fg_vals.min() + 1e-8) * 100

    row = {
        "center": center,
        "subject": subject,
        "phase": phase_note,
        "p99_ceiling": float(ceil99),
        "max": vmax,
        "pct_range_flattened": float(range_above),
        # fraction of ALL voxels above the ceiling
        "pct_vox_above_p99": float((dce > ceil99).mean() * 100),
    }

    if mask is not None and mask.any():
        roi = dce[mask]
        row["roi_p50"] = float(np.percentile(roi, 50))
        row["roi_p95"] = float(np.percentile(roi, 95))
        row["roi_max"] = float(roi.max())
        # THE key number: fraction of prostate voxels clipped by the DCE ceiling
        row["pct_roi_above_p99"] = float((roi > ceil99).mean() * 100)
        row["pct_roi_above_p995"] = float((roi > ceil995).mean() * 100)
        # where does the prostate's peak sit, as a percentile of the foreground?
        row["roi_max_as_fg_pctile"] = float((fg_vals < roi.max()).mean() * 100)
    else:
        row["pct_roi_above_p99"] = None
    return row


def main():
    if not IMG.exists():
        sys.exit(f"not found: {IMG}")
    rows = []
    for center_dir in sorted(IMG.iterdir()):
        if not center_dir.is_dir():
            continue
        for subject_dir in sorted(center_dir.iterdir()):
            if subject_dir.is_dir():
                r = measure_subject(center_dir.name, subject_dir)
                if r:
                    rows.append(r)

    print(f"\n{'='*100}\nPer-exam DCE clipping (harmonized ceiling = {PC_HIGH}th pct of foreground)\n{'='*100}")
    hdr = f"{'center':<12}{'subj':<18}{'phase':<16}{'%range_flat':>11}{'%vox>p99':>9}{'%ROI>p99':>9}{'%ROI>p99.5':>11}{'ROImax@fg%':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        roi99 = f"{r['pct_roi_above_p99']:.2f}" if r['pct_roi_above_p99'] is not None else "no-mask"
        roi995 = f"{r.get('pct_roi_above_p995', None):.2f}" if r.get('pct_roi_above_p995') is not None else "-"
        roifg = f"{r.get('roi_max_as_fg_pctile', None):.2f}" if r.get('roi_max_as_fg_pctile') is not None else "-"
        print(f"{r['center']:<12}{r['subject'][:17]:<18}{r['phase']:<16}"
              f"{r['pct_range_flattened']:>10.1f}%{r['pct_vox_above_p99']:>8.2f}%"
              f"{roi99:>8}%{roi995:>10}%{roifg:>10}%")

    # ---- aggregate ----
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    n_roi = sum(1 for r in rows if r.get("pct_roi_above_p99") is not None)
    n_roi_clipped = sum(1 for r in rows if (r.get("pct_roi_above_p99") or 0) > 0)
    print(f"\n{'='*100}\nSUMMARY ({len(rows)} exams, {n_roi} with masks)\n{'='*100}")
    print(f"  mean % of DCE intensity range flattened above the p99 ceiling : {avg('pct_range_flattened'):.1f}%")
    print(f"  mean % of all voxels above the p99 ceiling                    : {avg('pct_vox_above_p99'):.2f}%")
    print(f"  mean % of PROSTATE voxels clipped by the p99 ceiling          : {avg('pct_roi_above_p99'):.2f}%")
    print(f"  mean % of PROSTATE voxels clipped by the p99.5 ceiling        : {avg('pct_roi_above_p995'):.2f}%")
    print(f"  exams where >0% of prostate is clipped                        : {n_roi_clipped}/{n_roi}")
    print(f"  mean percentile of prostate-max within foreground            : {avg('roi_max_as_fg_pctile'):.2f}")
    print("\nInterpretation: if 'prostate-max within foreground' << 99, the lesion peak\n"
          "sits below the ceiling and little is lost. If it approaches/exceeds 99 and\n"
          "'% prostate clipped' > 0, real peak-enhancement signal is being flattened.")


if __name__ == "__main__":
    main()
