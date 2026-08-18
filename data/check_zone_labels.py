"""Which label in prostate_zones.nii.gz is the transition zone?

`data/preprocessing.py` maps 1->TZ and 2->PZ, but that mapping is a bare code
comment with no recorded provenance, and the UCSF zone masks did not come from the
same source as the Bao ones the comment was written against. Kang's recommendation
is to target the TRANSITION ZONE, so getting this backwards would silently invert
the whole analysis -- it is worth two minutes of anatomy.

Three discriminators, none of which needs the mapping we are trying to test:

  1. RADIUS. The PZ is the outer shell of the gland and the TZ is the central
     core, so PZ voxels sit farther from the gland centroid. Computed in mm via
     voxel spacing; a rotation preserves distance, so no direction handling needed.
  2. POSTERIOR. The PZ lies posterior to the TZ. SimpleITK physical coordinates
     are LPS, so +y is posterior: the PZ centroid should have the larger y.
  3. VARIABILITY. TZ volume is driven by BPH and varies enormously across an
     older cohort; PZ volume is comparatively stable. The label with the higher
     coefficient of variation across cases is the TZ.

1 and 2 are per-case and should agree with each other on nearly every case. 3 is a
cohort-level tiebreaker. If they disagree, do not proceed -- the masks are wrong.

    python -m tier1_static.data.check_zone_labels --n 40
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk

DS2 = "/mnt/fac/CX000018_DS2/yanglab/Prostate_data_all/UCSF_data/registered"


def zone_stats(path):
    """Per-label {radius_mm, posterior_y_mm, voxels, cc} for one case."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    labels = [int(v) for v in np.unique(arr) if v > 0]
    if len(labels) < 2:
        return None

    sp = np.asarray(img.GetSpacing(), float)   # (x, y, z)
    vox_cc = float(sp.prod()) / 1000.0
    gland = np.argwhere(arr > 0).astype(float)             # rows are (z, y, x)
    scale = sp[::-1]                                       # -> (z, y, x) mm
    centroid = gland.mean(axis=0)

    out = {}
    for L in labels:
        idx = np.argwhere(arr == L).astype(float)
        r = np.linalg.norm((idx - centroid) * scale, axis=1)
        mz, my, mx = idx.mean(axis=0)
        phys = img.TransformContinuousIndexToPhysicalPoint(
            (float(mx), float(my), float(mz)))
        out[L] = dict(radius_mm=float(r.mean()),
                      posterior_y=float(phys[1]),      # LPS: +y = posterior
                      voxels=int(len(idx)),
                      cc=len(idx) * vox_cc)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default=DS2)
    ap.add_argument("--n", type=int, default=40, help="cases to inspect")
    ap.add_argument("--verbose", action="store_true", help="print every case")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.main_root, "*", "prostate_zones.nii.gz")))
    print(f"  {len(files)} zone files; inspecting {min(a.n, len(files))}")

    rows, bad = [], 0
    for f in files[:a.n]:
        pid = os.path.basename(os.path.dirname(f))
        try:
            st = zone_stats(f)
        except Exception as e:
            print(f"  {pid}: {type(e).__name__}: {str(e)[:60]}")
            bad += 1
            continue
        if st is None:
            print(f"  {pid}: fewer than 2 zone labels; skipped")
            bad += 1
            continue
        if set(st) != {1, 2}:
            print(f"  {pid}: unexpected labels {sorted(st)}; skipped")
            bad += 1
            continue
        rows.append((pid, st))
        if a.verbose:
            print(f"  {pid}  "
                  + "  ".join(f"L{L}: r={st[L]['radius_mm']:5.1f}mm "
                              f"y={st[L]['posterior_y']:7.1f} "
                              f"{st[L]['cc']:5.1f}cc" for L in (1, 2)))

    if not rows:
        print("  no usable cases")
        return 1

    r1 = np.array([s[1]["radius_mm"] for _, s in rows])
    r2 = np.array([s[2]["radius_mm"] for _, s in rows])
    y1 = np.array([s[1]["posterior_y"] for _, s in rows])
    y2 = np.array([s[2]["posterior_y"] for _, s in rows])
    c1 = np.array([s[1]["cc"] for _, s in rows])
    c2 = np.array([s[2]["cc"] for _, s in rows])
    n = len(rows)

    print(f"\n  n={n} usable ({bad} skipped)\n")
    print("  test 1  RADIUS from gland centroid (outer shell = PZ)")
    print(f"    label 1: {r1.mean():5.1f} +/- {r1.std():4.1f} mm")
    print(f"    label 2: {r2.mean():5.1f} +/- {r2.std():4.1f} mm")
    outer_2 = int((r2 > r1).sum())
    print(f"    label 2 is the outer zone in {outer_2}/{n} cases "
          f"({100*outer_2/n:.0f}%)")

    print("\n  test 2  POSTERIOR position, LPS +y (posterior = PZ)")
    dy = y2 - y1
    post_2 = int((dy > 0).sum())
    print(f"    label 2 centroid is posterior to label 1 by "
          f"{dy.mean():+.1f} +/- {dy.std():.1f} mm")
    print(f"    label 2 is posterior in {post_2}/{n} cases ({100*post_2/n:.0f}%)")

    print("\n  test 3  VOLUME variability (BPH-driven spread = TZ)")
    for nm, c in (("label 1", c1), ("label 2", c2)):
        print(f"    {nm}: {c.mean():5.1f} +/- {c.std():4.1f} cc  "
              f"(CV {c.std()/c.mean():.2f}, range {c.min():.1f}-{c.max():.1f})")
    cv_hi = 1 if (c1.std() / c1.mean()) > (c2.std() / c2.mean()) else 2

    votes = [2 if outer_2 > n / 2 else 1,      # outer  -> PZ
             2 if post_2 > n / 2 else 1]       # posterior -> PZ
    pz = 2 if votes.count(2) == 2 else (1 if votes.count(1) == 2 else None)

    print("\n  VERDICT")
    if pz is None:
        print("    tests 1 and 2 DISAGREE -- do not trust these zone masks.")
        print("    Inspect them visually before using either label.")
        return 1
    tz = 1 if pz == 2 else 2
    print(f"    PZ = label {pz}   TZ = label {tz}   (radius and posterior agree)")
    print(f"    volume-variability tiebreaker says TZ = label {cv_hi}"
          + ("  [consistent]" if cv_hi == tz else "  [INCONSISTENT -- check]"))
    code = "1=TZ, 2=PZ"
    actual = f"{tz}=TZ, {pz}=PZ"
    print(f"\n    preprocessing.py assumes {code}; data says {actual}"
          + ("  -> OK" if tz == 1 else "  -> *** CODE IS BACKWARDS ***"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
