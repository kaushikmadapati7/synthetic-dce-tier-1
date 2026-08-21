"""Extract per-case DCE enhancement curves once, so target selection stops being
a property of the staged tree on disk.

WHY THIS EXISTS. Staging currently *chooses* the target phase (the one nearest
`--dce-target-time`, default 120 s) and writes that single volume per case. The
decision is therefore baked into ~11,370 files with no record of what the
alternatives would have been, so changing the rule means re-staging everything
and invalidating every run that came before. That is exactly what happened: the
120 s constant was set from five curves, never re-checked against 4,246 cases,
and when it was finally measured it turned out to land anywhere from 58% to 96%
of a given patient's own enhancement -- a different physiological state per
patient. See ASSUMPTIONS.md section A.

The fix is not a better constant, it is moving the decision. This pass reads
every 4D series ONCE and writes a few KB of JSON per case: raw region means for
every phase, with the real timestamps. After that, any selection rule -- plateau
onset, argmax, a fixed time, a lesion-anchored rule -- is an offline recompute
over the whole cohort in seconds, and switching rules tells you immediately which
cases move and by how much.

DELIBERATELY RECORDS RAW MEANS, NOT ENHANCEMENT. `enhancement = mean - mean[0]`
is derivable from raw, but not the reverse, and the baseline convention is itself
a choice (first phase? mean of the pre-contrast phases?). Storing raw defers it.

Likewise it records ALL regions -- whole gland, TZ, PZ, and both lesion masks --
because which ROI the rule should be computed on is an open question: Kang
recommended the transition zone, while our own measurement shows lesion voxels
are where the dynamics actually live (they wash out relative to size-matched
normal tissue, p=3.8e-05). With every region stored, that is settled from data
rather than by fiat.

NOTHING IS SKIPPED FOR SIZE. Oversized series fall back to per-phase streaming
rather than being dropped, because the large ones are systematically the
highest-resolution acquisitions (1024x1024) and silently excluding them is a
selection bias, not a memory limit.

    python -m tier1_static.data.extract_curves --out-dir curves
    python -m tier1_static.data.extract_curves --out-dir curves --shard 3 --n-shards 8
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

LESION_FILES = {"lesion_report": "lesion_probability_report_aligned.nii.gz",
                "lesion_picai": "lesion_probability_picai_stage2.nii.gz"}

_BYTES = {sitk.sitkUInt8: 1, sitk.sitkInt8: 1, sitk.sitkUInt16: 2, sitk.sitkInt16: 2,
          sitk.sitkUInt32: 4, sitk.sitkInt32: 4, sitk.sitkFloat32: 4,
          sitk.sitkFloat64: 8}


def write_atomic(obj, dst):
    """Temp name keeps the ORIGINAL extension -- SimpleITK picks its writer by
    extension, and a `.tmp` suffix silently broke the staging writer once."""
    tmp = os.path.join(os.path.dirname(dst), ".tmp_" + os.path.basename(dst))
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, dst)


def phase_arrays(path, n, whole, size4):
    """Yield 3D arrays phase by phase, either from one whole read or streamed."""
    if whole:
        img = sitk.ReadImage(path)
        for k in range(n):
            yield k, sitk.GetArrayFromImage(img[:, :, :, k])
        del img
    else:
        sx, sy, sz = size4[0], size4[1], size4[2]
        for k in range(n):
            r = sitk.ImageFileReader()
            r.SetFileName(path)
            r.ReadImageInformation()
            r.SetExtractIndex([0, 0, 0, k])
            r.SetExtractSize([sx, sy, sz, 0])
            yield k, sitk.GetArrayFromImage(r.Execute())


def time_status(t, n):
    """Describe the timestamps without acting on them. Recording beats filtering:
    a case dropped here is invisible later, a case flagged here is recoverable."""
    if len(t) != n:
        return "length_mismatch"
    if not np.isfinite(t).all():
        return "missing_values"
    if not np.all(np.diff(t) >= 0):
        return "non_monotonic"
    if t.max() >= 3600:
        return "implausible_span"
    return "ok"


def extract(pid, a):
    d = os.path.join(a.dce_root, pid, "DCE")
    p4d = os.path.join(d, "DCE_4D_to_T2W.nii.gz")

    r = sitk.ImageFileReader()
    r.SetFileName(p4d)
    r.ReadImageInformation()
    size4 = list(r.GetSize())
    nbytes = _BYTES.get(r.GetPixelID(), 4)
    gb = float(np.prod(size4, dtype=np.float64)) * nbytes / 1e9
    n = size4[3] if len(size4) > 3 else 1

    out = {"pid": pid, "dims": size4, "n_phases": int(n), "gb": round(gb, 2),
           "spacing": [round(float(x), 5) for x in r.GetSpacing()],
           "streamed": bool(gb > a.max_gb)}

    if n < 2:
        out["status"] = "not_dynamic"
        return out

    meta = json.load(open(os.path.join(d, "dce_times.json")))
    t = np.asarray([np.nan if p.get("rel_time_s") is None else p["rel_time_s"]
                    for p in meta.get("phases", [])], float)
    out["series"] = str(meta.get("series", "?"))
    out["times_s"] = [None if not np.isfinite(x) else round(float(x), 3) for x in t]
    out["times_status"] = time_status(t, n)

    gimg = sitk.ReadImage(os.path.join(a.main_root, pid, "prostate_mask.nii.gz"))
    gland = sitk.GetArrayFromImage(gimg) > 0
    if not gland.any():
        out["status"] = "empty_gland"
        return out

    regions = {"gland": gland}
    zp = os.path.join(a.main_root, pid, "prostate_zones.nii.gz")
    if os.path.exists(zp):
        z = sitk.GetArrayFromImage(sitk.ReadImage(zp))
        if z.shape == gland.shape:
            regions["TZ"] = gland & (z == a.tz_label)
            regions["PZ"] = gland & (z == a.pz_label)
    for key, fn in LESION_FILES.items():
        lp = os.path.join(a.main_root, pid, fn)
        if not os.path.exists(lp):
            continue
        lv = sitk.GetArrayFromImage(sitk.ReadImage(lp))
        if lv.shape != gland.shape:
            continue
        les = lv >= a.lesion_thresh
        if not les.any():
            continue
        regions[key] = les & gland
        out[key + "_frac_outside_gland"] = round(float((les & ~gland).sum())
                                                 / float(les.sum()), 4)
        out[key + "_vox_total"] = int(les.sum())

    regions = {k: v for k, v in regions.items() if v.any()}
    sup = np.zeros_like(gland)
    for v in regions.values():
        sup |= v
    idx = {k: v[sup] for k, v in regions.items()}

    sums = {k: np.zeros(n, float) for k in idx}
    shape_bad = False
    for k, arr in phase_arrays(p4d, n, not out["streamed"], size4):
        if arr.shape != gland.shape:
            shape_bad = True
            break
        v = arr[sup]
        for name, m in idx.items():
            sums[name][k] = float(v[m].mean())
    if shape_bad:
        out["status"] = "dce_mask_shape_mismatch"
        return out

    out["regions"] = {k: {"n_vox": int(m.sum()),
                          "mean_raw": [round(x, 4) for x in sums[k]]}
                      for k, m in idx.items()}
    out["lesion_thresh"] = a.lesion_thresh
    out["zone_labels"] = {"TZ": a.tz_label, "PZ": a.pz_label}

    pg = os.path.join(d, "pregad_4D_to_T2W.nii.gz")
    out["has_pregad"] = os.path.exists(pg)
    out["status"] = "ok"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-root", default=DS2)
    ap.add_argument("--dce-root", default=DS3)
    ap.add_argument("--out-dir", default="curves")
    ap.add_argument("--lesion-thresh", type=float, default=0.5,
                    help="nominal; report-aligned maps are already ~binary")
    ap.add_argument("--tz-label", type=int, default=1, help="manifest says 1=TZ")
    ap.add_argument("--pz-label", type=int, default=2, help="manifest says 2=PZ")
    ap.add_argument("--max-gb", type=float, default=6.0,
                    help="above this, stream phase-by-phase instead of one whole "
                         "read; NOTHING is skipped for size")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pids", nargs="*", default=[],
                    help="run only these case ids; for exercising the lesion "
                         "branch, which most cases do not reach")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    pids = sorted({os.path.relpath(f, a.dce_root).split(os.sep)[0]
                   for f in glob.glob(os.path.join(a.dce_root, "*", "DCE",
                                                   "DCE_4D_to_T2W.nii.gz"))})
    if a.pids:
        missing = [p for p in a.pids if p not in set(pids)]
        if missing:
            print(f"  WARNING: no DCE_4D for {missing}", flush=True)
        pids = [p for p in a.pids if p in set(pids)]
    else:
        pids = pids[a.shard::a.n_shards]
    if a.limit:
        pids = pids[:a.limit]
    print(f"  shard {a.shard}/{a.n_shards}: {len(pids)} cases -> {a.out_dir}",
          flush=True)

    done = err = 0
    stat = {}
    for i, pid in enumerate(pids):
        dst = os.path.join(a.out_dir, pid + ".json")
        if os.path.exists(dst) and not a.overwrite:
            stat["cached"] = stat.get("cached", 0) + 1
            continue
        try:
            rec = extract(pid, a)
        except Exception as e:
            print(f"  [{i+1}/{len(pids)}] {pid}: {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
            rec = {"pid": pid, "status": f"error:{type(e).__name__}",
                   "error": str(e)[:200]}
            err += 1
        write_atomic(rec, dst)
        s = rec.get("status", "?")
        stat[s] = stat.get(s, 0) + 1
        done += 1
        if done % 25 == 0 or rec.get("streamed"):
            print(f"  [{i+1}/{len(pids)}] {pid} status={s} "
                  f"n={rec.get('n_phases','?')} gb={rec.get('gb','?')}"
                  + ("  STREAMED" if rec.get("streamed") else ""), flush=True)

    print(f"\n  wrote {done} ({err} errors); status counts: "
          f"{sorted(stat.items(), key=lambda x: -x[1])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
