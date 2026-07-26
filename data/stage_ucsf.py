"""Stage the UCSF cohort for training: extract the target DCE phase once, and
co-locate every modality in one flat per-patient dir on fast local storage.

Why this exists. As delivered, UCSF is split across two fac mounts and the DCE is
a single ~1 GB 4D series per patient (512x512x28x35). Training straight off that
re-reads and decompresses the whole 4D on EVERY sample of EVERY epoch just to pull
one 3D phase -- utterly I/O-bound -- and `/mnt/fac` has historically dropped on
gpu nodes mid-run. This pass reads each 4D exactly once, writes the chosen phase
as a ~30 MB 3D volume, and copies the anatomy/masks alongside it:

    <out>/<pid>/  T2W.nii.gz  ADC_to_T2W.nii.gz  DWI_to_T2W.nii.gz
                  DCE_to_T2W.nii.gz          <- the time-selected phase (3D)
                  prostate_mask.nii.gz  prostate_zones.nii.gz
                  stage_meta.json            <- phase idx/time + QC enhancement

`UCSFDCEDataset` reads this tree directly when `--ucsf-dce-root` is omitted, so
training is just `--ucsf-main-root <out>`.

Phase choice. The UCSF enhancement curve is flat until ~55 s, rises steeply to
~90 s, then PLATEAUS near 2.5x baseline out past 300 s. Intensity-argmax would
therefore land on plateau noise (~280 s, into washout), so the phase is chosen by
ACQUISITION TIME (`--target-time`, default 120 s) -- inside the stable plateau,
past the timing-sensitive wash-in, before washout.

QC. `stage_meta.json` records the mask-mean enhancement ratio (chosen phase over
pre-contrast). Cases whose ratio is ~1.0 never enhanced (failed/mistimed injection)
and are worth excluding -- `--min-enh` drops them at staging time, and the summary
prints the distribution.

    python -m tier1_static.data.stage_ucsf \
        --main-root /mnt/fac/CX000018_DS2/.../UCSF_data/registered \
        --dce-root  /mnt/fac/CX000018_DS3/.../UCSF_data/registered \
        --out /mnt/scratch/user/$USER/UCSF_staged --target-time 120 --workers 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .dataset import UCSF_STEMS, _ucsf_dwi_stems, _resolve_stem, ucsf_phase_index
from .preprocessing import extract_phase_from_4d, load_sitk_4d

# copied verbatim (mask/zones are labels; anatomy is already registered to T2W)
COPY_FILES = ["prostate_mask.nii.gz", "prostate_zones.nii.gz"]

# sweep the full enhancement curve only when the selected phase looks weak
SWEEP_BELOW = 1.5


def _interleave_factor(dce_dir):
    """How many sub-series are interleaved in the 4D, inferred from repeated
    timestamps. Some UCSF studies stack e.g. Dixon water/fat or two echoes, so the
    4D holds k volumes per timepoint (n_phases = k * n_timepoints) and a plain index
    alternates between sub-series -- one of which barely enhances. Returns k (1 =
    a normal single series)."""
    try:
        meta = json.loads((Path(dce_dir) / "dce_times.json").read_text())
        phases = meta["phases"] if isinstance(meta, dict) and "phases" in meta else meta
        ts = [p.get("rel_time_s") for p in phases]
        uniq = len({t for t in ts if t is not None})
        return max(1, round(len(ts) / uniq)) if uniq else 1
    except Exception:
        return 1


def stage_one(pid, main_root, dce_root, out_root, target_time, t_max, dwi_bvalue,
              min_enh=0.0, overwrite=False):
    """Stage a single patient. Returns a dict record (or {'pid', 'skip': reason})."""
    subj = Path(main_root) / pid
    dst = Path(out_root) / pid
    meta_p = dst / "stage_meta.json"
    if meta_p.exists() and not overwrite:
        return {**json.loads(meta_p.read_text()), "cached": True}

    dce_dir = Path(dce_root) / pid / "DCE"
    t2 = _resolve_stem(subj, UCSF_STEMS["t2w"])
    adc = _resolve_stem(subj, UCSF_STEMS["adc"])
    dwi = _resolve_stem(subj, _ucsf_dwi_stems(dwi_bvalue))
    mask_p = subj / "prostate_mask.nii.gz"
    if not (t2 and adc and dwi and mask_p.exists()):
        return {"pid": pid, "skip": "missing inputs"}
    if not (dce_dir / "DCE_4D_to_T2W.nii.gz").exists() or not (dce_dir / "dce_times.json").exists():
        return {"pid": pid, "skip": "no DCE"}

    idx, t_sel = ucsf_phase_index(dce_dir, target_time, t_max)
    interleave = _interleave_factor(dce_dir)
    img4d = load_sitk_4d(dce_dir / "DCE_4D_to_T2W.nii.gz")
    n_phases = img4d.GetSize()[3] if img4d.GetDimension() > 3 else 1

    m = None
    try:
        mm = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_p))) > 0
        probe = sitk.GetArrayFromImage(extract_phase_from_4d(img4d, 0))
        m = mm if (mm.shape == probe.shape and mm.any()) else None
    except Exception:
        pass

    def _mean(j):
        return float(sitk.GetArrayFromImage(extract_phase_from_4d(img4d, j))[m].mean())

    # Interleaved 4D (k volumes per timepoint, e.g. Dixon water/fat): a plain index
    # alternates between sub-series and can land on the one that barely enhances.
    # Re-pick within the selected timepoint, keeping the sub-series that actually
    # enhances (each scored against its OWN t=0 volume).
    sub = None
    if interleave > 1 and m is not None and n_phases > interleave:
        grp = idx - (idx % interleave)
        best, best_e = idx, -1.0
        for s in range(interleave):
            j = grp + s
            if j >= n_phases:
                continue
            b = _mean(s)
            e = _mean(j) / b if abs(b) > 1e-6 else 0.0
            if e > best_e:
                best, best_e, sub = j, e, s
        idx = best

    phase = extract_phase_from_4d(img4d, idx)

    # QC: mask-mean enhancement of the chosen phase vs its sub-series pre-contrast.
    # If that looks low, sweep the WHOLE series for the max -- a low selected-phase
    # value can mean either a genuinely non-enhancing study (max also ~1.0 -> drop)
    # or a mis-selected phase (max is high -> keep and fix selection), and only the
    # max distinguishes them. The sweep is skipped on the healthy majority.
    enh = enh_max = enh_max_idx = None
    if m is not None:
        try:
            base = _mean(idx % interleave if interleave > 1 else 0)
            if abs(base) > 1e-6:
                enh = float(sitk.GetArrayFromImage(phase)[m].mean() / base)
                if enh < SWEEP_BELOW and n_phases > 1:
                    curve = [_mean(j) / base for j in range(n_phases)]
                    enh_max_idx = int(np.argmax(curve))
                    enh_max = float(curve[enh_max_idx])
        except Exception:
            pass
    del img4d
    if enh_max is None:
        enh_max = enh
    # filter on the MAX over the series: only that separates "never enhanced" from
    # "we picked the wrong phase" (e.g. interleaved sub-series).
    if min_enh and enh_max is not None and enh_max < min_enh:
        return {"pid": pid, "skip": f"no enhancement (max {enh_max:.2f})",
                "enh_ratio": enh, "enh_max": enh_max}

    dst.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(phase, str(dst / "DCE_to_T2W.nii.gz"), True)     # True = compress
    for src, name in ((t2, "T2W.nii.gz"), (adc, "ADC_to_T2W.nii.gz"),
                      (dwi, "DWI_to_T2W.nii.gz")):
        shutil.copyfile(src, dst / name)
    for name in COPY_FILES:
        p = subj / name
        if p.exists():
            shutil.copyfile(p, dst / name)

    rec = {"pid": pid, "phase_idx": idx, "rel_time_s": t_sel, "n_phases": n_phases,
           "target_time": target_time, "enh_ratio": enh, "enh_max": enh_max,
           "enh_max_idx": enh_max_idx, "interleave": interleave, "sub_series": sub,
           "dwi_src": Path(dwi).name}
    meta_p.write_text(json.dumps(rec, indent=1))
    return rec


def _worker(a):
    try:
        return stage_one(*a)
    except Exception as e:                        # never let one bad case kill the pass
        return {"pid": a[0], "skip": f"error: {type(e).__name__}: {e}"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-root", default="", help="UCSF DS2 registered tree (anatomy/masks)")
    p.add_argument("--dce-root", default="", help="UCSF DS3 registered tree (<pid>/DCE/)")
    p.add_argument("--out", required=True, help="output staged root (use fast local scratch)")
    p.add_argument("--target-time", type=float, default=120.0,
                   help="acquisition time (s) of the DCE phase to extract; default 120 "
                        "(stable plateau, past wash-in, before washout)")
    p.add_argument("--t-max", type=float, default=600.0)
    p.add_argument("--dwi-bvalue", default="", help="preferred DWI b-value (e.g. 1000); '' = highest")
    p.add_argument("--min-enh", type=float, default=0.0,
                   help="skip cases whose mask-mean enhancement ratio is below this "
                        "(e.g. 1.2 drops failed/mistimed injections); 0 = keep all")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    # SLURM array sharding: each task takes a strided slice of the patient list
    # (strided, not contiguous, so uneven per-case cost spreads evenly across tasks).
    p.add_argument("--shard", type=int, default=0, help="this task's index (0-based)")
    p.add_argument("--num-shards", type=int, default=1, help="total array tasks")
    p.add_argument("--report", action="store_true",
                   help="don't stage; scan <out>/*/stage_meta.json and print the aggregate "
                        "summary (use after a sharded array job finishes)")
    a = p.parse_args(argv)

    out = Path(a.out)
    if a.report:
        return _report(out)
    if not (a.main_root and a.dce_root):
        p.error("--main-root and --dce-root are required unless --report")

    pids = sorted(d.name for d in Path(a.dce_root).glob("*")
                  if (d / "DCE" / "DCE_4D_to_T2W.nii.gz").exists())
    if a.limit:
        pids = pids[:a.limit]
    n_all = len(pids)
    if a.num_shards > 1:
        pids = pids[a.shard::a.num_shards]
    out.mkdir(parents=True, exist_ok=True)
    shard_note = f", shard {a.shard}/{a.num_shards} of {n_all}" if a.num_shards > 1 else ""
    print(f"staging {len(pids)} patients -> {out}  (target_time={a.target_time}s, "
          f"workers={a.workers}{shard_note})", flush=True)

    jobs = [(pid, a.main_root, a.dce_root, a.out, a.target_time, a.t_max,
             a.dwi_bvalue, a.min_enh, a.overwrite) for pid in pids]
    recs, t0 = [], time.time()

    def _tick(k):
        if k % 20 == 0 or k == len(jobs):
            el = time.time() - t0
            print(f"  [{k}/{len(jobs)}] {el:.0f}s elapsed, {el/max(k,1):.1f}s/case", flush=True)

    if a.workers <= 1:                            # serial path (also keeps it importable)
        for k, j in enumerate(jobs, 1):
            recs.append(_worker(j)); _tick(k)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                recs.append(f.result()); _tick(k)

    ok = [r for r in recs if "skip" not in r]
    skipped = [r for r in recs if "skip" in r]
    name = "stage_summary.json" if a.num_shards == 1 else f"stage_summary_{a.shard:03d}.json"
    (out / name).write_text(json.dumps(
        {"n_ok": len(ok), "n_skipped": len(skipped), "target_time": a.target_time,
         "shard": a.shard, "num_shards": a.num_shards,
         "records": sorted(recs, key=lambda r: r["pid"])}, indent=1))

    print(f"\nstaged {len(ok)}  skipped {len(skipped)}  in {time.time()-t0:.0f}s")
    _stats(ok, skipped)
    if a.num_shards > 1:
        print(f"\n(shard {a.shard} done; after ALL array tasks finish run:\n"
              f"   python -m tier1_static.data.stage_ucsf --report --out {out})")
    else:
        print(f"\ntrain with:  --ucsf-main-root {out}   (no --ucsf-dce-root)")
    return 0


def _stats(ok, skipped=()):
    """Print phase/time/enhancement distributions for a set of staged records."""
    if skipped:
        from collections import Counter
        for reason, n in Counter(r["skip"].split(":")[0] for r in skipped).most_common():
            print(f"  skip: {reason} x{n}")
    idxs = [r["phase_idx"] for r in ok if r.get("phase_idx") is not None]
    ts = [r["rel_time_s"] for r in ok if r.get("rel_time_s") is not None]
    enh = [r["enh_ratio"] for r in ok if r.get("enh_ratio")]
    if idxs:
        print(f"  phase idx : median {int(np.median(idxs))}  range {min(idxs)}-{max(idxs)}")
    if ts:
        print(f"  phase time: median {np.median(ts):.1f}s  range {min(ts):.1f}-{max(ts):.1f}s")
    if enh:
        e = np.array(enh)
        print(f"  enhancement @ selected phase: median {np.median(e):.2f}  "
              f"p10 {np.percentile(e,10):.2f}  p90 {np.percentile(e,90):.2f}")
    # split the weak cases: never-enhanced (drop) vs mis-selected phase (recoverable)
    weak = [r for r in ok if (r.get("enh_ratio") or 9) < 1.2]
    dead = [r for r in weak if (r.get("enh_max") or 0) < 1.2]
    fixable = [r for r in weak if (r.get("enh_max") or 0) >= 1.2]
    if weak:
        print(f"  {len(weak)} case(s) < 1.2x at the selected phase:")
        print(f"     {len(dead)} never enhance anywhere in the series "
              f"(failed/mistimed injection) -> --min-enh 1.2 drops these")
        if fixable:
            print(f"     {len(fixable)} DO enhance elsewhere (max up to "
                  f"{max(r['enh_max'] for r in fixable):.2f}x) -> phase mis-selected, "
                  f"recoverable; see interleave below")
    inter = [r for r in ok if (r.get("interleave") or 1) > 1]
    if inter:
        ks = sorted({r["interleave"] for r in inter})
        print(f"  {len(inter)} case(s) have interleaved sub-series (k={ks}) -- the 4D "
              f"stacks >1 volume per timepoint (e.g. Dixon water/fat)")


def _report(out: Path):
    """Aggregate every per-patient stage_meta.json under `out` (post-array summary)."""
    recs = []
    for m in sorted(out.glob("*/stage_meta.json")):
        try:
            recs.append(json.loads(m.read_text()))
        except Exception:
            print(f"  unreadable: {m}")
    print(f"staged patients under {out}: {len(recs)}")
    _stats(recs)
    (out / "stage_summary.json").write_text(json.dumps(
        {"n_ok": len(recs), "records": sorted(recs, key=lambda r: r["pid"])}, indent=1))
    print(f"\ntrain with:  --ucsf-main-root {out}   (no --ucsf-dce-root)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
