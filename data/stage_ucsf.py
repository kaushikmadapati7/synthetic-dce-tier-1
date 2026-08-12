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
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .dataset import UCSF_STEMS, ucsf_dwi_path, _resolve_stem, ucsf_phase_index
from .preprocessing import extract_phase_from_4d, load_sitk_4d

# copied verbatim (mask/zones are labels; anatomy is already registered to T2W)
COPY_FILES = ["prostate_mask.nii.gz", "prostate_zones.nii.gz"]

# sweep the full enhancement curve only when the selected phase looks weak
SWEEP_BELOW = 1.5
# ...and if the sweep shows the study DOES enhance this much, the timing was wrong,
# not the study: re-select from the curve instead of trusting the timestamps.
RESELECT_ABOVE = 1.5
# curve-based target = first phase reaching this fraction of max (plateau onset)
PLATEAU_FRAC = 0.9


MAX_INTERLEAVE = 4


def _interleave_factor(dce_dir):
    """How many sub-series are interleaved in the 4D, inferred from REPEATED
    timestamps. Some UCSF studies stack e.g. Dixon water/fat or two echoes, so the
    4D holds k volumes per timepoint (n_phases = k * n_timepoints) and a plain index
    alternates between sub-series -- one of which barely enhances. Returns k (1 = a
    normal single series).

    Only genuinely repeated timestamps count: a series with MISSING (null) times
    would otherwise collapse to one unique value and masquerade as k = n_phases.
    """
    try:
        meta = json.loads((Path(dce_dir) / "dce_times.json").read_text())
        phases = meta["phases"] if isinstance(meta, dict) and "phases" in meta else meta
        ts = [p.get("rel_time_s") for p in phases]
        if not ts or any(t is None for t in ts):
            return 1                              # incomplete timing -> can't infer
        uniq = len(set(ts))
        if uniq == 0 or len(ts) % uniq:
            return 1                              # not a clean k-per-timepoint stack
        k = len(ts) // uniq
        return k if 1 <= k <= MAX_INTERLEAVE else 1
    except Exception:
        return 1


def _write_atomic(img, dst_path: Path):
    """Write via a temp file + os.replace, which is atomic on POSIX.

    Staging writes into the same tree training reads. A plain WriteImage leaves the
    file partially written for as long as compression takes, and any DataLoader worker
    that opens it in that window dies with "Unable to determine ImageIO reader" --
    which kills the whole run, since every job walks the full cohort. os.replace means
    a reader sees either the old file or the new one, never a half-written one.
    """
    tmp = dst_path.with_name(dst_path.name + ".tmp")
    sitk.WriteImage(img, str(tmp), True)
    os.replace(tmp, dst_path)


def _copy_atomic(src, dst_path: Path):
    tmp = dst_path.with_name(dst_path.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst_path)


def anchor_name(t: float) -> str:
    """Filename for the anchor phase nearest `t` seconds post-injection."""
    return f"DCE_t{int(round(t)):03d}_to_T2W.nii.gz"


def stage_one(pid, main_root, dce_root, out_root, target_time, t_max, dwi_bvalue,
              min_enh=0.0, overwrite=False, anchor_times=()):
    """Stage a single patient. Returns a dict record (or {'pid', 'skip': reason})."""
    subj = Path(main_root) / pid
    dst = Path(out_root) / pid
    meta_p = dst / "stage_meta.json"
    # "Already staged" means every artifact THIS INVOCATION would write is present --
    # including the pre-contrast phase and every requested anchor. Checking only
    # stage_meta.json (or a fixed file list) would make a cohort staged by an older
    # version look complete, so a re-stage adding new anchors would silently skip the
    # entire cohort and report success. Same failure class as the --overwrite pass that
    # once resurrected 729 pruned cases: the resume predicate must track the outputs.
    core = [dst / n for n in ("DCE_to_T2W.nii.gz", "DCE_pre_to_T2W.nii.gz", "T2W.nii.gz",
                              "ADC_to_T2W.nii.gz", "DWI_to_T2W.nii.gz")]
    core_ok = meta_p.exists() and all(p.exists() for p in core)
    need_anchor = [t for t in anchor_times if not (dst / anchor_name(t)).exists()]
    if core_ok and not need_anchor and not overwrite:
        return {**json.loads(meta_p.read_text()), "cached": True}
    # Already staged, only anchors missing -> ADD those and touch nothing else. Without
    # this an anchor backfill rewrites every core file for all 4246 cases, and any
    # training job reading the tree at the time dies.
    backfill_only = core_ok and bool(need_anchor) and not overwrite
    if backfill_only:
        anchor_times = tuple(need_anchor)

    dce_dir = Path(dce_root) / pid / "DCE"
    t2 = _resolve_stem(subj, UCSF_STEMS["t2w"])
    adc = _resolve_stem(subj, UCSF_STEMS["adc"])
    dwi = ucsf_dwi_path(subj, dwi_bvalue)
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
    # Pre-contrast T1 = phase 0 of the SELECTED sub-series (rel_time_s == 0, i.e.
    # before injection). Same series, same geometry, already registered to the
    # target -- so it is a better pre-contrast source than the separate pregad
    # acquisition, and free here since the 4D is already open. Staged for the
    # 4th-conditioning-channel / enhancement-residual experiments.
    pre_idx = idx % interleave if interleave > 1 else 0
    pre_phase = extract_phase_from_4d(img4d, pre_idx)

    # QC: mask-mean enhancement of the chosen phase vs its sub-series pre-contrast.
    # If that looks low, sweep the series for the max -- a low selected-phase value
    # can mean either a genuinely non-enhancing study (max also ~1.0 -> drop) or a
    # mis-selected phase (max is high -> re-select), and only the max distinguishes
    # them. The sweep is skipped on the healthy majority.
    enh = enh_max = enh_max_idx = None
    select_mode = "time"
    if m is not None:
        try:
            off = idx % interleave if interleave > 1 else 0
            base = _mean(off)
            if abs(base) > 1e-6:
                enh = float(sitk.GetArrayFromImage(phase)[m].mean() / base)
                if enh < SWEEP_BELOW and n_phases > 1:
                    js = list(range(off, n_phases, interleave))   # same sub-series only
                    curve = [_mean(j) / base for j in js]
                    k = int(np.argmax(curve))
                    enh_max_idx, enh_max = js[k], float(curve[k])
                    # Timing was unusable (e.g. corrupt right after phase 0, so we fell
                    # back to the pre-contrast volume) yet the study clearly enhances ->
                    # re-select from the curve at PLATEAU ONSET: the first phase reaching
                    # 90% of max. That is what target_time=120s approximates anyway, and
                    # it needs no trustworthy timestamps.
                    if enh_max >= RESELECT_ABOVE:
                        thr = PLATEAU_FRAC * enh_max
                        j = next((jj for jj, v in zip(js, curve) if v >= thr), enh_max_idx)
                        if j != idx:
                            idx, select_mode = j, "curve"
                            phase = extract_phase_from_4d(img4d, idx)
                            t_sel = None          # timing untrustworthy for this case
                            enh = float(curve[js.index(j)])
        except Exception:
            pass
    # Anchor phases for the multi-timepoint (Tier-2) track, on a COMMON time grid so
    # heterogeneous protocols (26-70 phases, 1.7-13s cadence) become comparable.
    # Extracted here because the 4D is already open -- one read, N phases.
    #
    # Only written when the timestamps are trustworthy: if we had to fall back to the
    # enhancement curve (select_mode == "curve"), the times are corrupt, so a
    # "time grid" built from them would be fiction. Those cases get anchors=None and
    # the time-series loader can skip them.
    anchors = None
    if anchor_times and select_mode == "time":
        off = idx % interleave if interleave > 1 else 0
        anchors = {}
        for a in anchor_times:
            try:
                j, t_a = ucsf_phase_index(dce_dir, a, t_max)
            except Exception:
                continue
            if interleave > 1:                 # stay inside the SELECTED sub-series,
                j = j - (j % interleave) + off  # else anchors alternate water/fat
            j = max(0, min(int(j), n_phases - 1))
            anchors[anchor_name(a)] = {"target_s": a, "idx": j, "rel_time_s": t_a,
                                       "phase": extract_phase_from_4d(img4d, j)}
    del img4d
    if enh_max is None:
        enh_max = enh
    # filter on the MAX over the series: only that separates "never enhanced" from
    # "we picked the wrong phase" (e.g. interleaved sub-series).
    if min_enh and enh_max is not None and enh_max < min_enh:
        if dst.exists():          # previously staged under a looser threshold -> remove,
            shutil.rmtree(dst)    # else the loader would keep picking up a rejected case
        return {"pid": pid, "skip": f"no enhancement (max {enh_max:.2f})",
                "enh_ratio": enh, "enh_max": enh_max}

    dst.mkdir(parents=True, exist_ok=True)
    if not backfill_only:
        _write_atomic(phase, dst / "DCE_to_T2W.nii.gz")
        _write_atomic(pre_phase, dst / "DCE_pre_to_T2W.nii.gz")
        for src, name in ((t2, "T2W.nii.gz"), (adc, "ADC_to_T2W.nii.gz"),
                          (dwi, "DWI_to_T2W.nii.gz")):
            _copy_atomic(src, dst / name)
        for name in COPY_FILES:
            p = subj / name
            if p.exists():
                _copy_atomic(p, dst / name)
    for fname, a in (anchors or {}).items():
        _write_atomic(a.pop("phase"), dst / fname)

    # `series` is the only scanner covariate that survived de-identification (the DICOM
    # Manufacturer tag is wiped to "NA"): TWIST -> Siemens, DISCO -> GE (Dixon, and the
    # likely source of the interleaved k=2 cases). `reg` is the DCE->T2w registration
    # quality, which both CEKWorld and ODEWorld name as the failure mode for generated
    # dynamics. Both were being discarded.
    times_j = {}
    try:
        times_j = json.loads((dce_dir / "dce_times.json").read_text())
    except Exception:
        pass
    rec = {"pid": pid, "phase_idx": idx, "rel_time_s": t_sel, "n_phases": n_phases,
           "target_time": target_time, "enh_ratio": enh, "enh_max": enh_max,
           "enh_max_idx": enh_max_idx, "interleave": interleave, "sub_series": sub,
           "select_mode": select_mode, "pre_idx": pre_idx, "dwi_src": Path(dwi).name,
           "series": times_j.get("series"), "reg": times_j.get("reg"),
           "anchors": {k: {kk: vv for kk, vv in v.items() if kk != "phase"}
                       for k, v in (anchors or {}).items()} or None}
    if backfill_only and meta_p.exists():        # keep the original record, add anchors
        try:
            rec = {**json.loads(meta_p.read_text()), "anchors": rec.get("anchors")}
        except Exception:
            pass
    tmp = meta_p.with_name(meta_p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, meta_p)
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
    p.add_argument("--anchor-times", type=float, nargs="*", default=[],
                   help="extra phases to stage on a COMMON time grid (s post-injection) "
                        "for the multi-timepoint track, e.g. --anchor-times 45 60 75 90 "
                        "150 180 240. 0s and 120s already ship as DCE_pre_to_T2W and "
                        "DCE_to_T2W. Sampling should be dense through wash-in (55-90s) "
                        "and sparse across the plateau, since that is where PI-RADS "
                        "curve types separate. Only written when timestamps are "
                        "trustworthy (select_mode == 'time').")
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
    p.add_argument("--prune-below", type=float, default=0.0,
                   help="with --report: DELETE already-staged cases whose max enhancement is "
                        "below this (no 4D re-read needed). Use after inspecting the summary, "
                        "e.g. --report --prune-below 1.4")
    a = p.parse_args(argv)

    out = Path(a.out)
    if a.report:
        return _report(out, a.prune_below)
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
    if a.anchor_times:
        print(f"  anchors: {sorted(a.anchor_times)}  (+{len(a.anchor_times)} phases/case)")
    print(f"staging {len(pids)} patients -> {out}  (target_time={a.target_time}s, "
          f"workers={a.workers}{shard_note})", flush=True)

    jobs = [(pid, a.main_root, a.dce_root, a.out, a.target_time, a.t_max,
             a.dwi_bvalue, a.min_enh, a.overwrite, tuple(a.anchor_times)) for pid in pids]
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
    curved = [r for r in ok if r.get("select_mode") == "curve"]
    if curved:
        e = [r["enh_ratio"] for r in curved if r.get("enh_ratio")]
        print(f"  {len(curved)} case(s) had unusable timing -> phase re-selected from the "
              f"enhancement curve (plateau onset)" +
              (f", enhancement now median {np.median(e):.2f}x" if e else ""))
    t0 = [r for r in ok if r.get("rel_time_s") == 0]
    if t0:
        print(f"  WARNING {len(t0)} case(s) still target t=0 (pre-contrast) -- "
              f"broken timing AND no measurable enhancement")


def _report(out: Path, prune_below: float = 0.0):
    """Aggregate every per-patient stage_meta.json under `out` (post-array summary).
    With `prune_below`, delete staged cases whose max enhancement is under it."""
    recs = []
    for m in sorted(out.glob("*/stage_meta.json")):
        try:
            recs.append(json.loads(m.read_text()))
        except Exception:
            print(f"  unreadable: {m}")
    print(f"staged patients under {out}: {len(recs)}")

    if prune_below:
        def _emax(r):                       # pre-QC-fix records have no enh_max
            return r.get("enh_max") if r.get("enh_max") is not None else r.get("enh_ratio")
        drop = [r for r in recs if _emax(r) is not None and _emax(r) < prune_below]
        stale = [r for r in recs if _emax(r) is None]
        for r in drop:
            shutil.rmtree(out / r["pid"], ignore_errors=True)
        recs = [r for r in recs if r not in drop]
        print(f"  pruned {len(drop)} case(s) with max enhancement < {prune_below} "
              f"-> {len(recs)} remain")
        if stale:
            print(f"  NOTE {len(stale)} case(s) predate the QC fix (no enh_max) and were "
                  f"NOT evaluated; re-stage with --overwrite to score them")
    _stats(recs)
    # NOT stage_summary.json -- that is the staging run's own output (it carries the
    # skip records), and clobbering it loses why cases were dropped.
    (out / "stage_report.json").write_text(json.dumps(
        {"n_ok": len(recs), "records": sorted(recs, key=lambda r: r["pid"])}, indent=1))
    print(f"\ntrain with:  --ucsf-main-root {out}   (no --ucsf-dce-root)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
