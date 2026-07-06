"""Reader-study export: blinded real-vs-synthetic DCE panels for a radiologist
Turing test -- the clinical-realism gold standard (a synth DCE is "realistic" if a
radiologist can't tell it from a real one). Complements the downstream csPCa-AUC
utility eval; together they cover realism + faithfulness.

Two idempotent stages:

  1. EXPORT (per model): render one clinically-windowed axial DCE slice per case
     for this model's synthetic output ->  <reader-out>/staging/<case>__<tag>.png.
     Run once per candidate model (each with its own --model + trained --output-dir),
     and once with --reader-include-real to add the REAL DCE panels.

  2. FINALIZE (--reader-finalize): shuffle every staged panel, assign anonymous IDs,
     copy to <reader-out>/images/NNNN.png, and write key.csv (hidden truth) +
     ratings_template.csv (for the radiologist). Blinding happens HERE, after all
     sources exist, so real/synth/model can't be inferred from a filename.

Fairness: all panels for a case share the REAL DCE's ROI [1,99]% window, so the
reader judges texture/structure realism, not a windowing artifact; masks, titles,
and annotations are never drawn. Draw the reader set from the SAME split/seed for
every model so each case has real + each-model panels on the identical grid.

  # export the real panels + each model's synth (same reader-out)
  python -m tier1_static.reader_study_export --eval-only --model gan \
      --data-root .../Bao_DCE --output-dir runs/gan_run \
      --reader-out reader_study --reader-include-real
  python -m tier1_static.reader_study_export --eval-only --model ldm_flow \
      --first-stage wavelet --wavelet-levels 2 --wavelet-loss energy \
      --data-root .../Bao_DCE --output-dir runs/flow_run --reader-out reader_study
  # blind + build the radiologist packet
  python -m tier1_static.reader_study_export --reader-finalize --reader-out reader_study
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .main import parse_args, build_data, set_seed, setup_logging
from .training import LOADERS

log = logging.getLogger("tier1")


def _model_tag(args) -> str:
    """Short source label for the staged filename (blinded away at finalize)."""
    if args.model != "ldm_flow":
        return args.model
    if getattr(args, "first_stage", "vae") == "wavelet":
        return "flow_wavelet"
    return "flow_vae"


def _most_prostate_slice(mask_vol: np.ndarray) -> int:
    """Axial slice (D index) with the most prostate-mask voxels; mid-slice if none."""
    if mask_vol.sum() > 0:
        return int(mask_vol.sum(axis=(1, 2)).argmax())
    return mask_vol.shape[0] // 2


def _render(dce2d, t2w2d, lo, hi, with_context, path):
    """Save one blinded grayscale panel (no axes/title/annotation). All sources are
    rendered identically so nothing but the image content can distinguish them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if with_context and t2w2d is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(t2w2d, cmap="gray", interpolation="nearest"); axes[0].axis("off")
        axes[1].imshow(dce2d, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"); axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(dce2d, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"); ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def export(args, device):
    reader_out = Path(args.reader_out)
    staging = reader_out / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    args.eval_only = True                     # load the trained ckpt + saved harmonizer
    train, val, test = build_data(args)
    loader = {"val": val, "test": test, "train": train}[getattr(args, "reader_split", "val")]
    if loader is None:
        log.error(f"no '{args.reader_split}' loader available; nothing to export")
        return
    _, gen = LOADERS[args.model](args, train, test, device)

    tag = _model_tag(args)
    include_real = getattr(args, "reader_include_real", False)
    with_ctx = getattr(args, "reader_with_context", False)
    limit = getattr(args, "reader_cases", 40)
    n = 0
    for batch in loader:
        if n >= limit:
            break
        synth = gen(batch["cond"].to(device)).clamp(-1, 1).cpu().numpy()   # (B,1,D,H,W)
        real = batch["target"].numpy()
        mask = batch["mask"].numpy()
        t2w = batch["cond"][:, 0:1].numpy()
        for i, cid in enumerate(batch["id"]):
            if n >= limit:
                break
            safe = cid.replace("/", "_")
            mvol = mask[i, 0]
            d = _most_prostate_slice(mvol)
            roi = real[i, 0][mvol > 0.5]
            if roi.size >= 16:
                lo, hi = (float(x) for x in np.percentile(roi, [1, 99]))
            else:
                lo, hi = float(real[i, 0].min()), float(real[i, 0].max())
            if hi <= lo:
                hi = lo + 1e-3
            t2sl = t2w[i, 0, d] if with_ctx else None
            _render(synth[i, 0, d], t2sl, lo, hi, with_ctx, staging / f"{safe}__{tag}.png")
            if include_real:
                _render(real[i, 0, d], t2sl, lo, hi, with_ctx, staging / f"{safe}__real.png")
            n += 1
    log.info(f"exported {n} '{tag}' panels{' + real' if include_real else ''} to {staging}")


def finalize(reader_out: Path, seed: int):
    staging = reader_out / "staging"
    images = reader_out / "images"
    images.mkdir(parents=True, exist_ok=True)
    panels = sorted(staging.glob("*__*.png"))
    if not panels:
        log.error(f"no staged panels in {staging}; run the per-model export first")
        return
    order = panels[:]
    random.Random(seed).shuffle(order)          # blinding: source order is randomized here
    key_rows = []
    for idx, p in enumerate(order, 1):
        case, source = p.stem.split("__", 1)
        anon = f"{idx:04d}"
        shutil.copyfile(p, images / f"{anon}.png")
        key_rows.append((anon, case, source))
    with open(reader_out / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "case_id", "source"])   # HIDDEN truth (do not give to readers)
        w.writerows(key_rows)
    with open(reader_out / "ratings_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "guess_real_or_synth", "realism_1to5", "confidence_1to5", "notes"])
        for anon, _, _ in key_rows:
            w.writerow([anon, "", "", "", ""])
    n_real = sum(s == "real" for _, _, s in key_rows)
    log.info(f"finalized {len(key_rows)} blinded panels ({n_real} real, {len(key_rows) - n_real} synth) "
             f"-> {images}; wrote key.csv (hidden) + ratings_template.csv")


def main():
    # finalize needs no data/model, so it takes a minimal arg set (not main.parse_args,
    # which requires --data-root)
    if "--reader-finalize" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--reader-finalize", action="store_true")
        ap.add_argument("--reader-out", required=True)
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args()
        setup_logging(Path(a.reader_out))
        finalize(Path(a.reader_out), a.seed)
        return

    args = parse_args()
    if not getattr(args, "reader_out", ""):
        raise SystemExit("--reader-out is required")
    set_seed(args.seed)
    setup_logging(Path(args.output_dir))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    export(args, device)


if __name__ == "__main__":
    main()
