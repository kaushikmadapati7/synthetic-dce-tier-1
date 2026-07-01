"""Generate synthetic DCE volumes from a trained model, for the downstream
clinical eval. Writes one `synth_DCE.nii.gz` per case at
    <synth-out>/<center>/<subject>/synth_DCE.nii.gz
in the *native DCE geometry* (uncropped, DCE spacing/origin/direction), which is
exactly what the collaborator's classifier `--mode bpmri+synth` expects (it does
its own resample/crop). Requires --reference dce (our default) so the model grid
is the DCE grid; the prediction is uncropped and stamped with the DCE geometry.

    python -m tier1_static.generate_synth --eval-only --model gan \
        --data-root .../Bao_DCE --output-dir runs/tierA_gan_clin --synth-out synth_dce/gan

NOTE (leakage): the synthesis model was trained on some of these cases, so synth
DCE on them is optimistic vs a clean cross-fold. Fine for a first read; flag it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from .main import parse_args, build_data, set_seed, setup_logging
from .training import LOADERS
from .data.dataset import _resolve_stem, MODALITY_STEMS
from .data.preprocessing import uncrop_pad, load_sitk

log = logging.getLogger("tier1")


def _dce_ref(data_root, image_subdir, case_id):
    """Native DCE image for `case_id` (= its target geometry)."""
    center, subject = case_id.split("/", 1)
    d = Path(data_root) / image_subdir / center / subject
    p = _resolve_stem(d, MODALITY_STEMS["dce"])
    return load_sitk(p) if p else None


@torch.no_grad()
def main():
    args = parse_args()
    args.eval_only = True                     # load the saved harmonizer, no re-fit
    set_seed(args.seed)
    setup_logging(Path(args.output_dir))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    synth_out = Path(getattr(args, "synth_out", None) or (Path(args.output_dir) / "synth_dce"))
    synth_out.mkdir(parents=True, exist_ok=True)
    log.info(f"generate_synth: model={args.model} -> {synth_out}")

    train, val, test = build_data(args)
    _, gen = LOADERS[args.model](args, train, test, device)

    n = 0
    for loader in (train, val, test):
        if loader is None:
            continue
        for batch in loader:
            cond = batch["cond"].to(device)
            preds = gen(cond).clamp(-1, 1).cpu().numpy()   # (B,1,D,H,W) on the DCE-cropped grid
            for i, cid in enumerate(batch["id"]):
                ref = _dce_ref(args.data_root, args.image_subdir, cid)
                if ref is None:
                    log.warning(f"no DCE ref for {cid}; skipping"); continue
                native = sitk.GetArrayFromImage(ref).shape           # (D,H,W)
                vol = uncrop_pad(preds[i, 0], native, pad_value=-1.0)
                img = sitk.GetImageFromArray(vol.astype(np.float32))
                img.CopyInformation(ref)                             # DCE geometry
                dst = synth_out / cid
                dst.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(img, str(dst / "synth_DCE.nii.gz"))
                n += 1
                if n % 50 == 0:
                    log.info(f"  wrote {n} synth_DCE volumes")
    log.info(f"done: {n} synth_DCE.nii.gz under {synth_out}")


if __name__ == "__main__":
    main()
