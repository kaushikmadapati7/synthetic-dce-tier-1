"""Reconstruct a full 3D synthetic DCE volume from a trained 2D model.

The 2D models (pix2pix GAN, pixel CFM, or MedVAE-latent flow) predict one axial
slice at a time. This runs the chosen model over EVERY slice of each case (batched
per case for speed) and stacks the predictions into a volume, written in the native
DCE geometry -- exactly like generate_synth.py for the 3D models, so it drops into
a viewer or the downstream csPCa classifier.

Note: the 2D models are trained on prostate-bearing slices, so non-prostate slices
are extrapolated. The downstream classifier crops to the prostate anyway; for a
clean visual, the prostate slices are the meaningful ones. Independently-generated
slices can have mild through-plane flicker (the 2D-vs-3D-coherence tradeoff).

  python -m tier1_static.generate_synth2d --model gan \
      --data-root .../Bao_DCE --output-dir runs/px2d_gan_realism \
      --spatial-size 32 256 256 --synth-out synth2d/gan
  python -m tier1_static.generate_synth2d --model ldm_flow --first-stage medvae \
      --medvae-model medvae_4_1_2d --data-root .../Bao_DCE \
      --output-dir runs/px2d_medvae --spatial-size 32 256 256 --synth-out synth2d/medvae
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from .main import parse_args, build_data, set_seed, setup_logging
from .models.gan2d import Generator2D
from .models.flow2d import FlowMatching2D, LatentFlowMatching2D
from .data.preprocessing import uncrop_pad, load_sitk
from .data.dataset import _resolve_stem, MODALITY_STEMS
from .data.slice_dataset import SliceDCEDataset
from .main2d import _fit_scaling_2d

log = logging.getLogger("tier1")


def _dce_ref(data_root, image_subdir, case_id):
    """Native DCE image for `case_id` (= its target geometry)."""
    center, subject = case_id.split("/", 1)
    d = Path(data_root) / image_subdir / center / subject
    p = _resolve_stem(d, MODALITY_STEMS["dce"])
    if p is None:                      # multi-phase (zhongyiyuan): DCE_ph*_to_T2W, not DCE_to_T2WI
        cands = sorted(d.glob("DCE*_to_T2W*.nii.gz")) or sorted(d.glob("DCE*.nii.gz"))
        # prefer ph3 (what the downstream classifier uses as zhongyiyuan's real DCE)
        p = next((c for c in cands if "ph3" in c.name), cands[0] if cands else None)
    return load_sitk(p) if p else None


@torch.no_grad()
def _build_gen(args, train_loader, device):
    """Rebuild the 2D model, load its best (or last) checkpoint, return gen(cond)."""
    out = Path(args.output_dir)
    is_flow = args.model != "gan"
    is_latent = is_flow and getattr(args, "first_stage", "vae") == "medvae"
    name = "flow2d" if is_flow else "gan2d"

    ckpt = out / f"{name}_best.pt"
    if not ckpt.exists():
        ckpt = out / f"{name}_last.pt"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    # infer base channels from the checkpoint so we don't depend on --base-ch matching
    base = int(sd["unet.cin.weight"].shape[0] if is_flow else sd["d1.0.weight"].shape[0])

    if is_latent:
        from .models import MedVAEFirstStage
        fs = MedVAEFirstStage(model_name=getattr(args, "medvae_model", "medvae_4_1_2d"),
                              modality=getattr(args, "medvae_modality", "mri")).to(device)
        sl = DataLoader(SliceDCEDataset(train_loader.dataset, args.spatial_size[0]),
                        batch_size=args.batch_size)
        _fit_scaling_2d(fs, sl, device)                # scaling_factor isn't in the ckpt
        model = LatentFlowMatching2D(fs, cond_ch=3, base=base).to(device)
    elif is_flow:
        model = FlowMatching2D(cond_ch=3, base=base,
                               source=getattr(args, "flow_source", "noise")).to(device)
    else:
        model = Generator2D(in_ch=3, out_ch=1, base=base).to(device)

    model.load_state_dict(sd)
    model.eval()
    log.info(f"loaded 2D {name} (base={base}) from {ckpt}")

    steps = args.sample_steps
    if is_flow:
        return lambda c: model.sample(c, steps=steps).clamp(-1, 1)
    return lambda c: model(c).clamp(-1, 1)


@torch.no_grad()
def main():
    args = parse_args()
    args.eval_only = True                              # load the saved harmonizer, no re-fit
    set_seed(args.seed)
    setup_logging(Path(args.output_dir))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    synth_out = Path(getattr(args, "synth_out", None) or (Path(args.output_dir) / "synth2d"))
    synth_out.mkdir(parents=True, exist_ok=True)
    log.info(f"generate_synth2d: model={args.model} first_stage={getattr(args,'first_stage','vae')} -> {synth_out}")

    train, val, test = build_data(args)
    gen = _build_gen(args, train, device)

    n = skipped = 0
    for loader in (train, val, test):
        if loader is None:
            continue
        for batch in loader:
            for i, cid in enumerate(batch["id"]):
                dst = synth_out / cid / "synth_DCE.nii.gz"
                if dst.exists() and (dst.parent / "target_DCE.nii.gz").exists():  # resumable
                    skipped += 1
                    continue
                # (3,D,H,W) -> (D,3,H,W): D slices as one batch through the 2D model
                cslices = batch["cond"][i].permute(1, 0, 2, 3).to(device)
                vol = gen(cslices)[:, 0].cpu().numpy()  # (D,H,W) on the cropped grid
                ref = _dce_ref(args.data_root, args.image_subdir, cid)
                if ref is None:
                    log.warning(f"no DCE ref for {cid}; skipping"); continue
                native = sitk.GetArrayFromImage(ref).shape
                vol = uncrop_pad(vol, native, pad_value=-1.0)
                img = sitk.GetImageFromArray(vol.astype(np.float32))
                img.CopyInformation(ref)               # native DCE geometry
                dst.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(img, str(dst))
                # matched preprocessed target in the SAME [-1,1] space + geometry, so
                # synth-vs-target intensities are directly comparable (the raw DCE on
                # disk is in scanner units and is NOT apples-to-apples with the synth)
                tvol = uncrop_pad(batch["target"][i, 0].cpu().numpy(), native, pad_value=-1.0)
                timg = sitk.GetImageFromArray(tvol.astype(np.float32))
                timg.CopyInformation(ref)
                sitk.WriteImage(timg, str(dst.parent / "target_DCE.nii.gz"))
                n += 1
                if n % 20 == 0:
                    log.info(f"  wrote {n} synth_DCE volumes ({skipped} skipped)")
    log.info(f"done: {n} synth_DCE.nii.gz written, {skipped} skipped, under {synth_out}")


if __name__ == "__main__":
    main()
