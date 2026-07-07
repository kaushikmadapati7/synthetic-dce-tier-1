"""End-to-end Tier-1 pipeline entrypoint: data -> harmonization -> model ->
train -> eval.

Task: predict the peak-contrast DCE volume from (T2w, DWI, ADC).

Per-model training loops live in tier1_static/training/ (one file per model);
evaluation lives in tier1_static/eval.py.

Run from the project root:
    python -m tier1_static.main --model ldm_flow --data-root /path/to/Bao_DCE \
        --output-dir runs/exp1 --epochs 100

Outputs written to --output-dir:
    config.json, harmonizer.json, train.log,
    checkpoints/, metrics.json, samples/{*.nii.gz, montage.png}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PreprocessConfig, Harmonizer, HarmonizationConfig,
                   build_tier1_datasets, CanonicalDCEDataset, NewbatchDCEDataset,
                   fit_harmonizer_from_dataset, CANONICAL_HOSPITALS, TIER1_TEST_HOSPITALS)
from .loss import CustomLoss
from .training import TRAINERS, LOADERS
from .eval import evaluate, save_indist_sample

log = logging.getLogger("tier1")


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train.log"); fh.setFormatter(fmt); log.addHandler(fh)


def resolve_medicalnet_weights(args):
    """Default --medicalnet-weights to the depth-matched MedicalNet file, and warn
    loudly if the perceptual term is on but no real weights are available (otherwise
    it silently runs on a randomly-initialized backbone).

    Because the code dir and the run/CWD may differ on a cluster, the auto-default
    searches several `pretrain/` roots (in order): $TIER1_PRETRAIN_DIR, the CWD,
    and the project root next to this package. Pass an absolute --medicalnet-weights
    to bypass the search entirely.
    """
    import os
    if args.perceptual <= 0:
        return
    fname = f"resnet_{args.perceptual_depth}.pth"
    if args.medicalnet_weights is None:
        roots = []
        if os.environ.get("TIER1_PRETRAIN_DIR"):
            roots.append(Path(os.environ["TIER1_PRETRAIN_DIR"]))
        roots.append(Path("pretrain"))                              # relative to CWD
        roots.append(Path(__file__).resolve().parent.parent / "pretrain")  # next to the code
        for cand in (r / fname for r in roots):
            if cand.exists():
                args.medicalnet_weights = str(cand)
                log.info(f"perceptual: using MedicalNet weights {cand}")
                return
    elif Path(args.medicalnet_weights).exists():
        log.info(f"perceptual: using MedicalNet weights {args.medicalnet_weights}")
        return
    log.warning(
        f"perceptual_weight={args.perceptual} but no MedicalNet weights found "
        f"(--medicalnet-weights={args.medicalnet_weights}, looked for {fname} in "
        f"$TIER1_PRETRAIN_DIR / ./pretrain / <code-dir>/pretrain). The perceptual "
        f"term will run on a RANDOMLY-INITIALIZED backbone — pass valid weights, set "
        f"$TIER1_PRETRAIN_DIR, or set --perceptual 0 to disable it."
    )


def make_criterion(args, device) -> CustomLoss:
    return CustomLoss(
        l1_weight=args.l1, ssim_weight=args.ssim, perceptual_weight=args.perceptual,
        roi_weight=args.roi_weight,
        radio_weight=args.radio_weight, focal_weight=args.focal_weight,
        perceptual_depth=args.perceptual_depth, perceptual_shortcut=args.perceptual_shortcut,
        medicalnet_weights=args.medicalnet_weights,
    ).to(device)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_data(args):
    cfg = PreprocessConfig(reference=args.reference, spatial_size=tuple(args.spatial_size),
                           tz_weight=args.tz_weight, pz_weight=args.pz_weight)
    out = Path(args.output_dir)
    layout = dict(image_subdir=args.image_subdir, mask_subdir=args.mask_subdir)
    test_hospitals = args.test_hospitals

    harmonizer = None
    if args.harmonize:
        saved = out / "harmonizer.json"
        if args.eval_only and saved.exists():
            harmonizer = Harmonizer.load(saved)
            log.info(f"eval-only: loaded harmonizer {saved} (skipping re-fit)")
        else:
            # build the config first so self.nyul reflects the methods (t2w/dce can
            # be per-image -> no Nyul, which generalizes to held-out hospitals)
            hcfg = HarmonizationConfig()
            hcfg.methods = dict(hcfg.methods, t2w=args.t2w_norm, dce=args.dce_norm)
            hcfg.dce_robust_k = args.dce_robust_k
            harmonizer = Harmonizer(hcfg)
            fit_hospitals = [h for h in CANONICAL_HOSPITALS if h not in test_hospitals]
            fit_ds = CanonicalDCEDataset(args.data_root, fit_hospitals, cfg, **layout)
            if len(fit_ds) == 0:
                log.warning("no canonical exams found to fit harmonizer; disabling harmonization")
                harmonizer = None
            elif harmonizer.nyul_modalities:   # only fit if something needs Nyul
                log.info(f"fitting Nyul {harmonizer.nyul_modalities} on "
                         f"{min(len(fit_ds), args.harmonize_max)} cases ...")
                fit_harmonizer_from_dataset(harmonizer, fit_ds, max_cases=args.harmonize_max)
                harmonizer.save(out / "harmonizer.json")
            else:                              # all per-image -> nothing to fit
                log.info(f"harmonizer all per-image ({harmonizer.cfg.methods}); no Nyul fit needed")
                harmonizer.save(out / "harmonizer.json")

    train = build_tier1_datasets(args.data_root, cfg, "train", harmonizer,
                                 test_hospitals=test_hospitals, dce_phase=args.dce_phase, **layout)
    test = build_tier1_datasets(args.data_root, cfg, "test", harmonizer,
                                test_hospitals=test_hospitals, dce_phase=args.dce_phase, **layout)
    if len(train) == 0:
        log.error(f"train split is EMPTY under data-root={args.data_root} "
                  f"(image_subdir={args.image_subdir}). Check the dataset layout/paths.")
    if args.limit:
        train = torch.utils.data.Subset(train, range(min(args.limit, len(train))))
        test = torch.utils.data.Subset(test, range(min(max(1, args.limit // 4), len(test))))

    # carve a random val split off the SILVER train for best-checkpoint selection
    # (no test leakage). Done BEFORE adding newbatch so the val set stays the same
    # in-distribution silver cohort across runs -> VAL metrics stay comparable.
    val = None
    if args.val_frac and len(train) >= 4:
        n_val = max(1, int(round(len(train) * args.val_frac)))
        n_train = len(train) - n_val
        if n_train >= 1 and n_val >= 1:
            g = torch.Generator().manual_seed(args.seed)
            train, val = torch.utils.data.random_split(train, [n_train, n_val], generator=g)

    # newbatch = extra TRAINING data only (data-scale lever); never in val/test, so
    # eval stays a clean comparison against the silver-only baseline.
    if args.newbatch_root:
        nb_ds = NewbatchDCEDataset(args.newbatch_root, cfg, harmonizer=harmonizer,
                                   target_time=args.newbatch_target_time,
                                   require_dwi=args.newbatch_require_dwi)
        if args.limit:
            nb_ds = torch.utils.data.Subset(nb_ds, range(min(args.limit, len(nb_ds))))
        train = torch.utils.data.ConcatDataset([train, nb_ds]) if len(train) else nb_ds
        log.info(f"+newbatch: {len(nb_ds)} training cases added")
    log.info(f"train cases: {len(train)}  val cases: {len(val) if val else 0}  test cases: {len(test)}")

    dl = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size, shuffle=shuf,
                                     num_workers=args.num_workers, drop_last=shuf,
                                     pin_memory=torch.cuda.is_available())
    return (dl(train, True),
            (dl(val, False) if val else None),
            (dl(test, False) if len(test) else None))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    resolve_medicalnet_weights(args)
    # don't clobber the training config.json when only re-evaluating
    cfg_name = "config_eval.json" if args.eval_only else "config.json"
    (out / cfg_name).write_text(json.dumps(vars(args), indent=2))
    log.info(f"device={device}  model={args.model}")
    log.info(f"config: {json.dumps(vars(args))}")

    train_loader, val_loader, test_loader = build_data(args)

    t0 = time.time()
    if args.eval_only:
        log.info(f"eval-only: loading {args.model} checkpoint from {args.output_dir}/checkpoints")
        _, gen = LOADERS[args.model](args, train_loader, test_loader, device)
        log.info(f"checkpoint loaded in {(time.time() - t0):.1f}s")
    else:
        criterion = make_criterion(args, device)
        trainer = TRAINERS[args.model]
        _, gen = trainer(args, train_loader, val_loader, test_loader, criterion, device)
        log.info(f"training done in {(time.time() - t0) / 60:.1f} min")

    metrics = evaluate(args, gen, test_loader, device)
    save_indist_sample(args, gen, val_loader, device)   # in-distribution (val) montage
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"artifacts written to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Tier-1 synthetic DCE pipeline")
    p.add_argument("--model", choices=list(TRAINERS), default="ldm_flow")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="runs/exp")
    # data / geometry
    p.add_argument("--spatial-size", type=int, nargs=3, default=[32, 192, 192])
    p.add_argument("--reference", choices=["t2w", "dce", "iso"], default="dce")
    p.add_argument("--harmonize", action="store_true", default=True)
    p.add_argument("--no-harmonize", dest="harmonize", action="store_false")
    p.add_argument("--harmonize-max", type=int, default=200)
    p.add_argument("--t2w-norm", choices=["nyul", "percentile"], default="nyul",
                   help="T2w INPUT normalization: 'nyul' (fit on train, doesn't generalize to "
                        "held-out sites) or 'percentile' (per-image, scanner-invariant). Use "
                        "'percentile' to stop T2w from misleading the model on the held-out hospital")
    p.add_argument("--dce-norm", choices=["percentile", "robust"], default="percentile",
                   help="DCE target normalization: 'percentile' (per-image, bladder-sensitive) "
                        "or 'robust' (body-tissue median+/-spread, bladder-insensitive -> aligns "
                        "the soft-tissue baseline across hospitals). Inputs unaffected")
    p.add_argument("--dce-robust-k", type=float, default=2.0,
                   help="robust DCE half-width in body-spread units (larger -> prostate maps "
                        "lower / less saturation). Tune on the per-hospital alignment probe")
    # silver layout: <data-root>/<image-subdir|mask-subdir>/<center>/<subject>/
    p.add_argument("--image-subdir", default="Image_volumes")
    p.add_argument("--mask-subdir", default="Prostate_masks")
    p.add_argument("--test-hospitals", nargs="*", default=list(TIER1_TEST_HOSPITALS),
                   help="held-out test center(s); default jiulong (silver-available)")
    p.add_argument("--dce-phase", default="early",
                   help="multi-phase (zhongyiyuan) target phase: 'early' (ph1, default — "
                        "matches the single-phase centers' early-phase DCE target), "
                        "'peak' (mask-mean argmax), or an int index. Single-phase centers unaffected.")
    # Bao_newbatch data-scale cohort (extra TRAINING data; held-out test unchanged)
    p.add_argument("--newbatch-root", default="",
                   help="path to the Bao_newbatch_2312_2512/registered tree (flat <case>/ dirs, "
                        "multi-phase DCE). Added as extra training data only; empty = off")
    p.add_argument("--newbatch-target-time", type=float, default=30.0,
                   help="target acquisition time (s post-baseline) for the newbatch DCE phase; the "
                        "phase with the closest valid rel_time_s is used (early wash-in ~30s), keeping "
                        "the target phase-consistent with the single-phase centers across vendors")
    p.add_argument("--newbatch-require-dwi", action="store_true", default=False,
                   help="keep only newbatch cases that have DWI (the full-3-modality subset, ~219). "
                        "Default off = use all DCE cases (~517), filling missing DWI with background "
                        "(pair with --modality-dropout for a principled missing-DWI signal)")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of train held out (random) for best-checkpoint "
                        "selection; 0 disables best-ckpt tracking (eval falls back to last)")
    p.add_argument("--select-metric", choices=["ssim_roi", "roi_pearson", "realism", "balanced"],
                   default="ssim_roi",
                   help="best-checkpoint selection metric. 'ssim_roi' (legacy, SMOOTHNESS-biased -> "
                        "rewards the blob); 'roi_pearson' (faithfulness/localization); 'realism' "
                        "(label-free texture+detail+intensity realism proxy); 'balanced' (realistic "
                        "AND faithful -- the clinical-realism objective). Use balanced for the "
                        "radiologist-reading goal")
    p.add_argument("--val-every", type=int, default=0,
                   help="epochs between val-score checkpoint selections; 0 = use --ckpt-every. "
                        "Set small (e.g. 2) for the GAN, whose ROI fidelity peaks early")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0, help="cap #cases (smoke testing)")
    # model size
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--n-upsamples", type=int, default=4, help="GAN up/down sampling depth")
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--first-stage", choices=["vae", "wavelet", "medvae"], default="vae",
                   help="LDM first stage: 'vae' (learned AutoencoderKL3D, default); 'wavelet' "
                        "(fixed invertible 3D Haar transform, FlowLet-style, lossless so the "
                        "--anchor-weight recon is exact); 'medvae' (frozen MedVAE foundation VAE "
                        "pretrained on ~1.6M medical images -- a data-rich alternative to our "
                        "from-scratch VAE, aimed at realism). latent_channels is set by the stage")
    p.add_argument("--medvae-model", default="medvae_4_1_3d",
                   help="MedVAE variant for --first-stage medvae: medvae_4_1_3d (4x, 1ch, 3D; "
                        "matches our latent grid) or medvae_8_1_3d (8x, more compression)")
    p.add_argument("--medvae-modality", default="mri",
                   help="MedVAE modality preset (mri|ct); DCE is MRI")
    p.add_argument("--wavelet-levels", type=int, default=1,
                   help="Haar packet levels for --first-stage wavelet: 1 -> 8ch at D/2 (FlowLet "
                        "default), 2 -> 64ch at D/4 (8x smaller feature maps, cheaper 3D). "
                        "D,H,W must each be divisible by 2**levels")
    p.add_argument("--wavelet-loss", choices=["uniform", "energy"], default="energy",
                   help="wavelet-flow velocity-loss channel weighting. Subbands are standardized "
                        "to unit variance, so 'uniform' MSE spends equal budget on near-flat HF "
                        "subbands (noise) as on the structural LF band. 'energy' (default) reweights "
                        "channels by natural subband variance (image-space L2), so the structural "
                        "band dominates. 'uniform' reproduces the pre-fix behavior")
    p.add_argument("--wavelet-loss-gamma", type=float, default=1.0,
                   help="exponent on subband variance for --wavelet-loss energy: 1.0 = variance-"
                        "proportional (true image-L2), 0.5 = std-proportional (gentler), 0 = uniform. "
                        "Lower keeps more gradient on detail bands (more texture, less LF emphasis)")
    p.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 4], help="VAE")
    p.add_argument("--unet-ch-mults", type=int, nargs="+", default=[1, 2, 4], help="diffusion UNet")
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--vae-adv-weight", type=float, default=0.0,
                   help="adversarial (patch-discriminator) weight in VAE training; 0=off. "
                        "Sharpens reconstructions the L1/SSIM/perceptual loss blurs (standard "
                        "LDM autoencoder recipe). Try 0.1-0.5 to reduce LDM blur")
    p.add_argument("--vae-adv-warmup", type=int, default=5,
                   help="epochs of pure reconstruction before the VAE adversarial term kicks in")
    p.add_argument("--anchor-weight", type=float, default=0.0,
                   help="flow trajectory-anchoring weight (FlowMI-style): decode the predicted "
                        "clean latent and add an image-space ROI recon loss; 0=off. Gives the "
                        "flow LDM the direct prostate supervision the GAN has. Try 0.5-2.0")
    p.add_argument("--pz-weight", type=float, default=1.0,
                   help="zone-aware loss: extra multiplier on peripheral-zone (PZ) voxels where "
                        "DCE is clinically read (1.0=off; try 2.0-3.0). Needs prostate_zones masks")
    p.add_argument("--tz-weight", type=float, default=1.0,
                   help="zone-aware loss: multiplier on transition-zone (TZ) voxels (1.0=off)")
    p.add_argument("--cond-dim", type=int, default=0,
                   help="cross-attention conditioning: encode bpMRI cond to this-many token "
                        "channels that the UNet attention blocks attend to (0=off, concat-only; "
                        "try 64/128). Stronger conditioning aimed at the localization ceiling")
    p.add_argument("--flow-source", choices=["noise", "t2w"], default="noise",
                   help="flow ODE start: 'noise' (standard) or 't2w' (image-to-image: start "
                        "from the encoded T2w so the model learns only the enhancement residual)")
    p.add_argument("--vae-joint", action="store_true", default=False,
                   help="train the VAE on DCE + T2w jointly so encode(T2w) is well-defined "
                        "(the sound first stage for --flow-source t2w)")
    p.add_argument("--flow-adv-weight", type=float, default=0.0,
                   help="flow adversarial head: conditional patch-disc on the decoded "
                        "one-shot prediction (0=off; try 0.1). Forces faithful/localized "
                        "texture vs the pure-velocity flow's hallucinated texture")
    p.add_argument("--flow-feat-weight", type=float, default=1.0,
                   help="feature-matching weight for the flow adversarial head (stabilizes)")
    p.add_argument("--flow-adv-warmup", type=int, default=10,
                   help="epochs of pure flow before the adversarial term starts")
    p.add_argument("--flow-adv-t", type=float, default=0.5,
                   help="noise level t at which the adversarial fake (decoded x0_hat) is drawn")
    p.add_argument("--anchor-t-max", type=float, default=1.0,
                   help="only apply the anchor for t < this (1.0 = all t). The clean-latent "
                        "estimate is noisy at high t; 0.5 restricts it to the reliable low-noise regime")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--x0-clamp", type=float, default=3.0,
                   help="DDIM x0-estimate bound (latents are ~[-2.4,1.4]; 0 disables). "
                        "Loose/0 lets the latent drift to garbage at high-noise steps")
    p.add_argument("--beta-schedule", choices=["cosine", "linear"], default="linear",
                   help="DDPM noise schedule; linear avoids cosine's explosive clamped-tail betas")
    p.add_argument("--cfg-dropout", type=float, default=0.0,
                   help="classifier-free guidance: prob of dropping the condition during "
                        "LDM training (0 = off; 0.1 typical). Train with this to enable guidance")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="CFG sampling scale w: pred = uncond + w*(cond-uncond). 1.0 = no "
                        "guidance; >1 sharpens conditioning (needs a cfg-dropout-trained model)")
    # Layer-1 missing-sequence robustness (LDM only)
    p.add_argument("--modality-dropout", action="store_true", default=False,
                   help="randomly drop input sequences during LDM training and append a "
                        "per-modality availability mask (cond_channels 3->6), so one model "
                        "handles arbitrary missing inputs. Eval subset set by --eval-modalities")
    p.add_argument("--modality-full-prob", type=float, default=0.3,
                   help="prob a training sample keeps all 3 sequences (else a random subset, "
                        ">=1 kept); preserves full-modality quality while teaching robustness")
    p.add_argument("--eval-modalities", default="111",
                   help="which input sequences are present at eval, as T2w,DWI,ADC bits "
                        "(111=all, 100=T2w only, 101=T2w+ADC, ...). Traces the degradation curve")
    p.add_argument("--latent-center", action="store_true", default=False,
                   help="center the VAE latent (zero-mean) before diffusion so a DDPM's "
                        "N(0,1) prior matches; recommended for ldm_ddpm")
    p.add_argument("--vae-ckpt", default="")
    # loss
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--ssim", type=float, default=1.0)
    p.add_argument("--perceptual", type=float, default=0.1)
    p.add_argument("--perceptual-depth", type=int, default=18)
    p.add_argument("--perceptual-shortcut", default="A")
    p.add_argument("--medicalnet-weights", default="")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--roi-weight", type=float, default=10.0,
                   help="how much more ROI voxels count in the recon loss (1.0 = off; "
                        "needs prostate masks in the data)")
    p.add_argument("--radio-weight", type=float, default=0.0,
                   help="ClinDCE regional-radiomics loss weight (0=off; try 0.1). Preserves "
                        "local enhancement feature maps inside the prostate; needs masks")
    p.add_argument("--focal-weight", type=float, default=0.0,
                   help="ClinDCE focal-enhancement loss weight (0=off; try 0.1). Matches "
                        "focal (PI-RADS-positive) regions + penalizes over-enhancement")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the generator/UNet weights (0=off). EMA weights are "
                        "used for best-ckpt scoring, eval, and the saved checkpoints")
    # evaluation
    p.add_argument("--synth-out", default="",
                   help="generate_synth.py: dir to write synth_DCE.nii.gz per case "
                        "(<synth-out>/<center>/<subject>/synth_DCE.nii.gz)")
    # reader_study_export.py (blinded real-vs-synth DCE panels for a radiologist Turing test)
    p.add_argument("--reader-out", default="",
                   help="reader-study output dir; per-model export writes <reader-out>/staging/")
    p.add_argument("--reader-cases", type=int, default=40,
                   help="number of cases to export for the reader study")
    p.add_argument("--reader-split", choices=["val", "test", "train"], default="val",
                   help="which split to draw reader cases from (val = in-distribution, has real "
                        "enhancement; jiulong test is near-flat and poor for a realism judgement)")
    p.add_argument("--reader-with-context", action="store_true", default=False,
                   help="render the real T2w beside each DCE panel for anatomical orientation "
                        "(T2w is a real input either way, so it doesn't leak real-vs-synth)")
    p.add_argument("--reader-include-real", action="store_true", default=False,
                   help="also export the REAL DCE panels (run once, alongside one model's export)")
    p.add_argument("--eval-only", action="store_true", default=False,
                   help="skip training: load the model from --output-dir/checkpoints and "
                        "just run evaluation (reuses the saved harmonizer.json)")
    p.add_argument("--compute-fid", action="store_true", default=True)
    p.add_argument("--no-fid", dest="compute_fid", action="store_false")
    p.add_argument("--fid-slices", type=int, default=8,
                   help="axial slices per volume for FID (2D Inception)")
    p.add_argument("--fid-batch-size", type=int, default=32)
    args = p.parse_args()
    if not args.medicalnet_weights:
        args.medicalnet_weights = None
    return args


if __name__ == "__main__":
    main()
