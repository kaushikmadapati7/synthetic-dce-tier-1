# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tier1_static` is the **Tier-1 static synthetic DCE generation** pipeline for a UCSF prostate-MRI project. The task: **predict a single peak-contrast DCE (dynamic contrast-enhanced) volume from three input modalities — T2w, DWI, ADC**. It is one package in a larger project tree (siblings `tier2_dynamic`, `mama_synth`, `exploration`, etc.); only `tier1_static` is documented here.

Everything is 3D and conditional: inputs are stacked as a 3-channel volume `(T2w, DWI, ADC)` and the model outputs a 1-channel DCE volume in `[-1, 1]` (tanh range).

## Running

This is a Python package run as a module **from the project root** (the parent dir containing `tier1_static/`, `venv/`, `pretrain/`):

```bash
python -m tier1_static.main --model ldm_flow --data-root /path/to/Bao_DCE \
    --output-dir runs/exp1 --epochs 100
```

- Models: `--model {ldm_flow, ldm_ddpm, gan}` (default `ldm_flow`).
- `--limit N` caps the number of cases — use it for fast smoke tests on CPU/laptop.
- `--no-harmonize` disables Nyul harmonization (on by default); `--no-fid` skips FID.
- Perceptual loss needs MedicalNet weights. If `--medicalnet-weights` is omitted, `main.resolve_medicalnet_weights` auto-defaults to the depth-matched file `pretrain/resnet_{perceptual_depth}.pth` (e.g. `pretrain/resnet_18.pth` for the default depth 18) when it exists. If no real weights can be found and `--perceptual > 0`, it logs a **loud warning** and the perceptual term runs on a randomly-initialized backbone (set `--perceptual 0` to disable it outright). The resolved path is recorded in `config.json`.
- Outputs land in `--output-dir`: `config.json`, `harmonizer.json`, `train.log`, `checkpoints/`, `metrics.json`, `samples/{*.nii.gz, montage.png}`.

Cluster runs use SLURM: `sbatch tier1_static/scripts/train.slurm`, overriding via env vars (`MODEL=gan EPOCHS=200 DATA_ROOT=... sbatch ...`). The slurm script expects a `venv/` at the project root.

There is **no test suite, linter, or build step** — validation is done by running `main.py` with `--limit` for a quick end-to-end smoke test. A CPU smoke run that exercises data → harmonizer fit → train → ROI loss/metrics → eval (uses the bundled `Bao_intern_package/processed_registered_samples`, itself a silver tree):

```bash
python -m tier1_static.main --model gan \
    --data-root Bao_intern_package/processed_registered_samples \
    --output-dir runs/smoke --limit 6 --epochs 1 --spatial-size 16 96 96 \
    --base-ch 8 --batch-size 2 --num-workers 0 --no-fid --perceptual 0 --device cpu
```

Expect `train cases: 6  test cases: 1` (jiulong held out) and ROI metrics (`mae_roi/psnr_roi/ssim_roi`) in the final line — their presence confirms masks resolved from the `Prostate_masks/` tree.

## Architecture

The pipeline is a strict chain: **data → harmonization → model → train → eval**, orchestrated entirely by `main.py`. Understanding it requires reading across the `data/`, `models/`, `training/`, and `loss/` packages because the contracts between them are implicit.

### Data layer (`data/`)
The loaders target the Bao_DCE **silver** layer on CHPC (`DATA_ROOT=.../Bao_DCE`), which splits images and masks into two parallel trees keyed by `<center>/<subject>`:

    <root>/Image_volumes/<center>/<subject>/   registered volumes (T2 reference space)
    <root>/Prostate_masks/<center>/<subject>/  prostate_mask.nii.gz (+ prostate_zones)

(Sub-tree names are overridable via `--image-subdir`/`--mask-subdir`.) The `<root>/00_raw_from_bao/exams/<center>/exam_*` **bronze** layer is immutable, unregistered, has 4D multi-phase `DCE.nii.gz`, and carries **no masks** — the loaders do *not* read it.

Only **four** centers exist in silver: `changshu, fuyiyuan, jiulong` (single-phase) + `zhongyiyuan` (multi-phase). `taizhou`/`zhangjiagang` are bronze-only, so the lead's intended held-out test (taizhou+zhangjiagang) is **not** runnable against silver. Two `Dataset` classes (`dataset.py`):
- `CanonicalDCEDataset` — single-phase centers; files resolve via `MODALITY_STEMS` (`T2WI`, `ADC_to_T2WI`, `DWI_to_T2WI`, `DCE_to_T2WI`, with `_to_T2W`/bare fallbacks). DCE is single-phase → used directly as target.
- `DescriptorDCEDataset` — `zhongyiyuan` (GE LAVA-Flex), multi-phase DCE (`DCE_pre` + `ph1..4`). Inputs resolve via `MODALITY_STEMS` (`T2W`, `ADC_to_T2W`, …); dynamic phases are globbed by `phase_glob` (`DCE_ph*_to_T2W*`, pre-contrast excluded). The target phase is set by `--dce-phase`: `early` (default, ph1), `peak` (mask-mean argmax via `peak_phase_index`), or an int index.

  **Target-phase consistency:** the three single-phase centers provide one *early-phase* DCE (≈ph1/ph2) as their target, so the default is `--dce-phase early` — it makes zhongyiyuan's target ph1, matching the single-phase centers. `--dce-phase peak` instead targets enhancement-peak (a phase mismatch across the pool, kept available for the early-vs-peak comparison). The canonical Tier-1 phase remains an open project decision.

`build_tier1_datasets(...)` merges both into a `ConcatDataset` and applies the **train/test split by hospital** — `TIER1_TEST_HOSPITALS = [jiulong]` is held out by default. Override at runtime with `--test-hospitals` (threaded through `build_data`); the module-level constant is the default. A split with no hospitals returns a 0-length `_EmptyDataset` so callers guard on `len()`.

Every sample is a dict: `{"cond": (3,D,H,W), "target": (1,D,H,W), "mask": (1,D,H,W), "id": str}`. Mask is zeros if unavailable.

`preprocessing.py` does all geometry: it resamples every modality onto **one reference grid** (configurable via `--reference {t2w,dce,iso}`, default `dce`) so volumes are voxel-aligned and channel-stackable, then normalizes intensities to `[-1,1]` and center-crops/pads to a fixed `--spatial-size` (default `32 192 192`, order `D H W`).

### Harmonization (`data/harmonization.py`)
Cross-scanner intensity harmonization, applied on raw resampled intensities **before** the final `[-1,1]` scaling, per modality:
- **T2w, DCE** → Nyul-Udupa landmark histogram standardization (must be **fit on the training split first**, then applied everywhere). `main.build_data` fits it on non-test canonical hospitals and saves `harmonizer.json`.
- **DWI** → per-image foreground z-score (no fitting).
- **ADC** → fixed physical clip only (preserves absolute mm²/s values; no remapping).

When a `Harmonizer` is supplied it **owns the intensity step** and the dataset skips its own percentile normalization (see `process_case`). ComBat is intentionally not implemented (needs the full multi-site population at once).

**DCE peak preservation (important for the prediction target).** The Nyul upper anchor `pc_high` doubles as a clip ceiling: `transform` uses `np.interp`, which clamps everything above the top landmark, so a low `pc_high` flattens the brightest DCE enhancement — clinically the signal most tied to lesion severity. To avoid this:
- `NyulConfig.pc_high = 99.9` (raised from 99.0). On the 20 sample exams this cut mean prostate-voxel clipping from 0.69% → 0.005% (worst case 7.5% → 0.10%) with negligible cost to standardization. Going to the literal max was rejected — a single hot vessel/artifact voxel would set the ceiling and is not robust across scanners.
- `NyulConfig.landmark_percentiles` adds `95` and `99`. Raising `pc_high` widened the 90→99.9 region into one coarse linear segment, yet ~13% of prostate voxels live there; the extra landmarks restore fine-grained cross-scanner alignment in the enhancement band.
- The `--no-harmonize` fallback (`preprocessing.normalize` / `PreprocessConfig.clip_percentiles`) upper bound was matched to `99.9` so both paths agree on the peak.
- These defaults only apply to a **freshly fit** harmonizer; a saved `harmonizer.json` keeps the values it was serialized with. The analysis scripts behind this live next to the exploration notebook: `data/measure_dce_clipping.py` and `data/sweep_dce_clip.py`.

### Models (`models/`) and the generator contract
Three model families, all dispatched through the `TRAINERS` registry (`training/__init__.py`):
- **`gan`** — single-stage 3D conditional GAN (`conditional_gan.py`), pix2pix-style, hinge loss + reconstruction loss. The condition volume is concatenated/fused; the discriminator is a projection discriminator.
- **`ldm_ddpm`** and **`ldm_flow`** — **two-stage latent diffusion**. Both share `AutoencoderKL3D` (3D VAE, `autoencoder3d.py`) as the frozen first stage and `UNet3D` (`unet3d.py`) as the denoiser. They differ only in the diffusion objective + sampler: DDPM predicts noise ε (`ldm_ddpm.py`, ancestral + DDIM samplers); flow matching predicts velocity on straight-line OT paths (`ldm_flow_matching.py`, Euler/Heun ODE solver). `common.py` holds shared `ResBlock3D`/`AttentionBlock3D`/`Downsample3D`/`Upsample3D`/`TimeEmbedding`.

**Key contract:** every trainer returns `(model, gen)` where `gen` is a callable `gen(cond) -> predicted DCE volume in [-1,1]`. `eval.py` only ever calls `gen`, so it is model-agnostic. When adding a model, register it in `TRAINERS` and have its trainer return this pair.

### Two-stage LDM training (`training/_ldm_base.py`)
Both LDMs share `train_ldm(...)`, parameterized by a `flow` flag. The flow is: (1) train (or load via `--vae-ckpt`) the VAE on target DCE volumes; (2) compute a **latent scaling factor** so latent std ≈ 1 (standard LDM practice, stored on `vae.scaling_factor`); (3) freeze the VAE; (4) train the U-Net in latent space, with the **conditioning volume downsampled to the latent grid** (`downsample_cond`). At sample time, condition is downsampled, latents are sampled, then decoded back to image space.

### Loss (`loss/loss.py`)
`CustomLoss` = `l1_weight·L1 + ssim_weight·(1−SSIM3D) + perceptual_weight·Perceptual3D`. The same criterion object is reused for the GAN generator's reconstruction term and the VAE's reconstruction term. SSIM is a custom 3D implementation (`ssim3d`, also reused by `metrics.py`; pass `return_map=True` for the per-voxel map). The perceptual term uses a frozen **3D ResNet (Tencent MedicalNet / Med3D)** — the architecture in this file matches the official repo so `.pth` weights load directly (`conv_seg.*` keys are the seg head and expected to be missing).

**ROI emphasis (prostate ≈ 1% of the volume).** Because an unweighted loss is dominated by background/padding, `CustomLoss.forward(pred, target, mask=None)` reweights toward the prostate when a mask is supplied and `--roi-weight > 1` (default **10**): the L1 term counts ROI voxels `roi_weight`× more, and an extra `(1−SSIM_roi)` term (SSIM map averaged over mask voxels) is added. With no mask, an empty mask, or `roi_weight ≤ 1` it is byte-for-byte the old unweighted loss. The mask is threaded in via `g_total_loss(..., mask=)` (GAN, `models/conditional_gan.py`) and `AutoencoderKL3D.loss(..., mask=)` (VAE). **The LDMs get ROI emphasis only through the VAE reconstruction stage** — their diffusion objective is latent-space MSE with no clean ROI mapping, so the VAE recon is where prostate image fidelity is set for `ldm_ddpm`/`ldm_flow`.

### Evaluation (`eval.py`, `metrics.py`)
`evaluate` runs `gen` over the test loader, reports per-volume **SSIM / PSNR / MAE** plus their ROI counterparts **MAE_roi / PSNR_roi / SSIM_roi** (computed only when a non-empty mask is present), optionally computes **FID** over 2D axial slices via `torch_fidelity` (Inception-v3), and saves the first case as NIfTI + a PNG montage. The global-vs-ROI gap is the intended signal: global metrics are ~99% background, so a large gap means the prostate is being reconstructed poorly despite good-looking global numbers. `aggregate` averages each key only over the batches that report it, so mixed mask availability is fine. FID/NIfTI/montage saves are wrapped in soft try/except — missing optional deps (`torch_fidelity`, `SimpleITK`, `matplotlib`) degrade gracefully.

**Masks are the prerequisite for both ROI loss and ROI metrics.** The loaders read `prostate_mask.nii.gz` from the parallel `Prostate_masks/<center>/<subject>/` tree (helper `_silver_mask`, with a `_find_mask` in-dir fallback). All four silver centers have masks, so `--roi-weight` fires and ROI metrics are reported. If a mask can't be found the dataset returns a zeros mask, in which case `--roi-weight` is a silent no-op and the ROI metrics are skipped for that case (`aggregate` averages each key only over cases that report it).

## Conventions and gotchas

- **Array axis order is `(D, H, W)`** throughout (SimpleITK's `GetArrayFromImage` convention), but SimpleITK *image* spacing/size are `(x, y, z)`. `--spatial-size` is given as `D H W`.
- All intensities live in **`[-1, 1]`** to match tanh outputs; SSIM/PSNR use `data_range=2.0` accordingly.
- For the **GAN**, `spatial_size` must be divisible by `2**n_upsamples` (`--n-upsamples`, default 4) or training raises.
- The harmonizer must be **fit before use** — it is fit only on training hospitals to avoid leakage.
- Dependencies are heavy and not pinned in a manifest here: `torch`, `SimpleITK`, `numpy`, optionally `torch_fidelity`, `matplotlib`. Use the project-root `venv/`.

## Reference data

`Bao_intern_package/` holds small sample data and the authoritative description of the cohort's file conventions (canonical vs. descriptor layouts, per-vendor DCE quirks for GE/Siemens/Philips/UIH). Read `Bao_intern_package/README.md` when working on the DICOM→NIfTI ingestion or phase-ordering logic. `data/data_exploration.ipynb` is the notebook whose peak-phase-selection logic `preprocessing.peak_phase_index` mirrors.
