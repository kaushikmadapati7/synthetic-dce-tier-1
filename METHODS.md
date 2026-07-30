# Methods — Tier-1 Static Synthetic DCE

Working methods document for the Tier-1 static task: **predict a single post-contrast
DCE volume from three non-contrast modalities (T2w, DWI, ADC)**. Everything is
conditional; inputs are stacked as a 3-channel volume and the model emits a 1-channel
DCE volume in `[-1, 1]`.

This file records (a) the UCSF cohort and how it is constructed, and (b) the pipeline
defects found and corrected during the 2026-07 audit, with an explicit note on which
corrections are reflected in the current results and which are still outstanding.
Status as of **2026-07-30**.

---

## 1. Data

### 1.1 Source

Single-center UCSF prostate MRI, delivered on two secured mounts:

| mount | contents |
|---|---|
| `CX000018_DS2/.../UCSF_data/registered/<pid>/` | `T2W`, `ADC_to_T2W`, `DWI_b*_to_T2W`, `prostate_mask`, `prostate_zones` — 5962 patients |
| `CX000018_DS3/.../UCSF_data/registered/<pid>/DCE/` | `DCE_4D_to_T2W.nii.gz` (4D, phases stacked) + `dce_times.json`; also `pregad_4D_to_T2W.nii.gz` + `pregad_times.json` |

All volumes are pre-registered into T2w space by the data provider. `dce_times.json`
gives per-phase acquisition times (`{"case", "n_phases", "time_source", "phases":
[{"idx", "rel_time_s", ...}]}`), with `idx 0` the pre-contrast phase at `t = 0`.

The pre-contrast T1 (`pregad`) is present on disk and is **not** currently used; it is
available as a 4th conditioning channel or for an enhancement-residual target
(`--use-pregad`).

### 1.2 Target-phase selection: by time, not by intensity

Acquisition protocols are heterogeneous — `n_phases` observed at 26/35/44/70 with
cadence 1.7–13 s — so a fixed **phase index** does not correspond to a fixed
post-injection time across patients. We therefore select the target phase by
**elapsed time**.

The enhancement curve was measured directly (mask-mean signal vs. the pre-contrast
phase): flat at ~1.0× until ~45 s, steep wash-in from 55–90 s (1.12× → 2.00×), then a
**plateau of 2.1–2.5× from ~90 s out past 300 s**. There is no sharp peak. Intensity
argmax therefore lands at ~280 s on plateau noise and drifts into washout, which is why
"peak contrast" is the wrong selection criterion for this cohort.

`--dce-target-time 120` (default) sits inside the stable plateau — past the
timing-sensitive wash-in, before washout — and recovers ~92% of maximum enhancement.
Realized selection times cluster tightly at 104–125 s.

Two protocol complications are handled explicitly:

- **Interleaved series.** Some studies stack *k* volumes per timepoint (duplicate
  timestamps, e.g. Dixon water/fat), so a plain index alternates between sub-series and
  can land on the non-enhancing one. Detected via repeated `rel_time_s`; the phase is
  re-picked within the timepoint, keeping the sub-series that actually enhances relative
  to its own `t = 0`. 27 cases affected.
- **Corrupt timing.** Some `dce_times.json` carry corrupt trailing values (e.g.
  67989 s). Only the clean monotonic prefix is trusted; when the time-selected phase
  still looks weak (< 1.5×), the phase is re-selected from the enhancement curve at
  **plateau onset** (first phase ≥ 90% of the curve maximum). 155 cases rescued.

### 1.3 Staging

Each 4D DCE series is ~1 GB. Training directly against them re-reads the full 4D volume
per sample per epoch (I/O-bound) and depends on a mount that is intermittently absent on
compute nodes. `data/stage_ucsf.py` reads each 4D **once**, writes the time-selected
phase as a 3D `DCE_to_T2W.nii.gz`, and co-locates T2w/ADC/DWI/mask/zones into a single
flat directory per patient with a `stage_meta.json` sidecar (selected index, realized
time, enhancement ratios, DWI source filename, interleave factor).

Staging is resumable and shards across a SLURM array (`--shard/--num-shards`). The
staged target is verified **bit-identical** to the raw-4D path.

### 1.4 Cohort construction

Quality filters are applied at two points. The enhancement filter runs **at staging** so
the tree is correct by construction; the remaining two are load-time flags.

| stage | criterion | remaining |
|---|---|---|
| patients with DCE present | — | 5439 |
| successfully staged | inputs resolvable, mask matches DCE grid | 4975 |
| `MIN_ENH=1.5` **at staging** | max enhancement over the whole series < 1.5× | **4246** |
| `--dwi-min-bvalue 600` | DWI resolves to b < 600 | 3684 |
| `--require-qc` | no enhancement QC recorded | **3667** |

- **Enhancement QC** filters on the **maximum over the whole series**, not the selected
  phase — only the max distinguishes "never enhanced" (drop) from "we picked the wrong
  phase" (recoverable by re-selection). ~12–13% of patients never enhance at any
  timepoint; this is a genuine cohort property and belongs in the paper's data section.
  17 further cases have no QC value at all (enhancement is only measurable when the mask
  matches the DCE grid and is non-empty), so they were staged unverified and are dropped.
- **DWI b-value.** UCSF DWI spans b50–b1400: b1000 (3485), b50 (562), b1400 (135),
  b1000_synth (29), b1400_synth (13), b0800 (7), b0600 (7), b1350 (5), b600 (2),
  b1200_synth (1). The 562 b50 acquisitions carry almost no diffusion weighting and read
  as anatomical/T2-like — a **different contrast** in the same input channel as the b1000
  the majority receive. `--dwi-min-bvalue 600` enforces channel consistency at the cost
  of 13% of the data; this is a clean one-flag ablation worth reporting. Note also that
  **43 cases use computed (`_synth_`) high-b images rather than acquired ones**.

### 1.5 Intensity normalization

Single center, so there is no cross-scanner distribution to align and Nyul landmark
harmonization is not used; normalization is per-image.

- **DCE** → `--dce-norm robust --dce-robust-k 1` (both must be passed explicitly; the
  CLI defaults are `percentile` and `k=2.0`). The robust scheme keys to the body-tissue
  median ± k·spread. Measured within-prostate target std over 12 staged cases: k=3 →
  0.073, k=2 → 0.109, **k=1 → 0.218**. Bao's k=3 was tuned by a *cross-hospital alignment*
  probe that does not apply to a single-center cohort, and it leaves the ROI target nearly
  flat — the same degeneracy that made held-out metrics meaningless previously, since
  `roi_pearson`/`ssim_roi` become noise on a near-uniform target. Cost of k=1: the
  contrast-filled bladder saturates at +1, irrelevant for a prostate task.
- **T2w** → percentile normalization. **ADC** → fixed physical clip (preserves absolute
  mm²/s). **DWI** → per-image foreground z-score.

### 1.6 Split

Single center, so the split is **patient-level** (`--ucsf-test-frac`, default 0.15) with
a held-out validation split for checkpoint selection. Splits are seeded. ID prefixes were
checked as a possible acquisition-batch marker and rejected — `8ii*` spans 512, 704, 736
and 1024 matrices alike, so the prefix carries no protocol information and no
stratification on it is warranted.

### 1.7 Geometry — OPEN ISSUE

`--spatial-size` currently specifies a **pixel count**, and is applied as a center
crop/pad on each case's native grid (`--reference dce`). UCSF in-plane resolution is
**not** uniform: matrices of 256/512/672/704/736/1024 at 0.176–0.781 mm. A fixed 256-pixel
crop therefore covers a variable physical field of view (full cohort, n = 4246):

| 256-px crop covers | cases | share |
|---|---|---|
| < 80 mm | 734 | 17% |
| 80–110 mm (intended framing) | 3346 | 79% |
| > 110 mm | 164 | 4% |

An **8× spread in anatomical scale** (40–320 mm). At the low end, 172 cases crop to
45 mm — *smaller than a prostate* — so the crop sits entirely inside the gland: the
prostate mask fills the frame, there is no surrounding anatomy, and ROI metrics
degenerate to whole-image metrics for those cases.

The through-plane axis carries the same defect at smaller scale: slice thickness is
3.00 mm for 4151/4246 cases (97.8%), but 44 cases are acquired at 0.70–0.80 mm, so a
32-slice crop covers a **25.6 mm slab** for them versus 96 mm for everyone else.

The fix is to resample to a common physical grid before cropping
(`--reference iso --iso-spacing 0.35 0.35 3.0`), which makes a 32×256×256 crop cover
90 × 90 × 96 mm for **every** patient. Validated against the cohort:

- the dominant 512 @ 0.352 mm in-plane group and the 97.8% at 3.00 mm slice thickness are
  left effectively unresampled, so the change is a no-op for the bulk of the data and
  acts only on the outliers it is meant to correct;
- the iso grid anchors on T2w, and **T2w and DCE share an identical grid in all 4246
  cases**, so the resample introduces no crop-center shift relative to the current
  `--reference dce` behaviour.

`--iso-spacing` was added to the CLI on 2026-07-30 (it did not previously exist, so
`--reference iso` would have silently used a 1 mm isotropic grid — a 256 mm field of view
with 3–4× z-upsampling).

**This is not yet reflected in any trained model.** See §6.

---

## 2. Preprocessing

All modalities are resampled onto one reference grid so channels are voxel-aligned and
stackable (linear interpolation for intensities, nearest-neighbour for mask/zones), then
intensity-normalized to `[-1, 1]` and center-cropped/padded to `--spatial-size`
(order `D H W`; SimpleITK image spacing is `(x, y, z)`).

Each sample is `{"cond": (3,D,H,W), "target": (1,D,H,W), "mask": (1,D,H,W), "id": str}`.
The 2D track wraps the 3D dataset and emits axial slices.

## 3. Models

All model families expose the same contract: the trainer returns `(model, gen)` where
`gen(cond) -> DCE prediction in [-1, 1]`, so evaluation is model-agnostic.

**3D.** Conditional GAN (pix2pix-style U-Net generator with full skips, hinge loss +
reconstruction, projection discriminator); latent DDPM (ε-prediction, ancestral + DDIM
samplers); latent flow matching. Both LDMs share a frozen first stage — either a 3D
`AutoencoderKL3D` trained on target DCE volumes, or the frozen **MedVAE** foundation VAE
(`--first-stage medvae`) — with the conditioning volume downsampled to the latent grid.

**2D.** `gan2d` (pix2pix), `flow2d` (pixel-space CFM), and MedVAE-latent flow.

**Flow-matching details.** Linear/rectified interpolant with `t = 0 → data` and
`t = 1 → source` (reversed relative to much of the literature):

- path: `z_t = (1 − (1 − σ_min)·t)·z0 + t·noise`
- target velocity: `v = noise − (1 − σ_min)·z0`, with `σ_min = 0` by default

`t ~ U(0,1)` is sampled per-example, so the network learns the velocity field across the
whole interpolation, not just the endpoints. `--flow-source` defaults to **noise**;
`t2w` is available but has not been used in any run to date. Solvers: 3D uses Heun
(2nd-order) by default with Euler available; **2D is Euler-only**.

An optional FlowMI-style **image-space anchor** (`--anchor-weight`) decodes the predicted
clean latent `z0_hat = z_t − t·v` — an identity that holds exactly for any `σ_min` — and
supervises it with the image-space criterion. `--anchor-t-max` restricts this to
low-noise steps, where `z0_hat` is sharp, to control cost.

## 4. Loss

`CustomLoss = l1_weight·L1 + ssim_weight·(1 − SSIM3D) + perceptual_weight·Perceptual3D`,
reused as both the GAN generator's reconstruction term and the VAE's.

Because the prostate is ~1% of the volume, an unweighted loss is dominated by background
and padding. With a mask and `--roi-weight > 1` (default 10), ROI voxels count
`roi_weight`× more in L1 and an additional `(1 − SSIM_roi)` term is added; zone weights
(`--tz-weight`, `--pz-weight`) further emphasize transition/peripheral zones. With no
mask or `roi_weight ≤ 1` the loss is byte-for-byte the unweighted original.

The perceptual term uses a frozen 3D ResNet (MedicalNet/Med3D).

**Note on the LDMs:** their diffusion objective is latent-space MSE with no clean ROI
mapping, so ROI emphasis reaches them only through the VAE reconstruction stage (or
through the image-space anchor, when enabled).

## 5. Evaluation

Per-volume **SSIM / PSNR / MAE** plus ROI counterparts **SSIM_roi / PSNR_roi / MAE_roi**
(computed only where a non-empty mask exists). The global-vs-ROI gap is the intended
signal: global metrics are ~99% background, so a large gap means the gland is poorly
reconstructed behind good-looking global numbers.

Beyond fidelity, two families:

- **Faithfulness** — `roi_pearson` (within-gland voxelwise co-localization, the hard
  metric and current ceiling) and `p75_corr` (cross-patient enhancement tracking).
- **Realism** — `roi_var_ratio` (heterogeneity, 1 = match), `roi_grad_ratio`
  (spatial-frequency/detail, 1 = match), `roi_w1` (ROI intensity-histogram distance), and
  FID over 2D axial slices. `realism_score` combines the first three; FID is excluded as
  too unstable at these sample sizes.

`selection_score` supports checkpoint selection on `ssim_roi`, `roi_pearson`, or a
`balanced` objective — realistic **and** faithful, since optimizing pure realism rewards
hallucination.

---

## 6. Pipeline corrections

Systematic audit, 2026-07. Grouped by what each defect corrupts. Every one was **silent**
— no crash, no warning, plausible-looking output.

### 6.1 Data-construction defects

| # | defect | effect | status |
|---|---|---|---|
| D1 | DWI resolved against a hardcoded stem list | ~1000 patients scanned at b400/b800/b1500/b2000 dropped as "missing inputs"; now globbed with nearest-b preference | fixed, in current cohort |
| D2 | Corrupt timestamps after phase 0 → `select_phase_by_time` returned idx 0 | shipped the **pre-contrast volume as the prediction target** for 155 cases — i.e. trained to predict a non-enhanced image | fixed (curve re-selection) |
| D3 | `_interleave_factor` divided by unique timestamps | null timings looked like k = n_phases, masking D2 | fixed (requires complete times, capped at k=4) |
| D4 | `--overwrite` re-stage without `MIN_ENH` | silently **re-created the 729 pruned non-enhancers**; a run trained on 4975 before it was noticed | fixed by filtering at staging |
| D5 | `_dwi_bval` read the staged (renamed) DWI filename | staging renames to canonical `DWI_to_T2W.nii.gz`, so every case scored b = −1 < 600 and **every case was dropped — `num_samples = 0`** | fixed (reads `dwi_src` from `stage_meta.json`; empty split now raises) |
| D6 | `--report` wrote to `stage_summary.json` | clobbered the staging run's skip records | fixed (`stage_report.json`) |

D2 and D4 are the consequential ones: both put wrong *targets* into training.

### 6.2 Training defects

| # | defect | effect | status |
|---|---|---|---|
| T1 | GAN conditioning pooled across the batch | the generator's condition was contaminated by other patients in the batch | fixed; **3D GAN required retraining** |
| T2 | Sampling unseeded in GAN, 2D flow, 3D flow and DDPM | validation integrated from different noise each epoch, so the val curve mixed model improvement with sampling variance — checkpoint selection was partly selecting on noise | fixed (seeded samplers) |

### 6.3 Measurement defects

These corrupt reported numbers but not weights — affected runs can be re-scored from
checkpoints rather than retrained.

| # | defect | effect | status |
|---|---|---|---|
| M1 | ROI scalars pooled over the batch before reduction (`roi_radiomics`, `zone_metrics`, `eval_metrics`) | catastrophic inflation: on pure noise this reports **`roi_pearson` +0.977 where the truth is +0.020**, because cross-patient brightness differences masquerade as within-gland correlation | fixed — all ROI scalars computed per sample, then averaged |
| M2 | 2D ROI mask rank mismatch | broadcast to a `(B, B, …)` outer product, silently mixing patients' masks | fixed; rank mismatches now raise |
| M3 | `ssim3d` Gaussian window larger than the depth axis | wrong window on thin volumes; depth-1 now yields exact 2D SSIM | fixed (per-axis window, `min(dim, window)` forced odd) |

M1 is the most important single finding in the audit: it is the reason earlier
`roi_pearson` figures looked encouraging, and it invalidated the two conclusions
previously drawn from them — that the **data-scale hypothesis was rejected** and that the
**~0.5 localization ceiling was information-bound**. Both were measured on crippled
configurations and are retracted; neither has been re-tested.

### 6.4 Infrastructure

`--mem=32G` OOM-killed on 70-phase series (→ 96G); `/mnt/fac` is absent (ENODEV) on some
CPU nodes (→ preflight mount check, exit 17, exclude the bad node); `train2d.slurm` was
missing the `HF_HOME`/`HF_HUB_OFFLINE`/`TORCH_HOME` exports that `train.slurm` has, which
killed the MedVAE run on a HuggingFace fetch.

### 6.5 Regression guard

There is no unit-test suite. `tier1_static/selfcheck.py` provides 26 preflight assertions
across metrics/loss, model/determinism and data/staging that run on CPU in seconds; it is
the intended gate before launching cluster runs. Two of the bugs above (D5, and an earlier
`enh_max` case) initially escaped because the **fixture did not mirror the real staged
tree** — fixtures must be built from staged filenames, not idealized ones.

---

## 7. Validity of current results

- The 2D GAN vs. 2D pixel-flow comparison was run after §6.1–6.3 were fixed, so it is
  free of the pooling, masking, conditioning and seeding defects. VAL ≈ TEST for both.
- **All current runs still train on the §1.7 FOV mixture.** They see an 8× range of
  anatomical scale, and for the 17% of cases cropped below 80 mm the ROI metrics are
  closer to whole-image metrics. The GAN-vs-flow *ranking* is still informative — both
  models saw the same mixture — but the absolute numbers are not a clean baseline and
  should not be quoted as one.
- Every result predating the audit is superseded.

## 8. Reproducing the current cohort

```bash
# stage once, with the QC filter inline (yields 4246)
MIN_ENH=1.5 sbatch --array=0-3 --exclude=gcpu2-17 tier1_static/scripts/stage_ucsf.slurm

# verify: expect 4246, median enh ~2.07, p10 1.70, no t=0 warning,
#         27 interleaved, 155 curve-reselected
python -m tier1_static.data.stage_ucsf --report
python -m tier1_static.selfcheck        # 26 assertions

# train (logs must show `min_b=600` and `562 below b600, 17 without enhancement QC`)
python -m tier1_static.main2d --model gan \
    --ucsf-main-root <staged> --dwi-min-bvalue 600 --require-qc \
    --dce-target-time 120 --dce-norm robust --dce-robust-k 1 \
    --spatial-size 32 256 256 --roi-weight 10
```

Add `--reference iso --iso-spacing 0.35 0.35 3.0` to address §1.7. Validated against the
cohort (no-op for the 97.8% at 3.00 mm and for the dominant in-plane group; no
crop-center shift, since T2w and DCE grids are identical in all 4246 cases) but **not yet
used by any trained model** — the next round of runs should adopt it, and it is the
natural candidate to become the default for UCSF.
