# Synthetic DCE — Methods & Results Brief

Presentation companion. `METHODS.md` is the full technical record; this is what you
need in the room, plus the numbers you'll be asked to defend.

**Status 2026-08-06.** Cohort and pipeline are settled and audited. Single-phase (Tier-1)
results are in and trustworthy. The multi-timepoint (Tier-2) track is scoped and staged
for, not yet run.

---

## 1. The one-paragraph version

We predict a post-contrast DCE volume from three non-contrast sequences (T2w, DWI, ADC)
on 3,667 UCSF prostate exams, to avoid gadolinium in patients who can't receive it and to
harmonize DCE across protocols. A 3D conditional GAN reaches **roi_pearson 0.48** inside
the gland, against **−0.04** for the strongest trivial baseline (copying the T2w channel),
so the model is genuinely predicting enhancement structure rather than passing anatomy
through. The substantive finding is a decomposition: **within-gland structure is
predictable, per-patient enhancement *amplitude* is not** — and amplitude is 51% of the
signal. That is a property of the task, not a tuning failure, and it scopes what synthetic
DCE can be used for.

---

## 2. Data

**Source.** Single-center UCSF, two secured mounts: anatomy/masks on DS2 (5,962 patients),
4D DCE + per-phase timing on DS3. All volumes pre-registered into T2w space.

**Cohort construction** — quote these numbers, they will be asked for:

| stage | criterion | remaining |
|---|---|---|
| patients with DCE | — | 5,439 |
| successfully staged | inputs resolvable, mask matches DCE grid | 4,975 |
| enhancement QC at staging | max enhancement over series ≥ 1.5× | **4,246** |
| DWI b-value filter | DWI resolves to b ≥ 600 | 3,684 |
| enhancement QC recorded | — | **3,667** |

Split **patient-level**: 2,805 train / 312 val / 550 test.

**Two cohort properties worth reporting as findings, not footnotes:**

- **~12–13% of patients never enhance at any timepoint.** Not a processing failure — a
  genuine property (failed or mistimed injection). Belongs in the data section of any paper.
- **13% of DWI resolves to b50**, which carries almost no diffusion weighting and reads
  T2-like. That's a *different contrast* in the same input channel as the b1000 the
  majority get. Excluding it costs 13% of the data and is a clean one-flag ablation.
  Also: 43 cases use *computed* (`_synth_`) high-b images rather than acquired ones.

**Target phase chosen by elapsed time, not index.** Protocols are heterogeneous — 26–70
phases at 1.7–13 s cadence — so a fixed index is a different physiological moment per
patient. Measured enhancement curve: flat to ~45 s, wash-in 55–90 s (1.12→2.00×), then a
**plateau of 2.1–2.5× from ~90 s past 300 s**. There is no sharp peak, so intensity-argmax
lands on plateau noise at ~280 s. We take **t = 120 s**: inside the plateau, past the
timing-sensitive wash-in, ~92% of maximum. Realized times cluster at 104–125 s.

Two protocol complications handled explicitly: **interleaved series** (Dixon water/fat
stacked per timepoint — 27 cases; a plain index alternates between sub-series and can land
on the non-enhancing one) and **corrupt timestamps** (155 cases re-selected from the
enhancement curve at plateau onset).

---

## 3. Pipeline

```
bpMRI (T2w, DWI, ADC)  →  resample to common physical grid  →  normalize to [-1,1]
                       →  crop 90 × 90 × 96 mm  →  model  →  DCE prediction
```

**Geometry — the fix that mattered most.** `--spatial-size` originally meant a *pixel
count*. UCSF in-plane resolution spans 0.176–0.781 mm across 256/512/672/704/736/1024
matrices, so a fixed 256-px crop covered anywhere from **40 mm to 320 mm** of anatomy — an
8× range of scale, with 172 cases cropped to 45 mm, *smaller than a prostate*. Now
`--reference iso` resamples everyone to a common grid first, so the crop is a fixed
**physical** field of view (90 × 90 × 96 mm) for every patient.

**Normalization** is per-image (single center, so no cross-scanner alignment to do):
DCE robust-scaled to the body-tissue median ± 1 spread, T2w percentile, ADC fixed physical
clip, DWI foreground z-score. No Nyul.

**Loss.** L1 + (1−SSIM) + perceptual (frozen 3D MedicalNet), with **10× ROI weighting** —
the prostate is ~5% of the crop, so an unweighted loss is dominated by background.

---

## 4. Metrics — what each one actually means

Audiences conflate these. Worth a slide.

| metric | measures | target |
|---|---|---|
| `roi_pearson` | **within-gland co-localization** — is the enhancement in the right *place* | higher |
| `ssim_roi` / `mae_roi` / `psnr_roi` | fidelity inside the gland | higher / lower / higher |
| `roi_var_ratio` | heterogeneity preserved (<1 = over-smoothed, >1 = over-textured) | **1.0** |
| `roi_grad_ratio` | fine detail preserved | **1.0** |
| `roi_w1` | ROI intensity-histogram distance | 0 |
| `FID` | distributional realism over 2D slices | lower |
| `p75_corr` | cross-patient enhancement tracking | higher |

**Two framing points to make explicitly:**

- **Global SSIM/PSNR are ~99% background** and near-meaningless here. The gap between
  global and ROI metrics is the real signal.
- **Faithfulness and realism are different axes and can move in opposite directions.**
  A model can be more "realistic" and less correct. Report both.

---

## 5. Headline results

3D models, **best checkpoint**, held-out test (n = 550):

| | 3D GAN | 3D flow (MedVAE) |
|---|---|---|
| **roi_pearson** | **0.482** | 0.369 |
| ssim_roi | **0.397** | 0.276 |
| mae_roi | **0.236** | 0.269 |
| roi_var_ratio (→1) | **0.912** | 1.711 |
| roi_grad_ratio (→1) | 0.711 | **1.059** |
| realism_score | **0.845** | 0.714 |
| PZ / TZ roi_pearson | 0.339 / 0.522 | 0.169 / 0.417 |

VAL ≈ TEST throughout — no overfitting.

**The GAN wins on faithfulness across four independent setups** (2D and 3D, pixel and
latent). The flow's one advantage is texture (`grad_ratio` 1.06 vs 0.71) — it is
*less smooth*, which reads as more realistic while localizing worse. That is the
realism/faithfulness tradeoff in one row.

**Peripheral zone is roughly half as good as transition zone** (0.339 vs 0.522). Report
this — PZ is where most clinically significant cancer arises and where DCE drives the
PI-RADS upgrade rule, so it is the clinically weakest spot. It is also the honest limitation
a reviewer will find if you don't.

---

## 6. The main scientific finding

Baseline audit, n = 550 test, 3D GAN:

| predictor | MAE_roi | roi_pearson | what it knows |
|---|---|---|---|
| const (cohort mean) | 0.2510 | — | nothing |
| **t2w copy** | 0.3372 | **−0.043** | identity baseline |
| **MODEL** | 0.2355 | **+0.482** | trained generator |
| level (oracle brightness) | **0.1851** | — | correct per-case brightness, no structure |

**Read it in two halves.**

**Structure is learned.** Copying T2w scores ≈ 0, so 0.482 is real localization — the
model is not laundering the anatomy it was given.

**Amplitude is not.** The model tracks per-patient brightness at only **r = 0.334** and
compresses between-case spread **2.4×** (target sd 0.231 → predicted 0.098). Since **51%
of ROI variance is between-case amplitude**, both models lose to a baseline that knows only
the correct brightness — and the flow is worse than a *constant* predictor on MAE while
still having genuine localization.

**Why this is a property of the task.** Per-case amplitude depends on gadolinium dose,
injection rate, cardiac output, renal clearance and scanner gain. None of that is visible
in T2w/DWI/ADC. Within-gland structure is. We checked the DICOM headers for the protocol
variables — **they were removed by de-identification** (dose, rate, patient weight and
field strength absent; manufacturer overwritten with `"NA"`). So the information is not
merely unused, it is not present.

**And it may not matter clinically.** PI-RADS curve types (I progressive / II plateau /
III washout) are **scale-invariant** — multiplying a curve by any constant leaves the type
unchanged. The component we can't predict is orthogonal to the endpoint that matters.
That converts a negative result into a scoping argument.

---

## 7. Credibility: what was wrong and is now fixed

Be upfront about this; it is a strength, not a liability.

| fix | evidence |
|---|---|
| **Checkerboard artifact** in all three decoders (transposed convs) | lag-1 autocorrelation of the first difference **−0.60** in predictions (white noise is −0.50) vs **+0.81** for real DCE; **FID 175 → 89** after the fix |
| **Physical FOV** instead of pixel count | 8× scale range → uniform 90 mm |
| **Best-checkpoint evaluation** | 3D metrics previously scored the *final* epoch; roi_pearson 0.419 → 0.482 |
| **Batch-pooled ROI metrics** | pooling reported roi_pearson **+0.977 on pure noise** where truth is +0.020 |
| **Harmonizer leakage** | Nyul was fit on the full cohort incl. test (never triggered by our configs, now impossible) |
| Wrong targets in training | 155 cases trained against the pre-contrast volume; an `--overwrite` pass resurrected 729 pruned non-enhancers |

A 28-assertion preflight (`python -m tier1_static.selfcheck`) now pins every one of these
so they cannot regress silently.

---

## 8. What we can and cannot claim

**Run-to-run variance is real and was measured.** Two config-identical runs at the same
seed differ:

| metric | Δ between identical runs |
|---|---|
| roi_pearson | **0.017** |
| roi_var_ratio | 0.055 |
| **p75_corr** | **0.276** |

Training is not deterministic (cuDNN autotuning, non-deterministic 3D conv atomics),
adversarial dynamics amplify it, and best-checkpoint selection turns a bumpy validation
curve into a noisy discrete choice of epoch.

**Rule of thumb: differences under ~0.02 roi_pearson are unresolved at n = 1.**

- **Safe to claim:** GAN > flow (0.113 gap, ~7× noise); model ≫ t2w baseline (~30×);
  PZ < TZ (paired within run); best > last checkpoint (~4×).
- **Do not claim:** anything from `p75_corr` at n = 1; the iso-vs-no-iso comparison
  (0.020 — inside noise; iso is justified on correctness grounds, not measured gain).

**A deeper observation worth making:** *reproducibility tracks identifiability.* The
component the inputs determine (structure) reproduces at ±0.017; the component they don't
(amplitude) drifts freely, because nothing in the loss pins it down. The instability is a
measurement of which part of the task is ill-posed.

**Other caveats.** 2D and 3D ROI metrics aren't comparable (2D keeps only gland-containing
slices, so the denominators differ). FID was measured on the last checkpoint, not the best.
Everything predating the audit is superseded.

---

## 9. Questions you should expect

**"Is it just copying the T2w?"** No — T2w-copy scores `roi_pearson` −0.043 against the
model's +0.482. That baseline is in the paper.

**"Why not just report SSIM/PSNR?"** They're ~99% background here. A model can score well
globally while reconstructing the gland poorly; the global-vs-ROI gap is the diagnostic.

**"Is 0.48 good?"** It is far above every trivial baseline and below what the task would
allow if amplitude were recoverable. We've decomposed exactly which part is missing and why.

**"Why is the GAN beating the diffusion/flow model?"** Consistently, across four setups, at
this data scale. The flow produces more realistic *texture* but localizes worse — a
faithfulness/realism tradeoff, not a bug.

**"Will more data fix it?"** Unknown, and we're careful not to claim otherwise — an earlier
"data scale doesn't help" conclusion was measured on a configuration with known bugs and
has been retracted.

**"Can this replace real DCE?"** Not on this evidence. Per-patient enhancement amplitude
isn't recoverable, and PZ — where it matters most — is the weakest region. The defensible
claim is about curve *shape*, which is scale-invariant.

---

## 10. What's next

| | status |
|---|---|
| Pre-contrast conditioning (`--use-pregad`) | **running** — tests whether amplitude is recoverable from the pre-contrast baseline. Decisive, since the protocol metadata is gone |
| PI-RADS / DCE-positivity labels | waiting on Hanxue; unlocks the clinical endpoint and a downstream readout |
| Multi-timepoint staging | code landed; anchors on a common time grid, goal state at **240 s** (91.9% coverage; 300 s reaches only 49.7%) |
| Tier-2 / kinetics | ODEWorld-style continuous-time model with the velocity field **constrained to the Tofts equation** rather than learned free-form — we know the governing ODE, which the method's own related work calls out as the limiting requirement |
| Multi-seed repeats | needed before any marginal comparison is publishable |
