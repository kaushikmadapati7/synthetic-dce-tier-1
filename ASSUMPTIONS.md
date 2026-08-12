# Embedded assumptions — provenance audit

Every constant in this pipeline that encodes a **choice about the data or the task**,
with what actually justifies it. Written 2026-08-12 because these had accumulated as
defaults over ~3 weeks without being flagged as assumptions, and were being discovered
one at a time.

**Evidence classes:**

| class | meaning |
|---|---|
| `MEASURED(n)` | derived from data, sample size given |
| `CHOSEN` | picked by judgement, never validated against this cohort |
| `CONVENTION` | standard practice / inherited from literature |
| `RESOURCE` | set by compute or wall-clock limits, not by the science |

Anything `CHOSEN` that feeds a **published number** needs either a justification or a
sensitivity check before submission.

---

## A. Target definition — the highest-stakes group

These decide *what the model is trained to predict*. An error here is unrecoverable
downstream.

| constant | value | class | provenance / risk |
|---|---|---|---|
| `--dce-target-time` | 120 s | **MEASURED(5)** ⚠️ | Curve measured on **five** cases: flat to ~45 s, wash-in 55–90 s, plateau 2.1–2.5× from ~90 s past 300 s. 120 s chosen as inside-plateau. **Never re-checked against 4,246 cases.** Whether a fixed time is right at all depends on whether curves peak or plateau — unresolved. |
| `SWEEP_BELOW` | 1.5 | **CHOSEN** | Threshold below which the time-selected phase is distrusted and the curve is swept. No sensitivity analysis. |
| `RESELECT_ABOVE` | 1.5 | **CHOSEN** | Evidence that the study *did* enhance, so timing was wrong rather than the study. Same number as above by construction, not by measurement. |
| `MIN_ENH` | 1.5 | **CHOSEN** | Cohort filter. Drops ~12–13% of patients (5439→4246 with other steps). **This number decides the cohort size**, and it is a judgement call. |
| `PLATEAU_FRAC` | 0.9 | **CHOSEN** | Plateau onset = first phase at 90% of max enhancement. Was applied to raw signal rather than baseline-subtracted enhancement until `abf1ad2` — fired at 70–85% of true enhancement. |
| `t_max` | 600 s | **CHOSEN** | Timestamps beyond this are treated as corrupt. Plausible, unverified. |
| `MAX_INTERLEAVE` | 4 | **CHOSEN** | Cap on detected Dixon-style interleaving. Observed k=2,3; the cap is a guard, not a measurement. |

## B. Preprocessing

| constant | value | class | provenance |
|---|---|---|---|
| `--iso-spacing` (3D) | 0.47 mm | **MEASURED(4246)** ✅ | 192 px × 0.47 ≈ 90 mm, matching the dominant 512 @ 0.352 mm group. FOV distribution measured cohort-wide. |
| `--iso-spacing` (2D) | 0.35 mm | **MEASURED(4246)** ✅ | 256 px × 0.35 ≈ 90 mm. Same measurement. |
| `--dce-robust-k` | 1 | **MEASURED(12)** ⚠️ | Within-prostate target sd: k=3→0.073, k=2→0.109, k=1→**0.218**. Twelve cases. Direction is clear; the exact value is not pinned. |
| `--spatial-size` | 32×192×192 / 32×256×256 | **RESOURCE** | Depth 32 and the in-plane size are memory-driven. 90 mm FOV follows from spacing, not from anatomy. |
| `--dwi-min-bvalue` | 600 | **MEASURED(4246)** ✅ | b-value distribution: b1000 (3485), b50 (562), b1400 (135)… The 562 b50 acquisitions carry a different contrast. Threshold separates them cleanly. |
| `--ucsf-test-frac` | 0.15 | **CONVENTION** | Standard split fraction. |
| `--val-frac` | 0.1 | **CONVENTION** | |
| Nyul `pc_high` | 99.9 | **MEASURED(20)** | Raised from 99.0; cut prostate-voxel clipping 0.69%→0.005%. Not used on UCSF (per-image norms). |

## C. Loss and training

| constant | value | class | provenance |
|---|---|---|---|
| `--roi-weight` | 10 | **CHOSEN** | Prostate is ~5% of the crop, so *some* weighting is clearly needed; **10 specifically is arbitrary**. No sweep. |
| `--l1` / `--ssim` | 1.0 / 1.0 | **CONVENTION** | |
| `--perceptual` | 0.1 | **CONVENTION** | Typical value from the synthesis literature. |
| `--lr` | 1e-4 (3D), 2e-4 (2D) | **CONVENTION** | |
| `--epochs` | 40 (3D), 100 (2D) | **RESOURCE** | Wall-clock, **not** a convergence criterion. 3D convergence was never demonstrated. |
| `--base-ch`, `--batch-size` | 32/4, 64/16 | **RESOURCE** | Memory-driven. |
| `--select-metric` | `ssim_roi` | **CHOSEN** ⚠️ | Smoothness-biased, and the GAN is the smoother model — so it partly favours one arm of the main comparison. A selection-metric ablation has not been run. |
| `--ema-decay` | 0.999 | **CONVENTION** | |

## D. Metrics

| constant | value | class | provenance |
|---|---|---|---|
| `var_ratio` / `grad_ratio` cap | 5.0 | **CHOSEN** | Prevents a near-flat target denominator blowing up the mean. Biases both estimators toward 5 from above. Fraction of cases hitting the cap is not logged. |
| `realism_score` weights | equal | **CHOSEN** ⚠️ | Unvalidated composite of `var_ratio`, `grad_ratio`, `w1`. Ranking heuristic only; should not appear in a results table without that caveat. |
| SSIM window / sigma | 7 / 1.5 | **CONVENTION** | Standard. |
| `data_range` | 2.0 | derived | Follows from the `[-1,1]` range. |
| min ROI voxels | 16 | **CHOSEN** | Below this a case is skipped in ROI metrics. |
| slice `min_area` | 50 | **CHOSEN** | 2D slice retention threshold; sets the slice count per case. |
| `MONTAGE_MAX_MASK_FRAC` | 0.35 | **CHOSEN** | Cosmetic (montage selection only). Added on a mistaken diagnosis — the real cause was the FOV bug. |

---

## Triage

**Blocking for publication** — these decide the cohort or the target, and are `CHOSEN`
or thinly measured:

1. `--dce-target-time 120` / whether a fixed time is the right selection rule at all
2. `MIN_ENH 1.5` — sets cohort size, drops ~13% of patients
3. `SWEEP_BELOW` / `RESELECT_ABOVE` / `PLATEAU_FRAC` — govern 155 re-selected cases
4. `--roi-weight 10` — shapes the objective
5. `--select-metric ssim_roi` — partly determines which model "wins"

**Defensible as-is** — measured cohort-wide: `--iso-spacing`, `--dwi-min-bvalue`, the
240 s Tier-2 anchor (measured on n=1367).

**Low stakes** — conventions and resource limits, worth stating in Methods but not
worth a sweep: learning rates, loss weights, EMA, SSIM window, batch/channel sizes.

## Rule going forward

Any new constant that touches the data or the objective gets a class label and a
sample size **at the point it is introduced**. `CHOSEN` is acceptable — silently
`CHOSEN` is not.
