"""Realism-metric development lab — Track-1 (CV-metrics) scaffolding.

The Track-1 contribution is *new realism metrics*, not a new model. A metric is
only credible if we can show it is a good metric, so this module is the engine
for DEVELOPING and VALIDATING metrics rather than any single metric itself.

Design:
  - A metric is a pure function  m(pred, target, mask=None) -> float, by convention
    a DEVIATION where 0 == "identical to the real target" and larger == "less real".
  - We validate a candidate metric by DEGRADING real images along a controlled axis
    (blur, noise, contrast-flattening, hallucinated blobs, latent-bottleneck) at
    increasing severity and asking: does the metric rise monotonically with severity?
    A good realism metric has high positive Spearman(severity, metric) for the
    failure modes it is meant to catch. The (metric x degradation) Spearman table is
    the core diagnostic — it exposes which metrics detect which failure modes.

Everything is numpy-only and works on 2D (H,W) or 3D (D,H,W) arrays in any intensity
range, with an optional ROI mask — so it ports from prostate to liver/breast with
no change. Run `python -m tier1_static.realism_lab` for a self-contained phantom
smoke that prints the validation table (no external data needed).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _roi(x, mask):
    if mask is None:
        return x.ravel()
    m = mask > 0.5
    return x[m] if m.any() else x.ravel()


def _grad_mag(x):
    g = np.gradient(x.astype(np.float32))
    return np.sqrt(sum(np.square(gi) for gi in g))


# ----------------------------------------------------------------------------- #
# candidate realism metrics: m(pred, target, mask) -> deviation (0 = matched)
# `pred` and `target` are assumed spatially aligned (they are, for degradations).
# ----------------------------------------------------------------------------- #
def m_l1(pred, target, mask=None):
    """Plain intensity error — the naive baseline a realism metric should beat."""
    return float(np.abs(_roi(pred, mask) - _roi(target, mask)).mean())


def m_grad_ratio_dev(pred, target, mask=None):
    """|1 - edge-energy ratio|. Detects over-smoothing (blob) / over-texturing.
    Blind to WHERE the edges are — see m_edge_struct_dev for that."""
    gt = _roi(_grad_mag(target), mask).mean()
    if gt < 1e-8:
        return 0.0
    return abs(1.0 - _roi(_grad_mag(pred), mask).mean() / gt)


def m_edge_struct_dev(pred, target, mask=None):
    """NEW candidate: 1 - corr(|grad pred|, |grad target|) over the ROI. Measures
    whether sharpness sits in the RIGHT places, not just in the right amount — so it
    fires on hallucinated texture that m_grad_ratio_dev (amount-only) misses."""
    gp = _roi(_grad_mag(pred), mask)
    gt = _roi(_grad_mag(target), mask)
    if gp.std() < 1e-8 or gt.std() < 1e-8:
        return 1.0
    return float(1.0 - np.clip(np.corrcoef(gp, gt)[0, 1], -1, 1))


def m_var_ratio_dev(pred, target, mask=None):
    """|1 - variance ratio|. Contrast/energy match within the ROI."""
    vt = _roi(target, mask).var()
    if vt < 1e-12:
        return 0.0
    return abs(1.0 - _roi(pred, mask).var() / vt)


def m_intensity_w1(pred, target, mask=None, q=64):
    """1-Wasserstein between ROI intensity histograms (quantile approximation).
    Detects distribution shift (over-bright / flattened) independent of location."""
    a, b = _roi(pred, mask), _roi(target, mask)
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.linspace(0, 1, q)
    return float(np.abs(np.quantile(a, grid) - np.quantile(b, grid)).mean())


def m_lap_ratio_dev(pred, target, mask=None):
    """NEW candidate: |1 - 2nd-order (Laplacian) energy ratio|. A finer sharpness
    probe than the 1st-order gradient — more sensitive to mild blur / bottlenecking."""
    def lap_energy(x):
        acc = np.zeros_like(x, dtype=np.float32)
        for ax in range(x.ndim):
            acc += np.square(np.gradient(np.gradient(x, axis=ax), axis=ax))
        return _roi(np.sqrt(acc), mask).mean()
    lt = lap_energy(target)
    if lt < 1e-8:
        return 0.0
    return abs(1.0 - lap_energy(pred) / lt)


METRICS = {
    "l1": m_l1,
    "grad_ratio": m_grad_ratio_dev,
    "edge_struct": m_edge_struct_dev,     # new
    "var_ratio": m_var_ratio_dev,
    "intensity_w1": m_intensity_w1,
    "lap_ratio": m_lap_ratio_dev,         # new
}


# ----------------------------------------------------------------------------- #
# controlled degradations: d(img, severity in [0,1], mask) -> degraded img.
# severity 0 returns the image unchanged; each models a realism failure mode.
# ----------------------------------------------------------------------------- #
def d_blur(img, s, mask=None):
    return gaussian_filter(img, sigma=3.0 * s) if s > 0 else img


def d_noise(img, s, mask=None, rng=None):
    if s <= 0:
        return img
    rng = rng or np.random.default_rng(0)
    return img + rng.normal(0, 0.3 * s, img.shape).astype(img.dtype)


def d_flatten(img, s, mask=None):
    """Pull the ROI toward its own mean — the smooth-blob / mode-collapse failure."""
    out = img.copy()
    m = np.ones_like(img, bool) if mask is None else (mask > 0.5)
    if m.any():
        out[m] = (1 - s) * img[m] + s * img[m].mean()
    return out


def d_hallucinate(img, s, mask=None):
    """Inject a localized false bright blob inside the ROI — plausible-but-wrong
    enhancement, the failure mode a realism-only metric must not reward."""
    if s <= 0:
        return img
    out = img.copy()
    m = np.ones_like(img, bool) if mask is None else (mask > 0.5)
    idx = np.argwhere(m)
    if len(idx) == 0:
        return out
    c = idx[len(idx) // 2]
    coords = np.indices(img.shape)
    r2 = sum(np.square(coords[i] - c[i]) for i in range(img.ndim))
    rad = max(2.0, 0.15 * min(img.shape))
    blob = np.exp(-r2 / (2 * rad ** 2)).astype(img.dtype)
    amp = s * (img[m].max() - img[m].mean() + 1e-3)
    return out + amp * blob * m


def d_bottleneck(img, s, mask=None):
    """Downsample then upsample — mimics a lossy latent (VAE) bottleneck."""
    if s <= 0:
        return img
    f = 1.0 / (1.0 + 3.0 * s)
    small = zoom(img, f, order=1)
    back = zoom(small, np.array(img.shape) / np.array(small.shape), order=1)
    return back.astype(img.dtype)


DEGRADATIONS = {
    "blur": d_blur,
    "noise": d_noise,
    "flatten": d_flatten,
    "hallucinate": d_hallucinate,
    "bottleneck": d_bottleneck,
}


# ----------------------------------------------------------------------------- #
# nuisance transforms: realism-PRESERVING intensity changes (scanner gain / bias).
# A structure-realism metric should stay ~flat under these; an intensity-fidelity
# metric will (correctly) respond. The lab exposes each metric's character.
# ----------------------------------------------------------------------------- #
def p_gain(img, s, mask=None):
    return img * (1.0 + 0.4 * s)


def p_bias(img, s, mask=None):
    return img + 0.2 * s


NUISANCES = {"gain": p_gain, "bias": p_bias}
ALL_PERT = {**DEGRADATIONS, **NUISANCES}


# ----------------------------------------------------------------------------- #
# validation harness
# ----------------------------------------------------------------------------- #
def sensitivity(reals, masks, perturbation, metric, severities=(0, .25, .5, .75, 1.)):
    """Metric-vs-severity curve (mean over the real set) + Spearman monotonicity.
    Monotonicity is necessary but NOT discriminating (most metrics rise with any
    degradation); use response_profile for the discriminating view."""
    curve = []
    for s in severities:
        vals = [metric(perturbation(reals[i], s, None if masks is None else masks[i]),
                       reals[i], None if masks is None else masks[i])
                if s > 0 else 0.0 for i in range(len(reals))]
        curve.append(float(np.mean(vals)))
    rho = spearmanr(severities, curve).correlation if len(set(curve)) > 1 else 0.0
    return np.array(curve), float(rho if rho == rho else 0.0)


def _mask_l1(reals, masks, pf, sev):
    """Mean masked-L1 induced by perturbation pf at severity sev (its 'distortion')."""
    return float(np.mean([
        np.abs(_roi(pf(reals[i], sev, None if masks is None else masks[i]) - reals[i],
                    None if masks is None else masks[i])).mean()
        for i in range(len(reals))]))


def calibrate(reals, masks, perturbations=ALL_PERT, target_l1=None, hi=4.0, iters=16):
    """Per-perturbation severity that induces ~`target_l1` masked-L1 from the real, so
    every perturbation is compared at MATCHED pixel-distortion (else the harshest one,
    e.g. noise, hijacks the normalization). Assumes distortion rises with severity.
    `target_l1=None` auto-picks a level every perturbation can reach (0.8x the weakest
    perturbation's max distortion) so the comparison is truly matched."""
    reach = {pn: _mask_l1(reals, masks, pf, hi) for pn, pf in perturbations.items()}
    if target_l1 is None:
        target_l1 = 0.8 * min(reach.values())
    sev = {}
    for pn, pf in perturbations.items():
        if reach[pn] <= target_l1:
            sev[pn] = hi; continue                       # can't reach target -> cap
        a, b = 0.0, hi
        for _ in range(iters):
            m = 0.5 * (a + b)
            a, b = (m, b) if _mask_l1(reals, masks, pf, m) < target_l1 else (a, m)
        sev[pn] = 0.5 * (a + b)
    return sev, target_l1


def response_profile(reals, masks, metric, perturbations=ALL_PERT, sev=None):
    """Per-metric response to each perturbation (at matched-distortion severities if
    `sev` is a dict from calibrate), normalized so the metric's strongest response = 1.
    Reveals WHAT perturbation type a metric is sensitive to, magnitude held equal."""
    def sv(pn):
        return sev[pn] if isinstance(sev, dict) else (0.6 if sev is None else sev)
    raw = {pn: float(np.mean([
        metric(pf(reals[i], sv(pn), None if masks is None else masks[i]),
               reals[i], None if masks is None else masks[i])
        for i in range(len(reals))])) for pn, pf in perturbations.items()}
    mx = max(raw.values()) or 1.0
    return {pn: raw[pn] / mx for pn in raw}


def profile_table(reals, masks=None, metrics=METRICS, perturbations=ALL_PERT, sev=None):
    return {mn: response_profile(reals, masks, mf, perturbations, sev)
            for mn, mf in metrics.items()}


def scorecard(table, degradations=DEGRADATIONS, nuisances=NUISANCES):
    """Realism-metric quality = sensitive to degradations, robust to nuisances.
    Score = mean(degradation response) - mean(nuisance response), in [-1, 1]:
    higher = flags real failures without false-alarming on benign intensity change."""
    return {mn: float(np.mean([row[d] for d in degradations])
                      - np.mean([row[n] for n in nuisances]))
            for mn, row in table.items()}


def print_profiles(table, perturbations=ALL_PERT, degradations=DEGRADATIONS,
                   nuisances=NUISANCES):
    cols = list(perturbations)
    score = scorecard(table, degradations, nuisances)
    hdr = f"{'metric':<14}" + "".join(f"{c:>12}" for c in cols) + f"{'SCORE':>9}"
    print(hdr); print("-" * len(hdr))
    for mn, row in sorted(table.items(), key=lambda kv: -score[kv[0]]):
        print(f"{mn:<14}" + "".join(f"{row[c]:>12.2f}" for c in cols)
              + f"{score[mn]:>9.2f}")
    print(f"\n  degradations (want HIGH): {', '.join(degradations)}")
    print(f"  nuisances    (want LOW ): {', '.join(nuisances)}")
    print("  cells = per-metric normalized response (max=1); "
          "SCORE = mean(deg) - mean(nuisance), higher is a better realism metric.")


# ----------------------------------------------------------------------------- #
# phantom smoke — self-contained, no external data
# ----------------------------------------------------------------------------- #
def _phantoms(n=6, shape=(16, 96, 96), seed=0):
    """Structured 'real' volumes: smooth base + a few bright ellipsoids + fine
    texture; ROI mask = central box. Stand-ins for peak-phase DCE until Bao's data
    lands — the harness is identical on real volumes."""
    rng = np.random.default_rng(seed)
    reals, masks = [], []
    coords = np.indices(shape)
    for _ in range(n):
        img = gaussian_filter(rng.normal(0, 1, shape), 4).astype(np.float32)
        for _ in range(rng.integers(2, 5)):
            c = [rng.integers(0, d) for d in shape]
            r2 = sum(np.square(coords[i] - c[i]) for i in range(len(shape)))
            img += rng.uniform(1, 3) * np.exp(-r2 / (2 * (0.12 * min(shape)) ** 2))
        img += 0.15 * gaussian_filter(rng.normal(0, 1, shape), 1)   # fine texture
        img = np.clip(img / (np.abs(img).max() + 1e-6), -1, 1)
        m = np.zeros(shape, np.float32)
        sl = tuple(slice(d // 4, 3 * d // 4) for d in shape)
        m[sl] = 1.0
        reals.append(img.astype(np.float32)); masks.append(m)
    return reals, masks


if __name__ == "__main__":
    reals, masks = _phantoms()
    print(f"phantom set: {len(reals)} volumes {reals[0].shape}, ROI-masked")
    sev, tgt = calibrate(reals, masks)             # auto matched-distortion severities
    print(f"matched-distortion (masked-L1~{tgt:.3f}): "
          + ", ".join(f"{k}={v:.2f}" for k, v in sev.items()) + "\n")
    table = profile_table(reals, masks, sev=sev)
    print_profiles(table)
