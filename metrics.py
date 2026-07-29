"""Image-space evaluation metrics for generated DCE volumes.

These are reported (not back-propagated) to gauge how close a generated volume
is to ground truth. SSIM reuses the 3D implementation from the loss module.

FID (Fréchet Inception Distance) compares the distribution of 2D axial slices
from predicted vs. reference volumes using Inception-v3 features (via
torch_fidelity). Lower is better.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .loss.loss import ssim3d

log = logging.getLogger("tier1")


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(data_range ** 2 / (mse + 1e-12))


@torch.no_grad()
def roi_radiomics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """ROI radiomic-fidelity metrics inside the prostate -- the numbers that
    separate a smooth, range-compressed prediction (good SSIM/PSNR, no texture)
    from one that preserves focal enhancement. SSIM/PSNR are ~99% background and
    even ROI-SSIM rewards the smooth blob; these target the enhancement signal
    directly. Computed over masked voxels (pred and target share the mask, so the
    voxel sets align 1:1), PER SAMPLE and then averaged -- never pooled across the
    batch, which would mix between-patient variance into every statistic.

      roi_pearson    voxelwise Pearson r(pred, target): spatial co-localization of
                     enhancement -- does the bright spot land in the right place
      roi_var_ratio  var(pred)/var(target): heterogeneity preserved (1 = match,
                     <1 = flattened/smoothed -> the smooth-blob detector)
      roi_p75_err    |p75(pred) - p75(target)|: upper-enhancement (peak) fidelity
      roi_w1         1-Wasserstein distance of the ROI intensity distributions,
                     normalized to [0,1] (0 = identical histograms)
    """
    # PER SAMPLE, then averaged. Pooling every masked voxel in the batch into one
    # vector (the previous behaviour) mixes BETWEEN-patient brightness variance into
    # each statistic, which is both wrong and batch-size dependent: on data where
    # per-patient brightness is predicted perfectly but within-gland structure is
    # pure noise, pooling reports roi_pearson +0.977 where the truth is +0.020, and
    # the value climbs with batch size (0.66 at B=2 -> 0.98 at B=8). Cross-patient
    # tracking is what p75_corr measures; roi_pearson must stay within-gland.
    acc, n = {}, 0
    for i in range(pred.shape[0]):
        m = mask[i] > 0.5
        if m.sum() < 16:
            continue
        p, t = pred[i][m].flatten().float(), target[i][m].flatten().float()
        pc, tc = p - p.mean(), t - t.mean()
        denom = (pc.norm() * tc.norm()).clamp(min=1e-8)
        w1 = (p.sort().values - t.sort().values).abs().mean() / 2.0  # /2: [-1,1] -> [0,1]
        # var_ratio is a ratio with a small denominator -> a near-flat target ROI
        # (low-enhancement case) blows it up and wrecks the mean. Cap per-case at 5
        # so "much too noisy" reads ~5 instead of thousands; aggregate stays robust.
        var_ratio = min(float(p.var(unbiased=False) / (t.var(unbiased=False) + 1e-6)), 5.0)
        for k, v in (("roi_pearson", float((pc * tc).sum() / denom)),
                     ("roi_var_ratio", var_ratio),
                     ("roi_p75_err", float((p.quantile(0.75) - t.quantile(0.75)).abs())),
                     ("roi_w1", float(w1))):
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in acc.items()} if n else {}


@torch.no_grad()
def roi_sharpness(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Sharpness / high-frequency-detail fidelity inside the ROI -- the direct
    quantifier of the 'blobby vs realistic' look that matters for a radiologist
    reading the synthetic DCE. Gradient-magnitude energy (3D finite differences)
    averaged over prostate voxels, as a pred/target ratio:

      roi_grad_ratio  mean|grad(pred)| / mean|grad(target)| over ROI voxels
                      1  = detail matches real, <1 = too smooth (blob), >1 = too
                      noisy. Complements roi_var_ratio (intensity spread) with a
                      spatial-frequency view -- a smooth blob has low grad_ratio
                      even if its variance happens to match.
    """
    def gmag(v):
        v = v.float()
        dims = tuple(d for d in (-3, -2, -1) if v.shape[d] >= 2)  # skip singleton (2D: depth=1)
        gs = torch.gradient(v, dim=dims)
        if not isinstance(gs, (list, tuple)):
            gs = (gs,)
        return torch.sqrt(sum(g ** 2 for g in gs) + 1e-12)
    # per sample, then averaged (mean of ratios, not ratio of batch means) so that
    # high-gradient cases cannot dominate and the value is batch-size independent --
    # matching roi_radiomics.
    gp, gt = gmag(pred), gmag(target)
    tot, n = 0.0, 0
    for i in range(pred.shape[0]):
        m = mask[i] > 0.5
        if m.sum() < 16:
            continue
        tot += min(float(gp[i][m].mean() / (gt[i][m].mean() + 1e-6)), 5.0)
        n += 1
    return {"roi_grad_ratio": tot / n} if n else {}


# Keys that make up the label-free "realism panel" (how real it looks to a reader),
# vs the faithfulness metrics (roi_pearson etc., is it in the right place).
REALISM_KEYS = ("fid", "roi_w1", "roi_var_ratio", "roi_grad_ratio")


def realism_score(agg: dict) -> float | None:
    """A single label-free realism proxy in [0,1] (higher = more realistic) from
    the texture/detail/intensity-distribution metrics -- a cheap stand-in for a
    reader study to rank checkpoints/models between reader sessions. Rewards
    var_ratio & grad_ratio near 1 (real amount of texture and detail) and a small
    ROI intensity-histogram distance (roi_w1). FID is NOT included here (too
    expensive per checkpoint); use it as the primary realism number at eval time."""
    def near1(v):
        return None if v is None else 1.0 - min(abs(v - 1.0), 1.0)
    parts = [near1(agg.get("roi_var_ratio")), near1(agg.get("roi_grad_ratio"))]
    if "roi_w1" in agg:
        parts.append(1.0 - min(agg["roi_w1"], 1.0))
    parts = [p for p in parts if p is not None]
    return float(sum(parts) / len(parts)) if parts else None


def selection_score(agg: dict, metric: str = "ssim_roi") -> float:
    """Map an aggregated-metrics dict to a scalar for best-checkpoint selection.

      ssim_roi     legacy default (SMOOTHNESS-biased -> rewards the blob)
      roi_pearson  faithfulness / within-gland localization
      realism      label-free realism proxy (texture+detail+intensity match)
      balanced     0.5*max(0,roi_pearson) + 0.5*realism -- realistic AND faithful,
                   the realism-primary-but-safety-guarded objective
    """
    if metric == "roi_pearson":
        return agg.get("roi_pearson", float("-inf"))
    if metric == "realism":
        r = realism_score(agg)
        return r if r is not None else float("-inf")
    if metric == "balanced":
        r = realism_score(agg)
        if r is None:
            return agg.get("ssim_roi", float("-inf"))
        return 0.5 * max(0.0, agg.get("roi_pearson", 0.0)) + 0.5 * r
    return agg.get("ssim_roi", agg.get("ssim", float("-inf")))   # default / "ssim_roi"


def _masked_pearson(pred, target, m) -> float | None:
    m = m > 0.5
    if m.sum() < 16:
        return None
    p, t = pred[m].flatten().float(), target[m].flatten().float()
    pc, tc = p - p.mean(), t - t.mean()
    denom = (pc.norm() * tc.norm()).clamp(min=1e-8)
    return float((pc * tc).sum() / denom)


@torch.no_grad()
def zone_metrics(pred: torch.Tensor, target: torch.Tensor, zones: torch.Tensor | None) -> dict:
    """Per-zone localization, split by the PZ/TZ label map (1=TZ, 2=PZ). DCE is
    clinically read in the PZ, so `roi_pearson_pz` is the zone number that matters
    -- and the one zone-weighted training is meant to lift. Whole-gland metrics
    average PZ and TZ together and can hide a PZ gain bought at TZ's expense."""
    out = {}
    if zones is None or zones.sum() == 0:
        return out
    # per sample then averaged, for the same reason as roi_radiomics: pooling the
    # batch lets between-patient brightness inflate a within-zone correlation.
    for name, lbl in (("pz", 2), ("tz", 1)):
        rs, es = [], []
        for i in range(pred.shape[0]):
            zm = (zones[i].round() == lbl).float()
            r = _masked_pearson(pred[i], target[i], zm)
            if r is None:
                continue
            rs.append(r)
            m = zm > 0.5
            es.append(float((pred[i][m].float().quantile(0.75)
                             - target[i][m].float().quantile(0.75)).abs()))
        if rs:
            out[f"roi_pearson_{name}"] = sum(rs) / len(rs)
            out[f"roi_p75_err_{name}"] = sum(es) / len(es)
    return out


@torch.no_grad()
def roi_p75(vol: torch.Tensor, mask: torch.Tensor) -> float | None:
    """Scalar ROI 75th-percentile (enhancement level) of one volume; None if the
    mask is empty. Used to build a cross-case real-vs-synth enhancement scatter."""
    m = mask > 0.5
    return float(vol[m].float().quantile(0.75)) if m.sum() >= 16 else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r over paired scalars; None if <3 pairs or zero variance."""
    import math
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    d = math.sqrt(sxx * syy)
    return float(sxy / d) if d > 0 else None


@torch.no_grad()
def eval_metrics(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None, zones: torch.Tensor | None = None) -> dict:
    """Per-batch metrics. If a mask is given, MAE plus the ROI radiomic-fidelity
    metrics (see ``roi_radiomics``) are also computed inside the ROI; if a PZ/TZ
    zone label map is given, per-zone localization is added (see ``zone_metrics``)."""
    out = {
        "ssim": float(ssim3d(pred, target)),
        "psnr": float(psnr(pred, target)),
        "mae": float(F.l1_loss(pred, target)),
    }
    if mask is not None and mask.sum() > 0:
        # ROI scalars per sample, then averaged: pooling batch voxels weights each
        # patient by gland size (and, for PSNR, log of a pooled MSE is not the mean
        # of per-case PSNRs). The global ssim/mae above are safe to pool because
        # every volume has the same fixed spatial_size.
        smap = ssim3d(pred, target, return_map=True)
        maes, psnrs, ssims = [], [], []
        for i in range(pred.shape[0]):
            mi = mask[i] > 0.5
            if mi.sum() < 16:
                continue
            maes.append(float((pred[i][mi] - target[i][mi]).abs().mean()))
            psnrs.append(float(psnr(pred[i][mi], target[i][mi])))
            ssims.append(float(smap[i][mi].mean()))
        if maes:
            out["mae_roi"] = sum(maes) / len(maes)
            out["psnr_roi"] = sum(psnrs) / len(psnrs)
            out["ssim_roi"] = sum(ssims) / len(ssims)
        out.update(roi_radiomics(pred, target, mask))
        out.update(roi_sharpness(pred, target, mask))
        out.update(zone_metrics(pred, target, zones))
    return out


def aggregate(metric_dicts: list[dict]) -> dict:
    """Mean over a list of per-batch metric dicts."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(sum(d[k] for d in metric_dicts if k in d) /
                     max(1, sum(k in d for d in metric_dicts))) for k in keys}


# ---------------------------------------------------------------------------
# FID (distribution metric over 2D axial slices)
# ---------------------------------------------------------------------------
def _to_uint8_rgb(slice_2d: torch.Tensor) -> torch.Tensor:
    """(H, W) in [-1, 1] -> (3, H, W) uint8 RGB for Inception."""
    x = ((slice_2d.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return x.unsqueeze(0).expand(3, -1, -1)


def volumes_to_slice_tensors(volumes: list[torch.Tensor], slices_per_volume: int = 8,
                             min_size: int = 64) -> list[torch.Tensor]:
    """Extract evenly spaced axial slices from (1, D, H, W) volumes."""
    slices = []
    for vol in volumes:
        v = vol[0] if vol.dim() == 4 else vol  # (D, H, W)
        d = v.shape[0]
        if d == 0:
            continue
        idxs = torch.linspace(0, d - 1, min(slices_per_volume, d)).long().tolist()
        for i in idxs:
            sl = v[i]
            if min(sl.shape) < min_size:
                sl = F.interpolate(sl[None, None], size=(min_size, min_size),
                                   mode="bilinear", align_corners=False)[0, 0]
            slices.append(_to_uint8_rgb(sl.cpu()))
    return slices


class _SliceDataset(Dataset):
    """torch_fidelity-compatible dataset of RGB uint8 slices."""

    def __init__(self, slices: list[torch.Tensor]):
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        return self.slices[i]


def compute_fid(preds: list[torch.Tensor], targets: list[torch.Tensor],
                device: torch.device | str = "cpu", slices_per_volume: int = 8,
                batch_size: int = 32) -> float | None:
    """FID between predicted and reference slice distributions.

    Requires at least a few slices in each set (torch_fidelity needs enough
    samples for a stable covariance estimate).
    """
    if not preds or not targets:
        return None
    pred_slices = volumes_to_slice_tensors(preds, slices_per_volume)
    tgt_slices = volumes_to_slice_tensors(targets, slices_per_volume)
    if len(pred_slices) < 2 or len(tgt_slices) < 2:
        log.warning(f"FID skipped: need >=2 slices per set (got {len(pred_slices)}/{len(tgt_slices)})")
        return None
    try:
        from torch_fidelity import calculate_metrics
        from torch_fidelity.metric_fid import KEY_METRIC_FID
    except ImportError:
        log.warning("torch_fidelity not installed; skipping FID")
        return None

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        result = calculate_metrics(
            input1=_SliceDataset(pred_slices),
            input2=_SliceDataset(tgt_slices),
            cuda=use_cuda,
            batch_size=min(batch_size, len(pred_slices), len(tgt_slices)),
            fid=True,
            isc=False, kid=False, prc=False, ppl=False,
            verbose=False,
        )
        return float(result[KEY_METRIC_FID])
    except Exception as e:
        log.warning(f"FID computation failed: {e}")
        return None
