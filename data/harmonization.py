"""Cross-scanner intensity harmonization for Tier-1 MRI volumes.

Applied on raw resampled intensities, *before* the final [-1, 1] scaling, and
each method is matched to whether the modality is qualitative or quantitative:

  T2w, DCE  (arbitrary units)  -> Nyul-Udupa landmark histogram standardization
                                  (fit on the training split, applied to all).
  DWI       (semi-quantitative)-> per-image z-score in the foreground (removes
                                  scanner gain without flattening diffusion
                                  contrast). No distribution remapping.
  ADC       (quantitative)     -> NO remapping; fixed physical clip only, so the
                                  absolute mm^2/s values are preserved.

ComBat is intentionally not implemented: it requires fitting over the full
multi-site population (and site labels) at once, which we may not reliably have.

Each method also produces the final [-1, 1] output so intensity handling lives
in one place; when a Harmonizer is supplied the dataset skips its own
percentile normalization.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _foreground(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Tissue voxels. Uses mask if given, else intensities above the mean
    (Nyul's standard background exclusion)."""
    if mask is not None and mask.shape == arr.shape and mask.any():
        return arr[mask > 0]
    fg = arr[arr > arr.mean()]
    return fg if fg.size else arr.ravel()


def _to_unit(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] then scale to [-1, 1]."""
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Nyul-Udupa landmark standardization (T2w, DCE)
# ---------------------------------------------------------------------------
@dataclass
class NyulConfig:
    pc_low: float = 1.0          # lower scale anchor percentile
    pc_high: float = 99.0        # upper scale anchor percentile
    landmark_percentiles: tuple = (10, 20, 30, 40, 50, 60, 70, 80, 90)
    i_min: float = 1.0           # standard scale endpoints
    i_max: float = 100.0


class NyulStandardizer:
    """Classic Nyul-Udupa two-pass landmark histogram standardization."""

    def __init__(self, cfg: NyulConfig | None = None):
        self.cfg = cfg or NyulConfig()
        self.standard_landmarks: np.ndarray | None = None

    def _landmarks(self, arr, mask=None):
        fg = _foreground(arr, mask)
        pcs = [self.cfg.pc_low, *self.cfg.landmark_percentiles, self.cfg.pc_high]
        return np.percentile(fg, pcs)

    def fit(self, volumes, masks=None):
        masks = masks or [None] * len(volumes)
        mapped = []
        for arr, m in zip(volumes, masks):
            lm = self._landmarks(arr, m)
            p1, p2 = lm[0], lm[-1]
            scaled = self.cfg.i_min + (lm - p1) * (self.cfg.i_max - self.cfg.i_min) / (p2 - p1 + 1e-8)
            mapped.append(scaled)
        # mean standard landmarks (drop the two anchors -> they map to i_min/i_max)
        self.standard_landmarks = np.mean(mapped, axis=0)
        return self

    def transform(self, arr, mask=None) -> np.ndarray:
        if self.standard_landmarks is None:
            raise RuntimeError("NyulStandardizer must be fit() before transform().")
        lm = self._landmarks(arr, mask)
        # piecewise-linear map this image's landmarks onto the standard ones
        return np.interp(arr, lm, self.standard_landmarks).astype(np.float32)

    def state(self):
        return {"cfg": asdict(self.cfg),
                "standard_landmarks": None if self.standard_landmarks is None
                else self.standard_landmarks.tolist()}

    def load_state(self, s):
        self.cfg = NyulConfig(**s["cfg"])
        self.standard_landmarks = (None if s["standard_landmarks"] is None
                                   else np.asarray(s["standard_landmarks"], dtype=np.float32))
        return self


# ---------------------------------------------------------------------------
# per-image methods (DWI, ADC) - no fitting required
# ---------------------------------------------------------------------------
def zscore_foreground(arr, mask=None, clip_sd: float = 3.0) -> np.ndarray:
    fg = _foreground(arr, mask)
    m, s = float(fg.mean()), float(fg.std()) + 1e-8
    z = (arr - m) / s
    return _to_unit(z, -clip_sd, clip_sd)


def clip_quantitative(arr, max_value: float) -> np.ndarray:
    """Preserve absolute (e.g. ADC) values via a fixed physical clip."""
    return _to_unit(arr, 0.0, max_value)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
@dataclass
class HarmonizationConfig:
    methods: dict = field(default_factory=lambda: {
        "t2w": "nyul", "dce": "nyul", "dwi": "zscore", "adc": "clip"})
    adc_max_value: float = 3000.0   # 10^-6 mm^2/s
    dwi_clip_sd: float = 3.0
    nyul_i_min: float = 1.0
    nyul_i_max: float = 100.0


class Harmonizer:
    """Per-modality harmonization. Fit the Nyul modalities on training raw
    volumes; per-image methods need no fitting. ``apply`` returns [-1, 1]."""

    def __init__(self, cfg: HarmonizationConfig | None = None,
                 nyul_cfg: NyulConfig | None = None):
        self.cfg = cfg or HarmonizationConfig()
        nyul_cfg = nyul_cfg or NyulConfig(i_min=self.cfg.nyul_i_min,
                                          i_max=self.cfg.nyul_i_max)
        self.nyul = {m: NyulStandardizer(nyul_cfg)
                     for m, meth in self.cfg.methods.items() if meth == "nyul"}

    def has(self, modality: str) -> bool:
        return modality in self.cfg.methods

    @property
    def nyul_modalities(self):
        return list(self.nyul.keys())

    def fit(self, volumes_by_modality: dict, masks_by_modality: dict | None = None):
        masks_by_modality = masks_by_modality or {}
        for m, std in self.nyul.items():
            if m not in volumes_by_modality:
                raise KeyError(f"no training volumes provided for Nyul modality '{m}'")
            std.fit(volumes_by_modality[m], masks_by_modality.get(m))
        return self

    def apply(self, modality: str, arr: np.ndarray, mask=None) -> np.ndarray:
        meth = self.cfg.methods.get(modality)
        if meth == "nyul":
            std = self.nyul[modality].transform(arr, mask)
            return _to_unit(std, self.cfg.nyul_i_min, self.cfg.nyul_i_max)
        if meth == "zscore":
            return zscore_foreground(arr, mask, self.cfg.dwi_clip_sd)
        if meth == "clip":
            return clip_quantitative(arr, self.cfg.adc_max_value)
        raise ValueError(f"unknown harmonization method '{meth}' for '{modality}'")

    # ---- persistence ----
    def save(self, path):
        state = {"cfg": {**asdict(self.cfg)},
                 "nyul": {m: std.state() for m, std in self.nyul.items()}}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text())
        cfg = HarmonizationConfig(**state["cfg"])
        obj = cls(cfg)
        for m, s in state["nyul"].items():
            obj.nyul[m] = NyulStandardizer().load_state(s)
        return obj


# ---------------------------------------------------------------------------
# fit convenience: pull raw resampled volumes from a dataset
# ---------------------------------------------------------------------------
def fit_harmonizer_from_dataset(harmonizer: Harmonizer, dataset, max_cases=None,
                                verbose=True):
    """Collect raw (un-normalized) volumes for the Nyul modalities from a
    dataset that exposes ``raw_modalities(i)`` and fit the harmonizer.
    Only the modalities that actually need fitting (Nyul) are loaded."""
    mods = harmonizer.nyul_modalities
    vols = {m: [] for m in mods}
    n = len(dataset) if max_cases is None else min(max_cases, len(dataset))
    for i in range(n):
        raw = dataset.raw_modalities(i)
        for m in mods:
            if m in raw:
                vols[m].append(raw[m])
        if verbose and (i + 1) % 25 == 0:
            print(f"[harmonizer] collected {i + 1}/{n}")
    return harmonizer.fit(vols)
