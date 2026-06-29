"""Geometry + intensity preprocessing for Tier-1 NIfTI volumes.

All cross-modality alignment is done by resampling onto a single reference grid
so that T2w/DWI/ADC/DCE end up voxel-aligned and channel-stackable. The choice
of reference grid is left configurable (it's an open design decision):

    reference = "t2w" -> resample everything onto the T2w grid (anatomical res)
    reference = "dce" -> resample everything onto the DCE grid (target native)
    reference = "iso" -> resample everything to a fixed isotropic spacing

After resampling, volumes are intensity-normalized (robust percentile clip to
[-1, 1], matching the tanh output range of the models) and center cropped/padded
to a fixed `spatial_size` so a batch can be stacked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk


@dataclass
class PreprocessConfig:
    reference: str = "dce"                 # "t2w" | "dce" | "iso"; dce keeps target native
    iso_spacing: tuple = (1.0, 1.0, 1.0)   # used when reference == "iso" (x, y, z mm)
    spatial_size: tuple | None = (32, 192, 192)  # (D, H, W); None = leave as resampled
    clip_percentiles: tuple = (0.5, 99.9)  # robust intensity window (99.9 upper preserves DCE peak; matches Nyul pc_high)
    adc_clip_value: float | None = 3000.0  # ADC is quantitative; hard clip instead of %iles
    pad_value: float = -1.0                # background after normalization to [-1, 1]
    # zone-aware loss weighting (prostate_zones: 1=TZ, 2=PZ). The emitted
    # 'zone_weight' map multiplies the ROI loss; DCE matters clinically in the PZ
    # (PI-RADS), so pz_weight>1 emphasizes it. 1.0/1.0 = no zone effect (default).
    tz_weight: float = 1.0
    pz_weight: float = 1.0


# ---------------------------------------------------------------------------
# IO + resampling (SimpleITK images carry spacing/origin/direction)
# ---------------------------------------------------------------------------
def load_sitk(path) -> sitk.Image:
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def make_iso_reference(img: sitk.Image, spacing) -> sitk.Image:
    """A blank reference image at `spacing` covering the same physical extent."""
    old_spacing = np.array(img.GetSpacing())
    old_size = np.array(img.GetSize())
    new_spacing = np.array(spacing, dtype=float)
    new_size = np.ceil(old_size * old_spacing / new_spacing).astype(int).tolist()
    ref = sitk.Image([int(s) for s in new_size], sitk.sitkFloat32)
    ref.SetSpacing([float(s) for s in new_spacing])
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    return ref


def resample_to_reference(img: sitk.Image, ref: sitk.Image,
                          interp=sitk.sitkLinear, pad=0.0) -> sitk.Image:
    return sitk.Resample(img, ref, sitk.Transform(), interp, pad, sitk.sitkFloat32)


# ---------------------------------------------------------------------------
# numpy intensity + shape ops  (arrays are (D, H, W) from GetArrayFromImage)
# ---------------------------------------------------------------------------
def to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)


def normalize(arr: np.ndarray, percentiles=(0.5, 99.9),
              hard_clip: float | None = None) -> np.ndarray:
    """Robust scale to [-1, 1]. If hard_clip given, clip to [0, hard_clip] first."""
    if hard_clip is not None:
        lo, hi = 0.0, float(hard_clip)
    else:
        lo, hi = np.percentile(arr, percentiles)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr * 2.0 - 1.0


def center_crop_pad(arr: np.ndarray, size, pad_value=-1.0) -> np.ndarray:
    """Center crop or pad a (D, H, W) array to exactly `size`."""
    out = np.full(size, pad_value, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for a, s in zip(arr.shape, size):
        if a >= s:
            start = (a - s) // 2
            src_slices.append(slice(start, start + s))
            dst_slices.append(slice(0, s))
        else:
            start = (s - a) // 2
            src_slices.append(slice(0, a))
            dst_slices.append(slice(start, start + a))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


# ---------------------------------------------------------------------------
# High-level: a set of modality images -> aligned, normalized arrays
# ---------------------------------------------------------------------------
_INTENSITY_INTERP = sitk.sitkLinear
_LABEL_INTERP = sitk.sitkNearestNeighbor


def build_reference(images: dict, cfg: PreprocessConfig) -> sitk.Image:
    if cfg.reference == "iso":
        anchor = images.get("t2w") or next(iter(images.values()))
        return make_iso_reference(anchor, cfg.iso_spacing)
    key = "t2w" if cfg.reference == "t2w" else "dce"
    if key not in images:
        raise KeyError(f"reference='{cfg.reference}' needs a '{key}' image; got {list(images)}")
    return images[key]


def resample_case(images: dict, cfg: PreprocessConfig,
                  mask: sitk.Image | None = None, zones: sitk.Image | None = None):
    """Resample all modalities (and mask/zones) onto the reference grid.

    Returns (raw_arrays, mask_arr, zones_arr) with intensities un-normalized and
    uncropped, so the same arrays can be reused to fit a harmonizer.
    """
    ref = build_reference(images, cfg)
    raw = {name: to_array(resample_to_reference(img, ref, _INTENSITY_INTERP))
           for name, img in images.items()}
    mask_arr = (to_array(resample_to_reference(mask, ref, _LABEL_INTERP))
                if mask is not None else None)
    zones_arr = (to_array(resample_to_reference(zones, ref, _LABEL_INTERP))
                 if zones is not None else None)
    return raw, mask_arr, zones_arr


def process_case(images: dict, cfg: PreprocessConfig,
                 mask: sitk.Image | None = None, harmonizer=None,
                 zones: sitk.Image | None = None) -> dict:
    """images: {'t2w','dwi','adc','dce'(, ...)} -> {name: (D,H,W) float32 array}.

    Intensities are scaled to [-1, 1]. If a `harmonizer` is given it owns the
    intensity step (cross-scanner harmonization + scaling); otherwise a robust
    per-image percentile normalization is used. A 'mask' label image is
    resampled with nearest-neighbour and returned under key 'mask'. When a
    `zones` label image (1=TZ, 2=PZ) is given, a per-voxel 'zone_weight' map
    (TZ->tz_weight, PZ->pz_weight, else 1.0) is also returned for the ROI loss.
    """
    raw, mask_arr, zones_arr = resample_case(images, cfg, mask, zones)
    out = {}
    for name, arr in raw.items():
        if harmonizer is not None and harmonizer.has(name):
            arr = harmonizer.apply(name, arr)
        else:
            hard = cfg.adc_clip_value if name == "adc" else None
            arr = normalize(arr, cfg.clip_percentiles, hard)
        if cfg.spatial_size is not None:
            arr = center_crop_pad(arr, cfg.spatial_size, cfg.pad_value)
        out[name] = arr
    if mask_arr is not None:
        if cfg.spatial_size is not None:
            mask_arr = center_crop_pad(mask_arr, cfg.spatial_size, 0.0)
        out["mask"] = (mask_arr > 0.5).astype(np.float32)
    if zones_arr is not None:
        if cfg.spatial_size is not None:
            zones_arr = center_crop_pad(zones_arr, cfg.spatial_size, 0.0)
        zl = np.round(zones_arr)
        zw = np.ones_like(zones_arr, dtype=np.float32)
        zw[zl == 1] = cfg.tz_weight        # transition zone
        zw[zl == 2] = cfg.pz_weight        # peripheral zone (DCE matters here)
        out["zone_weight"] = zw
        out["zones"] = zl.astype(np.float32)   # label map (1=TZ, 2=PZ) for zone-split eval
    return out


def peak_phase_index(phase_imgs: list, mask_img: sitk.Image | None) -> int:
    """Argmax of mean intensity inside the prostate mask across dynamic phases.

    Mirrors the exploration-notebook logic. If no mask is available, falls back
    to the global mean per phase.
    """
    means = []
    mask_arr = to_array(mask_img).astype(bool) if mask_img is not None else None
    for img in phase_imgs:
        arr = to_array(img)
        if mask_arr is not None and mask_arr.shape == arr.shape and mask_arr.any():
            means.append(arr[mask_arr].mean())
        else:
            means.append(arr.mean())
    return int(np.argmax(means))
