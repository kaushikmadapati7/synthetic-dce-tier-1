"""Tier-1 datasets: (T2w, DWI, ADC) -> peak-contrast DCE.

The on-disk source is the Bao_DCE **silver** layer on CHPC, which splits images
and masks into two parallel trees keyed by <center>/<subject>:

    <root>/Image_volumes/<center>/<subject>/   registered volumes (T2 ref space)
    <root>/Prostate_masks/<center>/<subject>/  prostate_mask.nii.gz

Two loaders, because the cohort mixes single- and multi-phase DCE:

  CanonicalDCEDataset  -> single-phase centers (changshu, fuyiyuan, jiulong).
                          DCE is one registered file, used as the target directly.
                          files: T2WI, ADC_to_T2WI, DWI_to_T2WI, DCE_to_T2WI

  DescriptorDCEDataset -> zhongyiyuan (GE LAVA-Flex), multi-phase DCE. The target
                          is the peak post-contrast phase among DCE_ph*_to_T2W,
                          chosen by mean-intensity argmax inside the prostate mask.
                          files: T2W, ADC_to_T2W, DWI_to_T2W, DCE_ph1..N_to_T2W

Both return:
    {
      "cond":   FloatTensor (3, D, H, W)  # stacked [T2w, DWI, ADC]
      "target": FloatTensor (1, D, H, W)  # peak-contrast DCE
      "mask":   FloatTensor (1, D, H, W)  # prostate ROI (zeros if unavailable)
      "id":     str
    }

NOTE on centers: only changshu/fuyiyuan/jiulong/zhongyiyuan exist in the silver
tree; taizhou/zhangjiagang are bronze-only (raw, unregistered, no masks). The
default Tier-1 held-out test is therefore `jiulong` (override via --test-hospitals).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from .preprocessing import (PreprocessConfig, load_sitk, process_case,
                            resample_case, peak_phase_index)

log = logging.getLogger("tier1")

# Silver centers only (taizhou/zhangjiagang are bronze-only and excluded here).
CANONICAL_HOSPITALS = ["changshu", "fuyiyuan", "jiulong"]
DESCRIPTOR_HOSPITALS = ["zhongyiyuan"]
TIER1_TEST_HOSPITALS = ["jiulong"]  # held-out test center (see module docstring)

INPUT_KEYS = ("t2w", "dwi", "adc")
DCE_KEY = "dce"

# Default silver sub-trees under the dataset root.
IMAGE_SUBDIR = "Image_volumes"
MASK_SUBDIR = "Prostate_masks"

# Candidate on-disk stems per modality. Single-phase centers use the `_to_T2WI`
# suffix; zhongyiyuan uses `_to_T2W`; bare names are kept as a bronze fallback.
MODALITY_STEMS = {
    "t2w": ["T2WI", "T2W"],
    "adc": ["ADC_to_T2WI", "ADC_to_T2W", "ADC"],
    "dwi": ["DWI_to_T2WI", "DWI_to_T2W", "DWI"],
    "dce": ["DCE_to_T2WI", "DCE_to_T2W", "DCE"],
}


def _resolve_stem(exam_dir: Path, candidates):
    """First existing `<candidate>.nii(.gz)` in `exam_dir`, trying stems in order."""
    for stem in candidates:
        for ext in (".nii.gz", ".nii"):
            p = exam_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _find_mask(exam_dir: Path):
    """In-dir mask fallback (bronze / co-located masks)."""
    for p in exam_dir.glob("*mask*.nii*"):
        return p
    return None


def _silver_mask(mask_root: Path, hosp: str, subject: str):
    """Locate the prostate mask in the parallel Prostate_masks tree."""
    d = mask_root / hosp / subject
    if not d.is_dir():
        return None
    for name in ("prostate_mask.nii.gz", "prostate_mask.nii"):
        p = d / name
        if p.exists():
            return p
    # any *mask* file, but never the zonal segmentation
    for p in sorted(d.glob("*mask*.nii*")):
        if "zone" not in p.name.lower():
            return p
    return None


def _silver_zones(mask_root: Path, hosp: str, subject: str):
    """Locate the PZ/TZ zonal segmentation in the parallel Prostate_masks tree."""
    d = mask_root / hosp / subject
    if not d.is_dir():
        return None
    for name in ("prostate_zones.nii.gz", "prostate_zones.nii"):
        p = d / name
        if p.exists():
            return p
    return None


def _stack_sample(arrays: dict, case_id: str, spatial_size) -> dict:
    cond = np.stack([arrays[k] for k in INPUT_KEYS], axis=0)
    target = arrays[DCE_KEY][None]
    if "mask" in arrays:
        mask = arrays["mask"][None]
    else:
        mask = np.zeros_like(target)
    # zone_weight defaults to all-ones (no zone effect) when zones are unavailable
    zw = arrays["zone_weight"][None] if "zone_weight" in arrays else np.ones_like(target)
    return {
        "cond": torch.from_numpy(cond).float(),
        "target": torch.from_numpy(target).float(),
        "mask": torch.from_numpy(mask).float(),
        "zone_weight": torch.from_numpy(zw).float(),
        "id": case_id,
    }


# ---------------------------------------------------------------------------
# Canonical (single-phase DCE centers, silver layout)
# ---------------------------------------------------------------------------
class CanonicalDCEDataset(Dataset):
    MODS = ("t2w", "dwi", "adc", "dce")

    def __init__(self, root, hospitals=None, cfg: PreprocessConfig | None = None,
                 harmonizer=None, image_subdir=IMAGE_SUBDIR, mask_subdir=MASK_SUBDIR,
                 subject_glob="*", require_all=True):
        self.root = Path(root)
        self.cfg = cfg or PreprocessConfig()
        self.harmonizer = harmonizer
        self.hospitals = hospitals or CANONICAL_HOSPITALS
        self.img_root = self.root / image_subdir
        self.mask_root = self.root / mask_subdir
        self.samples = []
        for hosp in self.hospitals:
            hosp_dir = self.img_root / hosp
            found = skipped = 0
            for subj in sorted(hosp_dir.glob(subject_glob)):
                if not subj.is_dir():
                    continue
                paths = {k: _resolve_stem(subj, MODALITY_STEMS[k]) for k in self.MODS}
                if require_all and any(v is None for v in paths.values()):
                    skipped += 1
                    continue
                mask = _silver_mask(self.mask_root, hosp, subj.name) or _find_mask(subj)
                zones = _silver_zones(self.mask_root, hosp, subj.name)
                self.samples.append((f"{hosp}/{subj.name}", paths, mask, zones))
                found += 1
            if found == 0:
                log.warning(f"[canonical] {hosp}: 0 usable subjects under {hosp_dir} "
                            f"({skipped} skipped for missing modalities)")
            else:
                log.info(f"[canonical] {hosp}: {found} subjects ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def _load_images(self, i):
        case_id, paths, mask_path, zones_path = self.samples[i]
        images = {k: load_sitk(p) for k, p in paths.items()}
        mask = load_sitk(mask_path) if mask_path else None
        zones = load_sitk(zones_path) if zones_path else None
        return case_id, images, mask, zones

    def __getitem__(self, i):
        case_id, images, mask, zones = self._load_images(i)
        arrays = process_case(images, self.cfg, mask, self.harmonizer, zones=zones)
        return _stack_sample(arrays, case_id, self.cfg.spatial_size)

    def raw_modalities(self, i) -> dict:
        """Un-normalized resampled arrays per modality (for fitting a harmonizer)."""
        _, images, mask, _ = self._load_images(i)
        raw, _, _ = resample_case(images, self.cfg, mask)
        return raw


# ---------------------------------------------------------------------------
# Descriptor (zhongyiyuan, multi-phase, peak selection)
# ---------------------------------------------------------------------------
class DescriptorDCEDataset(Dataset):
    """Multi-phase DCE: pick the peak post-contrast dynamic phase per subject.

    Inputs (T2w/ADC/DWI) resolve by the shared MODALITY_STEMS; the dynamic phases
    are globbed by `phase_glob` (default the silver `DCE_ph*_to_T2W*` series) and
    ordered by phase number. The pre-contrast baseline (`DCE_pre`) is excluded.
    """

    def __init__(self, root, hospitals=None, cfg: PreprocessConfig | None = None,
                 harmonizer=None, image_subdir=IMAGE_SUBDIR, mask_subdir=MASK_SUBDIR,
                 subject_glob="*", phase_glob="DCE_ph*_to_T2W*.nii*", phase_select="early"):
        self.root = Path(root)
        self.cfg = cfg or PreprocessConfig()
        self.harmonizer = harmonizer
        self.hospitals = hospitals or DESCRIPTOR_HOSPITALS
        self.img_root = self.root / image_subdir
        self.mask_root = self.root / mask_subdir
        self.phase_glob = phase_glob
        # which dynamic phase becomes the target: "peak" (mask-mean argmax),
        # "early" (first post-contrast = ph1), or an int index into the sorted phases.
        self.phase_select = phase_select
        self.samples = []
        for hosp in self.hospitals:
            hosp_dir = self.img_root / hosp
            found = skipped = 0
            skipped_ids = []
            for subj in sorted(hosp_dir.glob(subject_glob)):
                if not subj.is_dir():
                    continue
                # only keep subjects that have all inputs AND >=1 dynamic phase, so a
                # malformed subject is dropped here (with a count) rather than crashing
                # a DataLoader worker mid-epoch.
                inputs_ok = all(_resolve_stem(subj, MODALITY_STEMS[k]) for k in INPUT_KEYS)
                has_phase = any(subj.glob(self.phase_glob))
                if inputs_ok and has_phase:
                    self.samples.append((f"{hosp}/{subj.name}", subj, hosp))
                    found += 1
                else:
                    skipped += 1
                    if len(skipped_ids) < 8:
                        skipped_ids.append(subj.name)
            note = f", e.g. {skipped_ids}" if skipped_ids else ""
            log.info(f"[descriptor] {hosp}: {found} subjects ({skipped} skipped{note}) under {hosp_dir}")

    @staticmethod
    def _phase_sort_key(path: Path):
        m = re.search(r"ph(\d+)", path.name, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    def __len__(self):
        return len(self.samples)

    def _load_images(self, i):
        case_id, subj, hosp = self.samples[i]

        inputs = {k: _resolve_stem(subj, MODALITY_STEMS[k]) for k in INPUT_KEYS}
        missing = [k for k, v in inputs.items() if v is None]
        if missing:
            raise FileNotFoundError(f"{case_id}: missing input modalities {missing} in {subj}")

        phase_files = sorted(subj.glob(self.phase_glob), key=self._phase_sort_key)
        if not phase_files:
            raise FileNotFoundError(f"{case_id}: no dynamic DCE phases matched "
                                    f"'{self.phase_glob}' in {subj}")

        mask_path = _silver_mask(self.mask_root, hosp, subj.name) or _find_mask(subj)
        mask_img = load_sitk(mask_path) if mask_path else None
        zones_path = _silver_zones(self.mask_root, hosp, subj.name)
        zones_img = load_sitk(zones_path) if zones_path else None

        phase_imgs = [load_sitk(p) for p in phase_files]
        idx = self._select_phase(phase_imgs, mask_img)

        images = {k: load_sitk(v) for k, v in inputs.items()}
        images[DCE_KEY] = phase_imgs[idx]
        return case_id, images, mask_img, zones_img

    def _select_phase(self, phase_imgs, mask_img) -> int:
        sel = self.phase_select
        if sel == "peak":
            return peak_phase_index(phase_imgs, mask_img)
        if sel == "early":
            return 0  # phases are sorted ascending, so 0 == ph1 (first post-contrast)
        return max(0, min(int(sel), len(phase_imgs) - 1))

    def __getitem__(self, i):
        case_id, images, mask_img, zones_img = self._load_images(i)
        arrays = process_case(images, self.cfg, mask_img, self.harmonizer, zones=zones_img)
        return _stack_sample(arrays, case_id, self.cfg.spatial_size)

    def raw_modalities(self, i) -> dict:
        """Un-normalized resampled arrays per modality (for fitting a harmonizer)."""
        _, images, mask_img, _ = self._load_images(i)
        raw, _, _ = resample_case(images, self.cfg, mask_img)
        return raw


# ---------------------------------------------------------------------------
# Tier-1 convenience builder
# ---------------------------------------------------------------------------
class _EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError("empty dataset")


def build_tier1_datasets(bao_root, cfg: PreprocessConfig | None = None, split="train",
                         harmonizer=None, test_hospitals=None,
                         image_subdir=IMAGE_SUBDIR, mask_subdir=MASK_SUBDIR,
                         dce_phase="early"):
    """ConcatDataset over the canonical + descriptor cohorts for a given split.

    split: "train" -> all silver hospitals except the held-out test centers
           "test"  -> only the held-out test centers
           "all"   -> everything

    `test_hospitals` defaults to TIER1_TEST_HOSPITALS (jiulong). Returns an empty
    dataset (len 0) if a split has no hospitals, so callers can guard on len().
    """
    cfg = cfg or PreprocessConfig()
    test_hospitals = list(TIER1_TEST_HOSPITALS if test_hospitals is None else test_hospitals)

    if split == "train":
        canon = [h for h in CANONICAL_HOSPITALS if h not in test_hospitals]
        desc = [h for h in DESCRIPTOR_HOSPITALS if h not in test_hospitals]
    elif split == "test":
        canon = [h for h in CANONICAL_HOSPITALS if h in test_hospitals]
        desc = [h for h in DESCRIPTOR_HOSPITALS if h in test_hospitals]
    else:
        canon, desc = CANONICAL_HOSPITALS, DESCRIPTOR_HOSPITALS

    kw = dict(image_subdir=image_subdir, mask_subdir=mask_subdir, harmonizer=harmonizer)
    parts = []
    if canon:
        parts.append(CanonicalDCEDataset(bao_root, canon, cfg, **kw))
    if desc:
        parts.append(DescriptorDCEDataset(bao_root, desc, cfg, phase_select=dce_phase, **kw))
    return ConcatDataset(parts) if parts else _EmptyDataset()
