"""Tier-1 datasets: (T2w, DWI, ADC) -> peak-contrast DCE.

Two loaders, because the Bao cohort uses two different file layouts:

  CanonicalDCEDataset  -> 5 hospitals with fixed canonical names
                          exam_XXXX/{T2WI,ADC,DWI,DCE}.nii.gz
                          DCE is single-phase, so it is the target directly.

  DescriptorDCEDataset -> zhongyiyuan (GE LAVA-Flex), descriptor-based names
                          inputs globbed by descriptor; target is the peak phase
                          among the dynamic series, chosen by mean-intensity
                          argmax inside the prostate mask (notebook logic).

Both return:
    {
      "cond":   FloatTensor (3, D, H, W)  # stacked [T2w, DWI, ADC]
      "target": FloatTensor (1, D, H, W)  # peak-contrast DCE
      "mask":   FloatTensor (1, D, H, W)  # prostate ROI (zeros if unavailable)
      "id":     str
    }
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (PreprocessConfig, load_sitk, process_case,
                            resample_case, peak_phase_index)

CANONICAL_HOSPITALS = ["changshu", "fuyiyuan", "jiulong", "taizhou", "zhangjiagang"]
DESCRIPTOR_HOSPITALS = ["zhongyiyuan"]
TIER1_TEST_HOSPITALS = ["taizhou", "zhangjiagang"]  # candidate held-out

INPUT_KEYS = ("t2w", "dwi", "adc")


def _find_mask(exam_dir: Path):
    for p in exam_dir.glob("*mask*.nii*"):
        return p
    return None


def _stack_sample(arrays: dict, case_id: str, spatial_size) -> dict:
    cond = np.stack([arrays[k] for k in INPUT_KEYS], axis=0)
    target = arrays["dce"][None]
    if "mask" in arrays:
        mask = arrays["mask"][None]
    else:
        mask = np.zeros_like(target)
    return {
        "cond": torch.from_numpy(cond).float(),
        "target": torch.from_numpy(target).float(),
        "mask": torch.from_numpy(mask).float(),
        "id": case_id,
    }


# ---------------------------------------------------------------------------
# Canonical (5 hospitals, fixed names, single-phase DCE)
# ---------------------------------------------------------------------------
class CanonicalDCEDataset(Dataset):
    FILENAMES = {"t2w": "T2WI", "dwi": "DWI", "adc": "ADC", "dce": "DCE"}

    def __init__(self, root, hospitals=None, cfg: PreprocessConfig | None = None,
                 exam_glob="exam_*", require_all=True, harmonizer=None):
        self.root = Path(root)
        self.cfg = cfg or PreprocessConfig()
        self.harmonizer = harmonizer
        self.hospitals = hospitals or CANONICAL_HOSPITALS
        self.samples = []
        for hosp in self.hospitals:
            for exam in sorted((self.root / hosp).glob(exam_glob)):
                if not exam.is_dir():
                    continue
                paths = {k: self._resolve(exam, name) for k, name in self.FILENAMES.items()}
                if require_all and any(v is None for v in paths.values()):
                    continue
                self.samples.append((f"{hosp}/{exam.name}", paths, _find_mask(exam)))

    @staticmethod
    def _resolve(exam: Path, stem: str):
        for ext in (".nii.gz", ".nii"):
            p = exam / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def _load_images(self, i):
        case_id, paths, mask_path = self.samples[i]
        images = {k: load_sitk(p) for k, p in paths.items()}
        mask = load_sitk(mask_path) if mask_path else None
        return case_id, images, mask

    def __getitem__(self, i):
        case_id, images, mask = self._load_images(i)
        arrays = process_case(images, self.cfg, mask, self.harmonizer)
        return _stack_sample(arrays, case_id, self.cfg.spatial_size)

    def raw_modalities(self, i) -> dict:
        """Un-normalized resampled arrays per modality (for fitting a harmonizer)."""
        _, images, mask = self._load_images(i)
        raw, _ = resample_case(images, self.cfg, mask)
        return raw


# ---------------------------------------------------------------------------
# Descriptor-glob (zhongyiyuan, multi-phase, peak selection)
# ---------------------------------------------------------------------------
class DescriptorDCEDataset(Dataset):
    """Match modality files by descriptor substrings (case-insensitive regex).

    Defaults follow the slide examples; override `patterns` if names differ.
    """

    DEFAULT_PATTERNS = {
        "t2w": r"t2",
        "dwi": r"dwi",
        "adc": r"adc",
        "phase": r"(dyn|lava-flex\+c|ph\d+dyn)",  # dynamic DCE phases
    }

    def __init__(self, root, hospitals=None, cfg: PreprocessConfig | None = None,
                 exam_glob="exam_*", patterns=None, harmonizer=None):
        self.root = Path(root)
        self.cfg = cfg or PreprocessConfig()
        self.harmonizer = harmonizer
        self.hospitals = hospitals or DESCRIPTOR_HOSPITALS
        self.pat = {k: re.compile(v, re.IGNORECASE)
                    for k, v in (patterns or self.DEFAULT_PATTERNS).items()}
        self.samples = []
        for hosp in self.hospitals:
            for exam in sorted((self.root / hosp).glob(exam_glob)):
                if exam.is_dir():
                    self.samples.append((f"{hosp}/{exam.name}", exam))

    def _match(self, files, key):
        # exclude mask files; pick first match for single-volume modalities
        hits = [f for f in files if self.pat[key].search(f.name) and "mask" not in f.name.lower()]
        return hits

    @staticmethod
    def _phase_sort_key(path: Path):
        m = re.search(r"ph(\d+)", path.name, re.IGNORECASE)
        return int(m.group(1)) if m else -1  # +C water phase (no Ph#) sorts first

    def __len__(self):
        return len(self.samples)

    def _load_images(self, i):
        case_id, exam = self.samples[i]
        files = sorted(exam.glob("*.nii*"))

        inputs = {}
        for key in ("t2w", "dwi", "adc"):
            hits = self._match(files, key)
            if not hits:
                raise FileNotFoundError(f"{case_id}: no file matched '{key}' ({self.pat[key].pattern})")
            inputs[key] = hits[0]

        phase_files = sorted(self._match(files, "phase"), key=self._phase_sort_key)
        if not phase_files:
            raise FileNotFoundError(f"{case_id}: no dynamic DCE phase files matched")

        mask_path = _find_mask(exam)
        mask_img = load_sitk(mask_path) if mask_path else None

        phase_imgs = [load_sitk(p) for p in phase_files]
        peak = peak_phase_index(phase_imgs, mask_img)

        images = {k: load_sitk(v) for k, v in inputs.items()}
        images["dce"] = phase_imgs[peak]
        return case_id, images, mask_img

    def __getitem__(self, i):
        case_id, images, mask_img = self._load_images(i)
        arrays = process_case(images, self.cfg, mask_img, self.harmonizer)
        return _stack_sample(arrays, case_id, self.cfg.spatial_size)

    def raw_modalities(self, i) -> dict:
        """Un-normalized resampled arrays per modality (for fitting a harmonizer)."""
        _, images, mask_img = self._load_images(i)
        raw, _ = resample_case(images, self.cfg, mask_img)
        return raw


# ---------------------------------------------------------------------------
# Tier-1 convenience builder
# ---------------------------------------------------------------------------
def build_tier1_datasets(bao_root, cfg: PreprocessConfig | None = None, split="train",
                         harmonizer=None):
    """Return a ConcatDataset over the canonical + descriptor cohorts.

    split: "train"  -> all hospitals except the held-out test centers
           "test"   -> only the held-out test centers (taizhou, zhangjiagang)
           "all"    -> everything
    """
    from torch.utils.data import ConcatDataset

    cfg = cfg or PreprocessConfig()
    if split == "train":
        canon = [h for h in CANONICAL_HOSPITALS if h not in TIER1_TEST_HOSPITALS]
        desc = DESCRIPTOR_HOSPITALS
    elif split == "test":
        canon, desc = TIER1_TEST_HOSPITALS, []
    else:
        canon, desc = CANONICAL_HOSPITALS, DESCRIPTOR_HOSPITALS

    parts = []
    if canon:
        parts.append(CanonicalDCEDataset(bao_root, canon, cfg, harmonizer=harmonizer))
    if desc:
        parts.append(DescriptorDCEDataset(bao_root, desc, cfg, harmonizer=harmonizer))
    return ConcatDataset(parts)
