"""MedVAE foundation-VAE first stage for the 3D LDM.

Wraps a frozen MedVAE (StanfordMIMI, pretrained on ~1.6M medical images) as a
drop-in first stage mirroring ``AutoencoderKL3D``'s interface. Motivation (Cajas
et al. 2604.12152): the first-stage VAE, not the diffusion objective, is the
dominant fidelity/realism constraint, and our from-scratch ``AutoencoderKL3D`` is
domain-matched but DATA-STARVED (~700 DCE volumes) vs MedVAE's 1.6M.

``medvae_4_1_3d``: 4x downsample, 1 latent channel, 3D -> latent grid matches our
VAE (8x48x48 for a 32x192x192 volume). Verified interface: ``encode(img)`` and
``decode(latent)`` return plain tensors (deterministic, no posterior); MedVAE
expects ``[0,1]`` input (roundtrip MSE 0.006 at [0,1] vs 0.024 at [-1,1]), so the
wrapper maps our ``[-1,1]`` DCE to/from ``[0,1]`` around encode/decode.

The frozen MedVAE weights (~600M) are stored WITHOUT nn.Module registration, so
they never bloat the LDM checkpoint -- they are external pretrained weights,
rebuilt from the HuggingFace cache at load time. They are consistently absent from
the state_dict on both save and load, so a strict load still matches.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Dirac:
    """Deterministic 'posterior' so the stage matches the VAE interface."""

    def __init__(self, z):
        self.z = z

    def sample(self):
        return self.z

    def mode(self):
        return self.z

    def kl(self):
        return torch.zeros((), device=self.z.device)


def medvae_latent_channels(model_name: str) -> int:
    """Latent channel count from the model name (medvae_<down>_<ch>_<dim>)."""
    try:
        return int(model_name.split("_")[2])
    except (IndexError, ValueError):
        return 1


class MedVAEFirstStage(nn.Module):
    def __init__(self, model_name: str = "medvae_4_1_3d", modality: str = "mri",
                 latent_channels: int | None = None):
        super().__init__()
        from medvae import MVAE          # lazy: only needed to instantiate, not import
        mvae = MVAE(model_name=model_name, modality=modality)
        mod = mvae if isinstance(mvae, nn.Module) else getattr(mvae, "model", mvae)
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        # store un-registered so the 600M frozen foundation weights stay out of the
        # LDM checkpoint (rebuilt from the HF cache at load; absent from state_dict
        # on both save and load, so strict load still matches).
        object.__setattr__(self, "_mvae", mvae)
        object.__setattr__(self, "_mod", mod)
        self.model_name = model_name
        self.latent_channels = latent_channels or medvae_latent_channels(model_name)
        self.scaling_factor = 1.0
        self.latent_shift = 0.0

    def _sync_device(self, x):
        if next(self._mod.parameters()).device != x.device:
            self._mod.to(x.device)

    @staticmethod
    def _to01(x):
        return (x + 1.0) * 0.5          # [-1,1] -> [0,1] (MedVAE's expected range)

    @staticmethod
    def _to11(x):
        return x * 2.0 - 1.0            # [0,1] -> [-1,1] (our tanh range)

    def _encode_one(self, x01):
        return self._mvae.encode(x01)

    def _decode_one(self, z):
        return self._mvae.decode(z)

    def encode(self, x) -> _Dirac:
        self._sync_device(x)
        x01 = self._to01(x)
        # MedVAE encode may assume batch 1 -> loop for safety (small overhead)
        if x01.shape[0] == 1:
            z = self._encode_one(x01)
        else:
            z = torch.cat([self._encode_one(x01[i:i + 1]) for i in range(x01.shape[0])], dim=0)
        return _Dirac(z)

    def decoder(self, z):
        """Grad-enabled decode of RAW latents (name mirrors AutoencoderKL3D.decoder);
        called by the anchor / adversarial heads. Grad flows to z through the frozen
        decoder even though its params are frozen."""
        self._sync_device(z)
        if z.shape[0] == 1:
            x01 = self._decode_one(z)
        else:
            x01 = torch.cat([self._decode_one(z[i:i + 1]) for i in range(z.shape[0])], dim=0)
        return self._to11(x01)

    def decode(self, z):
        return self.decoder(z / self.scaling_factor + self.latent_shift)

    def forward(self, x):
        z = self.encode(x).sample()
        return self.decoder(z), _Dirac(z)
