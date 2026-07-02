"""Fixed invertible 3D Haar wavelet transform as an LDM first stage.

A drop-in alternative to the learned ``AutoencoderKL3D``: instead of a trained
VAE, decompose the target DCE volume with a single- (or multi-) level 3D Haar
discrete wavelet *packet* transform. The transform is orthonormal, so it is

  - **lossless / perfectly invertible** -- no reconstruction ceiling, nothing to
    train (the VAE recon stage is where the LDM's prostate fidelity is currently
    capped), and
  - **linear**, so its inverse (IDWT) is exact *and differentiable* -- which lets
    the flow's trajectory-anchor term (`--anchor-weight`) supervise the decoded
    clean prediction in IMAGE space through an EXACT inverse, giving the flow the
    clean, ROI-weighted image-space signal the latent VAE path smears.

This is the FlowLet idea (Danese et al., arXiv 2601.05212): flow matching in the
wavelet domain. One level maps ``(B,1,D,H,W) -> (B,8,D/2,H/2,W/2)`` = 1 low-freq
(LLL) + 7 high-freq subbands; ``L`` levels give ``8**L`` channels at ``D/2**L``.
Subbands are per-channel standardized so every subband lands at ~unit scale (a
single global factor cannot -- the LLL approximation band dominates the details).

Interface mirrors ``AutoencoderKL3D`` exactly (encode -> posterior with .sample(),
decode, decoder, scaling_factor, latent_shift, latent_channels), so it drops into
``LDM_FlowMatching`` / ``LDM_DDPM`` with no changes to those classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_bank() -> torch.Tensor:
    """The 8 separable 3D Haar analysis filters, shape (8,1,2,2,2), orthonormal.

    Rows are the tensor products of low/high 1D filters over (D, H, W); flattened
    they form an 8x8 orthonormal matrix, so a strided (kernel=stride=2) conv is a
    critically-sampled orthonormal analysis bank and its transpose reconstructs
    exactly (Parseval)."""
    l = torch.tensor([1.0, 1.0]) / (2 ** 0.5)    # low-pass over [even, odd]
    h = torch.tensor([-1.0, 1.0]) / (2 ** 0.5)   # high-pass
    filts = []
    for fa in (l, h):            # depth  D  (LLL is the all-low first filter)
        for fb in (l, h):        # height H
            for fc in (l, h):    # width  W
                filts.append(torch.einsum("a,b,c->abc", fa, fb, fc))
    return torch.stack(filts).unsqueeze(1)        # (8,1,2,2,2)


class _Dirac:
    """Deterministic 'posterior' so the wavelet stage matches the VAE interface
    (``encode(x).sample()``). KL is zero -- there is nothing stochastic."""

    def __init__(self, z):
        self.z = z

    def sample(self):
        return self.z

    def mode(self):
        return self.z

    def kl(self):
        return torch.zeros((), device=self.z.device)


class WaveletFirstStage3D(nn.Module):
    def __init__(self, levels: int = 1, in_channels: int = 1):
        super().__init__()
        assert in_channels == 1, "wavelet first stage targets the 1-channel DCE"
        self.levels = int(levels)
        self.latent_channels = 8 ** self.levels
        self.register_buffer("bank", _haar_bank())            # (8,1,2,2,2)
        # per-subband standardization (persistent -> saved in the LDM checkpoint,
        # so eval-only reload restores them without re-fitting)
        self.register_buffer("coeff_mean", torch.zeros(self.latent_channels))
        self.register_buffer("coeff_std", torch.ones(self.latent_channels))
        # the LDM wrapper applies (z - latent_shift) * scaling_factor on top; keep
        # it ~identity since per-subband standardization already lands coeffs at
        # unit scale. _set_scaling_factor recomputes these to ~1 / ~0 (harmless).
        self.scaling_factor = 1.0
        self.latent_shift = 0.0

    # ---- forward / inverse Haar packet transform (no normalization) ----
    def _dwt(self, x):
        for _ in range(self.levels):
            c = x.shape[1]
            self._check_even(x.shape[2:])
            w = self.bank.repeat(c, 1, 1, 1, 1)               # (8c,1,2,2,2), groups=c
            x = F.conv3d(x, w, stride=2, groups=c)            # (B, 8c, ./2)
        return x

    def _idwt(self, z):
        for _ in range(self.levels):
            c = z.shape[1] // 8
            w = self.bank.repeat(c, 1, 1, 1, 1)
            z = F.conv_transpose3d(z, w, stride=2, groups=c)  # exact inverse of _dwt
        return z

    @staticmethod
    def _check_even(shape):
        if any(int(s) % 2 for s in shape):
            raise ValueError(
                f"wavelet stage needs spatial dims divisible by 2**levels; got a "
                f"sub-level shape {tuple(int(s) for s in shape)} with an odd axis. "
                f"Pick a --spatial-size whose D,H,W are divisible by 2**--wavelet-levels.")

    # ---- normalization ----
    def _stats(self):
        m = self.coeff_mean.view(1, -1, 1, 1, 1)
        s = self.coeff_std.view(1, -1, 1, 1, 1)
        return m, s

    def _normalize(self, coeffs):
        m, s = self._stats()
        return (coeffs - m) / s

    def _denormalize(self, z):
        m, s = self._stats()
        return z * s + m

    # ---- VAE-compatible interface ----
    def encode(self, x) -> _Dirac:
        return _Dirac(self._normalize(self._dwt(x)))

    def decoder(self, z):
        """Grad-enabled exact inverse (name mirrors ``AutoencoderKL3D.decoder``);
        called on RAW (unscaled) coeffs by the anchor / adversarial heads."""
        return self._idwt(self._denormalize(z))

    def decode(self, z):
        # invert the LDM's (z - shift) * scaling normalization, then IDWT
        return self.decoder(z / self.scaling_factor + self.latent_shift)

    def forward(self, x):
        z = self.encode(x).sample()
        return self.decoder(z), _Dirac(z)

    @torch.no_grad()
    def fit(self, loader, device, n_batches: int = 8):
        """Estimate per-subband mean/std from a few training targets so every
        subband is standardized to ~unit scale for the flow/diffusion objective."""
        self.eval()
        coeffs = []
        for i, batch in enumerate(loader):
            coeffs.append(self._dwt(batch["target"].to(device)))
            if i + 1 >= n_batches:
                break
        c = torch.cat(coeffs)                                 # (N, C, ./2)
        dims = (0, 2, 3, 4)
        self.coeff_mean.copy_(c.mean(dim=dims))
        self.coeff_std.copy_(c.std(dim=dims).clamp_min(1e-6))
        return self
