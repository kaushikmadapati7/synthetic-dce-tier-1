"""3D Latent Diffusion Model with a DDPM backbone.

Operates in the latent space of `AutoencoderKL3D`. The U-Net is trained to
predict the noise (epsilon) added to a latent at timestep t. Sampling uses the
standard ancestral DDPM update (a DDIM sampler is also provided).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import UNet3D


def make_beta_schedule(timesteps: int, schedule: str = "cosine"):
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps)
    # cosine schedule (Nichol & Dhariwal)
    steps = timesteps + 1
    s = 0.008
    x = torch.linspace(0, timesteps, steps)
    acp = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-4, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = a.gather(0, t).float()
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


class LDM_DDPM(nn.Module):
    def __init__(
        self,
        autoencoder=None,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        unet_kwargs: dict | None = None,
    ):
        super().__init__()
        self.autoencoder = autoencoder  # frozen first stage (optional at construction)
        self.unet = UNet3D(**(unet_kwargs or {}))
        self.timesteps = timesteps

        betas = make_beta_schedule(timesteps, beta_schedule)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("alphas_cumprod_prev", acp_prev)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))
        self.register_buffer("posterior_var", betas * (1.0 - acp_prev) / (1.0 - acp))

    # ---- first stage helpers ----
    @torch.no_grad()
    def encode(self, x):
        z = self.autoencoder.encode(x).sample()
        return (z - self.autoencoder.latent_shift) * self.autoencoder.scaling_factor

    @torch.no_grad()
    def decode(self, z):
        return self.autoencoder.decode(z)

    # ---- training ----
    def q_sample(self, z0, t, noise):
        return _extract(self.sqrt_acp, t, z0.shape) * z0 + \
            _extract(self.sqrt_one_minus_acp, t, z0.shape) * noise

    def loss(self, z0, cond=None, labels=None):
        """z0: a clean latent (encode your volume first, or pass latents directly)."""
        b = z0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=z0.device)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)
        pred = self.unet(zt, t.float(), cond=cond, labels=labels)
        return F.mse_loss(pred, noise)

    # ---- sampling ----
    @torch.no_grad()
    def p_sample(self, zt, t, cond=None, labels=None):
        eps = self.unet(zt, t.float(), cond=cond, labels=labels)
        acp = _extract(self.alphas_cumprod, t, zt.shape)
        beta = _extract(self.betas, t, zt.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_acp, t, zt.shape)
        alpha = 1.0 - beta
        mean = (zt - beta / sqrt_one_minus * eps) / torch.sqrt(alpha)
        if (t == 0).all():
            return mean
        var = _extract(self.posterior_var, t, zt.shape)
        return mean + torch.sqrt(var) * torch.randn_like(zt)

    @torch.no_grad()
    def sample(self, shape, device, cond=None, labels=None, decode=True):
        zt = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            zt = self.p_sample(zt, t, cond=cond, labels=labels)
        return self.decode(zt) if decode and self.autoencoder is not None else zt

    @torch.no_grad()
    def ddim_sample(self, shape, device, steps=50, eta=0.0, cond=None, labels=None,
                    decode=True, x0_clamp=0.0):
        seq = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        zt = torch.randn(shape, device=device)
        for i in range(steps):
            t = torch.full((shape[0],), seq[i], device=device, dtype=torch.long)
            eps = self.unet(zt, t.float(), cond=cond, labels=labels)
            acp_t = _extract(self.alphas_cumprod, t, zt.shape)
            z0 = (zt - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            # Bound the x0 estimate: at high-noise steps alphas_cumprod -> 0, so
            # 1/sqrt(acp_t) blows up any eps error into a huge z0 that corrupts the
            # trajectory (drift -> saturated/garbage decode; diag_ddpm shows unbounded
            # sampling gives L1~1.4 vs ~0.24 when bounded). The centered/scaled latents
            # are empirically ~[-2.4, 1.4], so a clamp of ~3 keeps valid values and
            # kills the drift. 0.0 disables (do NOT use for eval).
            if x0_clamp:
                z0 = z0.clamp(-x0_clamp, x0_clamp)
            if i < steps - 1:
                t_next = torch.full((shape[0],), seq[i + 1], device=device, dtype=torch.long)
                acp_n = _extract(self.alphas_cumprod, t_next, zt.shape)
                sigma = eta * torch.sqrt((1 - acp_n) / (1 - acp_t) * (1 - acp_t / acp_n))
                zt = torch.sqrt(acp_n) * z0 + \
                    torch.sqrt((1 - acp_n - sigma ** 2).clamp(min=0)) * eps + \
                    sigma * torch.randn_like(zt)
            else:
                zt = z0
        return self.decode(zt) if decode and self.autoencoder is not None else zt
