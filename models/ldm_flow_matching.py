"""3D Latent Diffusion Model with a (rectified) flow-matching backbone.

Same latent space and U-Net as the DDPM variant, but trained with conditional
flow matching on straight-line (optimal-transport) paths:

    z_t = (1 - t) * z0 + t * noise          t in [0, 1]
    target velocity  v = dz_t/dt = noise - z0

The network predicts v(z_t, t); sampling integrates the ODE  dz/dt = v  from
t=1 (noise) back to t=0 (data) with an Euler or Heun solver.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import UNet3D
from ._cfg import CFGMixin, roi_weighted_mse


class LDM_FlowMatching(CFGMixin, nn.Module):
    def __init__(
        self,
        autoencoder=None,
        unet_kwargs: dict | None = None,
        time_scale: float = 1000.0,  # scales t in [0,1] before the sinusoidal embedding
        sigma_min: float = 0.0,      # >0 enables a small noise floor on the path
        cfg_dropout: float = 0.0,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = UNet3D(**(unet_kwargs or {}))
        self.time_scale = time_scale
        self.sigma_min = sigma_min
        self.cfg_dropout = cfg_dropout

    # ---- first stage helpers ----
    @torch.no_grad()
    def encode(self, x):
        z = self.autoencoder.encode(x).sample()
        return (z - self.autoencoder.latent_shift) * self.autoencoder.scaling_factor

    @torch.no_grad()
    def decode(self, z):
        return self.autoencoder.decode(z)

    # ---- training ----
    def loss(self, z0, cond=None, labels=None, mask=None, roi_weight=1.0,
             anchor_image=None, anchor_mask=None, anchor_criterion=None, anchor_weight=0.0):
        """z0: clean latent. t=0 -> data, t=1 -> noise. ``mask`` (latent-grid
        prostate mask) + roi_weight give the latent objective ROI emphasis.

        Trajectory anchoring (adapted from FlowMI for a noise->data flow): when
        anchor_weight>0, decode the predicted CLEAN latent z0_hat = z_t - t*v and
        supervise it in IMAGE space with anchor_criterion (L1/SSIM/perceptual, ROI-
        weighted). This gives the flow the same direct image-space prostate signal
        the GAN's recon loss has -- the latent-MSE objective alone never sees the
        decoder, which is why the LDM blurs."""
        b = z0.shape[0]
        t = torch.rand(b, device=z0.device)
        noise = torch.randn_like(z0)
        tb = t.view(b, *([1] * (z0.dim() - 1)))

        zt = (1.0 - (1.0 - self.sigma_min) * tb) * z0 + tb * noise
        target = noise - (1.0 - self.sigma_min) * z0  # dz_t/dt

        pred = self.unet(zt, t * self.time_scale, cond=self._drop_cond(cond), labels=labels)
        loss = roi_weighted_mse(pred, target, mask, roi_weight)

        if anchor_weight > 0 and anchor_image is not None and self.autoencoder is not None:
            z0_hat = zt - tb * pred                       # predicted clean latent (holds for any sigma_min)
            img = self.autoencoder.decoder(z0_hat / self.autoencoder.scaling_factor
                                           + self.autoencoder.latent_shift)  # grad-enabled decode
            loss = loss + anchor_weight * anchor_criterion(img, anchor_image, anchor_mask)[0]
        return loss

    # ---- sampling: integrate the probability-flow ODE from noise (t=1) to data (t=0) ----
    @torch.no_grad()
    def sample(self, shape, device, steps=50, cond=None, labels=None,
               solver="heun", decode=True, guidance_scale=1.0):
        z = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            dt = t_next - t  # negative
            tb = torch.full((shape[0],), t, device=device)
            v = self._guided(z, tb * self.time_scale, cond, labels, guidance_scale)
            if solver == "euler":
                z = z + dt * v
            else:  # heun (2nd order)
                z_pred = z + dt * v
                tb_n = torch.full((shape[0],), t_next, device=device)
                v_next = self._guided(z_pred, tb_n * self.time_scale, cond, labels, guidance_scale)
                z = z + dt * 0.5 * (v + v_next)
        return self.decode(z) if decode and self.autoencoder is not None else z
