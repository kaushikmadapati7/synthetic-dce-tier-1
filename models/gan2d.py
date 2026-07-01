"""2D pixel-space conditional GAN (pix2pix-style) for the DCE **2D validation lever**.

Reproduces the collaborator's Pix2pix baseline: a single (T2w, DWI, ADC) axial
slice -> the DCE slice. Working in 2D pixel-space sidesteps the 3D VAE + latent
bottleneck and multiplies training data (~one sample per prostate-bearing slice),
letting us render at native in-plane resolution cheaply. Purpose: validate that
our methodology reproduces their crisp look + scatter-r (~0.5) -- i.e. confirm we
aren't doing anything wrong -- not to beat the 3D pipeline clinically.

The generator is a U-Net (skip connections); the discriminator is a conditional
PatchGAN. Hinge GAN losses are the dimension-agnostic ones from conditional_gan.
Inputs are HxW divisible by 32 (five stride-2 downsamples).
"""
import torch
import torch.nn as nn


def _down(ic, oc, norm=True):
    layers = [nn.Conv2d(ic, oc, 4, 2, 1)]
    if norm:
        layers.append(nn.InstanceNorm2d(oc))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _up(ic, oc, drop=False):
    layers = [nn.ConvTranspose2d(ic, oc, 4, 2, 1), nn.InstanceNorm2d(oc)]
    if drop:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class Generator2D(nn.Module):
    """U-Net generator (pix2pix): cond (3ch) -> DCE (1ch), tanh output. Decoder
    dropout supplies the stochasticity pix2pix uses in place of a noise input."""

    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.d1 = _down(in_ch, base, norm=False)   # H/2
        self.d2 = _down(base, base * 2)            # /4
        self.d3 = _down(base * 2, base * 4)        # /8
        self.d4 = _down(base * 4, base * 8)        # /16
        self.d5 = _down(base * 8, base * 8)        # /32
        self.u1 = _up(base * 8, base * 8, drop=True)
        self.u2 = _up(base * 16, base * 4, drop=True)
        self.u3 = _up(base * 8, base * 2)
        self.u4 = _up(base * 4, base)
        self.out = nn.Sequential(nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], 1))
        u3 = self.u3(torch.cat([u2, d3], 1))
        u4 = self.u4(torch.cat([u3, d2], 1))
        return self.out(torch.cat([u4, d1], 1))


class PatchDiscriminator2D(nn.Module):
    """Conditional PatchGAN: (DCE, cond) -> per-patch real/fake logits."""

    def __init__(self, in_ch=1, cond_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            _down(in_ch + cond_ch, base, norm=False),
            _down(base, base * 2),
            _down(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1), nn.InstanceNorm2d(base * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, dce, cond):
        return self.net(torch.cat([dce, cond], dim=1))
