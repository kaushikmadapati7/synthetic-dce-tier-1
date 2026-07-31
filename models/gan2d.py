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
    """Nearest upsample + 3x3 conv, NOT ConvTranspose2d (checkerboard).

    The 3D counterpart of this decoder measurably stamped a period-2 pattern into
    its output (runs/v2_3d_gan: lag-1 autocorrelation of the first difference
    -0.60, vs +0.81 for real DCE and -0.50 for white noise). This 2D path uses the
    identical ConvTranspose(4,2,1) construction, so it is fixed on the same
    grounds; note it has NOT been measured directly here, because the saved 2D
    volumes are restored to native resolution and that 2x interpolation would
    smooth a pixel-scale artifact away before it could be detected.

    Shape is unchanged: ConvTranspose2d(4,2,1) and upsample(2)+conv3(p=1) both n -> 2n.
    """
    layers = [nn.Upsample(scale_factor=2, mode="nearest"),
              nn.Conv2d(ic, oc, 3, 1, 1), nn.InstanceNorm2d(oc)]
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
        # output layer: nothing follows it to smooth a checkerboard, so it matters most
        self.out = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(base * 2, out_ch, 3, 1, 1), nn.Tanh())

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
