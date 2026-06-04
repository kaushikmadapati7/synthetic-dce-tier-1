"""Per-model training loops, dispatched by the TRAINERS registry."""
from .gan import train_gan
from .ldm_ddpm import train_ldm_ddpm
from .ldm_flow import train_ldm_flow

TRAINERS = {
    "gan": train_gan,
    "ldm_ddpm": train_ldm_ddpm,
    "ldm_flow": train_ldm_flow,
}

__all__ = ["TRAINERS", "train_gan", "train_ldm_ddpm", "train_ldm_flow"]
