"""Per-model training loops (TRAINERS) and checkpoint loaders (LOADERS)."""
from .gan import train_gan, load_gan
from .ldm_ddpm import train_ldm_ddpm, load_ldm_ddpm
from .ldm_flow import train_ldm_flow, load_ldm_flow

TRAINERS = {
    "gan": train_gan,
    "ldm_ddpm": train_ldm_ddpm,
    "ldm_flow": train_ldm_flow,
}

# eval-only: rebuild the model from a checkpoint and return (model, gen) without training
LOADERS = {
    "gan": load_gan,
    "ldm_ddpm": load_ldm_ddpm,
    "ldm_flow": load_ldm_flow,
}

__all__ = ["TRAINERS", "LOADERS", "train_gan", "train_ldm_ddpm", "train_ldm_flow",
           "load_gan", "load_ldm_ddpm", "load_ldm_flow"]
