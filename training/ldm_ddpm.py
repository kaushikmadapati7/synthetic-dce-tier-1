"""Training loop for the 3D LDM with a DDPM backbone."""
from ._ldm_base import train_ldm, load_ldm


def train_ldm_ddpm(args, train_loader, val_loader, test_loader, criterion, device):
    return train_ldm(args, train_loader, val_loader, test_loader, criterion, device, flow=False)


def load_ldm_ddpm(args, train_loader, test_loader, device):
    return load_ldm(args, train_loader, test_loader, device, flow=False)
