"""Training loop for the 3D LDM with a flow-matching backbone."""
from ._ldm_base import train_ldm, load_ldm


def train_ldm_flow(args, train_loader, test_loader, criterion, device):
    return train_ldm(args, train_loader, test_loader, criterion, device, flow=True)


def load_ldm_flow(args, train_loader, test_loader, device):
    return load_ldm(args, train_loader, test_loader, device, flow=True)
