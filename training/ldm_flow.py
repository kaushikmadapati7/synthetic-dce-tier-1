"""Training loop for the 3D LDM with a flow-matching backbone."""
from ._ldm_base import train_ldm


def train_ldm_flow(args, train_loader, test_loader, criterion, device):
    return train_ldm(args, train_loader, test_loader, criterion, device, flow=True)
