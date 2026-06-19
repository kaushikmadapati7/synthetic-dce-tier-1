"""Classifier-free guidance (CFG) helpers shared by the LDM backbones.

CFG trains one network to model both the conditional and the unconditional
score/velocity, then at sampling extrapolates away from the unconditional one to
sharpen adherence to the condition:

    pred = pred_uncond + w * (pred_cond - pred_uncond)

`w == 1` recovers the plain conditional model (no guidance, single forward);
`w > 1` amplifies the conditioning. For our spatial, channel-concatenated
conditioning the "null" condition is simply a zero cond — no extra parameters,
so enabling CFG does not change a checkpoint's state_dict (older non-CFG
checkpoints still load, and `cfg_dropout=0 / guidance_scale=1` is a byte-for-byte
no-op). A learned null embedding is a possible later refinement.
"""
import torch


class CFGMixin:
    cfg_dropout: float = 0.0  # training: prob a sample's cond is replaced by null

    def _drop_cond(self, cond):
        """Training-time conditioning dropout: zero out a random subset of the
        batch's conditioning so the net also learns the unconditional path."""
        if cond is None or self.cfg_dropout <= 0:
            return cond
        b = cond.shape[0]
        keep = (torch.rand(b, device=cond.device) >= self.cfg_dropout)
        return cond * keep.view(b, *([1] * (cond.dim() - 1))).to(cond.dtype)

    def _guided(self, zt, t, cond, labels, guidance_scale):
        """One denoiser evaluation with CFG. Batches the cond/uncond passes into
        a single forward. Falls back to a plain call when guidance is off."""
        if cond is None or guidance_scale == 1.0:
            return self.unet(zt, t, cond=cond, labels=labels)
        b = zt.shape[0]
        z2 = torch.cat([zt, zt], dim=0)
        t2 = torch.cat([t, t], dim=0)
        c2 = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        l2 = torch.cat([labels, labels], dim=0) if labels is not None else None
        pred_c, pred_u = self.unet(z2, t2, cond=c2, labels=l2).chunk(2, dim=0)
        return pred_u + guidance_scale * (pred_c - pred_u)
