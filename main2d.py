"""2D pixel-space pix2pix entry point -- the reproducibility / data-scale lever.

Reuses the full 3D pipeline (data, harmonization, loss, metrics) and only changes
two things: the data is viewed as 2D axial slices (SliceDCEDataset) and the model
is a 2D pix2pix GAN (models/gan2d). 2D slices are wrapped to (C,1,H,W) so the 3D
CustomLoss / eval_metrics apply unchanged. Purpose: confirm we reproduce the
collaborator's crisp look + scatter-r (~0.5) at native in-plane resolution, and
test whether slice-scale data (~20x samples) changes the GAN-vs-flow picture.

Run:  python -m tier1_static.main2d --data-root ... --output-dir runs/px2d \
        --spatial-size 32 256 256 --epochs 100 --batch-size 16
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .main import parse_args, build_data, make_criterion, set_seed, setup_logging
from .models import d_hinge_loss, g_hinge_loss
from .models.gan2d import Generator2D, PatchDiscriminator2D
from .models.flow2d import FlowMatching2D
from .data.slice_dataset import SliceDCEDataset
from .metrics import eval_metrics, aggregate, roi_p75, pearson
from .eval import save_samples

log = logging.getLogger("tier1")


def _w5(t):                       # (B,C,H,W) -> (B,C,1,H,W) so 3D loss/metrics apply
    return t.unsqueeze(2) if t is not None else None


def _slice_loader(loader3d, depth, batch_size, workers, shuffle):
    ds = SliceDCEDataset(loader3d.dataset, depth)
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=shuffle)


@torch.no_grad()
def evaluate2d(gen, loader, device, tag):
    if loader is None:
        return {}, None
    per, p75r, p75p, first = [], [], [], None
    for b in loader:
        cond = b["cond"].to(device); target = b["target"].to(device); mask = b["mask"].to(device)
        zones = b["zones"].to(device) if "zones" in b else None
        pred = gen(cond).clamp(-1, 1)
        per.append(eval_metrics(_w5(pred), _w5(target), _w5(mask), _w5(zones)))
        for i in range(pred.size(0)):
            rr = roi_p75(_w5(target[i:i+1]), _w5(mask[i:i+1])); pp = roi_p75(_w5(pred[i:i+1]), _w5(mask[i:i+1]))
            if rr is not None and pp is not None:
                p75r.append(rr); p75p.append(pp)
        if first is None:
            first = (cond.cpu(), target.cpu(), pred.cpu(), mask.cpu(), b["id"][0])
    m = aggregate(per)
    pc = pearson(p75r, p75p)
    if pc is not None:
        m["p75_corr"] = pc
    log.info(f"{tag} metrics: {json.dumps({k: round(v, 4) for k, v in m.items()})}")
    return m, first


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"device={device}  model=pix2pix-2D")

    train3d, val3d, test3d = build_data(args)
    depth = args.spatial_size[0]
    train = _slice_loader(train3d, depth, args.batch_size, args.num_workers, True)
    val = _slice_loader(val3d, depth, args.batch_size, args.num_workers, False)
    test = _slice_loader(test3d, depth, args.batch_size, args.num_workers, False)
    log.info(f"2D slices/epoch: train {len(train.dataset)}  "
             f"val {len(val.dataset) if val else 0}  test {len(test.dataset) if test else 0}")

    criterion = make_criterion(args, device)
    is_flow = args.model != "gan"
    name = "flow2d" if is_flow else "gan2d"

    if is_flow:                                   # pixel-space 2D CFM
        model = FlowMatching2D(cond_ch=3, base=args.base_ch,
                               source=getattr(args, "flow_source", "noise")).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        gen = lambda c: model.sample(c, steps=args.sample_steps).clamp(-1, 1)
        log.info(f"2D flow: {sum(p.numel() for p in model.parameters())/1e6:.1f}M source={model.source}")
    else:                                         # pix2pix 2D GAN
        model = Generator2D(in_ch=3, out_ch=1, base=args.base_ch).to(device)
        disc = PatchDiscriminator2D(in_ch=1, cond_ch=3, base=args.base_ch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
        gen = lambda c: model(c)
        log.info(f"2D GAN: G={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
                 f"D={sum(p.numel() for p in disc.parameters())/1e6:.1f}M")

    best = float("-inf")
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); agg = {}
        for b in train:
            cond = b["cond"].to(device); real = b["target"].to(device); mask = b["mask"].to(device)
            zw = b["zone_weight"].to(device) if "zone_weight" in b else None
            if is_flow:
                loss = model.loss(real, cond, mask=_w5(mask), roi_weight=args.roi_weight)
                opt.zero_grad(); loss.backward(); opt.step()
                agg["diff"] = agg.get("diff", 0.0) + float(loss.detach())
            else:
                fake = model(cond)
                d_loss = d_hinge_loss(disc(real, cond), disc(fake.detach(), cond))
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                g_adv = g_hinge_loss(disc(fake, cond))
                rec, parts = criterion(_w5(fake), _w5(real), _w5(mask), zone_weight=_w5(zw))
                g_loss = args.adv_weight * g_adv + rec
                opt.zero_grad(); g_loss.backward(); opt.step()
                for k, v in {"d": float(d_loss.detach()), "g": float(g_loss.detach()),
                             "adv": float(g_adv.detach()), **parts}.items():
                    agg[k] = agg.get(k, 0.0) + v
        msg = "  ".join(f"{k}={v/max(1,len(train)):.4f}" for k, v in agg.items())
        log.info(f"[epoch {epoch+1}/{args.epochs}] {msg}  ({time.time()-t0:.1f}s)")

        if val is not None and ((epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs):
            model.eval()
            m, _ = evaluate2d(gen, val, device, "VAL")
            score = m.get("ssim_roi", m.get("ssim", float("-inf")))
            if score > best:
                best = score
                torch.save(model.state_dict(), out / f"{name}_best.pt")
                log.info(f"  ** new best 2D {name}: val_ssim_roi={score:.4f} -> {name}_best.pt")
    torch.save(model.state_dict(), out / f"{name}_last.pt")

    # final: load best, eval test + val, save an in-distribution montage
    if (out / f"{name}_best.pt").exists():
        model.load_state_dict(torch.load(out / f"{name}_best.pt", map_location=device))
    model.eval()
    evaluate2d(gen, test, device, "TEST")
    _, first = evaluate2d(gen, val, device, "VAL")
    if first is not None:
        cond, target, pred, mask, cid = first
        save_samples(out / "samples", _w5(cond), _w5(target), _w5(pred), _w5(mask),
                     "indist_" + cid, montage_name="montage_indist")
    log.info(f"artifacts written to {out}")


if __name__ == "__main__":
    main()
