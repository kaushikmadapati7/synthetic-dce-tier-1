"""Preflight self-check: assert the invariants that silent bugs previously broke.

Nine bugs were found in this pipeline that degraded results WITHOUT ever raising an
error -- silent tensor broadcasting, shape assumptions, unseeded sampling, and
statistics pooled across the batch. Each cost real GPU time and produced numbers that
looked plausible. This file pins those invariants so a regression is caught in seconds
on CPU instead of after a 24h run.

Run before any formal training sweep:

    python -m tier1_static.selfcheck
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except Exception as e:
            FAIL.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return deco


# --------------------------------------------------------------------------- #
def run_metric_checks():
    import torch.nn.functional as F
    from .loss.loss import ssim3d, CustomLoss
    from .metrics import roi_radiomics, roi_sharpness, zone_metrics, eval_metrics
    from .models._cfg import roi_weighted_mse
    print("\n[metrics / loss]")

    @check("ssim3d on depth-1 slices equals true 2D SSIM (not the zero-padded 3D window)")
    def _():
        def ref2d(x, y, ws=7, sigma=1.5, dr=2.0):
            c = x.shape[1]
            co = torch.arange(ws, dtype=x.dtype) - ws // 2
            g = torch.exp(-(co ** 2) / (2 * sigma ** 2)); g /= g.sum()
            w = (g[:, None] * g[None, :]).expand(c, 1, ws, ws).contiguous(); p = ws // 2
            mx = F.conv2d(x, w, padding=p, groups=c); my = F.conv2d(y, w, padding=p, groups=c)
            sx = F.conv2d(x * x, w, padding=p, groups=c) - mx ** 2
            sy = F.conv2d(y * y, w, padding=p, groups=c) - my ** 2
            sxy = F.conv2d(x * y, w, padding=p, groups=c) - mx * my
            c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
            return float((((2 * mx * my + c1) * (2 * sxy + c2)) /
                          ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))).mean())
        torch.manual_seed(0)
        x = torch.randn(4, 1, 64, 64) * 0.3
        for y in (x.clone(), torch.randn_like(x) * 0.3, F.avg_pool2d(x, 5, 1, 2), x * 0.5):
            a = float(ssim3d(x.unsqueeze(2), y.unsqueeze(2)))
            b = ref2d(x, y)
            assert abs(a - b) < 1e-5, f"depth-1 SSIM {a:.6f} != 2D {b:.6f}"

    @check("ssim3d unchanged on full 3D volumes (identical -> 1.0)")
    def _():
        v = torch.randn(2, 1, 32, 48, 48) * 0.3
        assert abs(float(ssim3d(v, v)) - 1.0) < 1e-5
        assert float(ssim3d(v, torch.randn_like(v))) < 0.2

    @check("ROI metrics are PER-CASE, so batch pooling cannot inflate roi_pearson")
    def _():
        torch.manual_seed(0)
        B = 8
        mask = torch.zeros(B, 1, 1, 32, 32); mask[:, :, :, 10:22, 10:22] = 1
        lvl = torch.linspace(-0.6, 0.4, B).view(B, 1, 1, 1, 1)
        t = lvl + 0.05 * torch.randn(B, 1, 1, 32, 32)
        p = lvl + 0.05 * torch.randn(B, 1, 1, 32, 32)   # brightness shared, structure independent
        r = roi_radiomics(p, t, mask)["roi_pearson"]
        assert abs(r) < 0.25, f"roi_pearson {r:+.3f} -- batch brightness is leaking in"
        per = [roi_radiomics(p[i:i+1], t[i:i+1], mask[i:i+1])["roi_pearson"] for i in range(B)]
        assert abs(r - sum(per) / B) < 1e-6, "not equal to the per-case average"

    @check("ROI metrics are batch-size invariant")
    def _():
        torch.manual_seed(1)
        B = 8
        mask = torch.zeros(B, 1, 1, 32, 32); mask[:, :, :, 10:22, 10:22] = 1
        t = torch.randn(B, 1, 1, 32, 32) * 0.3
        p = t + 0.1 * torch.randn_like(t)
        full = roi_radiomics(p, t, mask)["roi_pearson"]
        halves = [roi_radiomics(p[i:i+4], t[i:i+4], mask[i:i+4])["roi_pearson"] for i in (0, 4)]
        assert abs(full - sum(halves) / 2) < 1e-5, "depends on how the batch is split"

    @check("eval_metrics ROI scalars weight patients equally, not by gland size")
    def _():
        B = 4
        mask = torch.zeros(B, 1, 4, 32, 32)
        mask[0, :, :, 4:28, 4:28] = 1                      # one big gland
        for i in range(1, B):
            mask[i, :, :, 14:18, 14:18] = 1                # three small ones
        t = torch.randn(B, 1, 4, 32, 32).clamp(-1, 1)
        p = t.clone(); p[0] = t[0] + 0.5                   # only the big-gland case is wrong
        mae = eval_metrics(p, t, mask)["mae_roi"]
        assert abs(mae - 0.125) < 1e-3, f"mae_roi {mae:.4f}, expected ~0.125 (1 of 4 cases)"

    @check("zone_metrics are per-case too")
    def _():
        torch.manual_seed(0)
        B = 8
        z = torch.zeros(B, 1, 1, 32, 32); z[:, :, :, 8:16, 8:24] = 2; z[:, :, :, 16:24, 8:24] = 1
        lvl = torch.linspace(-0.6, 0.4, B).view(B, 1, 1, 1, 1)
        t = lvl + 0.05 * torch.randn(B, 1, 1, 32, 32)
        p = lvl + 0.05 * torch.randn(B, 1, 1, 32, 32)
        out = zone_metrics(p, t, z)
        assert abs(out["roi_pearson_pz"]) < 0.25, out

    @check("roi_weighted_mse RAISES on a mask/pred rank mismatch (was a silent (B,B,..) product)")
    def _():
        p = torch.randn(4, 1, 32, 32); t = torch.randn(4, 1, 32, 32)
        m5 = torch.ones(4, 1, 1, 32, 32)
        try:
            roi_weighted_mse(p, t, m5, roi_weight=10.0)
        except ValueError:
            return
        raise AssertionError("no error raised on rank mismatch")

    @check("CustomLoss RAISES on a mask rank mismatch")
    def _():
        c = CustomLoss(l1_weight=1., ssim_weight=1., perceptual_weight=0., roi_weight=10.)
        p = torch.randn(2, 1, 4, 16, 16); t = torch.randn(2, 1, 4, 16, 16)
        try:
            c(p, t, torch.ones(2, 1, 4, 16, 16, 1))
        except ValueError:
            return
        raise AssertionError("no error raised on rank mismatch")


def run_model_checks():
    import argparse
    from .training.gan import _build_gan, _gan_gen
    from .models.conditional_gan import UNetGenerator3D, _stride_schedule
    from .models.flow2d import FlowMatching2D
    from .models.ldm_flow_matching import LDM_FlowMatching
    from .models.ldm_ddpm import LDM_DDPM
    print("\n[models / determinism]")
    dev = torch.device("cpu")

    @check("3D GAN defaults to the U-Net generator (skips, no pooled thumbnail)")
    def _():
        a = argparse.Namespace(z_dim=32, base_ch=8, n_upsamples=2, spatial_size=(8, 32, 32),
                               modality_dropout=False, use_pregad=False, seed=0)
        g = _build_gan(a, dev)
        assert g.generator_type == "unet" and g.z_dim == 0, g.generator_type
        assert isinstance(g.generator, UNetGenerator3D)

    @check("U-Net generator keeps depth >= 2 on anisotropic input and returns full size")
    def _():
        assert _stride_schedule((32, 192, 192), 5)[-1] == (1, 2, 2)
        for ss in [(32, 192, 192), (16, 96, 96), (8, 32, 32)]:
            for ch in (3, 4):
                g = UNetGenerator3D(in_channels=ch, base_ch=8, spatial_size=ss)
                with torch.no_grad():
                    assert g(torch.randn(1, ch, *ss)).shape == (1, 1, *ss)

    @check("GAN eval is deterministic (repeat evals of one checkpoint agree)")
    def _():
        for gt in ("unet", "resnet"):
            a = argparse.Namespace(z_dim=32, base_ch=8, n_upsamples=2, spatial_size=(8, 32, 32),
                                   modality_dropout=False, use_pregad=False, seed=0,
                                   gan_generator=gt)
            g = _build_gan(a, dev); g.eval()
            gen = _gan_gen(g, a, dev); c = torch.randn(2, 3, 8, 32, 32)
            with torch.no_grad():
                assert torch.allclose(gen(c), gen(c)), gt

    @check("2D flow samplers are deterministic when seeded (and not when unseeded)")
    def _():
        m = FlowMatching2D(cond_ch=3, base=8).eval(); c = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert torch.allclose(m.sample(c, steps=3, seed=0), m.sample(c, steps=3, seed=0))
            assert not torch.allclose(m.sample(c, steps=3), m.sample(c, steps=3))

    @check("3D flow and DDPM samplers are deterministic when seeded")
    def _():
        kw = dict(in_channels=2, out_channels=2, base_ch=8, ch_mults=(1, 2), cond_channels=3)
        cond = torch.randn(2, 3, 4, 16, 16); shape = (2, 2, 4, 16, 16)
        f = LDM_FlowMatching(autoencoder=None, unet_kwargs=kw).eval()
        d = LDM_DDPM(autoencoder=None, unet_kwargs=kw).eval()
        with torch.no_grad():
            assert torch.allclose(f.sample(shape, dev, steps=3, cond=cond, decode=False, seed=0),
                                  f.sample(shape, dev, steps=3, cond=cond, decode=False, seed=0))
            assert torch.allclose(d.ddim_sample(shape, dev, steps=3, cond=cond, decode=False, seed=0),
                                  d.ddim_sample(shape, dev, steps=3, cond=cond, decode=False, seed=0))

    @check("3D GAN completes a full train step (fwd+bwd) with both generators, 3 and 4 cond ch")
    def _():
        from .models import d_hinge_loss, g_total_loss
        for gt in ("unet", "resnet"):
            for pg, ch in ((False, 3), (True, 4)):
                a = argparse.Namespace(z_dim=32, base_ch=8, n_upsamples=2,
                                       spatial_size=(8, 32, 32), modality_dropout=False,
                                       use_pregad=pg, seed=0, gan_generator=gt, adv_weight=1.0)
                g = _build_gan(a, dev)
                cond = torch.randn(2, ch, 8, 32, 32)
                real = torch.randn(2, 1, 8, 32, 32).clamp(-1, 1)
                fake = g.generate(cond)
                assert fake.shape == real.shape, (gt, ch, fake.shape)
                d_hinge_loss(g.discriminator(real, cond_vol=cond),
                             g.discriminator(fake.detach(), cond_vol=cond)).backward()
                g_total_loss(g.discriminator(g.generate(cond), cond_vol=cond),
                             g.generate(cond), real, None, adv_weight=1.0)[0].backward()

    @check("3D LDM flow: latent-grid ROI mask keeps rank, loss+backward works")
    def _():
        from .training.utils import downsample_cond
        from .loss.loss import CustomLoss
        from .models.autoencoder3d import AutoencoderKL3D
        vae = AutoencoderKL3D(latent_channels=2, base_ch=8, ch_mults=(1, 2),
                              num_res_blocks=1, attn_resolutions=())
        vae.scaling_factor, vae.latent_shift = 1.0, 0.0
        kw = dict(in_channels=2, out_channels=2, base_ch=8, ch_mults=(1, 2), cond_channels=3)
        ldm = LDM_FlowMatching(autoencoder=vae, unet_kwargs=kw)
        img = torch.randn(2, 1, 8, 32, 32).clamp(-1, 1)
        cond = torch.randn(2, 3, 8, 32, 32)
        mask = (torch.rand(2, 1, 8, 32, 32) > 0.8).float()
        with torch.no_grad():
            z0 = ldm.encode(img)
        cond_ds = downsample_cond(cond, z0.shape[2:])
        mask_ds = downsample_cond(mask, z0.shape[2:])
        assert mask_ds.dim() == z0.dim() == 5, (mask_ds.shape, z0.shape)
        ldm.loss(z0, cond=cond_ds, mask=mask_ds, roi_weight=10.0).backward()

    @check("3D LDM flow: image-space anchor runs and gradients reach the UNet")
    def _():
        from .training.utils import downsample_cond
        from .loss.loss import CustomLoss
        from .models.autoencoder3d import AutoencoderKL3D
        vae = AutoencoderKL3D(latent_channels=2, base_ch=8, ch_mults=(1, 2),
                              num_res_blocks=1, attn_resolutions=())
        vae.scaling_factor, vae.latent_shift = 1.0, 0.0
        kw = dict(in_channels=2, out_channels=2, base_ch=8, ch_mults=(1, 2), cond_channels=3)
        ldm = LDM_FlowMatching(autoencoder=vae, unet_kwargs=kw)
        crit = CustomLoss(l1_weight=1., ssim_weight=1., perceptual_weight=0., roi_weight=10.)
        img = torch.randn(2, 1, 8, 32, 32).clamp(-1, 1)
        cond = torch.randn(2, 3, 8, 32, 32)
        mask = (torch.rand(2, 1, 8, 32, 32) > 0.8).float()
        with torch.no_grad():
            z0 = ldm.encode(img)
        torch.manual_seed(0)
        loss = ldm.loss(z0, cond=downsample_cond(cond, z0.shape[2:]),
                        mask=downsample_cond(mask, z0.shape[2:]), roi_weight=10.0,
                        anchor_image=img, anchor_mask=mask, anchor_criterion=crit,
                        anchor_weight=1.0, anchor_t_max=1.0)   # t_max=1 -> always anchors
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in ldm.unet.parameters()), "no gradient reached the UNet"

    @check("cond channels follow --use-pregad (3 -> 4, doubled under modality dropout)")
    def _():
        from .training._ldm_base import _cond_channels
        mk = lambda pg, md: argparse.Namespace(use_pregad=pg, modality_dropout=md)
        assert (_cond_channels(mk(False, False)), _cond_channels(mk(True, False)),
                _cond_channels(mk(True, True))) == (3, 4, 8)


def run_data_checks():
    import SimpleITK as sitk
    from .data.stage_ucsf import stage_one, main as stage_main
    from .data import UCSFDCEDataset
    from .data.preprocessing import PreprocessConfig
    from .data.harmonization import Harmonizer, HarmonizationConfig
    print("\n[data / staging]")
    root = Path(tempfile.mkdtemp())
    MAIN, DCE, OUT = root / "ds2", root / "ds3", root / "staged"
    rng = np.random.default_rng(0)
    curve = np.array([1, 1, 1, 1.1, 1.6, 2.0, 2.3, 2.45, 2.5, 2.5, 2.5, 2.49])
    times = [round(i * 9.08, 2) for i in range(12)]

    def case(pid, ts, inter=False, flat=False, bval="b1500"):
        (MAIN / pid).mkdir(parents=True); (DCE / pid / "DCE").mkdir(parents=True)
        base = rng.random((6, 16, 16)).astype(np.float32) + 1
        m = np.zeros((6, 16, 16), np.uint8); m[2:4, 6:10, 6:10] = 1
        for n in ("T2W", "ADC_to_T2W", f"DWI_{bval}_to_T2W"):
            sitk.WriteImage(sitk.GetImageFromArray(base), str(MAIN / pid / f"{n}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(m), str(MAIN / pid / "prostate_mask.nii.gz"))
        c = np.ones_like(curve) if flat else curve
        vols = []
        for v in c:
            vols += ([base * 1.0, base * v] if inter else [base * v])
        sitk.WriteImage(sitk.GetImageFromArray(np.stack(vols).astype(np.float32), isVector=False),
                        str(DCE / pid / "DCE" / "DCE_4D_to_T2W.nii.gz"))
        (DCE / pid / "DCE" / "dce_times.json").write_text(json.dumps(
            {"case": pid, "n_phases": len(ts), "phases": [{"idx": i, "rel_time_s": t}
                                                          for i, t in enumerate(ts)]}))

    case("NORMAL", times)
    case("INTER", [t for t in times for _ in (0, 1)], inter=True)
    case("DEAD", times, flat=True)
    case("BADTIME", [0.0] + [None] * 11)
    case("ODDBVAL", times, bval="b0400")          # b-value outside the old hardcoded list
    common = dict(main_root=str(MAIN), dce_root=str(DCE), out_root=str(OUT),
                  target_time=72.0, t_max=600.0, dwi_bvalue="1000")

    @check("DWI resolves for a b-value outside the old fixed stem list")
    def _():
        r = stage_one("ODDBVAL", **common, min_enh=0.0)
        assert "skip" not in r, r
        assert "b0400" in r["dwi_src"], r["dwi_src"]

    @check("pre-contrast T1 (DCE phase 0) is staged for the 4th cond channel")
    def _():
        stage_one("NORMAL", **common, min_enh=0.0)
        assert (OUT / "NORMAL" / "DCE_pre_to_T2W.nii.gz").exists()

    @check("interleaved 4D picks the ENHANCING sub-series")
    def _():
        r = stage_one("INTER", **common, min_enh=0.0)
        assert r["interleave"] == 2 and r["phase_idx"] % 2 == 1 and r["enh_ratio"] > 2.0, r

    @check("unusable timing re-selects from the enhancement curve, not phase 0")
    def _():
        r = stage_one("BADTIME", **common, min_enh=0.0)
        assert r["select_mode"] == "curve" and r["enh_ratio"] > 2.0, r

    @check("--min-enh drops never-enhancing studies and removes any stale staged dir")
    def _():
        stage_one("DEAD", **common, min_enh=0.0)            # stage it first
        assert (OUT / "DEAD").exists()
        r = stage_one("DEAD", **common, min_enh=1.2, overwrite=True)
        assert "skip" in r and not (OUT / "DEAD").exists(), r

    @check("staged loader == raw-4D loader (bit-identical target), and pregad adds a 4th channel")
    def _():
        cfg = PreprocessConfig(spatial_size=(6, 16, 16), reference="dce")
        hc = HarmonizationConfig()
        hc.methods = dict(hc.methods, t2w="percentile", dce="robust", pre="robust")
        h = Harmonizer(hc)
        st = UCSFDCEDataset(OUT, None, cfg, harmonizer=h, pids=["NORMAL"])
        rw = UCSFDCEDataset(MAIN, DCE, cfg, harmonizer=h, target_time=72.0, pids=["NORMAL"])
        assert torch.allclose(st[0]["target"], rw[0]["target"], atol=1e-5)
        p4 = UCSFDCEDataset(OUT, None, cfg, harmonizer=h, use_pregad=True, pids=["NORMAL"])[0]
        assert p4["cond"].shape[0] == 4
        assert torch.allclose(p4["cond"][:3], st[0]["cond"]), "first 3 channels changed"
        assert p4["cond"][3].std() > 1e-3, "pre-contrast channel is constant"

    @check("cohort filters read the b-value from stage_meta (staged files are renamed)")
    def _():
        # Mirror what stage_ucsf ACTUALLY writes: DWI is copied to a canonical
        # DWI_to_T2W.nii.gz, so the b-value survives only in stage_meta's dwi_src.
        # Reading it off the staged filename returns -1 and drops the whole cohort.
        tmp = Path(tempfile.mkdtemp()); T = tmp / "staged"
        def mk(pid, bsrc, measured=True):
            d = T / pid; d.mkdir(parents=True)
            a = (np.random.rand(4, 16, 16).astype(np.float32) + 1)
            m = np.zeros((4, 16, 16), np.uint8); m[1:3, 6:10, 6:10] = 1
            for n in ("T2W", "ADC_to_T2W", "DWI_to_T2W", "DCE_to_T2W", "DCE_pre_to_T2W"):
                sitk.WriteImage(sitk.GetImageFromArray(a), str(d / f"{n}.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(m), str(d / "prostate_mask.nii.gz"))
            (d / "stage_meta.json").write_text(json.dumps(
                {"pid": pid, "phase_idx": 8, "rel_time_s": 72.0, "n_phases": 12,
                 "target_time": 120.0, "select_mode": "time", "dwi_src": bsrc,
                 "enh_ratio": (2.0 if measured else None),
                 "enh_max": (2.0 if measured else None)}))
        mk("HIGH", "DWI_b1000_to_T2W.nii.gz")
        mk("LOW", "DWI_b50_to_T2W.nii.gz")
        mk("NOQC", "DWI_b1000_to_T2W.nii.gz", measured=False)
        cfg = PreprocessConfig(spatial_size=(4, 16, 16), reference="dce")
        assert sorted(UCSFDCEDataset(T, None, cfg).pids) == ["HIGH", "LOW", "NOQC"]
        assert sorted(UCSFDCEDataset(T, None, cfg, dwi_min_bvalue=600).pids) == ["HIGH", "NOQC"]
        assert sorted(UCSFDCEDataset(T, None, cfg, require_qc=True).pids) == ["HIGH", "LOW"]
        assert sorted(UCSFDCEDataset(T, None, cfg, dwi_min_bvalue=600,
                                     require_qc=True).pids) == ["HIGH"]
        shutil.rmtree(tmp, ignore_errors=True)

    @check("a filter that removes EVERY case raises instead of yielding an empty split")
    def _():
        tmp = Path(tempfile.mkdtemp()); T = tmp / "staged"
        d = T / "ONLYLOW"; d.mkdir(parents=True)
        a = (np.random.rand(4, 16, 16).astype(np.float32) + 1)
        m = np.zeros((4, 16, 16), np.uint8); m[1:3, 6:10, 6:10] = 1
        for n in ("T2W", "ADC_to_T2W", "DWI_to_T2W", "DCE_to_T2W", "DCE_pre_to_T2W"):
            sitk.WriteImage(sitk.GetImageFromArray(a), str(d / f"{n}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(m), str(d / "prostate_mask.nii.gz"))
        (d / "stage_meta.json").write_text(json.dumps(
            {"pid": "ONLYLOW", "dwi_src": "DWI_b50_to_T2W.nii.gz",
             "enh_ratio": 2.0, "enh_max": 2.0, "select_mode": "time"}))
        cfg = PreprocessConfig(spatial_size=(4, 16, 16), reference="dce")
        try:
            UCSFDCEDataset(T, None, cfg, dwi_min_bvalue=600)
        except RuntimeError:
            shutil.rmtree(tmp, ignore_errors=True)
            return
        shutil.rmtree(tmp, ignore_errors=True)
        raise AssertionError("empty split did not raise")

    @check("--report --prune-below removes weak cases and is idempotent")
    def _():
        stage_main(["--report", "--prune-below", "1.2", "--out", str(OUT)])
        n1 = len(list(OUT.glob("*/stage_meta.json")))
        stage_main(["--report", "--prune-below", "1.2", "--out", str(OUT)])
        assert len(list(OUT.glob("*/stage_meta.json"))) == n1

    shutil.rmtree(root, ignore_errors=True)


def main():
    print("tier1_static preflight self-check")
    run_metric_checks()
    run_model_checks()
    run_data_checks()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("all invariants hold -- safe to launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
