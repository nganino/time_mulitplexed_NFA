"""
sgd_hologram_project.py — SGD hologram optimisation (pure digital simulation)
=============================================================================
Workflow per epoch:
  1. Compute complex field from learnable SLM phase (latent → sin mapping).
  2. Propagate to the target plane (angular spectrum method).
  3. Compare intensity with target image — backprop via AdamW.
  4. Phase quantized to 8-bit (256 uniform levels) via FakeQuantize (STE).

Usage:
    python sgd_hologram_project.py
"""

import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FakeQuantize
from PIL import Image as PILImage
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

# ── Target image ─────────────────────────────────────────────
TARGET_IMAGE   = 'sensor_target_3x3_9_0_700_700.png'

# ── SLM geometry ─────────────────────────────────────────────
SLM_H          = 1080
SLM_W          = 1920
PAD_H          = SLM_H // 2
PAD_W          = SLM_W // 2
TOTAL_H        = SLM_H * 2
TOTAL_W        = SLM_W * 2

# ── Target region ────────────────────────────────────────────
RESIZE_TARGET  = False
TARGET_SIZE    = 800

# ── Optical parameters ───────────────────────────────────────
WLENGTH_VC     = 635e-9           # wavelength [m]
RIDX_AIR       = 1.0
FOCUS_DIST_M   = 0.03960           # focus distance [m]
DX             = 8e-6             # pixel pitch [m]
DY             = 8e-6
THETA0         = 75               # NA half-angle [deg]

# ── AdamW optimiser ──────────────────────────────────────────
NUM_EPOCHS     = 5000
LR             = 1e-3

# ── Loss weights ─────────────────────────────────────────────
LAMBDA_ENERGY  = 0.1

# ── Phase quantization ───────────────────────────────────────
BIT_DEPTH        = 8
NUM_PHASE_LEVELS = 2 ** BIT_DEPTH   # 256 uniform levels in [0, 2π)

# ── Logging ──────────────────────────────────────────────────
SAVE_PHASE_EVERY = 500

RUN_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sgd_hologram_data", RUN_TIME)
SIM_DIR   = os.path.join(OUT_DIR, "simulated")
PHASE_DIR = os.path.join(OUT_DIR, "phase")


# ═══════════════════════════════════════════════════════════════
# PHASE QUANTIZATION (straight-through via FakeQuantize)
# ═══════════════════════════════════════════════════════════════

def build_phase_quantizer(device):
    """FakeQuantize for 8-bit uniform phase in [0, 2pi] — 256 levels."""
    fq = FakeQuantize(
        observer=torch.ao.quantization.MinMaxObserver,
        quant_min=0, quant_max=NUM_PHASE_LEVELS - 1,
        dtype=torch.quint8,
    ).to(device)
    fq.enable_observer()
    fq(torch.tensor([0.0, 2.0 * np.pi], device=device))
    fq.disable_observer()
    return fq


# ═══════════════════════════════════════════════════════════════
# ANGULAR SPECTRUM PROPAGATOR  (from optimized_hologram.py)
# ═══════════════════════════════════════════════════════════════

class FreeSpaceProp(nn.Module):
    def __init__(self, wlength_vc, ridx_air,
                 total_x_num, total_y_num, dx, dy, theta0, prop_z):
        super().__init__()
        self.prop_z       = prop_z
        self.total_x_num  = total_x_num
        self.total_y_num  = total_y_num
        self.dx           = dx
        self.dy           = dy
        self.wlengtheff   = wlength_vc / ridx_air

        dfx = 1 / (total_x_num * dx)
        dfy = 1 / (total_y_num * dy)

        self.fx, self.fy = torch.meshgrid(
            (torch.arange(total_x_num) - total_x_num / 2) * dfx,
            (torch.arange(total_y_num) - total_y_num / 2) * dfy,
            indexing='ij',
        )
        self.f0 = 1 / self.wlengtheff

        if theta0:
            self.Q = (self.fx**2 + self.fy**2) <= \
                (self.f0 * torch.sin(torch.tensor(theta0 * np.pi / 180.))) ** 2
        else:
            self.Q = (self.fx**2 + self.fy**2) <= (self.f0**2)

        prop_window   = self.Q * (self.fx**2 + self.fy**2) * (self.wlengtheff**2)
        phase_change  = 2 * np.pi * self.f0 * prop_z * \
                        torch.sqrt(torch.clamp(1 - prop_window, min=0, max=1))
        phase_cplx    = torch.complex(torch.cos(phase_change),
                                      torch.sin(phase_change)) * self.Q
        phase_cplx    = torch.fft.ifftshift(phase_cplx)[None, None, ...]
        self.register_buffer('phase_cplx', phase_cplx)

    def forward(self, x):
        spec = torch.fft.fft2(x)
        return torch.fft.ifft2(self.phase_cplx * spec)




# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    for d in [OUT_DIR, SIM_DIR, PHASE_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load target ──────────────────────────────────────────
    target_pil = PILImage.open(TARGET_IMAGE).convert('L')
    if RESIZE_TARGET:
        target_pil = target_pil.resize((TARGET_SIZE, TARGET_SIZE), PILImage.LANCZOS)

    tgt_w_raw, tgt_h_raw = target_pil.size
    if tgt_h_raw > SLM_H or tgt_w_raw > SLM_W:
        crop_h = min(tgt_h_raw, SLM_H)
        crop_w = min(tgt_w_raw, SLM_W)
        top    = (tgt_h_raw - crop_h) // 2
        left   = (tgt_w_raw - crop_w) // 2
        target_pil = target_pil.crop((left, top, left + crop_w, top + crop_h))
        print(f"[CROP]  {tgt_w_raw}x{tgt_h_raw} -> {crop_w}x{crop_h} (center crop)")

    target_pil.save(os.path.join(OUT_DIR, 'target.png'))

    tgt_h, tgt_w = target_pil.size[1], target_pil.size[0]
    target_row0  = (SLM_H - tgt_h) // 2
    target_col0  = (SLM_W - tgt_w) // 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Device: {device}")
    target_t = transforms.ToTensor()(target_pil).unsqueeze(0).to(device)  # (1,1,tgt_h,tgt_w)

    # ── Propagator ──────────────────────────────────────────
    fsp = FreeSpaceProp(
        wlength_vc=WLENGTH_VC, ridx_air=RIDX_AIR,
        total_x_num=TOTAL_H, total_y_num=TOTAL_W,
        dx=DX, dy=DY, theta0=THETA0,
        prop_z=FOCUS_DIST_M,
    ).to(device)

    # ── Phase quantizer ──────────────────────────────────────
    phase_quantizer = build_phase_quantizer(device)

    # ── Learnable latent (mapped to phase via sin) ───────────
    slm_latent = nn.Parameter(
        torch.zeros((1, 1, SLM_H, SLM_W), device=device)
    )

    # ── Optimiser ────────────────────────────────────────────
    optimizer = torch.optim.AdamW([slm_latent], lr=LR)
    criterion = nn.MSELoss()
    pad = (PAD_W, PAD_W, PAD_H, PAD_H)
    TWO_PI = 2.0 * np.pi

    loss_history = []

    print("=" * 60)
    print("  SGD HOLOGRAM OPTIMISATION")
    print(f"  SLM FOV  : {SLM_H}x{SLM_W}  ->  propagation field {TOTAL_H}x{TOTAL_W}")
    print(f"  Quant    : {BIT_DEPTH}-bit uniform ({NUM_PHASE_LEVELS} levels)")
    print(f"  Target   : {TARGET_IMAGE}")
    print(f"  Focus    : {FOCUS_DIST_M*1e3:.1f} mm")
    print(f"  Epochs   : {NUM_EPOCHS}   LR={LR}")
    print(f"  Output   : {OUT_DIR}")
    print("=" * 60 + "\n")

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        # Latent -> phase via (sin(x)+1)/2 * 2pi -> quantize
        phase_rad = (torch.sin(slm_latent) + 1.0) / 2.0 * TWO_PI
        phase_rad = phase_quantizer(phase_rad)
        field = torch.exp(1j * phase_rad)
        field_pad = F.pad(field, pad, mode='constant', value=0)

        # Propagate -> crop -> intensity
        output = fsp(field_pad)
        output = torch.abs(output) ** 2
        output = output[:, :, PAD_H:PAD_H + SLM_H,
                               PAD_W:PAD_W + SLM_W]

        # Loss on target region
        output_target = output[:, :,
                                target_row0:target_row0 + tgt_h,
                                target_col0:target_col0 + tgt_w]
        loss_mse = criterion(output_target, target_t)

        # Energy efficiency penalty
        energy_target = output_target.sum()
        energy_total  = output.sum().clamp(min=1e-12)
        loss_energy   = -energy_target / energy_total

        loss = loss_mse + LAMBDA_ENERGY * loss_energy
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:5d} | Loss {loss_val:.6f}"
                  f"  (MSE {loss_mse.item():.6f}"
                  f"  Eff {(-loss_energy).item():.4f})")

        # Periodic saves
        if epoch % SAVE_PHASE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            with torch.no_grad():
                phase_vis = (torch.sin(slm_latent.detach()) + 1.0) / 2.0
                phase_vis_q = phase_quantizer(phase_vis * TWO_PI) / TWO_PI
            phase_np = phase_vis_q.cpu().squeeze().numpy()
            plt.imsave(os.path.join(PHASE_DIR, f"epoch_{epoch:05d}.png"),
                       phase_np, cmap='gray')
            np.save(os.path.join(PHASE_DIR, f"epoch_{epoch:05d}.npy"),
                    slm_latent.detach().cpu().squeeze().numpy())
            sim = output.detach().cpu().squeeze().numpy()
            sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
            plt.imsave(os.path.join(SIM_DIR, f"epoch_{epoch:05d}.png"),
                       sim_norm, cmap='gray')

    print("\n[SGD] Optimisation complete.")

    # ── Save final outputs ───────────────────────────────────
    with torch.no_grad():
        phase_final = (torch.sin(slm_latent.detach()) + 1.0) / 2.0 * TWO_PI
        phase_final_q = phase_quantizer(phase_final)
        phase_vis_np = (phase_final_q / TWO_PI).cpu().squeeze().numpy()
    np.save(os.path.join(OUT_DIR, 'final_slm_latent.npy'),
            slm_latent.detach().cpu().squeeze().numpy())
    phase_uint8 = (phase_vis_np * 255).astype(np.uint8)
    PILImage.fromarray(phase_uint8, mode='L').save(
        os.path.join(OUT_DIR, 'final_slm_phase.bmp'))

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SGD Hologram Optimisation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'), dpi=150)
    plt.close()

    print(f"\n[DONE]  Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
