"""Stand-in trainer + tester so the tuning harness can be exercised without a simulation.

Accepts the flags tune_common.card_args produces (unknown flags are ignored), sleeps briefly,
and writes the files the harness expects: checkpoints/best_model.pth, TRAINING_COMPLETE,
metrics.json {"mse": ...}, TEST_COMPLETE.  The fake MSE has a known optimum (z_factor 0.5,
opening 0.7 features, lr 1.2e-2, init 0.15, beta1 0.95, warm-up 8) plus card offsets and noise,
and improves with the epoch budget, so pruning / ranking behaviour can be checked.
Delete once the real train.py / test.py are wired in.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--card", default="card")
    ap.add_argument("--z-mm", type=float, default=30.0)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--wav-set", default="500nm")
    ap.add_argument("--n-p-out", type=int, default=16)
    ap.add_argument("--detector-width-px", type=int, default=3)
    ap.add_argument("--sensor-subsamples", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--init-std", type=float, default=0.2)
    ap.add_argument("--beta1", type=float, default=0.97)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a, _unknown = ap.parse_known_args()

    out = Path(a.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    z_factor = a.z_mm / (30.0 * a.r)
    feature_px = 2.7 * z_factor
    open_over_feature = a.detector_width_px / feature_px
    log_mse = -3.0 - 0.6 * (a.n_p_out - 16) / 12 + {"500nm": 0.0, "2wav": -0.4, "5wav": -1.2}.get(a.wav_set, 0.0)
    log_mse += 0.5 * (math.log10(z_factor / 0.5)) ** 2
    log_mse += 0.8 * (open_over_feature - 0.7) ** 2
    log_mse += 0.6 * (math.log10(a.lr / 1.2e-2)) ** 2
    log_mse += 0.3 * (math.log10(a.init_std / 0.15)) ** 2
    log_mse += 20.0 * (a.beta1 - 0.95) ** 2
    log_mse += 0.002 * (a.warmup_epochs - 8) ** 2
    log_mse += 0.3 * max(0.0, math.log10(1000.0 / max(a.epochs, 1)))   # budget effect
    # quadrature trap: a 1-px opening trained with < 9 nodes/side looks better than it is
    if a.detector_width_px <= 1 and a.sensor_subsamples < 9:
        log_mse -= 0.5
    seed = int(hashlib.md5(json.dumps(vars(a), sort_keys=True).encode()).hexdigest()[:8], 16)
    log_mse += random.Random(seed).gauss(0.0, 0.03)
    time.sleep(0.05 if a.smoke else 0.3)
    (out / "checkpoints" / "best_model.pth").write_bytes(b"")
    (out / "TRAINING_COMPLETE").write_text(time.strftime("%F %T"))
    (out / "metrics.json").write_text(json.dumps({"mse": 10 ** log_mse, "epochs": a.epochs, "card": a.card}))
    (out / "TEST_COMPLETE").write_text(time.strftime("%F %T"))
    print(f"[mock] {a.card} gpu{a.gpu} epochs {a.epochs} mse {10 ** log_mse:.3e}")


if __name__ == "__main__":
    main()
