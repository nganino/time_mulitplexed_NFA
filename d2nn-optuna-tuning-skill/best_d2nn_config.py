"""Best D2NN physical configuration of the det600_600_handoff simulation study (multi-layer stack,
codes/, September 2026).  Stand-alone: only the standard library is needed.

  DET600_SIM        the frozen optical stack, the best card (r 1.5, N_p,in 8, N_p,out 16, 9wav ->
                    mean test MSE 1.31e-6) and the training recipe that produced it (tag hp-specw-k16)
  det600_geometry   layer size N, propagation grid M and inter-plane distance z_ll for any card
  RETRAIN_CMD       the run_nwav.py command that reproduces the best card

Sources: README.md, markdown/physics.md, markdown/specw_search.md, markdown/hp_tuning.md and
runs/N_f=25-r=1.50-K=3-dx=200nm-lam=300nm-det=600-600-hp-specw-k16-NwScale/.../N_p_out=16/training_metadata.json.
"""
import json
import math

DET600_SIM = {
    "name": "det600 simulation stack, tag hp-specw-k16 (trainable illumination spectrum)",
    "optical_stack": {
        "trainable_phase_layers_K": 3,          # single D2NN of 4 modules, diffractive_network_2 frozen
        "pixel_pitch_m": 200e-9,                # dx
        "layout_wavelength_m": 300e-9,          # lambda used for z_ll and M (needs 2 dx > lambda)
        "refractive_index": 1.5,                # lossless thickness modulation
        "phase_encoding": "0-2pi at lambda_max; phi(lambda) = phi_max * lambda_max / lambda",
        "band_nm": (400, 600),
        "wav_sets": ("600nm", "400nm", "2wav400600", "3wav400600", "5wav400600", "9wav400600"),
        "input": "compact packing of 2 N_p,in + 1 live pixels (N_p,in 8: 4x4+1 on a 5x4 canvas; "
                 "N_p,in 16: 5x6 + 3 centred pixels on 6x6)",
        "detector": {"opening_m": 600e-9, "gap_m": 600e-9, "period_m": 1200e-9, "grid": (5, 5),
                     "roi_m": 6e-6, "readout": "incoherent sum_lambda w_lambda |E_lambda|^2",
                     "cli": "--detector-width-nm 600 --detector-gap-nm 600 (not --output-pitch-factor)"},
        "targets": "direct Hermitian intensity, seed 404, nested; loss = mean MSE of normalised intensities",
        "N_f": 25,
        "neuron_rule": "K N^2 = 2 r N_f N_p,in N_w, N even-rounded up",
        "z_rule": "z_ll = N dx sqrt((2 dx / lambda_layout)^2 - 1) (z_scale 1); propagation grid M = 2 N",
        "intensity_support": "mono +-2 N_p,in; multi-wavelength +-ceil(2 N_p,in lambda_max / lambda_min) = +-3 N_p,in",
    },
    "best_card": {
        "r": 1.5, "N_p_in": 8, "N_p_out": 16, "wav_set": "9wav400600", "N_w": 9, "K": 3,
        "N": 44, "M": 88, "layer_width_m": 8.8e-6, "z_ll_m": 7.7609e-6, "train_grid": 256,
        "wavelengths_nm": [400, 425, 450, 475, 500, 525, 550, 575, 600],
        "learned_spectrum": [4.1e-9, 4.4e-4, 1.9e-3, 4.6e-3, 1.3e-2, 2.5e-2, 0.103, 0.207, 0.645],
        "mean_test_mse": 1.31e-6,                # hp-specw-k16 (trainable spectrum), 2048 x 2048 dense test
        "mean_test_mse_equal_weights": 1.65e-6,  # hp-adam8e-3-cos
        "mean_test_mse_baseline": 4.87e-6,       # hp-lr3e-4 (Adam 3e-4, plateau)
        "note": "9wav is lowest on every N_f=25 grid; MSE rises with N_p,out; N_p,out 28 (in8) and 48 (in16) "
                "lie beyond the +-3 N_p,in multi-wavelength support.",
    },
    "training_recipe": {
        "tag": "hp-specw-k16",
        "optimizer": "adam", "lr": 8e-3, "beta1": 0.97, "beta2": 0.9988,
        "scheduler": "cosine", "min_lr": 8e-7, "warmup_epochs": 10,
        "init": "latent N(0, 0.2), init seed 0", "epochs": 1000, "early_stop": False,
        "iterations_per_epoch": 100, "cuda_graph": True,
        "spectral_weights": {"trainable": True, "init_logits": "-kappa (lambda_max / lambda - 1), kappa 16",
                             "start_epoch": 300, "lr": 0.03, "schedule": "constant", "floor": 0.0,
                             "note": "monos: equal weights; multi-wavelength cards learn to switch the shortest wavelengths off"},
        "result": "N_w chain min(mono) > 2wav > 3wav > 5wav > 9wav monotonic on 24/24 grids; the equal-weight "
                  "recipe alone (hp-adam8e-3-cos) gives gm MSE x0.89 vs hp-lr3e-4 (9wav x0.70, 5wav x0.81, monos x1.0)",
        "cli": "--lr 0.008 --beta1 0.97 --beta2 0.9988 --scheduler cosine --min-lr 8e-7 --warmup-epochs 10 "
               "--init-std 0.2 --patience 1000 --cuda-graph --trainable-spectral-weights --spec-w-init-kappa 16 "
               "--spec-w-start-epoch 300 --spec-w-lr 0.03 --spec-w-schedule constant --spec-w-min 0",
    },
}

RETRAIN_CMD = (
    "python codes/run_nwav.py --gpu 0 --num-functions 25 --k 3 --band 400-600 --n-p-in 8 --r 1.5 "
    "--wav-set 9wav400600 --n-p-out 16 --skip-stars --pixel-size-nm 200 --layout-wavelength-nm 300 "
    "--input-layout compact --detector-width-nm 600 --detector-gap-nm 600 --folder-tag hp-specw-k16 "
    "--allow-any-env " + DET600_SIM["training_recipe"]["cli"]
)


def det600_geometry(r, n_p_in, n_w, n_f=25, k=3, dx=200e-9, lambda_layout=300e-9, z_scale=1.0):
    """Layer size N, propagation grid M and inter-plane distance z_ll of a det600 card."""
    n = 2 * math.ceil(math.sqrt(2 * r * n_f * n_p_in * n_w / k) / 2)
    ratio = 2 * dx / lambda_layout
    z_ll = z_scale * n * dx * math.sqrt(ratio * ratio - 1)
    return {"N": n, "M": 2 * n, "layer_width_m": n * dx, "z_ll_m": z_ll}


if __name__ == "__main__":
    b = DET600_SIM["best_card"]
    print("best card:", json.dumps({k: b[k] for k in ("r", "N_p_in", "N_p_out", "wav_set", "N", "M", "z_ll_m", "mean_test_mse")}))
    print("rule check:", det600_geometry(b["r"], b["N_p_in"], b["N_w"]))
    print("layer sizes at r 1.5:", {(i, w): det600_geometry(1.5, i, w)["N"] for i in (8, 16) for w in (1, 9)})
    print("retrain:", RETRAIN_CMD)
