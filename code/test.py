'''
Time-Multiplexed NFA — test / evaluation (Nonlinear Function Approximation)

Usage:
    python test.py --ckpt logs/.../model/best.pth
    python test.py --ckpt logs/.../model/epoch=050.pth --out_dir out/

Outputs saved to <out_dir>/:
    test_summary.png            — combined summary figure (see _save_test_summary):
                                   error-distribution histogram (top-left), input
                                   encoding-plane phase with sim-pixel grid (top-middle),
                                   per-detector RMSE map (top-right),
                                   and target-vs-approx curves for representative
                                   functions -- best-fit / worst-fit / mid-error
                                   (bottom row, paper Fig. 2a/2b/2c-style)
    plane_layout.png            — phase key 0 | encoding plane | layer 1 | detector-array
                                   extent, each on the full sim grid (relative sizes)
    phase_keys_final.png        — learned phase-key masks (one per key, M total)
    layer_masks_final.png       — learned diffractive-layer phase masks
    decoder_mask.png            — E+D decoder phase(s), 1 row, one panel per decoder
    phase_key_similarity.png    — pairwise phase similarity between the M keys
                                   (low similarity = keys are learning genuinely
                                   different things -- see _save_mask_similarity)
    key_evolution.png           — per key: the phase key, the intensity and phase of its field at
                                   the encoding plane, its own contribution c_m(a) to the output
                                   (sum over keys = f_hat), and the spectrum of c_m (split: frame
                                   m's harmonic block shaded)
    encoding_frames.png         — split mode only: encoding phase of every frame
    per_function_rmse.npy       — [Nf] test RMSE per function / detector
Optionally appends a summary row (loss, RMSE mean/max/min) to --csv, if given.

--readout global (default): min-max over the whole test grid AND all detectors (paper Eq. 9), targets likewise.
--readout perfunc (2026-11-16): each function k min-max'd over the test grid on its own, f_hat and target (the bench
    readout; matches validation of the train_readout perfunc* runs). Default out_dir is then <run>/test_perfunc.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from config import init_params, recompute_derived, apply_saved_config, check_checkpoint_format
from model import TimeMultiplexedNFA
from dataloader import get_function_approx_dataloaders
from loss import FunctionApproxLoss


# ─────────────────────────── evaluation loop ──────────────────────────────── #

def _eval_loop(model, loader, criterion, device, desc='Evaluating', per_function=False):
    '''
    Single evaluation pass over the dense a-grid loader. The min-max readout uses Pmin / Pmax over the
    WHOLE grid and all detectors (the exact definition), so f_hat is formed after the pass.

    Returns
    -------
    agg : dict with -- all sorted by ascending `a` for gap-free plotting --
        'a'      : [N] input values
        'f_hat'  : [N, Nf] normalized model output
        'target' : [N, Nf] true function values, min-max normalised over this grid (as f_hat)
        'p_range': (Pmin, Pmax) of the frame-averaged intensity over the grid (scalars; per_function: [1, Nf] each)
        'loss'   : [full-grid MSE] (a 1-element list)
        'de'     : [N, T] diffraction efficiency per sample and phase key
                   (diagnostic only, see loss.FunctionApproxLoss.diffraction_efficiency)
        'I'      : [N, T, Nf] raw per-key detector readout (k = r*cols + c)
    '''
    agg = defaultdict(list)
    was_training_model = model.training
    was_training_crit  = criterion.training
    model.eval(); criterion.eval()

    with torch.no_grad():
        incident = model.incident_energy()
        for a, target in tqdm(loader, desc=desc):
            a = a.to(device); target = target.to(device)
            I_vec = model(a)

            agg['a'].append(a.cpu())
            agg['target'].append(target.cpu())
            agg['de'].append(criterion.diffraction_efficiency(I_vec, incident).cpu())
            agg['I'].append(I_vec.reshape(I_vec.shape[0], I_vec.shape[1], -1).cpu())

    if was_training_model:
        model.train()
    if was_training_crit:
        criterion.train()

    agg['a']      = torch.cat(agg['a'])
    agg['target'] = torch.cat(agg['target'])
    agg['de']     = torch.cat(agg['de'])
    agg['I']      = torch.cat(agg['I'])

    order = torch.argsort(agg['a'])
    for key in ('a', 'target', 'de', 'I'):
        agg[key] = agg[key][order]

    # targets min-max'd over the SAME grid as f_hat (their 2001-point calibration misses the grid's true range)
    # per_function (--readout perfunc): every function over a on its own, f_hat and target alike
    agg['target'] = criterion.normalize(agg['target'], *criterion.minmax_range(agg['target'], per_function))
    I_mean = agg['I'].mean(1)                                 # [N, Nf] frame average, as in loss._detector_to_function
    agg['p_range'] = criterion.minmax_range(I_mean, per_function)
    agg['f_hat'] = criterion.normalize(I_mean, *agg['p_range'])
    agg['loss'] = [torch.mean((agg['f_hat'] - agg['target']) ** 2).item()]
    return agg


# ──────────────────────────── print summary ───────────────────────────────── #

def _print_summary(agg, tag='Test'):
    rmse = FunctionApproxLoss.per_function_rmse(agg['f_hat'], agg['target'])   # [Nf]

    print(f'\n── {tag} Results ──────────────────────────────────────────')
    print(f'  Loss (MSE)          : {np.mean(agg["loss"]):.4e}')
    print(f'  Per-function RMSE   : mean={rmse.mean():.4e}  '
          f'max={rmse.max():.4e}  min={rmse.min():.4e}')
    if 'de' in agg:
        de = agg['de']                                     # [N, T]
        print(f'  Diffraction eff.    : mean={de.mean():.4e}  '
              f'max={de.max():.4e}  min={de.min():.4e}')
        if de.shape[1] > 1:
            per_key = '  '.join(f'{v:.3e}' for v in de.mean(dim=0).tolist())
            print(f'    per key (mean)    : {per_key}')
    print(f'──────────────────────────────────────────────────────────\n')
    return rmse


# ──────────────────────────── save figures ────────────────────────────────── #

def _enc_tag(config):
    '''Np / N_alpha / mode text for figure titles.'''
    if getattr(config, 'encoding_mode', 'constant') == 'split':
        return f'Np={config.Np}/frame  N_alpha={config.N_alpha} (split)'
    return f'Np={config.Np}'


def _bg_tag(model):
    '''Background text for figure titles: opaque (K > 0) or transparent (K = 0).'''
    return ('background OPAQUE (only encoding pixels pass light)' if model.opaque_background
            else 'background transparent (phase 0)')


def _encoding_boxes(model):
    '''(row0, col0, height, width) in sim px of every encoding patch copy (one for 'single').'''
    h, w, N = model.encoding_y_num_sim, model.encoding_x_num_sim, model.N_sim
    y0, x0 = (N - h) // 2, (N - w) // 2                       # centred patch, as _embed_in_sim
    return [(y0 + dy, x0 + dx, h, w) for dy, dx in model.encoding_tile_shifts]


def _save_test_summary(agg, rmse, config, model, save_path, n_show=4, dpi=200, enc_a=0.3):
    '''
    Combined test-summary figure -- one PNG instead of separate files:
      - top-left:   histogram of per-function RMSE (paper Fig. 2a-style)
      - top-middle: input encoding-plane phase at a = enc_a, zoomed on the
        encoding patch with the sim-pixel grid overlaid (title carries the
        input bin / gap settings)
      - top-right:  spatial map of per-function RMSE over the detector plane
      - bottom row (full width): target-vs-approx curves for representative
        functions -- best-fit, worst-fit, and two spread across the middle
        of the error distribution (paper Fig. 2b/2c-style)

    All RMSE values shown as text (titles/legends) use scientific notation
    (`.2e`) rather than fixed-point -- at the accuracy this project now
    reaches (down to ~1e-6/1e-7 after fixing detector crosstalk, see
    logs/SWEEP_HANDOVER.txt), `.3f`/`.4f` formatting would just print
    "0.000" and hide the actual value.
    '''
    a, f_hat, target = agg['a'].numpy(), agg['f_hat'].numpy(), agg['target'].numpy()
    rmse_np = rmse.numpy()
    Nf = rmse.shape[0]

    # Rank-based labels come from the ERROR-sorted position (index 0 = best
    # fit, Nf-1 = worst fit). `label_of` is built in best->worst order and
    # (relying on dict insertion order) `show_idx` below is left as-is rather
    # than re-sorted by function id `k`, so the bottom row displays left to
    # right as best -> middle -> middle -> worst.
    order_by_err = torch.argsort(rmse)
    rank_positions = [0, Nf // 3, 2 * Nf // 3, Nf - 1]
    rank_labels    = ['best', 'middle', 'middle', 'worst']
    label_of = {}
    for pos, lab in zip(rank_positions, rank_labels):
        k = int(order_by_err[pos])
        label_of.setdefault(k, lab)   # keep first label if Nf is small enough to collide
    show_idx = list(label_of.keys())[:n_show]

    # ncols = 3*n_show divides evenly by n_show (bottom row's equal columns);
    # the top row instead splits unevenly -- histogram gets back close to the
    # ~half-width it had before this panel existed, the (square) input-plane
    # image gets the least, heatmap gets the rest
    # (it needs room for its colorbar without feeling squeezed).
    ncols = 3 * n_show
    hist_w = ncols * 5 // 12
    cx_w   = ncols * 3 // 12
    heat_w = ncols - hist_w - cx_w
    fig = plt.figure(figsize=(4 * n_show, 7.2), dpi=dpi)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.15, 1])

    # ---- top-left: error distribution histogram ----------------------------
    ax_hist = fig.add_subplot(gs[0, :hist_w])
    ax_hist.hist(rmse_np, bins=min(30, max(len(rmse_np), 1)), color='steelblue', alpha=0.85)
    ax_hist.axvline(rmse_np.mean(), color='crimson', linestyle='--',
                     label=f'mean={rmse_np.mean():.2e}')
    ax_hist.set_xlabel('RMSE'); ax_hist.set_ylabel('# functions')
    ax_hist.set_title(f'Error distribution (Nf={len(rmse_np)})', fontsize=10)
    ax_hist.legend(fontsize=8)

    # ---- top-middle: input encoding arrangement (phase), zoomed on the patch --
    # Encoding phase ALONE (never the decoder): encoding pixels at a = enc_a on the phase-0 background (K = 0,
    # transparent) or on black (K > 0, opaque background). Crop = patch + a small margin (4 SLM px), cyan box = patch copy.
    ax_enc = fig.add_subplot(gs[0, hist_w:hist_w + cx_w])
    with torch.no_grad():
        a_show = torch.tensor([enc_a], device=model.slm_aperture.device)
        phi_enc = model._encode_input(a_show)[0, 0]            # [N, N], encoding only (0 elsewhere)
    step = model.enc_ratio                                     # sim px per encoding-SLM pixel
    N, margin = config.N_sim, int(np.ceil(4 * step))
    Ph, Pw = model.encoding_y_num_sim, model.encoding_x_num_sim  # one patch copy, sim px
    X = model.encoding_extent_sim                              # all patch copies
    W = X + 2 * margin
    c0 = (N - X) // 2 - margin
    phase = np.mod(phi_enc.cpu().numpy(), 2 * np.pi)[c0:c0 + W, c0:c0 + W]
    cmap_enc = plt.get_cmap('twilight').copy(); cmap_enc.set_bad('black')
    if model.opaque_background:                                # blocked pixels drawn black
        phase = np.ma.masked_where(model.background_amp[0, 0].cpu().numpy()[c0:c0 + W, c0:c0 + W] == 0, phase)
    im_enc = ax_enc.imshow(phase, cmap=cmap_enc, vmin=0, vmax=2 * np.pi, interpolation='nearest')
    ax_enc.set_xticks([]); ax_enc.set_yticks([])
    for r0, q0, h, w in _encoding_boxes(model):               # cyan box per patch copy
        ax_enc.add_patch(Rectangle((q0 - c0 - 0.5, r0 - c0 - 0.5), w, h, fill=False,
                                   edgecolor='cyan', linewidth=1.2))
    ax_enc.set_title(f'Input encoding phase ({'asymmetric' if config.encoding_zero_based else 'symmetric'}, bin = {config.encoding_patch_scale}, '
                     f'gap = {config.encoding_gap_blocks})\n'
                     f'a = {enc_a}, patch {Ph}x{Pw} sim px, tiles {model.encoding_tile_pattern}'
                     f'{", frame 0" if model.encoding_mode == "split" else ""}'
                     f'{", no decoder shown" if getattr(model, "decoder", False) else ""}\n'
                     f'{_bg_tag(model)}',
                     fontsize=9)
    cb_enc = fig.colorbar(im_enc, ax=ax_enc, fraction=0.046, pad=0.03)
    cb_enc.set_ticks([0, np.pi, 2 * np.pi]); cb_enc.set_ticklabels(['0', 'π', '2π'])

    # ---- top-right: spatial error heatmap -----------------------------------
    # Detector (r, c) reads function k = r*pd_num_cols + c (matches model.py's
    # _integrate_photodiode_array), so the RMSE vector reshapes straight into
    # that grid. The dead space between detectors is collapsed: each detector
    # is one equal-size block colored by its own RMSE (no interpolation), so
    # this is a table of detectors, not a physical-scale map. origin='upper' (row 0 at the
    # top), same orientation as the input-plane panel, so low sim-grid indices are top-left
    # in both (was origin='lower' before 2026-10-08, which flipped this panel vertically).
    ax_heat = fig.add_subplot(gs[0, hist_w + cx_w:])
    rows, cols = config.pd_num_rows, config.pd_num_cols
    grid = rmse_np.reshape(rows, cols)   # row-major k = r*cols + c, matches model.py

    im = ax_heat.imshow(grid, cmap='hot', origin='upper', interpolation='nearest',
                         aspect='equal')
    ax_heat.set_xlabel(f'detector column (CtoC spacing {config.pd_col_spacing_px} sim px = {1e6 * config.pd_col_spacing:.3g} µm)')
    ax_heat.set_ylabel(f'detector row (CtoC spacing {config.pd_row_spacing_px} sim px = {1e6 * config.pd_row_spacing:.3g} µm)')
    ax_heat.set_title(f'Per-detector RMSE (pd size = {config.photodiode_pixels} sim px = '
                      f'{1e6 * config.photodiode_size:.3g} µm)', fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE')
    # Format each tick as its own scientific-notation value rather than a
    # shared power-of-ten offset -- matplotlib's default offset text box
    # renders right above the axes and collides with the title above at the
    # tiny RMSE values this project now reaches (~1e-6/1e-7).
    cbar.formatter = FuncFormatter(lambda val, pos: f'{val:.1e}')
    cbar.update_ticks()

    # ---- bottom row: target-vs-approx curves --------------------------------
    for i, k in enumerate(show_idx):
        ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3])
        ax.plot(a, target[:, k], '--', color='tab:green', label='target')
        ax.plot(a, f_hat[:, k], '-.', color='tab:red', label='approx')
        ax.set_title(f'f_{k}  (RMSE={rmse[k]:.2e}) [{label_of[k]}]', fontsize=9)
        ax.set_xlabel('a'); ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Test summary ({getattr(config, "test_readout", "global")} min-max readout)   mean RMSE={rmse.mean():.2e}   max RMSE={rmse.max():.2e}\nr={config.r}  {_enc_tag(config)}  Nf={config.Nf}  M={config.M}  K={config.num_layers}  {_bg_tag(model)}\nkey_to_input_dist = {config.key_to_enc_spacing} key_size = {config.phase_key_size}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Test summary saved → {save_path}')


def _save_mask_similarity(model, out_dir, dpi=200):
    '''
    Pairwise circular-phase similarity between the M phase keys, on the
    model's raw learned phase (post-sigmoid, at its native phase_key_size
    resolution -- the actual learnable parameter, not the upsampled/embedded
    sim-grid version).

    similarity(i, j) = | mean_pixels( exp(i * (phi_i - phi_j)) ) |, in [0, 1].
    '''
    M = model.M
    with torch.no_grad():
        phi = model.key_phases()[:, 0]   # [M, phase_key_size, phase_key_size]
    phi = phi.cpu().numpy().reshape(M, -1)

    if M < 2:
        print('[mask_similarity] M < 2 -- nothing to compare, skipping.')
        return None

    sim = np.eye(M)
    for i in range(M):
        for j in range(i + 1, M):
            s = np.abs(np.mean(np.exp(1j * (phi[i] - phi[j]))))
            sim[i, j] = sim[j, i] = s

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=dpi)
    im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('key'); ax.set_ylabel('key')
    ax.set_xticks(np.arange(M)); ax.set_yticks(np.arange(M))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='similarity')

    off_diag = sim[~np.eye(M, dtype=bool)]
    mean_sim = float(off_diag.mean())
    ax.set_title(f'Phase-key similarity (M={M})  mean off-diag={mean_sim:.3f}', fontsize=9)

    path = os.path.join(out_dir, 'phase_key_similarity.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Phase-key similarity saved → {path}  (mean off-diagonal = {mean_sim:.3f})')
    return sim


def _save_encoding_frames(model, config, out_dir, dpi=200, enc_a=0.3):
    '''Split mode only: encoding phase of every frame (key) at a = enc_a -- the encoding alone, WITHOUT the decoder --
    zoomed on the encoding extent. Non-encoding pixels show their true phase 0 (the base distribution the decoder is
    added onto), black box per patch copy. Saved as encoding_frames.png.'''
    if model.encoding_mode != 'split':
        return
    with torch.no_grad():
        dev = model.slm_aperture.device
        phi = model._encode_input(torch.tensor([enc_a], device=dev))[0]    # [M, N, N], encoding only (0 elsewhere)
    N, X, margin = config.N_sim, model.encoding_extent_sim, int(np.ceil(4 * model.enc_ratio))
    W = X + 2 * margin
    c0 = (N - X) // 2 - margin
    cmap = plt.get_cmap('twilight')
    M = model.M
    fig, axes = plt.subplots(1, M, figsize=(3.4 * M, 3.8), dpi=dpi, squeeze=False)
    for m in range(M):
        ax = axes[0, m]
        phase = np.mod(phi[m].cpu().numpy(), 2 * np.pi)[c0:c0 + W, c0:c0 + W]
        im = ax.imshow(phase, cmap=cmap, vmin=0, vmax=2 * np.pi,
                       interpolation='nearest')
        for r0, q0, h, w in _encoding_boxes(model):
            ax.add_patch(Rectangle((q0 - c0 - 0.5, r0 - c0 - 0.5), w, h, fill=False,
                                   edgecolor='black', linewidth=1.0))
        ax.set_xticks([]); ax.set_yticks([])
        s = model.encoding_shift
        ax.set_title(f'key {m}: displayed alpha {m * model.Np - s}..{(m + 1) * model.Np - 1 - s}', fontsize=9)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    cb.set_ticks([0, np.pi, 2 * np.pi]); cb.set_ticklabels(['0', 'π', '2π'])
    fig.suptitle(f'Encoding phase per frame (split), encoding only (decoder not shown, phase 0 elsewhere; {_bg_tag(model)}), a = {enc_a}   {_enc_tag(config)}  '
                 f'bin {config.encoding_patch_scale}, gap {config.encoding_gap_blocks}, '
                 f'tiles {model.encoding_tile_pattern}', fontsize=10)
    path = os.path.join(out_dir, 'encoding_frames.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Encoding frames saved → {path}')


def _save_plane_layout(model, config, out_dir, dpi=200, enc_a=0.3):
    '''
    One row, every panel the full N_sim x N_sim grid, so the planes' relative sizes are
    directly comparable: phase key 0 | encoding plane (a = enc_a, cyan border = encoding
    patch) | diffractive layer 1 | detector plane (cyan border = extent of the detector
    array). Phases wrapped to [0, 2pi); blocked (zero-amplitude) pixels are black -- e.g.
    everything outside the layer aperture. Outside the key the key plane is lit at phase 0.
    '''
    N, dx_um = config.N_sim, config.sim_dx * 1e6
    cmap = plt.get_cmap('twilight').copy(); cmap.set_bad('black')
    # Large grids are block-averaged (complex field, then angle) to <= ~1000 px before drawing: drawing every sim px with
    # 'nearest' at a few hundred screen px subsamples them and turns smooth key structure into noise.
    f = max(1, int(np.ceil(N / 1000)))
    def show(U):                                               # complex [N, N] -> masked phase image [N/f, N/f]
        U = U[None, None]
        if f > 1:
            U = torch.complex(F.avg_pool2d(U.real, f, ceil_mode=True), F.avg_pool2d(U.imag, f, ceil_mode=True))
        U = U[0, 0]
        return np.ma.masked_where((U.abs() == 0).cpu().numpy(), torch.remainder(torch.angle(U), 2 * np.pi).cpu().numpy())
    with torch.no_grad():
        dev = model.slm_aperture.device
        key = show(model._key_field()[0, 0])                   # lit at phase 0 outside the key
        enc = show(model.input_plane_field(torch.tensor([enc_a], device=dev))[0, 0])   # K > 0: opaque background black
        lay = show(model._layer_field(0)[0, 0]) if model.num_layers > 0 else None       # K = 0: panel left blank

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), dpi=dpi)
    kp, ep = config.key_slm_dx * 1e6, config.enc_slm_dx * 1e6            # SLM pitches, um
    py, px = config.encoding_y_num_px, config.encoding_x_num_px
    panels = [(key, f'Phase key 0  ({config.phase_key_size} slm px = {config.phase_key_size * kp:.1f} µm)'),
              (enc, f'Encoding plane, a = {enc_a}  (patch {py}x{px} slm px = {py * ep:.1f}x{px * ep:.1f} µm,\n'
                    f'bin {config.encoding_patch_scale}, gap {config.encoding_gap_blocks}, '
                    f'tiles {model.encoding_tile_pattern})'),
              (lay, f'Diffractive layer 1  ({config.layer_size} slm px = {config.layer_size * ep:.1f} µm)' if lay is not None
                    else 'No diffractive layers (K = 0)')]
    im = None
    for ax, (img, title) in zip(axes[:3], panels):
        if img is None:                                        # K = 0 layer panel
            ax.imshow(np.zeros((N, N)), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        else:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=2 * np.pi, interpolation='nearest',
                           extent=(-0.5, N - 0.5, N - 0.5, -0.5))   # sim-px coordinates even when block-averaged
        ax.set_title(title, fontsize=9)
    for r0, q0, h, w in _encoding_boxes(model):               # cyan box per patch copy
        axes[1].add_patch(Rectangle((q0 - 0.5, r0 - 0.5), w, h, fill=False, edgecolor='cyan', linewidth=1.2))

    # detector plane: grey canvas, border around the array (outermost detector pixel edges)
    ax = axes[3]
    ax.imshow(np.zeros((N, N)), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    r0, r1 = int(model.pd_rows_idx.min()), int(model.pd_rows_idx.max())
    c0, c1 = int(model.pd_cols_idx.min()), int(model.pd_cols_idx.max())
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False, edgecolor='cyan', linewidth=1.2))
    ax.set_title(f'Detector plane  ({config.pd_num_rows} x {config.pd_num_cols} detectors, '
                 f'pdsize = {config.photodiode_pixels} sim px = {1e6 * config.photodiode_size:.3g} µm, '
                 f'pd_spacing = {config.pd_row_spacing_px} sim px = {1e6 * config.pd_row_spacing:.3g} µm,\n'
                 f'array {r1 - r0 + 1} sim px = {(r1 - r0 + 1) * dx_um:.1f} µm)', fontsize=9)
    for ax in axes:
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
        ax.set_xticks([0, N // 2, N - 1]); ax.set_yticks([0, N // 2, N - 1]); ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes[:3], fraction=0.015, pad=0.01)
    cbar.set_ticks([0, np.pi, 2 * np.pi]); cbar.set_ticklabels(['0', 'π', '2π'])
    fig.suptitle(f'Plane layout on the {N} x {N} sim grid ({N * dx_um:.1f} µm; 1 px = {dx_um * 1e3:.0f} nm; '
                 f'key SLM px {kp:.3g} µm, encoding SLM px {ep:.3g} µm, camera px {config.camera_dx * 1e6:.3g} µm)\n'
                 f'{_enc_tag(config)}  Nf={config.Nf}  K={config.num_layers}  M={config.M}  {_bg_tag(model)}'
                 f'{"  (encoding plane: frame 0)" if model.encoding_mode == "split" or getattr(model, "decoder_per_key", False) else ""}', fontsize=10)
    path = os.path.join(out_dir, 'plane_layout.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Plane layout saved → {path}')


def _save_phase_keys(model, out_dir, dpi=200):
    '''Grid of the learned phase-key masks, one per key (M total).'''
    with torch.no_grad():
        masks = model.key_phases().cpu()                          # already in [0, 2pi)
    M = masks.shape[0]

    fig, axes = plt.subplots(1, M, figsize=(2.2 * M, 2.4), dpi=dpi, squeeze=False)
    im = None
    for m in range(M):
        ax = axes[0, m]
        im = ax.imshow(masks[m, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
        ax.set_title(f'key {m}', fontsize=8)
        ax.axis('off')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels(['0', 'π', '2π'])
    fig.suptitle('Optimized phase-key masks')
    path = os.path.join(out_dir, 'phase_keys_final.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Phase-key masks saved → {path}')


def _save_layer_masks(model, out_dir, dpi=200):
    '''Grid of the learned diffractive-layer phase masks (one per layer,
    shared across all M phase keys).'''
    if model.num_layers == 0:                                  # K = 0: nothing to draw
        print('No diffractive layers (K = 0): layer_masks_final.png skipped')
        return
    with torch.no_grad():
        masks = model._quantize(torch.sigmoid(model.layer_phases) * 2 * np.pi).cpu()   # as displayed (slm_phase_bits)
    K = masks.shape[0]

    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3), dpi=dpi, squeeze=False)
    im = None
    for k in range(K):
        ax = axes[0, k]
        im = ax.imshow(masks[k, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
        ax.set_title(f'layer {k}', fontsize=8)
        ax.axis('off')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels(['0', 'π', '2π'])
    fig.suptitle('Diffractive layer phase masks')
    path = os.path.join(out_dir, 'layer_masks_final.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Layer masks saved → {path}')


def _save_decoder_mask(model, config, out_dir, dpi=200):
    '''E+D architecture: the learned decoder phase(s) on the encoding plane, one value per SLM pixel, wrapped to
    [0, 2pi). Laid out like phase_keys_final.png: 1 row, one column per decoder (M with decoder_per_key, decoder m is
    shown with key m; else 1 shared). With decoder_over_encoding = False the encoding pixels (decoder adds exactly 0
    there) are lime, a colour outside the twilight map (black would read as pi on it). Saved as decoder_mask.png;
    skipped when there is no decoder.'''
    if not getattr(model, 'decoder', False):
        return
    y0, x0, D = model.dec_y0, model.dec_x0, model.decoder_size
    with torch.no_grad():
        phi = model.decoder_phases_px()[0, :, y0:y0 + D, x0:x0 + D].cpu().numpy()     # [n_decoders, n, n] SLM px
        on = model.decoder_mask_px[y0:y0 + D, x0:x0 + D].cpu().numpy() > 0.5
    cmap = plt.get_cmap('twilight').copy(); cmap.set_bad('lime')
    n, nd = model.decoder_size, phi.shape[0]
    fig, axes = plt.subplots(1, nd, figsize=(2.6 * nd + 0.8, 3.0), dpi=dpi, squeeze=False)
    im = None
    for m in range(nd):
        ax = axes[0, m]
        im = ax.imshow(np.ma.masked_where(~on, np.mod(phi[m], 2 * np.pi)), cmap=cmap, vmin=0, vmax=2 * np.pi,
                       interpolation='nearest')
        ax.set_title(f'decoder {m}' if model.decoder_per_key else 'decoder (shared by all keys)', fontsize=8)
        ax.axis('off')
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_ticks([0, np.pi, 2 * np.pi]); cb.set_ticklabels(['0', 'π', '2π'])
    lime = '' if on.all() else '; lime = encoding pixels (decoder adds 0)'
    fig.suptitle(f'Encoding-plane decoder phase{"s, one per key" if model.decoder_per_key else ""}, '
                 f'{n} x {n} SLM px ({n * config.enc_slm_dx * 1e3:.2f} mm){lime}\n'
                 f'{_enc_tag(config)}  Nf={config.Nf}  M={config.M}  K={config.num_layers}', fontsize=9, y=1.08)
    path = os.path.join(out_dir, 'decoder_mask.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Decoder mask saved → {path}')


def _save_key_evolution(model, config, agg, out_dir, dpi=200):
    '''
    M rows x 5 columns, one row per phase key m:
      col 1: learned phase key m
      col 2: intensity key m deposits on the encoding plane (the key's field only, before the encoding / decoder
             phase and before the opaque-background mask at K > 0), cropped to the layer aperture (K = 0: the key),
             encoding patch copies outlined, mean intensity over them vs. the unit incident intensity
      col 3: phase of key m's field at the encoding plane (same crop / outline, wrapped to [0, 2pi))
      col 4: key m's OWN contribution to the output, c_m(a) = (I_m(a) - Pmin) / (M (Pmax - Pmin)), for the
             median-error function of the full model: the min-max readout of the frame average is exactly
             f_hat = sum_m c_m, so c_m is what frame m adds (not a cumulative sum over keys 0..m)
      col 5: spectrum of c_m over a: |DFT| at integer frequency nu = 0 .. N_alpha - 1 (the test grid is uniform over
             one period of a), mean over all Nf functions, normalised to the largest bin over all keys. Split mode:
             frame m's displayed harmonics |alpha| are shaded (interference with the transparent background puts the
             frame's own alphas into the intensity; differences within the frame land at nu < Np).
    At K > 0 only the outlined encoding pixels pass on (opaque background); at K = 0 the whole field does.
    '''
    M = model.M
    # per-key contributions in the min-max readout units (sum over m = f_hat)
    Pmin, Pmax = (v.double() for v in agg['p_range'])                  # scalars, or [1, Nf] with --readout perfunc
    C = ((agg['I'].double() - Pmin) / (M * (Pmax - Pmin))).numpy()     # [N, M, Nf]
    rmse_full = FunctionApproxLoss.per_function_rmse(agg['f_hat'], agg['target'])
    k_show = int(torch.argsort(rmse_full)[len(rmse_full) // 2])
    a_np = agg['a'].numpy()
    NA = model.N_alpha
    spec = np.abs(np.fft.rfft(C, axis=0))[:NA].mean(2).T                # [M, NA] (a grid uniform over one period)
    spec = spec / max(spec[:, 1:].max(), 1e-300)                        # DC excluded from the scale
    shift = model.encoding_shift
    with torch.no_grad():
        key_I, key_phi = model.key_intensity_at_encoding()       # [M, N_sim, N_sim] each
        key_I = key_I.cpu().numpy()
        key_phi = torch.remainder(key_phi, 2 * np.pi).cpu().numpy()   # angle() is (-pi, pi]; wrap to [0, 2pi)
        phases = torch.remainder(model.key_phases(), 2 * np.pi).cpu().numpy()[:, 0]   # wrapped to [0, 2pi)

    N = model.N_sim
    L = model.layer_size_sim if model.num_layers > 0 else model.phase_key_size_sim   # K = 0: crop to the key
    ly0 = (N - L) // 2
    # pixels the mean intensity is taken over: every encoding patch copy (K = 0), or only the encoding pixels themselves
    # when the background is opaque (K > 0: the only light that passes on)
    if model.opaque_background:
        in_patch = model.background_amp[0, 0].cpu().numpy() > 0.5
    else:
        in_patch = np.zeros((N, N), dtype=bool)
        for r0, q0, h, w in _encoding_boxes(model):
            in_patch[r0:r0 + h, q0:q0 + w] = True
    foot = 'on encoding pixels' if model.opaque_background else 'in footprint'

    fig, axes = plt.subplots(M, 5, figsize=(24, 3.6 * M), dpi=dpi, squeeze=False)
    def phase_cbar(im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_ticks([0, np.pi, 2 * np.pi]); cb.set_ticklabels(['0', 'π', '2π'])
    def boxes(ax):
        for r0, q0, h, w in _encoding_boxes(model):
            ax.add_patch(Rectangle((q0 - ly0 - 0.5, r0 - ly0 - 0.5), w, h, fill=False, edgecolor='cyan', linewidth=1.2))

    for m in range(M):
        ax = axes[m, 0]
        im = ax.imshow(phases[m], cmap='twilight', vmin=0, vmax=2 * np.pi)
        phase_cbar(im, ax)
        ax.set_ylabel(f'key {m}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if m == 0:
            ax.set_title('Phase key', fontsize=11, fontweight='bold')

        ax = axes[m, 1]
        I = key_I[m]
        enh = I[in_patch].mean()                              # incident (unit-amplitude) intensity == 1
        crop = I[ly0:ly0 + L, ly0:ly0 + L]
        # vmin=0: otherwise a uniform field (z=0: |exp(j*phi)|^2 == 1 everywhere) gets its
        # ~1e-7 float rounding stretched over the whole colormap and looks like speckle
        im = ax.imshow(crop, cmap='inferno', vmin=0, vmax=max(crop.max(), 1.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('× incident', fontsize=8)
        boxes(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'mean intensity {foot} = {enh:.2f}x incident', fontsize=9)
        if m == 0:
            ax.set_title('Intensity at encoding plane', fontsize=11, fontweight='bold')

        ax = axes[m, 2]
        im = ax.imshow(key_phi[m][ly0:ly0 + L, ly0:ly0 + L], cmap='twilight', vmin=0, vmax=2 * np.pi)
        phase_cbar(im, ax)
        boxes(ax); ax.set_xticks([]); ax.set_yticks([])
        if m == 0:
            ax.set_title('Phase at encoding plane', fontsize=11, fontweight='bold')

        ax = axes[m, 3]
        ax.plot(a_np, C[:, m, k_show], '-', color='tab:red', lw=1.4)
        ax.set_xlim(a_np[0], a_np[-1])
        if m == M - 1:
            ax.set_xlabel('a')
        ax.set_ylabel(f'c_{m}(a)', fontsize=9)
        ax.set_title(f'f_{k_show}: key {m} contribution, mean {C[:, m, k_show].mean():.3f}, '
                     f'swing {np.ptp(C[:, m, k_show]):.3f}', fontsize=9)
        if m == 0:
            ax.annotate('Contribution of this key alone', xy=(0.5, 1), xytext=(0, 22), xycoords='axes fraction',
                        textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax = axes[m, 4]
        nu = np.arange(NA)
        if model.encoding_mode == 'split':
            al = np.arange(m * model.Np, (m + 1) * model.Np) - shift          # alphas displayed in frame m
            lo, hi = (0 if al.min() <= 0 <= al.max() else np.abs(al).min()), np.abs(al).max()
            ax.axvspan(lo - 0.5, hi + 0.5, color='#2a78d6', alpha=0.15, lw=0, label=f'frame {m}: |alpha| {lo}-{hi}')
            ax.axvspan(-0.5, model.Np - 0.5, color='0.5', alpha=0.10, lw=0, label=f'within-frame differences < {model.Np}')
            inb = spec[m][(nu >= lo) & (nu <= hi) & (nu > 0)].sum() / max(spec[m][1:].sum(), 1e-300)
            ax.set_title(f'{100 * inb:.0f}% of the non-DC spectrum inside |alpha| {lo}-{hi}', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
        ax.bar(nu[1:], spec[m][1:], width=0.8, color='tab:red')
        ax.set_xlim(-0.5, NA - 0.5); ax.set_ylim(0, 1.05)
        if m == M - 1:
            ax.set_xlabel('frequency nu (cycles per unit a)')
        ax.set_ylabel('|DFT|, mean over functions', fontsize=8)
        if m == 0:
            ax.annotate('Spectrum of this key\'s contribution', xy=(0.5, 1), xytext=(0, 22), xycoords='axes fraction',
                        textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle(
        "Phase keys, each key's field at the encoding plane (before the encoding phase; cyan box = encoding footprint), "
        f"and each key's own contribution c_m to the output (sum over m = f_hat; median-error function f_{k_show})\n"
        + r'($N = r \cdot 2 N_\alpha N_f$,  '
        + f'r={config.r}  {_enc_tag(config)}  Nf={config.Nf}  M={config.M}  K={config.num_layers};  {_bg_tag(model)})',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.5, h_pad=2.0)
    path = os.path.join(out_dir, 'key_evolution.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Key evolution saved → {path}')


def _readout_fits(agg):
    '''
    Readout diagnostics from the frame-averaged intensity P = agg['I'].mean(1) [N, Nf] and the targets f [N, Nf]
    (float64). Three readouts of the SAME intensities, mean per-function RMSE each:
      minmax   : the model's readout, (P - Pmin) / (Pmax - Pmin) over the whole grid and all detectors
      global   : best single scale s and offset b for all detectors (least squares, f ~ s * P + b) -- the old
                 learned readout at its optimum
      detector : best s_k, b_k per detector (lower bound for any affine readout)
    plus, per detector, the fit f_hat_k ~ gain_k * f_k + offset_k of the min-max output against the target (ideal 1 / 0)
    and the modulation depth of P_k over a, (max - min) / (max + min). Pmin / Pmax locations are (a index, detector).
    '''
    P = agg['I'].mean(1).double(); f = agg['target'].double(); fh = agg['f_hat'].double()
    rmse = lambda x: float(torch.sqrt(((x - f) ** 2).mean(0)).mean())
    A = torch.stack([P.reshape(-1), torch.ones(P.numel(), dtype=P.dtype)], 1)
    s, b = torch.linalg.lstsq(A, f.reshape(-1, 1)).solution[:, 0]
    Pc, fc = P - P.mean(0), f - f.mean(0)
    s_k = (Pc * fc).sum(0) / (Pc ** 2).sum(0).clamp_min(1e-300)
    b_k = f.mean(0) - s_k * P.mean(0)
    fhc = fh - fh.mean(0)
    gain = (fhc * fc).sum(0) / (fc ** 2).sum(0).clamp_min(1e-300)
    offset = fh.mean(0) - gain * f.mean(0)
    Pmax_k, Pmin_k = P.max(0).values, P.min(0).values
    i_max, i_min = int(P.argmax()), int(P.argmin())
    return dict(rmse_minmax=rmse(fh), rmse_global_affine=rmse(s * P + b), rmse_detector_affine=rmse(s_k * P + b_k),
                gain=gain.numpy(), offset=offset.numpy(),
                depth=((Pmax_k - Pmin_k) / (Pmax_k + Pmin_k).clamp_min(1e-300)).numpy(),
                where_max=divmod(i_max, P.shape[1]), where_min=divmod(i_min, P.shape[1]))


def _save_readout_calibration(agg, fits, config, model, out_dir, dpi=200):
    '''
    readout_calibration.png: 2-D histogram of min-max output f_hat against target f over every (a, function), identity
    line, the points that set Pmax / Pmin marked; text box with the three readout RMSEs (_readout_fits). If min-max ~
    global affine, the readout costs nothing and the error is in the optics; if min-max >> global affine, a few extreme
    points set Pmin / Pmax badly; detector affine << global points at uneven detector gain / offset.
    '''
    f = agg['target'].numpy(); fh = agg['f_hat'].numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=dpi)
    lo, hi = min(f.min(), fh.min(), 0.0), max(f.max(), fh.max(), 1.0)
    h = ax.hist2d(f.reshape(-1), fh.reshape(-1), bins=200, range=[[lo, hi], [lo, hi]], cmap='Blues',
                  norm=matplotlib.colors.LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03).set_label('(a, function) samples per bin')
    ax.plot([lo, hi], [lo, hi], '-', color='0.3', lw=1, label='identity')
    for (n, k), lab, mk in ((fits['where_max'], 'sets Pmax', '^'), (fits['where_min'], 'sets Pmin', 'v')):
        ax.plot(f[n, k], fh[n, k], mk, color='#eb6834', ms=9, mec='white', mew=1,
                label=f'{lab}: f_{k}, a = {agg["a"][n].item():+.3f}')
    r = fits
    ax.text(0.03, 0.97, f'mean per-function RMSE\n'
            f'  min-max ({getattr(config, 'test_readout', 'global'):7s}):  {r["rmse_minmax"]:.3e}\n'
            f'  best global scale+offset: {r["rmse_global_affine"]:.3e}  (min-max {r["rmse_minmax"] / r["rmse_global_affine"]:.2f}x)\n'
            f'  best per-detector:        {r["rmse_detector_affine"]:.3e}  (min-max {r["rmse_minmax"] / r["rmse_detector_affine"]:.2f}x)',
            transform=ax.transAxes, va='top', ha='left', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    pf = getattr(config, 'test_readout', 'global') == 'perfunc'
    ax.set_xlabel(f'target f ({"per-function" if pf else "global"} min-max)')
    ax.set_ylabel(f'model output f_hat ({"per-function " if pf else ""}min-max over the test grid)')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'Readout calibration   {_enc_tag(config)}  Nf={config.Nf}  M={config.M}  K={config.num_layers}\n'
                 f'{_bg_tag(model)}', fontsize=9)
    fig.tight_layout(); path = os.path.join(out_dir, 'readout_calibration.png')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f'Readout calibration saved → {path}')


# ──────────────────────────── main entry point ────────────────────────────── #

def _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse, fits=None, readout='global'):
    '''Append one results row to the sweep CSV, creating the file if needed.'''
    row = {
        'sweep'     : sweep_name or '',
        'label'     : run_label  or '',
        'ckpt'      : ckpt_path  or '',
        'n_samples' : len(agg['a']),
        'loss_mean' : float(np.mean(agg['loss'])),
        'rmse_mean' : float(rmse.mean()),
        'rmse_max'  : float(rmse.max()),
        'rmse_min'  : float(rmse.min()),
        'diff_eff_mean' : float(agg['de'].mean()) if 'de' in agg else float('nan'),
        'readout'   : readout,
    }
    if fits is not None:   # readout diagnostics (_readout_fits)
        row.update(rmse_global_affine=fits['rmse_global_affine'], rmse_detector_affine=fits['rmse_detector_affine'],
                   gain_median=float(np.median(fits['gain'])), depth_median=float(np.median(fits['depth'])))
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    if file_exists:
        # Appending to a CSV written before a column existed: keep ITS header so
        # rows stay aligned (the extra column is dropped for that file only).
        with open(csv_path, newline='') as f:
            header = next(csv.reader(f), None)
        if header:
            fieldnames = header
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Results appended → {csv_path}')


def _load_and_evaluate(ckpt_path, readout='global', test_batch=None):
    '''
    Load a checkpoint (config + model + criterion) and run one eval pass over its own dense test grid. 
    '''
    config = init_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preserve local paths -- machine-specific, must not be overwritten by whatever was saved inside the checkpoint.
    _path_keys = ('log_dir', 'model_dir', 'tfboard_dir')
    saved_paths = {k: getattr(config, k, None) for k in _path_keys}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    check_checkpoint_format(ckpt, ckpt_path)
    apply_saved_config(config, ckpt['config'])
    for k, v in saved_paths.items():
        if v is not None:
            setattr(config, k, v)
    recompute_derived(config)
    if test_batch:   # --test_batch: memory only, the results do not depend on it
        config.test_batch_size = int(test_batch)

    model     = TimeMultiplexedNFA(config).to(device)
    criterion = FunctionApproxLoss(config).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    print(f'Loaded: {ckpt_path}  (epoch {ckpt["epoch"]})')
    model.eval(); criterion.eval()

    _, _, test_loader, target_fn = get_function_approx_dataloaders(config)
    config.test_readout = readout
    agg = _eval_loop(model, test_loader, criterion, device, desc='Testing', per_function=readout == 'perfunc')
    return agg, config, model, criterion


def evaluate(ckpt_path=None, out_dir=None, csv_path=None, sweep_name=None, run_label=None, readout='global', test_batch=None):
    ckpt_path = ckpt_path or init_params().ckpt_to_load
    assert readout in ('global', 'perfunc'), f'unknown readout {readout!r}'

    if ckpt_path is not None:
        agg, config, model, criterion = _load_and_evaluate(ckpt_path, readout, test_batch)
    else:
        print('[WARNING] No checkpoint provided — evaluating random init.')
        config = init_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model     = TimeMultiplexedNFA(config).to(device)
        criterion = FunctionApproxLoss(config).to(device)
        model.eval(); criterion.eval()
        _, _, test_loader, target_fn = get_function_approx_dataloaders(config)
        config.test_readout = readout
        agg = _eval_loop(model, test_loader, criterion, device, desc='Testing', per_function=readout == 'perfunc')

    rmse = _print_summary(agg, tag=f'Test ({readout} min-max readout)')

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(ckpt_path) if ckpt_path else '.', '..', 'test' if readout == 'global' else 'test_perfunc'
        )
    os.makedirs(out_dir, exist_ok=True)

    fits = _readout_fits(agg)
    print(f'  Readout RMSE        : min-max ({readout}) {fits["rmse_minmax"]:.4e}  best global affine {fits["rmse_global_affine"]:.4e}  '
          f'best per-detector affine {fits["rmse_detector_affine"]:.4e}')
    if csv_path is not None:
        _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse, fits, readout)

    # Per-function RMSE [Nf], index k = detector (row*pd_num_cols + col).
    # Saved so per-detector maps / asymmetry stats can be redone without re-testing.
    np.save(os.path.join(out_dir, 'per_function_rmse.npy'), rmse.cpu().numpy())

    _save_test_summary(agg, rmse, config, model, os.path.join(out_dir, 'test_summary.png'))
    _save_readout_calibration(agg, fits, config, model, out_dir)
    _save_plane_layout(model, config, out_dir)
    _save_encoding_frames(model, config, out_dir)

    # ── Phase keys / layers / key similarity ───────────────────────────────
    _save_phase_keys(model, out_dir)
    _save_layer_masks(model, out_dir)
    _save_decoder_mask(model, config, out_dir)
    _save_mask_similarity(model, out_dir)
    if getattr(config, 'prop_method', 'asm') != 'spectral':   # needs the sim-grid key -> LUNA propagation
        _save_key_evolution(model, config, agg, out_dir)

    print(f'\nAll outputs saved to: {os.path.abspath(out_dir)}')


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    # ── Hardcoded fallback (used when running test.py directly, no CLI args) ──
    CKPT_PATH = None   # <-- set to a checkpoint path, or always pass --ckpt

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default=None, help='Path to checkpoint')
    parser.add_argument('--out_dir', default=None, help='Output directory for plots')
    parser.add_argument('--csv',     default=None, help='CSV file to append results to')
    parser.add_argument('--sweep',   default=None, help='Sweep name (written to CSV)')
    parser.add_argument('--label',   default=None, help='Run label (written to CSV)')
    parser.add_argument('--readout', default='global', choices=('global', 'perfunc'),
                        help="min-max readout: 'global' (whole grid + all detectors, default) or 'perfunc' (each function "
                             "over a on its own, the bench readout); perfunc writes to <run>/test_perfunc by default")
    parser.add_argument('--test_batch', type=int, default=None,
                        help='override the checkpoint test_batch_size (GPU memory only; results do not depend on it)')
    args = parser.parse_args()

    evaluate(
        ckpt_path  = args.ckpt or CKPT_PATH,
        out_dir    = args.out_dir,
        csv_path   = args.csv,
        sweep_name = args.sweep,
        run_label  = args.label,
        readout    = args.readout,
        test_batch = args.test_batch,
    )
