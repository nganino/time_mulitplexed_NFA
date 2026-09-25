'''
Time-Multiplexed NFA — test / evaluation (Nonlinear Function Approximation)

Usage:
    python test.py --ckpt logs/.../model/best.pth
    python test.py --ckpt logs/.../model/epoch=050.pth --out_dir out/

Outputs saved to <out_dir>/:
    test_summary.png            — combined summary figure (see _save_test_summary):
                                   error-distribution histogram (top-left), spatial
                                   RMSE heatmap over the detector plane (top-right),
                                   and target-vs-approx curves for representative
                                   functions -- best-fit / worst-fit / mid-error
                                   (bottom row, paper Fig. 2a/2b/2c-style)
    phase_keys_final.png        — learned phase-key masks (one per key, M total)
    layer_masks_final.png       — learned diffractive-layer phase masks
    phase_key_similarity.png    — pairwise phase similarity between the M keys
                                   (low similarity = keys are learning genuinely
                                   different things -- see _save_mask_similarity)
Optionally appends a summary row (loss, RMSE mean/max/min) to --csv, if given.

NOTE: rewritten in place for the NFA pivot (image classification -> parallel
nonlinear function approximation), same as train.py -- the old classification
version (confusion matrix, per-class F1, misclassified-samples grid, arbitrary
image-dataset selection) doesn't carry over conceptually to a regression task,
so this isn't a side-by-side addition, it's a replacement.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from config import init_params, recompute_derived
from model import TimeMultiplexedNFA
from dataloader import get_function_approx_dataloaders
from loss import FunctionApproxLoss


# ─────────────────────────── evaluation loop ──────────────────────────────── #

def _eval_loop(model, loader, criterion, device, desc='Evaluating'):
    '''
    Single evaluation pass over the dense a-grid loader.

    Returns
    -------
    agg : dict with -- all sorted by ascending `a` for gap-free plotting --
        'a'      : [N] input values
        'f_hat'  : [N, Nf] normalized model output
        'target' : [N, Nf] true (already [0,1]-normalized) function values
        'loss'   : per-batch MSE (one value per batch, NOT sorted -- just a list)
    '''
    agg = defaultdict(list)
    was_training_model = model.training
    was_training_crit  = criterion.training
    model.eval(); criterion.eval()

    with torch.no_grad():
        for a, target in tqdm(loader, desc=desc):
            a = a.to(device); target = target.to(device)
            I_vec = model(a)
            loss, f_hat = criterion(I_vec, target)

            agg['loss'].append(loss.item())
            agg['a'].append(a.cpu())
            agg['f_hat'].append(f_hat.cpu())
            agg['target'].append(target.cpu())

    if was_training_model:
        model.train()
    if was_training_crit:
        criterion.train()

    agg['a']      = torch.cat(agg['a'])
    agg['f_hat']  = torch.cat(agg['f_hat'])
    agg['target'] = torch.cat(agg['target'])

    order = torch.argsort(agg['a'])
    agg['a']      = agg['a'][order]
    agg['f_hat']  = agg['f_hat'][order]
    agg['target'] = agg['target'][order]

    return agg


# ──────────────────────────── print summary ───────────────────────────────── #

def _print_summary(agg, tag='Test'):
    rmse = FunctionApproxLoss.per_function_rmse(agg['f_hat'], agg['target'])   # [Nf]

    print(f'\n── {tag} Results ──────────────────────────────────────────')
    print(f'  Loss (MSE)          : {np.mean(agg["loss"]):.4e}')
    print(f'  Per-function RMSE   : mean={rmse.mean():.4e}  '
          f'max={rmse.max():.4e}  min={rmse.min():.4e}')
    print(f'──────────────────────────────────────────────────────────\n')
    return rmse


# ──────────────────────────── save figures ────────────────────────────────── #

def _save_test_summary(agg, rmse, config, save_path, n_show=4, dpi=200):
    '''
    Combined test-summary figure -- one PNG instead of separate files:
      - top-left:   histogram of per-function RMSE (paper Fig. 2a-style)
      - top-middle: RMSE vs. each function's target dynamic range (max-min
        over the domain) -- a simple difficulty/complexity proxy, testing
        whether more-oscillatory/higher-range functions are the ones fit
        worst. The same best/middle/worst functions highlighted in the
        bottom row are marked here too, tying the two panels together.
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
    # ~half-width it had before this panel existed, the complexity scatter
    # (simple, doesn't need much width) gets the least, heatmap gets the rest
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

    # ---- top-middle: RMSE vs. target dynamic range (difficulty proxy) -------
    ax_cx = fig.add_subplot(gs[0, hist_w:hist_w + cx_w])
    target_range = target.max(axis=0) - target.min(axis=0)   # [Nf], each in [0,1]
    ax_cx.scatter(target_range, rmse_np, s=14, alpha=0.5, color='steelblue', label='_nolegend_')
    _rank_colors = {'best': 'tab:green', 'middle': 'tab:orange', 'worst': 'tab:red'}
    _seen_labels = set()
    for k, lab in label_of.items():
        ax_cx.scatter(target_range[k], rmse_np[k], s=45, marker='*',
                       color=_rank_colors[lab], edgecolors='black', linewidths=0.5,
                       label=(lab if lab not in _seen_labels else '_nolegend_'))
        _seen_labels.add(lab)
    ax_cx.set_yscale('log')
    ax_cx.set_xlabel('target dynamic range (max - min over a)')
    ax_cx.set_ylabel('RMSE')
    ax_cx.set_title('Error vs. target complexity', fontsize=10)
    ax_cx.legend(fontsize=7)

    # ---- top-right: spatial error heatmap -----------------------------------
    # Detector (r, c) reads function k = r*pd_num_cols + c (matches model.py's
    # _integrate_photodiode_array), so the RMSE vector reshapes straight into
    # that grid. The dead space between detectors is collapsed: each detector
    # is one equal-size block colored by its own RMSE (no interpolation), so
    # this is a table of detectors, not a physical-scale map.
    ax_heat = fig.add_subplot(gs[0, hist_w + cx_w:])
    rows, cols = config.pd_num_rows, config.pd_num_cols
    grid = rmse_np.reshape(rows, cols)   # row-major k = r*cols + c, matches model.py

    im = ax_heat.imshow(grid, cmap='hot', origin='lower', interpolation='nearest',
                         aspect='equal')
    ax_heat.set_xlabel(f'detector column (CtoC spacing {config.pd_col_spacing_px} px)')
    ax_heat.set_ylabel(f'detector row (CtoC spacing {config.pd_row_spacing_px} px)')
    ax_heat.set_title(f'Per-detector RMSE (pd size = {config.photodiode_pixels} px)', fontsize=10)
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

    fig.suptitle(f'Test summary   mean RMSE={rmse.mean():.2e}   max RMSE={rmse.max():.2e}\nr={config.r}  Np={config.Np}  Nf={config.Nf}  M={config.M}  K={config.num_layers}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Test summary saved → {save_path}')


def _save_mask_similarity(model, out_dir, dpi=200):
    '''
    Pairwise circular-phase similarity between the M phase keys, on the
    model's raw learned phase (post-sigmoid, at its native slm_x_num
    resolution -- the actual learnable parameter, not the upsampled/embedded
    sim-grid version).

    similarity(i, j) = | mean_pixels( exp(i * (phi_i - phi_j)) ) |, in [0, 1].
    '''
    M = model.M
    with torch.no_grad():
        phi = (torch.sigmoid(model.slm_phases) * 2 * np.pi)[:, 0]   # [M, slm_x_num, slm_x_num]
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


def _save_phase_keys(model, out_dir, dpi=200):
    '''Grid of the learned phase-key masks, one per key (M total).'''
    with torch.no_grad():
        masks = (torch.sigmoid(model.slm_phases) * 2 * np.pi).cpu()
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
    with torch.no_grad():
        masks = (torch.sigmoid(model.layer_phases) * 2 * np.pi).cpu()
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


def _save_key_evolution(model, config, ckpt_path, device, out_dir, dpi=200):
    '''
    M rows x 3 columns, one row per phase key m:
      col 1: learned phase key m
      col 2: intensity key m deposits on the encoding plane (cropped to the
             layer aperture, footprint of the encoding region outlined, mean
             footprint intensity vs. the unit
             incident intensity -- NOT a power fraction, since the key aperture
             fills the whole grid and most power is unmodulated background)
      col 3: target vs. approximation using keys 0..m only (mean of the first
             m+1 per-key detector readouts through the trained affine readout),
             for the median-error function of the FULL model, held fixed across
             rows. The last row reproduces the ordinary test result.
    Key order is arbitrary (the keys are averaged symmetrically), and the
    readout scale/bias were trained for the full M-average, so partial sums
    need not improve monotonically.
    '''
    M = model.M
    if M < 2:
        print('[key_evolution] M < 2 -- skipping.')
        return

    criterion = FunctionApproxLoss(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get('criterion') is not None:
        criterion.load_state_dict(ckpt['criterion'], strict=False)
    criterion.eval()

    _, _, test_loader, _ = get_function_approx_dataloaders(config)
    a_l, I_l, t_l = [], [], []
    with torch.no_grad():
        for a, target in tqdm(test_loader, desc='Per-key eval'):
            I_l.append(model(a.to(device)).cpu())      # [B, M, rows, cols]
            a_l.append(a); t_l.append(target)
    a_all = torch.cat(a_l); target = torch.cat(t_l)
    order = torch.argsort(a_all)
    a_all, target = a_all[order], target[order]
    I_all = torch.cat(I_l)[order]
    I_all = I_all.reshape(I_all.shape[0], M, -1)       # [N, M, Nf], k = r*cols + c

    with torch.no_grad():
        f_cum = [criterion.normalize(I_all[:, :m + 1].mean(1).to(device)).cpu() for m in range(M)]
        key_I = model.key_intensity_at_encoding().cpu().numpy()   # [M, N_sim, N_sim]
        phases = (torch.sigmoid(model.slm_phases) * 2 * np.pi).cpu().numpy()[:, 0]

    rmse_cum = [FunctionApproxLoss.per_function_rmse(f, target) for f in f_cum]
    k = int(torch.argsort(rmse_cum[-1])[len(rmse_cum[-1]) // 2])

    N = model.N_sim
    L = model.layer_size_sim
    E = model.encoding_x_num_sim
    ly0 = (N - L) // 2
    ey0 = (N - E) // 2

    fig, axes = plt.subplots(M, 3, figsize=(13, 3.2 * M), dpi=dpi, squeeze=False)
    tgt = target[:, k].numpy(); a_np = a_all.numpy()
    for m in range(M):
        ax = axes[m, 0]
        ax.imshow(phases[m], cmap='twilight', vmin=0, vmax=2 * np.pi)
        ax.set_ylabel(f'key {m}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if m == 0:
            ax.set_title('Phase key', fontsize=11, fontweight='bold')

        ax = axes[m, 1]
        I = key_I[m]
        enh = I[ey0:ey0 + E, ey0:ey0 + E].mean()   # incident (unit-amplitude) intensity == 1
        crop = I[ly0:ly0 + L, ly0:ly0 + L]
        ax.imshow(crop, cmap='inferno')
        ax.add_patch(Rectangle((ey0 - ly0 - 0.5, ey0 - ly0 - 0.5), E, E,
                                fill=False, edgecolor='cyan', linewidth=1.2))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'mean intensity in footprint = {enh:.1f}x incident', fontsize=9)
        if m == 0:
            ax.set_title('Intensity at encoding plane (cyan = footprint)', fontsize=11, fontweight='bold')

        ax = axes[m, 2]
        ax.plot(a_np, tgt, '-', color='black', label='target')
        ax.plot(a_np, f_cum[m][:, k].numpy(), '-.', color='tab:red', label='approx')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'keys 0..{m}   RMSE(f_{k})={rmse_cum[m][k]:.2e}   mean RMSE={rmse_cum[m].mean():.2e}',
                     fontsize=9)
        if m == M - 1:
            ax.set_xlabel('a')
        if m == 0:
            ax.legend(fontsize=7, loc='upper right')
            ax.annotate('Cumulative approximation', xy=(0.5, 1), xytext=(0, 26),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle(
        'Phase-key evolution  (function shown: median-error f_%d of the full model)\n' % k +
        r'($N = r \cdot 2 N_p N_f$,  ' +
        f'r={config.r}  Np={config.Np}  Nf={config.Nf}  M={config.M}  K={config.num_layers})',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, 'key_evolution.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Key evolution saved → {path}')


# ──────────────────────────── main entry point ────────────────────────────── #

def _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse):
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
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Results appended → {csv_path}')


def _load_and_evaluate(ckpt_path):
    '''
    Load a checkpoint (config + model + criterion) and run one eval pass over
    its own dense test grid. Returns (agg, config, model, device) -- the raw
    building block shared by evaluate() and cross-run comparison scripts
    (e.g. plot_sweep_grid.py). Callers get rmse (and a printed summary) via
    _print_summary(agg).
    '''
    config = init_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preserve local paths -- machine-specific, must not be overwritten by
    # whatever was saved inside the checkpoint.
    _path_keys = ('log_dir', 'image_dir', 'model_dir', 'tfboard_dir')
    saved_paths = {k: getattr(config, k, None) for k in _path_keys}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'config' in ckpt:
        config.__dict__.update(ckpt['config'])
        for k, v in saved_paths.items():
            if v is not None:
                setattr(config, k, v)
        recompute_derived(config)

    model     = TimeMultiplexedNFA(config).to(device)
    criterion = FunctionApproxLoss(config).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    if ckpt.get('criterion') is not None:
        criterion.load_state_dict(ckpt['criterion'], strict=False)
    print(f'Loaded: {ckpt_path}  (epoch {ckpt["epoch"]})')
    model.eval(); criterion.eval()

    _, _, test_loader, target_fn = get_function_approx_dataloaders(config)
    agg = _eval_loop(model, test_loader, criterion, device, desc='Testing')
    return agg, config, model, device


def evaluate(ckpt_path=None, out_dir=None, csv_path=None, sweep_name=None, run_label=None):
    ckpt_path = ckpt_path or init_params().ckpt_to_load

    if ckpt_path is not None:
        agg, config, model, device = _load_and_evaluate(ckpt_path)
    else:
        print('[WARNING] No checkpoint provided — evaluating random init.')
        config = init_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model     = TimeMultiplexedNFA(config).to(device)
        criterion = FunctionApproxLoss(config).to(device)
        model.eval(); criterion.eval()
        _, _, test_loader, target_fn = get_function_approx_dataloaders(config)
        agg = _eval_loop(model, test_loader, criterion, device, desc='Testing')

    rmse = _print_summary(agg)

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(ckpt_path) if ckpt_path else '.', '..', 'test'
        )
    os.makedirs(out_dir, exist_ok=True)

    if csv_path is not None:
        _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse)

    _save_test_summary(agg, rmse, config, os.path.join(out_dir, 'test_summary.png'))

    # ── Phase keys / layers / key similarity ───────────────────────────────
    _save_phase_keys(model, out_dir)
    _save_layer_masks(model, out_dir)
    _save_mask_similarity(model, out_dir)
    if ckpt_path is not None:
        _save_key_evolution(model, config, ckpt_path, device, out_dir)

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
    args = parser.parse_args()

    evaluate(
        ckpt_path  = args.ckpt or CKPT_PATH,
        out_dir    = args.out_dir,
        csv_path   = args.csv,
        sweep_name = args.sweep,
        run_label  = args.label,
    )
