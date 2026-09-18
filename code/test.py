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
    Combined test-summary figure -- one PNG instead of three separate files:
      - top-left:  histogram of per-function RMSE (paper Fig. 2a-style)
      - top-right: spatial map of per-function RMSE over the detector plane
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

    fig = plt.figure(figsize=(4 * n_show, 7.2), dpi=dpi)
    gs = fig.add_gridspec(2, n_show, height_ratios=[1.15, 1])
    half = n_show // 2   # top row splits n_show columns into left/right halves

    # ---- top-left: error distribution histogram ----------------------------
    ax_hist = fig.add_subplot(gs[0, :half])
    ax_hist.hist(rmse_np, bins=min(30, max(len(rmse_np), 1)), color='steelblue', alpha=0.85)
    ax_hist.axvline(rmse_np.mean(), color='crimson', linestyle='--',
                     label=f'mean={rmse_np.mean():.2e}')
    ax_hist.set_xlabel('RMSE'); ax_hist.set_ylabel('# functions')
    ax_hist.set_title(f'Error distribution (Nf={len(rmse_np)})', fontsize=10)
    ax_hist.legend(fontsize=8)

    # ---- top-right: spatial error heatmap -----------------------------------
    # Detector (r, c) reads function k = r*pd_num_cols + c (matches model.py's
    # _integrate_photodiode_array), so the RMSE vector reshapes straight into
    # that grid and is placed at the detectors' true physical spacing.
    # Detectors don't tile the plane (real gap = pd_row/col_spacing -
    # photodiode_size, see config.py) -- imshow's own bilinear resampling
    # fills that gap purely for legibility (a visualization smoothing, not a
    # physical measurement), and the true detector footprints are drawn to
    # scale (actual photodiode_size, not a cosmetic marker) so the real fill
    # factor is visible.
    ax_heat = fig.add_subplot(gs[0, half:])
    rows, cols = config.pd_num_rows, config.pd_num_cols
    grid = rmse_np.reshape(rows, cols)   # row-major k = r*cols + c, matches model.py

    height_um = (rows - 1) * config.pd_row_spacing * 1e6
    width_um  = (cols - 1) * config.pd_col_spacing * 1e6
    extent    = [-width_um / 2, width_um / 2, -height_um / 2, height_um / 2]

    im = ax_heat.imshow(grid, cmap='hot', origin='lower', extent=extent,
                         interpolation='bilinear')
    ys = np.linspace(-height_um / 2, height_um / 2, rows)
    xs = np.linspace(-width_um / 2, width_um / 2, cols)
    pd_w_um = config.photodiode_size * 1e6
    for yy in ys:
        for xx in xs:
            ax_heat.add_patch(Rectangle(
                (xx - pd_w_um / 2, yy - pd_w_um / 2), pd_w_um, pd_w_um,
                facecolor='none', edgecolor='white', linewidth=0.7, alpha=0.85))
    ax_heat.set_xlabel('detector-plane x (µm)')
    ax_heat.set_ylabel('detector-plane y (µm)')
    ax_heat.set_title('Spatial RMSE map (interpolated between detectors)', fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE')
    # Format each tick as its own scientific-notation value rather than a
    # shared power-of-ten offset -- matplotlib's default offset text box
    # renders right above the axes and collides with the title above at the
    # tiny RMSE values this project now reaches (~1e-6/1e-7).
    cbar.formatter = FuncFormatter(lambda val, pos: f'{val:.1e}')
    cbar.update_ticks()

    # ---- bottom row: target-vs-approx curves --------------------------------
    # Rank-based labels come from the ERROR-sorted position (index 0 = best
    # fit, Nf-1 = worst fit) -- assigned here, before `show_idx` gets re-sorted
    # by function id `k` for left-to-right display order, so the label always
    # reflects true error rank regardless of display order.
    order_by_err = torch.argsort(rmse)
    rank_positions = [0, Nf // 3, 2 * Nf // 3, Nf - 1]
    rank_labels    = ['best', 'middle', 'middle', 'worst']
    label_of = {}
    for pos, lab in zip(rank_positions, rank_labels):
        k = int(order_by_err[pos])
        label_of.setdefault(k, lab)   # keep first label if Nf is small enough to collide
    show_idx = sorted(label_of.keys())[:n_show]

    for i, k in enumerate(show_idx):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(a, target[:, k], '--', color='tab:green', label='target')
        ax.plot(a, f_hat[:, k], '-.', color='tab:red', label='approx')
        ax.set_title(f'f_{k}  (RMSE={rmse[k]:.2e}) [{label_of[k]}]', fontsize=9)
        ax.set_xlabel('a'); ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Test summary   mean RMSE={rmse.mean():.2e}   max RMSE={rmse.max():.2e}',
                 fontsize=12)
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

    This is a circular (phase-aware) similarity, not a naive Euclidean/cosine
    one -- raw phase values wrap at 2pi, so comparing them directly would
    treat e.g. 0.01 and 2*pi-0.01 as maximally different when they're
    actually almost identical. It's also deliberately invariant to a constant
    phase offset between two keys: a uniform additive shift to an entire key
    doesn't change its own detected intensity (it cancels under |field|^2),
    so two keys differing only by such a constant really do produce
    redundant measurements and should score as maximally similar (1.0), not
    different.

    1.0 = identical up to a constant offset (fully redundant, no benefit from
    averaging this pair); 0.0 = pixel-wise phase differences are spread
    uniformly around the circle (no consistent relationship -- these two
    keys are learning genuinely different things, which is what the "wisdom
    of the crowd" ensembling this project adds is supposed to produce).

    Saves one M x M heatmap to out_dir/phase_key_similarity.png.
    Returns the [M, M] numpy similarity array (None if M < 2).
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


# ──────────────────────────── main entry point ────────────────────────────── #

def _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse):
    '''Append one results row to the sweep CSV, creating the file if needed.'''
    row = {
        'sweep'     : sweep_name or '',
        'label'     : run_label  or '',
        'ckpt'      : ckpt_path  or '',
        'n_samples' : len(agg['a']),
        'loss_mean' : round(float(np.mean(agg['loss'])), 6),
        'rmse_mean' : round(float(rmse.mean()), 6),
        'rmse_max'  : round(float(rmse.max()), 6),
        'rmse_min'  : round(float(rmse.min()), 6),
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Results appended → {csv_path}')


def evaluate(ckpt_path=None, out_dir=None, csv_path=None, sweep_name=None, run_label=None):
    config = init_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint (config first, then model) ────────────────────────
    # Preserve local paths -- machine-specific, must not be overwritten by
    # whatever was saved inside the checkpoint.
    _path_keys = ('log_dir', 'image_dir', 'model_dir', 'tfboard_dir')
    saved_paths = {k: getattr(config, k, None) for k in _path_keys}

    ckpt_path = ckpt_path or config.ckpt_to_load
    ckpt = None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'config' in ckpt:
            config.__dict__.update(ckpt['config'])
            for k, v in saved_paths.items():
                if v is not None:
                    setattr(config, k, v)
            recompute_derived(config)

    model     = TimeMultiplexedNFA(config).to(device)
    criterion = FunctionApproxLoss(config).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'], strict=False)
        if ckpt.get('criterion') is not None:
            criterion.load_state_dict(ckpt['criterion'], strict=False)
        print(f'Loaded: {ckpt_path}  (epoch {ckpt["epoch"]})')
    else:
        print('[WARNING] No checkpoint provided — evaluating random init.')
    model.eval(); criterion.eval()

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(ckpt_path) if ckpt_path else '.', '..', 'test'
        )
    os.makedirs(out_dir, exist_ok=True)

    # ── Test loader: dense a-grid, deterministic from config's Np/Nf/func_seed ──
    _, _, test_loader, target_fn = get_function_approx_dataloaders(config)

    agg  = _eval_loop(model, test_loader, criterion, device, desc='Testing')
    rmse = _print_summary(agg)

    if csv_path is not None:
        _write_csv(csv_path, sweep_name, run_label, ckpt_path, agg, rmse)

    _save_test_summary(agg, rmse, config, os.path.join(out_dir, 'test_summary.png'))

    # ── Phase keys / layers / key similarity ───────────────────────────────
    _save_phase_keys(model, out_dir)
    _save_layer_masks(model, out_dir)
    _save_mask_similarity(model, out_dir)

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
