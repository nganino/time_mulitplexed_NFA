'''
Time-Multiplexed NFA — per-sweep-point "grid" summary plot

For one sweep directory (e.g. logs/M_sweep_pd2px/), builds a single figure
with one ROW per sweep point (e.g. per M value): the error-distribution
histogram, plus best / middle / worst target-vs-approx function curves for
that specific run -- so you can see, at a glance, how the whole error
distribution AND its extremes shift as the swept parameter changes, not just
the mean/max summary statistic plot_sweep.py produces.

Reuses test.py's own model-loading/eval machinery (_load_and_evaluate) so
each row's numbers are computed exactly the same way as that run's own
test_summary.png, just laid out across sweep points instead of within one
run.

Usage:
    python plot_sweep_grid.py logs/M_sweep_pd2px --param M
    python plot_sweep_grid.py logs/pdspacing_sweep_M1 --param pdspacing --out logs/pdspacing_sweep_M1/grid.png
'''

import sys, os, re, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from test import _load_and_evaluate, _print_summary


def _discover_runs(sweep_dir, param):
    '''Find <sweep_dir>/<param>_<value>/model/best.pth runs, sorted by value.
    Matches this project's sweep-script convention (e.g. M_1, M_2, ...,
    pdspacing_2x, keyspacing_0.5x) -- a trailing "x" multiplier suffix is
    stripped before parsing as a float.'''
    runs = []
    for name in os.listdir(sweep_dir):
        m = re.fullmatch(rf'{re.escape(param)}_([0-9.]+)x?', name)
        if not m:
            continue
        ckpt = os.path.join(sweep_dir, name, 'model', 'best.pth')
        if os.path.exists(ckpt):
            runs.append((float(m.group(1)), ckpt))
    runs.sort(key=lambda r: r[0])
    return runs


def plot_sweep_grid(sweep_dir, param, out_path=None, dpi=150):
    runs = _discover_runs(sweep_dir, param)
    if not runs:
        raise ValueError(f"No '{param}_<value>/model/best.pth' runs found under {sweep_dir}")

    if out_path is None:
        out_path = os.path.join(sweep_dir, f'{os.path.basename(sweep_dir.rstrip("/\\"))}_grid.png')

    ncols = 4   # error distribution + best + middle + worst
    fig, axes = plt.subplots(len(runs), ncols, figsize=(4 * ncols, 3 * len(runs)),
                              dpi=dpi, squeeze=False)
    col_headers = ['Error distribution', 'Best fit', 'Middle', 'Worst fit']

    for row, (val, ckpt) in enumerate(runs):
        agg, config, model, device = _load_and_evaluate(ckpt)
        rmse = _print_summary(agg, tag=f'{param}={val:g}')
        a, f_hat, target = agg['a'].numpy(), agg['f_hat'].numpy(), agg['target'].numpy()
        rmse_np = rmse.numpy()
        Nf = rmse.shape[0]
        order_by_err = torch.argsort(rmse)

        # ---- error distribution (this run's own RMSE histogram) ----
        ax_hist = axes[row, 0]
        ax_hist.hist(rmse_np, bins=min(30, max(len(rmse_np), 1)), color='steelblue', alpha=0.85)
        ax_hist.axvline(rmse_np.mean(), color='crimson', linestyle='--',
                         label=f'mean={rmse_np.mean():.2e}')
        ax_hist.set_ylabel(f'{param}={val:g}\n# functions', fontsize=9)
        ax_hist.legend(fontsize=7)
        if row == len(runs) - 1:
            ax_hist.set_xlabel('RMSE')

        # ---- best / middle / worst target-vs-approx curves for THIS run ----
        picks = [(0, 'best'), (Nf // 2, 'middle'), (Nf - 1, 'worst')]
        for col, (pos, lab) in enumerate(picks, start=1):
            k = int(order_by_err[pos])
            ax = axes[row, col]
            ax.plot(a, target[:, k], '--', color='tab:green', label='target')
            ax.plot(a, f_hat[:, k], '-.', color='tab:red', label='approx')
            ax.set_title(f'f_{k}  (RMSE={rmse[k]:.2e})', fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            if row == len(runs) - 1:
                ax.set_xlabel('a')
            if row == 0 and col == 1:
                ax.legend(fontsize=7)

    # Column headers drawn once, above the top row only.
    for col, header in enumerate(col_headers):
        axes[0, col].annotate(header, xy=(0.5, 1), xytext=(0, 28),
                               xycoords='axes fraction', textcoords='offset points',
                               ha='center', va='bottom', fontsize=12, fontweight='bold')

    fig.suptitle(f'Per-{param} error distribution and representative fits', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Sweep grid saved → {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', help="Sweep directory (e.g. logs/M_sweep_pd2px)")
    parser.add_argument('--param', required=True, help="Swept parameter's prefix in each run's directory name (e.g. M, pdspacing, keyspacing)")
    parser.add_argument('--out', default=None, help='Output PNG path (default: <sweep_dir>/<sweep_dir>_grid.png)')
    args = parser.parse_args()
    plot_sweep_grid(args.sweep_dir, args.param, args.out)
