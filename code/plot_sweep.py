'''
Time-Multiplexed NFA — reusable sweep-summary plot

Every hyperparameter sweep in this project (LR, train_a_samples,
key_to_enc_spacing, pd_row/col_spacing, ...) gets the same comparison plot:
mean + max per-function RMSE vs. the swept parameter, read straight from the
results.csv that test.py's --csv/--sweep/--label flags append to (see
logs/SWEEP_HANDOVER.txt for the standing convention). This used to be
re-written ad hoc (inline python) for each sweep -- this script formalizes it
so every sweep plot looks the same and carries the same fixed-context tag.

Every plot carries a small legend-only tag with r/Np/Nf/M/K(num_layers) --
the run's fixed physical/architectural context -- pulled from the sweep's own
config.json (NOT typed in by hand, so it can't drift from what actually ran).
Deliberately excluded from that tag (per project convention, keeps the plot
uncluttered): batch_size (64 is the project default now), max_epoch, learning
rates, train_a_samples -- these are the things sweeps themselves usually vary
or that don't change the accuracy ceiling being illustrated.

RMSE values are plotted on a log y-axis (already the right way to show
values spanning 1e-7 to 1e-2), not annotated as text on the plot itself.

Usage:
    python plot_sweep.py logs/<sweep_dir>/results.csv --xlabel "train_a_samples"
    python plot_sweep.py logs/<sweep_dir>/results.csv --xlabel "pd spacing (x photodiode_size)" --out logs/<sweep_dir>/summary.png
'''

import sys, os, csv, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _parse_x(label):
    '''label looks like "key=value" (e.g. "trainA=20000") or "key=valuex"
    (e.g. "mult=2x") -- extract the numeric value.'''
    val = label.split('=', 1)[1]
    val = val.rstrip('xX')
    return float(val)


def _context_tag(ckpt_path):
    '''Pull r/Np/Nf/M/K(num_layers) from the run's own config.json (sits two
    directories up from its checkpoint: <run>/model/best.pth -> <run>/config.json)
    so the tag can't drift from what the sweep actually ran.'''
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    return f"r={cfg['r']}  Np={cfg['Np']}  Nf={cfg['Nf']}  M={cfg['M']}  K={cfg['num_layers']}"


def plot_sweep(csv_path, xlabel, out_path=None, title=None, dpi=150):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f'{csv_path} has no rows to plot')

    # test.py's --csv rounds to 6 decimals, so a near-perfect fit can log as
    # an exact 0.0 -- floor-clip before plotting since log-scale can't render
    # zero/negative values (a real 0.0 would otherwise just silently vanish
    # from the line instead of showing up as "very good").
    floor = 1e-8
    x         = [_parse_x(r['label']) for r in rows]
    mean_rmse = [max(float(r['rmse_mean']), floor) for r in rows]
    max_rmse  = [max(float(r['rmse_max']),  floor) for r in rows]
    tag       = _context_tag(rows[0]['ckpt'])

    if out_path is None:
        sweep_dir = os.path.dirname(csv_path)
        out_path  = os.path.join(sweep_dir, os.path.basename(sweep_dir.rstrip('/\\')) + '_rmse.png')

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)
    ax.plot(x, mean_rmse, 'o-', label='mean RMSE (over Nf functions)')
    ax.plot(x, max_rmse, 's-', label='max RMSE (over Nf functions)')
    ax.plot([], [], ' ', label=tag)   # text-only legend entry, not a data series
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average Test RMSE')
    ax.set_yscale('log')
    # When every swept value is a whole number (spacing multipliers, M, ...),
    # pin ticks to just those values -- matplotlib's default locator would
    # otherwise add in-between ticks (e.g. 2.5, 3.5) that don't correspond to
    # any run and are meaningless for a parameter that only takes integers.
    if all(float(xi).is_integer() for xi in x):
        ax.set_xticks(sorted(set(x)))
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(title or f'Effect of {xlabel} on function-approximation accuracy', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Sweep summary saved → {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to the sweep\'s results.csv')
    parser.add_argument('--xlabel', required=True, help='x-axis label (the swept parameter)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: <sweep_dir>/<sweep_dir>_rmse.png)')
    parser.add_argument('--title', default=None, help='Plot title (default: auto-generated from --xlabel)')
    args = parser.parse_args()
    plot_sweep(args.csv_path, args.xlabel, args.out, args.title)
