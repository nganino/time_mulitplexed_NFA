"""
paths.py - single source of truth for every filesystem location this repo needs.

Why this exists: the training and evaluation scripts used to hardcode absolute Windows
paths, which meant a fresh clone could not run without editing several files. Every
location is resolved here and overridable by environment variable.

Environment overrides
---------------------
    SPQPI_PROJECT_ROOT   parent project dir (holds data/)
    SPQPI_DATA_ROOT      object datasets                     (default <project>/data)
    SPQPI_CKPT_ROOT      where trained checkpoints live

The SPQPI_ prefix is inherited from the single-pixel QPI project this code came from,
and is kept deliberately so one set of environment variables serves both repos on the
same machine.

Training output goes to <repo>/logs/, which is gitignored.

Run `python code/paths.py` to print the resolved locations and whether each exists.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../<repo>/code


def _env(name, default):
    v = os.environ.get(name)
    return v if v else default


# -- roots --------------------------------------------------------------------- #
REPO_ROOT    = os.path.dirname(_HERE)                       # .../time-multiplexed-classifier

# Object datasets live outside this repo; set SPQPI_DATA_ROOT to point at them.
PROJECT_ROOT = _env('SPQPI_PROJECT_ROOT', REPO_ROOT)
DATA_ROOT    = _env('SPQPI_DATA_ROOT',    os.path.join(PROJECT_ROOT, 'data'))
CKPT_ROOT    = _env('SPQPI_CKPT_ROOT',    os.path.join(PROJECT_ROOT, 'checkpoints'))

# -- training output (untracked) ----------------------------------------------- #
LOG_DIR = os.path.join(_HERE, 'logs')                        # .../<repo>/code/logs


def require(path, what='file'):
    """Fail loudly and usefully instead of deep inside torch.load."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {what}: {path}\n"
            f"  Datasets and checkpoints are not tracked in git.\n"
            f"  Set SPQPI_DATA_ROOT / SPQPI_CKPT_ROOT to your copies."
        )
    return path


if __name__ == '__main__':
    rows = [('REPO_ROOT', REPO_ROOT), ('PROJECT_ROOT', PROJECT_ROOT),
            ('DATA_ROOT', DATA_ROOT), ('CKPT_ROOT', CKPT_ROOT),
            ('LOG_DIR', LOG_DIR)]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{'OK ' if os.path.exists(v) else 'MISSING'}  {k:<{w}}  {v}")
