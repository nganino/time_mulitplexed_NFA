"""
paths.py - filesystem locations for this repo.

Training output goes to code/logs/ (gitignored); config.py builds every run directory from LOG_DIR.
Run `python code/paths.py` to print the resolved locations and whether each exists.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../<repo>/code

REPO_ROOT = os.path.dirname(_HERE)                          # .../<repo>
LOG_DIR   = os.path.join(_HERE, 'logs')                     # .../<repo>/code/logs


if __name__ == '__main__':
    for k, v in (('REPO_ROOT', REPO_ROOT), ('LOG_DIR', LOG_DIR)):
        print(f"{'OK ' if os.path.exists(v) else 'MISSING'}  {k:<9}  {v}")
