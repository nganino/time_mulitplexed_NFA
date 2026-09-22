"""Shared pieces of an Optuna study over a grid of simulation "cards" (configurations).

Used by tune_worker.py (search), tune_final.py (best trial on the full grid) and tune_report.py.
Everything project-specific lives in the ADAPT block: the card grid, the search subset, the
search space, the baseline / probe settings, how a trial's parameters become trainer CLI flags,
and how the test metric is read back.  The rest (subprocess runner with resume + retries,
journal storage, trial directories) is generic.

Out of the box the template drives mock_train.py so the whole chain can be exercised:
    bash run_tune.sh            (mock, ~1 min)      python tune_worker.py --smoke --enqueue
"""
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# ====================================== ADAPT ======================================
STUDY = "my_tune"                       # Optuna study name; also names runs/tune-<STUDY>
RUNS = HERE / "runs"                    # where per-tag card folders live: RUNS/<tag>/<card>
TUNE = RUNS / f"tune-{STUDY}"           # journal, trial dirs, logs, csv of this study
TAG_TUNED = f"{STUDY}-tuned"            # final-phase folder tags (new tags, never overwrite)
TAG_BASE = f"{STUDY}-base"

TRIAL_EPOCHS = 500                      # proxy budget of a search trial (>= 1/2 of FINAL_EPOCHS)
FINAL_EPOCHS = 1000                     # full budget of the final grid

# Trainer / tester entry points.  TEST = None when the trainer already writes the metric.
TRAIN = HERE / "mock_train.py"
TEST = None                             # e.g. HERE / "test.py"
TRAIN_MARKER = "TRAINING_COMPLETE"      # files the trainer / tester create when done
TEST_MARKER = "TEST_COMPLETE"
CHECKPOINT = "checkpoints/best_model.pth"   # present -> resume instead of restart
RESUME_FLAG = ["--resume"]

# The card grid.  AXES is ordered; a card is a tuple with one value per axis.
AXES = {"r": (0.5, 1.0, 1.5), "wav": ("500nm", "2wav", "5wav"), "out": (16, 20, 24, 28)}
CARDS = [(r, w, o) for r in AXES["r"] for w in AXES["wav"] for o in AXES["out"]]

# Search cards: cover every axis value at least once, cheapest first (pruner stops bad trials
# early), and include the cards that decide the goal.
SUBSET = [
    (0.5, "500nm", 16),
    (1.0, "2wav", 20),
    (0.5, "5wav", 28),
    (1.5, "500nm", 20),
    (1.0, "5wav", 24),
    (1.5, "2wav", 28),
]

# name -> (kind, low, high, log) with kind in {"float", "int"}; or ("cat", [choices]).
# Physical knobs dimensionless (ratio to a rule) so one setting fits every card.
SPACE = {
    "z_factor": ("float", 0.2, 2.5, True),
    "open_ratio": ("float", 0.4, 2.0, False),
    "lr": ("float", 2e-3, 4e-2, True),
    "init_std": ("float", 0.05, 0.6, True),
    "beta1": ("float", 0.90, 0.99, False),
    "warmup_epochs": ("int", 0, 20, False),
    # "scheduler": ("cat", ["cosine", "plateau"]),
}

# The current default recipe / geometry written in the search space (must lie inside SPACE).
BASELINE_PARAMS = {"z_factor": 1.0, "open_ratio": 1.1, "lr": 8e-3, "init_std": 0.2, "beta1": 0.97, "warmup_epochs": 5}
PROBE_PARAMS = [
    dict(BASELINE_PARAMS, z_factor=0.5, open_ratio=0.75),
    dict(BASELINE_PARAMS, lr=1.5e-2),
]

# Ordering constraints as (better_card, worse_card) pairs: hinge on log10 difference + MARGIN.
# Empty list -> plain mean log10 metric.  Both cards must be in SUBSET (and in CARDS).
PAIRS = []            # e.g. [((1.0, "2wav", 20), (1.0, "500nm", 20))]
MARGIN = 0.05
PAIR_WEIGHT = 1.0


def card_name(card):
    parts = []
    for (k, _), v in zip(AXES.items(), card):
        parts.append(f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}")
    return "_".join(parts)


def card_dir(tag, card):
    return RUNS / tag / card_name(card)


def resolve_geometry(params, card):
    """Turn dimensionless knobs into per-card physical values.  Replace with the project's
    geometry rule (e.g. z = z_factor * z_rule(D); w_px = round(open_ratio * lambda_min z / D / pitch)
    clamped to [1, W_MAX]; quadrature nodes chosen from w_px)."""
    r, wav, out = card
    z_rule_mm = 30.0 * r                                   # placeholder rule
    z_mm = float(params["z_factor"]) * z_rule_mm
    feature_px = 2.7 * float(params["z_factor"])           # placeholder finest feature in pixels
    w_px = int(min(6, max(1, math.floor(float(params["open_ratio"]) * feature_px + 0.5))))
    subsamples = 11 if w_px <= 1 else 5 if w_px == 2 else 3   # quadrature lesson: >= 9 nodes/side for 1-px openings
    return {"z_mm": z_mm, "w_px": w_px, "gap_px": w_px, "subsamples": subsamples,
            "open_over_feature": w_px / feature_px}


def common_args(card, epochs):
    r, wav, out = card
    return ["--r", r, "--wav-set", wav, "--n-p-out", out, "--epochs", epochs, "--card", card_name(card)]


def card_args(params, card, epochs):
    """Trainer CLI flags (without --output-dir / --gpu) for one card under `params`.
    Returns (args, info); info values are stored as trial user attrs (resolved geometry)."""
    g = resolve_geometry(params, card)
    args = common_args(card, epochs) + [
        "--z-mm", f"{g['z_mm']:.6f}", "--detector-width-px", g["w_px"], "--detector-gap-px", g["gap_px"],
        "--sensor-subsamples", g["subsamples"],
        "--lr", f"{float(params['lr']):.6g}", "--init-std", f"{float(params['init_std']):.6g}",
        "--beta1", f"{float(params['beta1']):.6g}", "--warmup-epochs", int(params["warmup_epochs"]),
    ]
    return args, g


def baseline_args(card, epochs):
    """The default recipe / geometry as the trainer would run it without the study
    (should equal card_args(BASELINE_PARAMS, ...) up to rounding)."""
    return card_args(BASELINE_PARAMS, card, epochs)


def gpu_args(gpu):
    """How the trainer selects a GPU.  Alternative: return [] and set CUDA_VISIBLE_DEVICES in _run."""
    return ["--gpu", str(gpu)]


def read_metric(out_dir):
    """Test metric of a finished card (lower is better).  Return None if missing."""
    f = Path(out_dir) / "metrics.json"
    if not f.is_file():
        return None
    return float(json.loads(f.read_text())["mse"])


def score(mses):
    """Objective from {card_name: metric} of the cards finished so far (partial for pruning)."""
    logs = {n: math.log10(v) for n, v in mses.items()}
    value = statistics.fmean(logs.values())
    for better, worse in PAIRS:
        b, w = card_name(better), card_name(worse)
        if b in logs and w in logs:
            value += PAIR_WEIGHT * max(0.0, logs[b] - logs[w] + MARGIN)
    return value
# =================================== end ADAPT =====================================


def suggest(trial):
    p = {}
    for k, spec in SPACE.items():
        kind = spec[0]
        if kind == "cat":
            p[k] = trial.suggest_categorical(k, list(spec[1]))
        elif kind == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
        else:
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
    return p


def in_space(params):
    for k, spec in SPACE.items():
        if k not in params:
            return False
        v = params[k]
        if spec[0] == "cat":
            if v not in spec[1]:
                return False
        elif not (spec[1] <= v <= spec[2]):
            return False
    return True


def same_params(a, b):
    for k in SPACE:
        if k not in a or k not in b:
            return False
        va, vb = a[k], b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > 1e-9 * max(1.0, abs(float(va))):
                return False
        elif va != vb:
            return False
    return True


def _run(cmd, log_path):
    with open(log_path, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT).returncode


def run_card(out_dir, train_args, gpu, log_path, retries=2, smoke=False, extra_test_args=()):
    """Train (resume if a checkpoint exists) and test one card; returns the metric or None.
    Finished cards (both markers present) are not re-run.  Broken output dirs are deleted so a
    retry starts clean; retries also absorb sporadic CUDA OOMs (WSL2 with several procs/GPU)."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    extra = ["--smoke"] if smoke else []
    for _ in range(retries + 1):
        if not (out_dir / TRAIN_MARKER).is_file():
            resume = (out_dir / CHECKPOINT).is_file()
            if out_dir.exists() and not resume:
                shutil.rmtree(out_dir)
            rc = _run([PY, TRAIN, "--output-dir", out_dir, *gpu_args(gpu), *train_args,
                       *(RESUME_FLAG if resume else []), *extra], log_path)
            if rc != 0:
                if out_dir.exists() and not (out_dir / CHECKPOINT).is_file():
                    shutil.rmtree(out_dir)
                time.sleep(5)
                continue
        if TEST is not None and not (out_dir / TEST_MARKER).is_file():
            rc = _run([PY, TEST, "--output-dir", out_dir, *gpu_args(gpu), *extra_test_args, *extra], log_path)
            if rc != 0:
                time.sleep(5)
                continue
        return read_metric(out_dir)
    return None


def tune_dir(smoke=False):
    return TUNE.parent / (TUNE.name + "_smoke") if smoke else TUNE


def journal_path(smoke=False):
    return tune_dir(smoke) / "optuna_journal.log"


def load_study(smoke=False, sampler=None, pruner=None):
    import optuna
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend
    except ImportError:  # optuna < 4
        from optuna.storages import JournalFileStorage as JournalFileBackend
    tune_dir(smoke).mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path(smoke))))
    return optuna.create_study(study_name=STUDY + ("_smoke" if smoke else ""), storage=storage,
                               direction="minimize", load_if_exists=True, sampler=sampler, pruner=pruner)


def trial_dir(number, smoke=False):
    return tune_dir(smoke) / "trials" / f"t{int(number):04d}"


def completed(study):
    import optuna
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def best_trial(study, number=None):
    if number is not None:
        return study.trials[number]
    done = completed(study)
    return min(done, key=lambda t: t.value) if done else None


def gm(vals):
    vals = [v for v in vals if v is not None and v > 0]
    return math.exp(statistics.fmean(math.log(v) for v in vals)) if vals else None
