"""Optuna worker: one process per GPU, trains the SUBSET cards of each trial one after another,
reports the running objective after every card so the pruner can stop a bad trial.

  python tune_worker.py --worker A --gpu 0 --n-trials 150 --time-limit-h 11 --enqueue
  python tune_worker.py --worker B --gpu 1 --n-trials 150 --time-limit-h 11
  python tune_worker.py --smoke --enqueue           # 3 cards, tiny budget, separate _smoke study

Restart safety: `--enqueue` (worker A only) marks trials left RUNNING by a killed worker as FAIL
and (re-)enqueues the baseline / probe trials that are missing.  The study lives in a journal
file, so workers can be killed and restarted at any time; finished cards are skipped.
"""
import argparse
import json
import math
import time

import optuna

import tune_common as T


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="A")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-trials", type=int, default=150, help="Counts pruned trials too: set high, time-box with --time-limit-h")
    ap.add_argument("--time-limit-h", type=float, default=11.0, help="Stop asking for new trials after this wall time")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--enqueue", action="store_true", help="Fail stale RUNNING trials, enqueue missing baseline/probes")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def enqueue_references(study, worker):
    S = optuna.trial.TrialState
    for t in study.trials:
        if t.state == S.RUNNING:
            try:
                study.tell(t.number, state=S.FAIL)
                print(f"[{worker}] stale trial {t.number} marked FAIL", flush=True)
            except Exception as exc:  # pragma: no cover
                print(f"[{worker}] could not fail stale trial {t.number}: {exc}", flush=True)
    have = [t.params if t.state != S.WAITING else t.system_attrs.get("fixed_params", {})
            for t in study.trials if t.state in (S.COMPLETE, S.RUNNING, S.WAITING, S.PRUNED)]
    refs = [(T.BASELINE_PARAMS, "baseline")] + [(p, f"probe {i}") for i, p in enumerate(T.PROBE_PARAMS)]
    n_new = 0
    for p, note in refs:
        if not T.in_space(p):
            raise SystemExit(f"reference trial '{note}' lies outside SPACE: {p}")
        if not any(T.same_params(h, p) for h in have):
            study.enqueue_trial(p, user_attrs={"note": note})
            n_new += 1
    print(f"[{worker}] enqueued {n_new} reference trials ({len(study.trials)} trials in the study)", flush=True)


def main():
    a = parse()
    epochs = a.epochs or (5 if a.smoke else T.TRIAL_EPOCHS)
    cards = T.SUBSET[:3] if a.smoke else T.SUBSET
    names = [T.card_name(c) for c in cards]
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, n_startup_trials=8, seed=None)
    # 75th percentile rather than the median: the first (easy) cards barely separate recipes.
    pruner = optuna.pruners.PercentilePruner(75.0, n_startup_trials=6, n_warmup_steps=3, n_min_trials=4)
    study = T.load_study(a.smoke, sampler=sampler, pruner=pruner)
    if a.enqueue:
        enqueue_references(study, a.worker)
    logdir = T.tune_dir(a.smoke) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        p = T.suggest(trial)
        trial.set_user_attr("worker", a.worker)
        trial.set_user_attr("epochs", epochs)
        tdir = T.trial_dir(trial.number, a.smoke)
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "params.json").write_text(json.dumps(p, indent=2))
        print(f"[{a.worker}] trial {trial.number} start {time.strftime('%H:%M:%S')} params {json.dumps(p)}", flush=True)
        mses = {}
        for k, (card, name) in enumerate(zip(cards, names)):
            args, info = T.card_args(p, card, epochs)
            t0 = time.time()
            mse = T.run_card(tdir / name, args, a.gpu, logdir / f"t{trial.number:04d}_{name}.log", smoke=a.smoke)
            dt = time.time() - t0
            if mse is None or not math.isfinite(mse) or mse <= 0:
                trial.set_user_attr("failed_card", name)
                raise RuntimeError(f"card {name} failed")
            mses[name] = mse
            trial.set_user_attr(f"mse_{name}", mse)
            trial.set_user_attr(f"min_{name}", round(dt / 60, 2))
            for key, val in info.items():
                if isinstance(val, (int, float)):
                    trial.set_user_attr(f"{key}_{name}", val if isinstance(val, int) else round(float(val), 4))
            running = T.score(mses)
            print(f"[{a.worker}] trial {trial.number} card {k + 1}/{len(cards)} {name}: {mse:.3e} "
                  f"({dt / 60:.1f} min) running {running:.3f}", flush=True)
            trial.report(running, step=k + 1)
            if k + 1 < len(cards) and trial.should_prune():        # never prune after the last card
                trial.set_user_attr("pruned_after", k + 1)
                print(f"[{a.worker}] trial {trial.number} pruned after {k + 1} cards ({running:.3f})", flush=True)
                raise optuna.TrialPruned()
        value = T.score(mses)
        print(f"[{a.worker}] trial {trial.number} done {time.strftime('%H:%M:%S')} value {value:.4f}", flush=True)
        return value

    study.optimize(objective, n_trials=a.n_trials, timeout=a.time_limit_h * 3600,
                   catch=(RuntimeError,), gc_after_trial=True)
    done = T.completed(study)
    if done:
        b = T.best_trial(study)
        print(f"[{a.worker}] finished: {len(study.trials)} trials, {len(done)} complete; "
              f"best #{b.number} value {b.value:.4f} {json.dumps(b.params)}", flush=True)
    else:
        print(f"[{a.worker}] finished without a completed trial", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
