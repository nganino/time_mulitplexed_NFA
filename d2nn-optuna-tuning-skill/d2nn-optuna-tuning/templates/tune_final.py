"""Final validation: the best trial's parameters on ALL cards at the full budget (tag TAG_TUNED)
next to the default recipe / geometry (tag TAG_BASE), one training queue per GPU, memory-heavy
cards first, finished cards skipped.

  python tune_final.py --gpus 0,1
  python tune_final.py --gpus 0,1 --best-trial 12 --skip-base
  python tune_final.py --smoke                # 2 cards per tag, '-smoke' tags
"""
import argparse
import json
import queue
import threading
import time

import tune_common as T


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--best-trial", type=int, default=None, help="Trial number to validate (default: best complete)")
    ap.add_argument("--tag-tuned", default=T.TAG_TUNED)
    ap.add_argument("--tag-base", default=T.TAG_BASE)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    epochs = a.epochs or (5 if a.smoke else T.FINAL_EPOCHS)
    study = T.load_study(a.smoke)
    best = T.best_trial(study, a.best_trial)
    if best is None:
        raise SystemExit("no completed trial in the study")
    params = best.params
    tag_tuned = a.tag_tuned + ("-smoke" if a.smoke else "")
    tag_base = a.tag_base + ("-smoke" if a.smoke else "")
    print(f"best trial #{best.number} value {best.value} params {json.dumps(params)}", flush=True)

    jobs = []
    for card in T.CARDS:
        args, info = T.card_args(params, card, epochs)
        jobs.append((T.card_dir(tag_tuned, card), args, info, "tuned"))
        if not a.skip_base:
            bargs, binfo = T.baseline_args(card, epochs)
            jobs.append((T.card_dir(tag_base, card), bargs, binfo, "base"))
    tdir = T.tune_dir(a.smoke)
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "final_params.json").write_text(json.dumps(
        {"trial": best.number, "value": best.value, "params": params, "epochs": epochs,
         "tag_tuned": tag_tuned, "tag_base": tag_base}, indent=2))
    if a.smoke:
        jobs = [j for j in jobs if j[3] == "tuned"][:2] + [j for j in jobs if j[3] == "base"][:2]
    jobs = [j for j in jobs if not (j[0] / (T.TEST_MARKER if T.TEST else T.TRAIN_MARKER)).is_file()]
    jobs.sort(key=lambda j: -float(j[2].get("g_mb", 0)))       # memory-bound cards first if known
    print(f"{len(jobs)} cards to run ({sum(j[3] == 'tuned' for j in jobs)} tuned, "
          f"{sum(j[3] == 'base' for j in jobs)} base) at {epochs} epochs", flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    logdir = tdir / "logs_final"
    results, lock = {}, threading.Lock()

    def worker(gpu):
        while True:
            try:
                d, args, info, kind = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            mse = T.run_card(d, args, gpu, logdir / f"{kind}_{d.name}.log", smoke=a.smoke)
            with lock:
                results[(kind, d.name)] = mse
                print(f"[gpu{gpu}] {kind} {d.name}: {'FAILED' if mse is None else f'{mse:.3e}'} "
                      f"({(time.time() - t0) / 60:.1f} min, {q.qsize()} left)", flush=True)

    threads = [threading.Thread(target=worker, args=(g.strip(),), daemon=True) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [k for k, v in results.items() if v is None]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
