# Launch / restart checklist

## Before the first launch
- [ ] Goal sentence, allowed knobs, frozen parts, success criterion and infeasibility rule agreed.
- [ ] Baseline params expressed in the search space and enqueued; probes inside the space
      (categoricals in the choices).
- [ ] SUBSET covers every axis, cheap cards first, contains the goal's critical cards.
- [ ] Proxy budget ≥ ½ final; every schedule length scales with the budget.
- [ ] Objective = mean log10 (+ hinge penalties with margin); `trial.report` per card;
      no prune after the last card.
- [ ] `--smoke` run of worker → final → report finished without error.
- [ ] One real card timed at the proxy budget; NT and LIMIT_H set from it.
- [ ] One process per GPU (or ≤ 3 on WSL2); GPU ids correct; `--cuda-graph` where supported.
- [ ] Chain launched with `setsid nohup bash run_tune.sh > logs/tune.log 2>&1 &`; log paths noted.
- [ ] Memory note written (study, storage, logs, subset, objective, decision rule, expected end).

## Every check-in
- [ ] `python tune_report.py` → trials.csv: counts complete/pruned/failed, best vs baseline.
- [ ] Best trials on a bound? Extend the range, note the time.
- [ ] Pruned share > 80 %? Relax the pruner.
- [ ] Any FAIL: read the card log before restarting.
- [ ] GPU utilisation ≈ 100 % per worker (`nvidia-smi`); zombies from killed workers removed.

## Restart
- [ ] Kill workers only (`pkill -f tune_worker.py`), not the chain if the final phase must not start.
- [ ] Worker A `--enqueue` marks stale RUNNING trials FAIL and re-enqueues missing references.
- [ ] Did the parameter key set change? Then do NOT re-enqueue old probes.
- [ ] Finished tag folders of the final phase untouched.

## Before declaring the result
- [ ] Full grid at full budget under a new tag; baseline re-run in the same env.
- [ ] Evaluation matches the exact test integral; quadrature adequate for the tuned geometry.
- [ ] Seed check on marginal cards.
- [ ] Report + figure + README/memory updated; resume command recorded.
