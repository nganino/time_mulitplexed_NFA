---
name: d2nn-optuna-tuning
description: Design, launch, monitor and validate an Optuna hyper-parameter / geometry study for D2NN (diffractive neural network) or similar physics-simulation training runs on local GPUs. Use when asked to "tune", "optimise", "search" training recipes (lr, optimizer, schedule, init) or dimensionless physical knobs (propagation distance, detector opening) across a grid of configurations, or to resume/report an existing Optuna study. Covers subset objectives, pruning, multi-GPU workers, restart safety, final full-grid validation and the report.
argument-hint: "[goal, e.g. 'tune lr/schedule so 2wav beats mono on all grids']"
---

# D2NN Optuna tuning

Distilled from two multi-day studies (det600 training-recipe search, Sept 2026; experiment_codes
geometry + recipe search `exp_specw_tune`). Follow the phases in order. Copy the scripts in
`templates/` as the starting point of a new harness; they run end to end against `mock_train.py`
so the harness can be smoke-tested before a single real GPU-hour is spent.

Read `reference/lessons.md` before designing the search space or the objective, and
`reference/checklist.md` before every launch or restart.

## Phase 0 — pin the goal down (do this in chat before writing code)

1. Write the goal as one sentence with a measurable success criterion
   (e.g. "geometric-mean test MSE over the 48-card grid lower than tag X" or
   "MSE ordering mono > 2wav > 3wav on all 24 grids with 5 % margin").
2. List which knobs are **allowed** (training recipe only? geometry too?) and which are **frozen**
   (optical stack, wavelength endpoints, sampling). Never widen this list on your own.
3. Fix the **decision rule for infeasibility** up front: the improvement factor each losing card
   needs, and a probe budget after which you stop and ask the user (AskUserQuestion) instead of
   burning GPU-hours. Goals of the form "A beats B on every grid" are often structural, not a
   training issue; the rule keeps you from chasing them for a night.
4. Agree on the compute budget (GPUs, hours) and whether the run may be left unattended.

## Phase 1 — design the harness

One global study, one `tune_common.py` holding everything the worker, final and report scripts share:

- **Cards**: the full grid as tuples; a **SUBSET** of 6–8 search cards that covers every axis
  (each r, each N_p,out, each wavelength set at least once), cheapest cards first so a pruner can
  stop bad trials early. Losing/critical cards of the goal must be in the subset.
- **Search space** as a dict `name -> (kind, lo, hi, log)`. Make physical knobs
  **dimensionless** (z_factor = z / z_rule, open_ratio = opening / finest feature) so one
  setting applies to every card; resolve to per-card physical values inside `card_args` and
  record the resolved values (`w_px`, `z_mm`) as trial user attributes.
- **Objective**: mean log10 of the test metric over the subset (a geometric mean; MSEs span
  decades). Ordering / window constraints go in as hinge penalties on log10 differences with a
  margin, weighted so a violation outweighs a small MSE gain. Report the running mean after each
  card with `trial.report(value, step)`.
- **Budget proxy**: train subset cards for a reduced epoch budget (½ of the final is safe;
  ¼ over-rewarded fast-decaying recipes in the det600 study) and scale every schedule
  length (warm-up, cosine, start epochs) with the budget. Final validation always uses the full
  budget.
- **Reference trials**: enqueue the current default recipe as the *baseline* trial plus 1–3
  hand-picked probes. The baseline gives the ratio in the report and checks the harness
  reproduces known numbers. Every enqueued value must lie inside the space (categorical values
  in the choices) or a worker crashes.
- **Sampler / pruner**: `TPESampler(multivariate=True, constant_liar=True, n_startup_trials≈8)`
  for parallel workers; `PercentilePruner(75, n_warmup_steps≈3)` rather than the median pruner
  (early cards barely separate recipes, the median rule pruned nearly every proposal). Never
  prune after the last card. Optuna ≥ 4 always feeds pruned trials to TPE (no kwarg).
- **Storage**: `JournalStorage` on a local file (no sqlite locking trouble on WSL / network
  drives; sqlite needs `connect_args timeout=300`). Trial directories `runs/tune-<name>/trials/tNNNN/<card>`.
- **Execution**: the trainer runs as a subprocess with CLI flags; `run_card` resumes from a
  checkpoint, retries twice, deletes broken output dirs, and honours completion markers
  (`TRAINING_COMPLETE`, `TEST_COMPLETE`) so re-runs skip finished cards.
- **Workers**: one process per GPU when the training step is launch-bound (small models,
  `--cuda-graph`); at most 3 per GPU on WSL2 (spurious CUDA OOM). Each worker has its own log.
- **Time boxing**: `n_trials` counts pruned trials, so set it high and rely on `timeout`.
- **Chain script** (`run_tune.sh`): workers → report → final validation → report, launched
  with `setsid nohup ... &`, re-runnable (study persists, finished cards are skipped).

## Phase 2 — smoke test, then launch

1. `--smoke` mode: 2–3 cards, tiny epoch budget, a separate `_smoke` study. Run worker, final and
   report end to end. Fix crashes here, not at 03:00.
2. Time one real trial card at the proxy budget; set NT / the time cap from it.
3. Launch the chain detached, note the log paths, then start a monitor (`Monitor` tool or a loop)
   on the worker logs and the trials CSV. Save the study name, storage path, logs, subset,
   objective and the decision rule to memory immediately (a context reset must resume, not redesign).

## Phase 3 — monitor and steer

Check every 30–60 min: trials complete / pruned / failed, best value vs baseline, where the best
params sit. Act on:

- **Best trials on a bound** → extend the range mid-study (Optuna warns "inconsistent parameter
  values" and samples the changed params independently; acceptable). Write down the time of the change.
- **Almost everything pruned** → pruner too aggressive (raise the percentile / warm-up steps).
- **Two knobs degenerate** (e.g. z and opening scale together, only quantisation differs) → note
  it, later report the physical invariant, do not add more knobs.
- **Repeated FAIL** → read the card log (OOM, NaN, bad enqueued value) before restarting.
- **Restart**: stale RUNNING trials must be marked FAIL and missing reference trials re-enqueued
  (the worker `--enqueue` flag does this); do not re-enqueue after adding a parameter key
  (duplicates). Killing a worker mid-final-phase must not wipe finished tag folders.

Stop the search when the top 5 complete trials are within ~0.01 log10 of each other, none sits on
a bound and ≥ ~50 trials completed, or at the time cap.

## Phase 4 — validate before believing

1. **Full grid, full budget, new folder tag**; re-run the **baseline in the same environment**
   under its own tag (numbers from another machine/env differ 5–20 %). Compare with
   geometric-mean ratios per axis (per wavelength set, per N_p,out, per r) and count cards improved.
2. **Physics / evaluation sanity**: make sure the proxy did not exploit the evaluator. In the
   specw study the winner (1-px opening) trained under a 3-node quadrature that was inaccurate
   for sub-2-feature openings; the exact 11×11 test integral was 5–50× worse. Any opening
   < ~2 features needs ≥ 9 nodes per pixel side in training. Check energy split between
   channels for multi-wavelength models, and evaluate at the exact test integral.
3. **Seed check** (2–3 init seeds) on the marginal cards before declaring an ordering goal met.
4. If the goal is infeasible with the allowed knobs, say so with the evidence (required vs
   achieved factor, oracle/bound experiments) and ask the user how to change scope; do not
   quietly widen the knob list.

## Phase 5 — report and hand over

`tune_report.py` writes `trials.csv`, a markdown report (study summary, best vs baseline per
parameter with ranges, per-card table, final-grid ratios per axis) and a scatter figure. Add an
interpretation paragraph (why the recipe works, degeneracies, caveats, practical advice such as
"build w = 2 px at z_rule rather than the 1-px optimum"). Update README / memory with the tags,
paths, and the resume command.

## Template map

| file | role | adapt |
|---|---|---|
| `templates/tune_common.py` | study name, cards, SUBSET, SPACE, baseline/probes, `card_args`, `read_metric`, `run_card`, storage | everything in the `ADAPT` block |
| `templates/tune_worker.py` | one worker per GPU, enqueue + restart safety, pruning | rarely |
| `templates/tune_final.py` | best trial + baseline on the full grid, one queue per GPU | tag names |
| `templates/tune_report.py` | CSV, markdown, scatter | axis names in the final table |
| `templates/run_tune.sh` | detached chain | NT / LIMIT_H / GPU ids |
| `templates/mock_train.py` | fake trainer so the harness runs without a simulation | delete once real |

Quick start: copy `templates/*` into the project, run `bash run_tune.sh` with the mock (finishes in
~1 min), then replace `TRAIN`/`TEST`/`card_args`/`read_metric` with the real trainer and re-run
`--smoke`.
