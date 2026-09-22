# Lessons from the det600 / experiment_codes Optuna studies (Sept 2026)

Concrete numbers are from D2NN simulations (single or few phase layers, 200 nm–8 µm pixels,
multi-wavelength illumination, detector-array read-out, MSE on target coefficients). The mechanisms
transfer to any "train a physical model per configuration" grid.

## Search-space design

- **Dimensionless physical knobs.** `z_factor = z / z_rule(D)` and `open_ratio = opening / finest
  diffraction feature (λ_min z / D)` let one trial setting apply to cards of every size and
  wavelength. Resolve per card, clamp (`1 ≤ w ≤ W_MAX px`, gap = w) and store the resolved values
  as trial user attrs so the report can show them.
- **Expect degeneracies.** z and opening are nearly degenerate: scaling both keeps the angular
  layout, only the Fresnel number and pixel quantisation differ (w = 2 px at z_rule ≈ w = 1 px at
  0.5 z_rule). Recognise it from the trial scatter, state the invariant (opening ≈ 0.5–1
  feature, gap ≈ opening) and give practical advice instead of chasing the exact optimum.
- **Ranges.** Start with 3–5× around the default; extend when the best trials sit on a bound
  (happened three times in one study: z_factor, spec_w_lr, kappa). Log scale for lr, std, kappa.
- **Training-recipe knobs that mattered** (Adam family): lr (largest effect; 2e-3–1.5e-2 with
  cosine annealing to lr/1000 and 5–10 warm-up epochs beat 3e-4 plateau by gm 0.86–0.89),
  beta1 0.93–0.97, init std (zero or ≤0.2; ≥0.3 worse). Neutral or harmful: NAdam/RAdam, L-BFGS,
  grad clipping, early stopping patience, weight decay, cosine restarts.
- **Trainable auxiliary parameters** (spectral weights): their own param group with own lr,
  a start epoch as a *fraction* of the budget, an init logit scale (kappa); a floor on the
  weights prevents collapse to a degenerate single-channel solution.
- **Categorical enqueued values must be in the choices** (start=300 not in {0,100,200,400} crashed a worker).

## Objective and proxy

- Mean log10 metric over the subset = geometric mean; MSEs span 1e-6..1e-2, an arithmetic mean
  would be one card.
- Ordering goals: hinge on log10 differences with a margin (0.05–0.1) times a weight
  (0.2 × mean log MSE + violations worked). Fixed references (baseline mono numbers) in the penalty
  stop a recipe from "closing the gap" by making its own reference card worse.
- 250 of 1000 epochs over-rewarded fast decay: the proxy leader was the worst at 1000 epochs.
  500 of 1000 ranked correctly. If a full-budget trial takes < 10 min, do not use a proxy.
- Schedules that end at the budget (cosine) scale naturally; plateau schedulers do not.
- Subset of 8 cards, one full-budget re-check of the top 3–5 trials before the full grid.

## Pruning and sampling

- `MedianPruner` pruned nearly every TPE proposal because the first cards (easy monos) barely
  separate recipes (spread 0.03 log10). `PercentilePruner(75, n_startup_trials=6,
  n_warmup_steps=3, n_min_trials=4)` fixed it.
- Only call `should_prune` when another card follows; pruning after the last card throws away
  a finished trial.
- Optuna 4.x `TPESampler` has no `consider_pruned_trials`; pruned trials are always used.
- Pruned trials count toward `n_trials`; a pruned trial costs ~1.5 min, a complete one 6–30 min.
  Use a high `n_trials` and `timeout` to time-box.
- Convergence seen at ~175 trials / 69 complete: top 6 within 0.006 log10.

## Execution

- Training small D2NNs is **launch-bound**: one process per GPU with CUDA graphs beats several
  concurrent processes; on WSL2, > 3 processes per 2080 Ti sporadically die with "CUDA error:
  out of memory" at 2 GB used. Retries in `run_card` absorb it.
- Memory-bound cards first when queuing the final grid (largest Green's-function / kernel first).
- Probes far outside the usual geometry (z = 7–10× rule) took hours because the propagation
  kernel grid grows as (2R+1)²; check the kernel size before launching a probe.
- Restart safety: `JournalStorage` + trial dirs + completion markers; on restart mark RUNNING
  trials FAIL and re-enqueue missing reference trials (WAITING trials keep their params in
  `system_attrs["fixed_params"]`); do not re-enqueue after adding a parameter key (duplicates
  drained with `study.tell(n, state=FAIL)`).
- A crashed worker made the chain script fall through into the final phase, which then got
  killed and wiped its tag folders — guard the final phase with a check that the search finished.
- `argparse` treats `--tag-suffix -e2000` as a flag: write `--tag-suffix=-e2000`.

## Validation traps

- **Evaluation quadrature**: 1-px openings on 1.3-px features trained under a 3-node-per-pixel
  rule reached 1.3e-6 but the exact 11×11 integral gave 6e-6..6e-5 (default 3-px opening: both
  2e-6). Rule: opening < ~2 features needs ≥ 9–11 nodes per detector side in training; keep the
  sample count (w·s)² near the default.
- **Same-environment baseline**: re-run the baseline tag next to the tuned tag; published numbers
  from another env differed 5–20 % (gm 0.98–1.01 per set once re-run).
- **Seed variance**: 1–10 % on stable cards, but a few seeds of low-MSE configs end 100× worse
  (basin failures); use 2–3 seeds on marginal cards and report trimmed means for seed sweeps.
- **Structural limits**: a 2wav model at 400/600 nm could not beat the 400 nm mono on grids where
  N_p,out ≤ 2 N_p,in for any recipe (16 full-budget trials, 3000 epochs, seeds): the dominant
  channel only produces harmonics on a 1.5× grid. Diagnose with a channel split (energy /
  variance per wavelength at the detector) and an **oracle** run (one channel weight set to 0) to
  separate capacity limits from training limits. Report and ask; do not keep searching.

## Reporting

- `trials.csv` with every trial (state, value, params, per-card metric, worker, pruned_after, note).
- Best vs baseline per parameter with the search range; per-card MSE best/baseline/ratio with the
  resolved geometry; final-grid gm ratios per axis and count of cards improved.
- Interpretation: mechanism (why the winner works), degeneracies, caveats (quadrature, proxy),
  practical recipe for the next experiment.
- Memory note: study name, storage path, worker logs, chain log, tags, decision rule, state.
