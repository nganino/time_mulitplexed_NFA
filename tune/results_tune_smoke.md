# Tuning report: my_tune_smoke

## Study `my_tune_smoke`

12 trials: 12 complete, 0 pruned, 0 failed, 0 running.  Objective = mean log10 metric over the 3 search cards (lower is better).

Best trial **#1**: objective -3.0566 (baseline trial #0: -2.8522, geometric-mean ratio 0.62).

| param | best | baseline | search range |
|---|---:|---:|---|
| z_factor | 0.5 | 1 | 0.2 每 2.5 (log) |
| open_ratio | 0.75 | 1.1 | 0.4 每 2 |
| lr | 0.008 | 0.008 | 0.002 每 0.04 (log) |
| init_std | 0.2 | 0.2 | 0.05 每 0.6 (log) |
| beta1 | 0.97 | 0.97 | 0.9 每 0.99 |
| warmup_epochs | 5 | 5 | 0 每 20 |

### Search cards: best trial vs baseline trial

| card | MSE best | MSE baseline | ratio | resolved (best) |
|---|---:|---:|---:|---|
| r0.50_wav500nm_out16 | 5.28e-03 | 8.64e-03 | 0.61 | z_mm 7.5, w_px 1, gap_px 1, subsamples 11, open_over_feature 0.7407 |
| r1.00_wav2wav_out20 | 1.39e-03 | 2.10e-03 | 0.66 | z_mm 15.0, w_px 1, gap_px 1, subsamples 11, open_over_feature 0.7407 |
| r0.50_wav5wav_out28 | 9.23e-05 | 1.53e-04 | 0.60 | z_mm 7.5, w_px 1, gap_px 1, subsamples 11, open_over_feature 0.7407 |

### Top 5 complete trials

| # | value | z_factor | open_ratio | lr | init_std | beta1 | warmup_epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.0566 | 0.5 | 0.75 | 0.008 | 0.2 | 0.97 | 5 |
| 11 | -3.0170 | 0.423 | 0.874 | 0.0104 | 0.266 | 0.988 | 8 |
| 2 | -2.8772 | 1 | 1.1 | 0.015 | 0.2 | 0.97 | 5 |
| 0 | -2.8522 | 1 | 1.1 | 0.008 | 0.2 | 0.97 | 5 |
| 7 | -2.8126 | 1.56 | 1.05 | 0.00479 | 0.201 | 0.965 | 5 |

![scatter](figures\tune_scatter_smoke.png)

## Final grid: `my_tune-tuned-smoke` (trial #1) vs `my_tune-base-smoke` at 5 epochs

2 cards with both tags: geometric-mean ratio tuned/base **0.63**, tuned lower on 2/2.

| axis | value | cards | gm ratio |
|---|---|---:|---:|
| r | 0.5 | 2 | 0.63 |
| wav | 500nm | 2 | 0.63 |
| out | 16 | 1 | 0.62 |
| out | 20 | 1 | 0.63 |

| card | tuned | base | ratio |
|---|---:|---:|---:|
| r0.50_wav500nm_out16 | 5.55e-03 | 8.94e-03 | 0.62 |
| r0.50_wav500nm_out20 | 3.33e-03 | 5.25e-03 | 0.63 |
