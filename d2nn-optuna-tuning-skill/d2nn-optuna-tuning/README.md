# d2nn-optuna-tuning — Claude Code skill

Optuna hyper-parameter / geometry search for D2NN-style simulation grids: workflow, design
rules, lessons and runnable template scripts. Written from the det600 and experiment_codes
studies of September 2026.

## Install

Project skill (shared with everyone who opens the repo):

    cp -r d2nn-optuna-tuning  <repo>/.claude/skills/

Personal skill (available in every project on your machine):

    cp -r d2nn-optuna-tuning  ~/.claude/skills/

Claude Code picks the skill up on the next session. Invoke it with `/d2nn-optuna-tuning <goal>`
or let Claude load it automatically when you ask for a tuning / optimisation study.

## Contents

| path | what |
|---|---|
| `SKILL.md` | the instructions Claude follows: phases 0–5, harness design rules, template map |
| `reference/lessons.md` | what worked and what failed (search space, proxy budget, pruner, execution, validation traps) |
| `reference/checklist.md` | launch / check-in / restart / sign-off checklist |
| `templates/tune_common.py` | shared config: cards, SUBSET, SPACE, baseline & probes, `card_args`, `read_metric`, `score`, runner, storage — edit the `ADAPT` block |
| `templates/tune_worker.py` | one Optuna worker per GPU with pruning, reference-trial enqueue, restart safety |
| `templates/tune_final.py` | best trial + baseline on the full grid at the full budget, one queue per GPU |
| `templates/tune_report.py` | trials.csv, markdown report, scatter figure, final-grid ratios per axis |
| `templates/run_tune.sh` | detached chain: workers → report → final → report (`SMOKE=1` for a 1-min dry run) |
| `templates/mock_train.py` | fake trainer so the chain runs without a simulation |

Requirements for the templates: Python ≥ 3.9, `optuna` ≥ 3 (tested with 4.6), `matplotlib`
optional (scatter figure).

## Try it

    cp templates/* /some/empty/dir && cd /some/empty/dir
    SMOKE=1 bash run_tune.sh              # ~1 min, mock trainer
    NT=40 LIMIT_H=0.1 bash run_tune.sh    # ~6 min, 70 trials with pruning, 36-card final grid
    cat results_tune.md

Then point `TRAIN` / `TEST` / `card_args` / `read_metric` at the real trainer and run
`python tune_worker.py --smoke --enqueue` before launching for real.
