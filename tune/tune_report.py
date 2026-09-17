"""Report of the study -> <tune dir>/trials.csv, results_tune.md, figures/tune_scatter.png and,
once tune_final.py has run, the full-grid comparison (tuned vs base, gm ratio per axis).

  python tune_report.py            python tune_report.py --smoke
"""
import argparse
import csv
import json
import math
from collections import defaultdict

import optuna

import tune_common as T


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.2e}"


def study_section(study, smoke, lines):
    S = optuna.trial.TrialState
    trials = study.trials
    n = {s: sum(1 for t in trials if t.state == s) for s in S}
    cards = T.SUBSET[:3] if smoke else T.SUBSET
    names = [T.card_name(c) for c in cards]
    lines.append(f"## Study `{study.study_name}`\n")
    lines.append(f"{len(trials)} trials: {n[S.COMPLETE]} complete, {n[S.PRUNED]} pruned, {n[S.FAIL]} failed, "
                 f"{n[S.RUNNING]} running.  Objective = mean log10 metric over the {len(cards)} search cards"
                 f"{' + ordering penalties' if T.PAIRS else ''} (lower is better).\n")
    keys = list(T.SPACE)
    with open(T.tune_dir(smoke) / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "value", *keys, *[f"mse_{x}" for x in names], "worker", "pruned_after", "note"])
        for t in trials:
            w.writerow([t.number, t.state.name, "" if t.value is None else t.value,
                        *[t.params.get(k, "") for k in keys], *[t.user_attrs.get(f"mse_{x}", "") for x in names],
                        t.user_attrs.get("worker", ""), t.user_attrs.get("pruned_after", ""), t.user_attrs.get("note", "")])
    done = T.completed(study)
    if not done:
        lines.append("No completed trial yet.\n")
        return None, None
    best = min(done, key=lambda t: t.value)
    base = next((t for t in done if t.user_attrs.get("note") == "baseline"), None)
    lines.append(f"Best trial **#{best.number}**: objective {best.value:.4f}"
                 + (f" (baseline trial #{base.number}: {base.value:.4f}, geometric-mean ratio "
                    f"{10 ** (best.value - base.value):.2f})" if base else "") + ".\n")
    lines.append("| param | best | baseline | search range |\n|---|---:|---:|---|")
    for k, spec in T.SPACE.items():
        bv, pv = best.params.get(k), T.BASELINE_PARAMS.get(k)
        rng = ", ".join(map(str, spec[1])) if spec[0] == "cat" else f"{spec[1]:g} – {spec[2]:g}{' (log)' if spec[3] else ''}"
        f_ = lambda v: v if isinstance(v, str) else f"{v:.4g}"
        lines.append(f"| {k} | {f_(bv)} | {f_(pv)} | {rng} |")
    lines.append("\n### Search cards: best trial vs baseline trial\n")
    lines.append("| card | MSE best | MSE baseline | ratio | resolved (best) |\n|---|---:|---:|---:|---|")
    for x in names:
        mb = best.user_attrs.get(f"mse_{x}")
        m0 = base.user_attrs.get(f"mse_{x}") if base else None
        ratio = f"{mb / m0:.2f}" if mb and m0 else "—"
        geo = ", ".join(f"{k[:-len(x) - 1]} {v}" for k, v in best.user_attrs.items()
                        if k.endswith("_" + x) and not k.startswith(("mse_", "min_")))
        lines.append(f"| {x} | {fmt(mb)} | {fmt(m0)} | {ratio} | {geo} |")
    lines.append("\n### Top 5 complete trials\n")
    lines.append("| # | value | " + " | ".join(keys) + " |\n|---|---:|" + "---:|" * len(keys))
    for t in sorted(done, key=lambda t: t.value)[:5]:
        lines.append(f"| {t.number} | {t.value:.4f} | " + " | ".join(
            (v if isinstance(v, str) else f"{v:.3g}") for v in (t.params.get(k, "") for k in keys)) + " |")
    lines.append("")
    return best, base


def scatter(study, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    done = T.completed(study)
    keys = [k for k, s in T.SPACE.items() if s[0] != "cat"]
    if not done or not keys:
        return False
    ncol = min(4, len(keys))
    nrow = math.ceil(len(keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    vals = [t.value for t in done]
    for ax, k in zip(axes.flat, keys):
        ax.scatter([t.params[k] for t in done], vals, s=12, c=[t.number for t in done], cmap="viridis")
        if T.SPACE[k][3]:
            ax.set_xscale("log")
        ax.axvline(T.BASELINE_PARAMS.get(k, float("nan")), color="0.6", ls="--", lw=0.8)
        ax.set_xlabel(k)
        ax.set_ylabel("objective")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{study.study_name}: {len(done)} complete trials (colour = trial number)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def final_section(smoke, lines):
    fp = T.tune_dir(smoke) / "final_params.json"
    if not fp.is_file():
        return
    info = json.loads(fp.read_text())
    tuned, base = info["tag_tuned"], info["tag_base"]
    rows = []
    for card in T.CARDS:
        mt = T.read_metric(T.card_dir(tuned, card))
        mb = T.read_metric(T.card_dir(base, card))
        if mt is not None or mb is not None:
            rows.append((card, mt, mb))
    if not rows:
        return
    lines.append(f"## Final grid: `{tuned}` (trial #{info['trial']}) vs `{base}` at {info['epochs']} epochs\n")
    both = [(c, t, b) for c, t, b in rows if t and b]
    if both:
        ratios = [t / b for _, t, b in both]
        lines.append(f"{len(both)} cards with both tags: geometric-mean ratio tuned/base "
                     f"**{T.gm(ratios):.2f}**, tuned lower on {sum(r < 1 for r in ratios)}/{len(both)}.\n")
        lines.append("| axis | value | cards | gm ratio |\n|---|---|---:|---:|")
        for i, (axis, values) in enumerate(T.AXES.items()):
            for v in values:
                sel = [t / b for c, t, b in both if c[i] == v]
                if sel:
                    lines.append(f"| {axis} | {v} | {len(sel)} | {T.gm(sel):.2f} |")
        lines.append("")
    lines.append("| card | tuned | base | ratio |\n|---|---:|---:|---:|")
    for c, t, b in rows:
        lines.append(f"| {T.card_name(c)} | {fmt(t)} | {fmt(b)} | {f'{t / b:.2f}' if t and b else '—'} |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    study = T.load_study(a.smoke)
    lines = [f"# Tuning report: {study.study_name}\n"]
    study_section(study, a.smoke, lines)
    png = T.HERE / "figures" / f"tune_scatter{'_smoke' if a.smoke else ''}.png"
    if scatter(study, png):
        lines.append(f"![scatter]({png.relative_to(T.HERE)})\n")
    final_section(a.smoke, lines)
    out = T.HERE / ("results_tune_smoke.md" if a.smoke else "results_tune.md")
    out.write_text("\n".join(lines))
    print(f"wrote {out} and {T.tune_dir(a.smoke) / 'trials.csv'}")


if __name__ == "__main__":
    main()
