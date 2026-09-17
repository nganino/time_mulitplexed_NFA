import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = r"P:\Nicholas\time_multiplexed_NFA\tune\test_results.csv"
OUT_PATH = r"P:\Nicholas\time_multiplexed_NFA\tune\accuracy_comparison.png"

rows = []
with open(CSV_PATH, newline='') as f:
    for row in csv.DictReader(f):
        rows.append(row)

labels = [r['label'] for r in rows]
acc = [float(r['accuracy']) * 100 for r in rows]

BAR_COLOR = '#4C72B0'
TEXT_COLOR = '#333333'
GRID_COLOR = '#DDDDDD'

fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
bars = ax.bar(labels, acc, width=0.5, color=BAR_COLOR)

for bar, val in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, color=TEXT_COLOR)

ax.set_ylabel('CIFAR-10 test accuracy (%)', color=TEXT_COLOR, fontsize=10)
ax.set_title('Test accuracy vs. number of countries (C)\nmajority-voting sweep, M=1 members/country',
             fontsize=11, color=TEXT_COLOR)
ax.set_ylim(0, max(acc) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', length=0, labelsize=10, colors=TEXT_COLOR)
ax.tick_params(axis='y', length=0, labelsize=9, colors=TEXT_COLOR)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches='tight')
print(f'Saved -> {OUT_PATH}')
