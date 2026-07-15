"""
Figure 4: E8 multi-turn debugging session — recall by condition.

Shows that naive window compression loses debugging context across turns
(mean recall 0.64 vs baseline 0.67), while Credence achieves perfect recall (1.0).

Data from e8_results.json.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Per-question recall ────────────────────────────────────────────────────────
# From e8_results.json
QUESTIONS = [
    "Q1: Exact error\n& file/line",
    "Q2: Two uncertain\nhypotheses",
    "Q3: Fix attempted\n& outcome",
]

# recall_score per question per condition (from callback_logs)
# baseline
BASELINE = [1.0, 1.0, 0.0]
# naive_window
NAIVE    = [0.6, 0.333, 1.0]
# credence
CREDENCE = [1.0, 1.0, 1.0]

MEANS = {
    "Baseline (no compression)":  np.mean(BASELINE),   # 0.667
    "Naive sliding window":        np.mean(NAIVE),      # 0.644
    "Credence":                    np.mean(CREDENCE),   # 1.000
}

COLORS = {
    "Baseline (no compression)":  "#969696",
    "Naive sliding window":        "#d73027",
    "Credence":                    "#4dac26",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2),
                                gridspec_kw={"width_ratios": [1.1, 2]})

# ── Left: mean recall bar chart ───────────────────────────────────────────────
names  = list(MEANS.keys())
vals   = [MEANS[n] for n in names]
colors = [COLORS[n] for n in names]

bars = ax1.bar(range(3), vals, width=0.55, color=colors, zorder=3,
               edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xticks(range(3))
ax1.set_xticklabels(["Baseline", "Naive\nwindow", "Credence"], fontsize=8.5)
ax1.set_ylabel("Mean recall score", fontsize=9)
ax1.set_ylim(0, 1.18)
ax1.set_title("(a)  Mean recall\nacross 3 questions", fontsize=9, loc="left", pad=5)
ax1.axhline(1.0, color="#aaaaaa", linewidth=0.7, linestyle="--")
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Right: per-question grouped bars ─────────────────────────────────────────
x = np.arange(len(QUESTIONS))
width = 0.26

r1 = ax2.bar(x - width,     BASELINE, width, color=COLORS["Baseline (no compression)"],
             label="Baseline", zorder=3, edgecolor="white")
r2 = ax2.bar(x,              NAIVE,    width, color=COLORS["Naive sliding window"],
             label="Naive window", zorder=3, edgecolor="white")
r3 = ax2.bar(x + width,     CREDENCE,  width, color=COLORS["Credence"],
             label="Credence", zorder=3, edgecolor="white")

def annotate_bars(bars_group, ax):
    for bar in bars_group:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

annotate_bars(r1, ax2)
annotate_bars(r2, ax2)
annotate_bars(r3, ax2)

ax2.set_xticks(x)
ax2.set_xticklabels(QUESTIONS, fontsize=8.5)
ax2.set_ylabel("Recall score", fontsize=9)
ax2.set_ylim(0, 1.18)
ax2.set_title("(b)  Per-question recall by condition", fontsize=9, loc="left", pad=5)
ax2.axhline(1.0, color="#aaaaaa", linewidth=0.7, linestyle="--")
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 4.  E8 multi-turn debugging session: naive context compression loses uncertainty context\n"
    "across turns (Q2 recall 0.33); Credence constraint registry achieves 1.0 on all three questions.",
    fontsize=9, y=1.02, x=0.02, ha="left"
)

plt.tight_layout()
plt.savefig("fig4_e8_recall.pdf", bbox_inches="tight")
plt.savefig("fig4_e8_recall.png", dpi=300, bbox_inches="tight")
print("Saved fig4_e8_recall.pdf + .png")
