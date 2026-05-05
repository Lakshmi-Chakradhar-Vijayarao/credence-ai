"""
Figure 3: Ghost constraint BothRate — naive sliding window vs Credence.

Ghost constraints: uncertain values stated WITHOUT explicit hedging language.
Standard marker detection gets 0% registration; Credence heuristic gets 100%.

Data from ghost_gauntlet_results.json, n=10 sessions naive, n=20 sessions credence.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Per-session BothRate data ──────────────────────────────────────────────────
# naive_window: 10 sessions
naive_sessions = [0.333, 0.0, 0.333, 0.333, 0.0, 0.0, 0.333, 0.667, 0.0, 0.0]
# credence (v1 + eg2 variants): 20 sessions
credence_sessions = [1.0] * 20

naive_mean  = np.mean(naive_sessions)    # 0.200
credence_mean = np.mean(credence_sessions)  # 1.000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0),
                                gridspec_kw={"width_ratios": [1.8, 1]})

# ── Left: per-session dots + mean bar ─────────────────────────────────────────
NAIVE_C   = "#d73027"
CRED_C    = "#4dac26"
JITTER    = 0.08

rng = np.random.default_rng(42)
x_naive    = np.zeros(len(naive_sessions))    + rng.uniform(-JITTER, JITTER, len(naive_sessions))
x_credence = np.ones(len(credence_sessions))  + rng.uniform(-JITTER, JITTER, len(credence_sessions))

ax1.bar([0, 1], [naive_mean, credence_mean], width=0.45,
        color=[NAIVE_C, CRED_C], alpha=0.25, zorder=2, edgecolor="none")
ax1.scatter(x_naive,    naive_sessions,    color=NAIVE_C,  alpha=0.75, s=35, zorder=4, label="Naive window")
ax1.scatter(x_credence, credence_sessions, color=CRED_C,   alpha=0.75, s=35, zorder=4, label="Credence")

# Mean annotations
ax1.text(0, naive_mean + 0.04, f"mean={naive_mean:.2f}", ha="center", fontsize=8.5, color=NAIVE_C, fontweight="bold")
ax1.text(1, credence_mean + 0.02, f"mean={credence_mean:.2f}", ha="center", fontsize=8.5, color="#276419", fontweight="bold")

ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Naive sliding window\n(n=10 sessions)", "Credence\n(n=20 sessions)"], fontsize=9)
ax1.set_ylabel("BothRate\n(value recalled AND uncertainty preserved)", fontsize=8.5)
ax1.set_ylim(-0.08, 1.18)
ax1.set_title("(a)  Per-session BothRate — ghost constraints\n(values stated without hedging language)", fontsize=9, loc="left", pad=5)
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=8, loc="upper left")

# ── Right: breakdown of naive failures ────────────────────────────────────────
# Of the 10 naive sessions: 5 both_rate=0.0, 4 both_rate=0.333, 1 both_rate=0.667
naive_counts = {0.0: 5, 0.333: 4, 0.667: 1}
btrates = sorted(naive_counts.keys())
counts  = [naive_counts[b] for b in btrates]
bar_colors = ["#d73027", "#f4a582", "#fddbc7"]

wedge_labels = [f"BothRate = {b:.2f}\n({c} session{'s' if c > 1 else ''})"
                for b, c in zip(btrates, counts)]
ax2.barh([2, 1, 0], counts, color=bar_colors, edgecolor="white", height=0.55)

for i, (c, b) in enumerate(zip(counts, btrates)):
    ax2.text(c + 0.08, [2, 1, 0][i], f"{b:.2f}", va="center", fontsize=8.5)

ax2.set_yticks([])
ax2.set_xlabel("Number of sessions", fontsize=9)
ax2.set_xlim(0, 6.5)
ax2.set_title("(b)  Naive window BothRate\ndistribution (n=10)", fontsize=9, loc="left", pad=5)

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in
                  zip(bar_colors, ["0.00 (full failure)", "0.33 (partial)", "0.67 (partial)"])]
ax2.legend(handles=legend_patches, fontsize=7.5, loc="lower right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

fig.suptitle(
    "Figure 3.  Ghost constraint recall: the naive sliding window loses uncertainty context\n"
    "in 9/10 sessions; Credence domain-keyword heuristic achieves BothRate = 1.0 in all 20.",
    fontsize=9, y=1.02, x=0.02, ha="left"
)

plt.tight_layout()
plt.savefig("fig3_ghost_bothrate.pdf", bbox_inches="tight")
plt.savefig("fig3_ghost_bothrate.png", dpi=300, bbox_inches="tight")
print("Saved fig3_ghost_bothrate.pdf + .png")
