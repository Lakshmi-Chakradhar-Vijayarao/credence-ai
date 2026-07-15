"""
Figure 5: Credence system overhead — per-component latency (P50/P95/P99).

Shows that the entire system adds <0.1ms per turn.
Data from latency_report_results.json.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data (from latency_report_results.json) ───────────────────────────────────
COMPONENTS = [
    ("Faithfulness Probe\n(CP1, 423 markers)",  0.0173, 0.0217, 0.0251),
    ("J-score\n(CredenceProxy)",                0.0135, 0.0150, 0.0189),
    ("Consistency Enforcer\n(CP2, 52 clusters)",0.0060, 0.0090, 0.0186),
    ("Generation Scanner\n(CP3, GTS)",          0.0256, 0.0299, 0.0416),
    ("Registry write\n(SQLite)",                0.0220, 0.0273, 0.0453),
    ("Registry read\n(SQLite)",                 0.0020, 0.0022, 0.0041),
]

labels = [c[0] for c in COMPONENTS]
p50 = np.array([c[1] for c in COMPONENTS])
p95 = np.array([c[2] for c in COMPONENTS])
p99 = np.array([c[3] for c in COMPONENTS])

y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.5, 4.5))

# P99 as background bar
ax.barh(y, p99, height=0.55, color="#d9d9d9", label="P99", zorder=2)
# P95
ax.barh(y, p95, height=0.55, color="#9ecae1", label="P95", zorder=3)
# P50
ax.barh(y, p50, height=0.55, color="#2166ac", label="P50 (median)", zorder=4)

# Value annotations on P50
for i, (val50, val99) in enumerate(zip(p50, p99)):
    ax.text(val50 + 0.0005, i, f"{val50:.3f}", va="center", ha="left",
            fontsize=7.8, color="#2166ac", fontweight="bold", zorder=5)
    ax.text(val99 + 0.0005, i - 0.22, f"P99={val99:.3f}", va="center", ha="left",
            fontsize=6.8, color="#666666", zorder=5)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Latency (ms)", fontsize=9.5)
ax.set_xlim(0, 0.065)

# Annotate "total per-turn ≈ 0.07ms" region
total_p50 = sum(p50)
ax.axvline(total_p50, color="#d73027", linewidth=1.0, linestyle="--", zorder=6)
ax.text(total_p50 + 0.001, len(labels) - 0.6,
        f"Sum P50 ≈ {total_p50:.2f} ms",
        color="#d73027", fontsize=8, va="top")

ax.set_title(
    "Figure 5.  Per-component latency (P50 / P95 / P99, n = 500–1000 each)\n"
    "Total system overhead < 0.1 ms per turn at P50.",
    fontsize=9, pad=6, loc="left"
)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fig5_latency.pdf", bbox_inches="tight")
plt.savefig("fig5_latency.png", dpi=300, bbox_inches="tight")
print("Saved fig5_latency.pdf + .png")
