"""
Figure 2: Compressor comparison — Haiku summarisation vs LLMLingua-style vs Probe-blocked.

Shows that different compressors produce different EQLR, but the probe
intercepts all of them at 0.00. Haiku also shows proper FCR (2%) vs LLMLingua (2%).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (from compression_faithfulness_n50_results.json) ─────────────────────
# EQLR = 1 - qualifier_survival_rate
# Haiku:      qualifier_survival = 0.54  → EQLR = 0.46  (n=50, CI [0.318, 0.607])
# LLMLingua:  qualifier_survival = 0.32  → EQLR = 0.68  (n=50, CI [0.536, 0.800])
# Probe:      EQLR = 0.00  (CI [0.000, 0.000])
CONDITIONS = [
    ("Haiku\nsummarisation",     0.46, 0.318, 0.607,  0.02, "#2166ac"),
    ("LLMLingua-style\ntoken compression", 0.68, 0.536, 0.800, 0.02, "#d73027"),
    ("Credence\nprobe-blocked",  0.00, 0.000, 0.000,  0.00, "#4dac26"),
]

labels   = [c[0] for c in CONDITIONS]
eqlr     = np.array([c[1] for c in CONDITIONS])
ci_lo    = np.array([c[1] - c[2] for c in CONDITIONS])
ci_hi    = np.array([c[3] - c[1] for c in CONDITIONS])
fcr      = np.array([c[4] for c in CONDITIONS])
colors   = [c[5] for c in CONDITIONS]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2),
                                gridspec_kw={"width_ratios": [2, 1.2]})

# ── Left: EQLR grouped bar ────────────────────────────────────────────────────
bars = ax1.bar(x, eqlr, width=0.5, color=colors, zorder=3,
               edgecolor="white", linewidth=0.5)
ax1.errorbar(x, eqlr, yerr=[ci_lo, ci_hi],
             fmt="none", color="#1a1a1a", capsize=5, linewidth=1.3, zorder=4)

for bar, val in zip(bars, eqlr):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.025,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("EQLR", fontsize=10)
ax1.set_ylim(0, 0.9)
ax1.set_title("(a)  EQLR by compression method\n(n = 50 scenarios, 95% CI)", fontsize=9, loc="left", pad=6)
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Right: FCR (proper scorer v3) ─────────────────────────────────────────────
fcr_colors = ["#2166ac", "#d73027", "#4dac26"]
bars2 = ax2.bar(x, fcr, width=0.5, color=fcr_colors, zorder=3,
                edgecolor="white", linewidth=0.5)

for bar, val in zip(bars2, fcr):
    label = f"{val:.0%}" if val > 0 else "0%"
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
             label, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("False Certainty Rate (proper FCR)", fontsize=9)
ax2.set_ylim(0, 0.12)
ax2.set_title("(b)  Downstream false certainty\n(v3 scorer — epistemic erasure excluded)", fontsize=9, loc="left", pad=6)
ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 2.  Two different compression methods both produce substantial EQLR;\n"
    "the Credence probe eliminates EQL without compressor-specific tuning.",
    fontsize=9, y=1.01, x=0.02, ha="left"
)

plt.tight_layout()
plt.savefig("fig2_compressor_comparison.pdf", bbox_inches="tight")
plt.savefig("fig2_compressor_comparison.png", dpi=300, bbox_inches="tight")
print("Saved fig2_compressor_comparison.pdf + .png")
