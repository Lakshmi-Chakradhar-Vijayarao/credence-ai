"""
Figure 1: EQLR across 8 models — unguarded vs probe-blocked.

The headline empirical result: qualifier loss is model-agnostic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (from multimodel_eqlr_results.json) ──────────────────────────────────
# Ordered by unguarded EQLR ascending so the range is visually clear
MODELS = [
    ("Llama-3.2-3B\n(Meta, 3B)",     0.4098, 0.2787, 0.5410),
    ("Llama-3.1-8B\n(Meta, 8B)",     0.4262, 0.3115, 0.5574),
    ("Phi-3.5-mini\n(Microsoft, 3.8B)", 0.4426, 0.3279, 0.5574),
    ("Qwen-2.5-1.5B\n(Alibaba, 1.5B)", 0.500,  0.422,  0.578),
    ("Mistral-7B\n(Mistral AI, 7B)", 0.6066, 0.4754, 0.7213),
    ("Gemma-2-9B\n(Google, 9B)",     0.6230, 0.5082, 0.7541),
    ("Claude Haiku\n(Anthropic, small)", 0.46, 0.318, 0.607),
    ("Qwen-2.5-7B\n(Alibaba, 7B)",   0.7541, 0.6393, 0.8525),
]

# Sort by unguarded EQLR
MODELS.sort(key=lambda x: x[1])

labels    = [m[0] for m in MODELS]
eqlr      = np.array([m[1] for m in MODELS])
ci_lo     = np.array([m[1] - m[2] for m in MODELS])   # error below
ci_hi     = np.array([m[3] - m[1] for m in MODELS])   # error above
probe_eqlr = np.zeros(len(MODELS))

# ── Layout ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.8))

y = np.arange(len(labels))
height = 0.35

BAR_COLOR   = "#2166ac"   # blue — unguarded
PROBE_COLOR = "#d9d9d9"   # light gray — probe-blocked (all zero)
CI_COLOR    = "#1a1a1a"

bars1 = ax.barh(y + height / 2, eqlr,      height, color=BAR_COLOR,   label="Unguarded EQLR", zorder=3)
bars2 = ax.barh(y - height / 2, probe_eqlr, height, color=PROBE_COLOR, label="Probe-blocked EQLR", zorder=3)

# Error bars (asymmetric CI) on unguarded bars
ax.errorbar(eqlr, y + height / 2,
            xerr=[ci_lo, ci_hi],
            fmt="none", color=CI_COLOR, capsize=3, linewidth=1.2, zorder=4)

# Reference line at 0.5
ax.axvline(0.5, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
ax.text(0.505, len(labels) - 0.4, "0.5", fontsize=7.5, color="#888888", va="top")

# Annotate bar values
for bar, val in zip(bars1, eqlr):
    ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", fontsize=7.8, color=CI_COLOR)
for bar in bars2:
    ax.text(0.008, bar.get_y() + bar.get_height() / 2,
            "0.00", va="center", ha="left", fontsize=7.8, color="#555555")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("EQLR (Epistemic Qualifier Loss Rate)", fontsize=9.5)
ax.set_xlim(0, 0.98)
ax.set_title(
    "Figure 1.  EQLR across 8 models from 6 organizations\n"
    "Whiskers = 95% bootstrap CI.  Probe-blocked EQLR = 0.00 for all models.",
    fontsize=9, pad=8, loc="left"
)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color=BAR_COLOR,   label="Unguarded EQLR (95% CI)"),
    mpatches.Patch(color=PROBE_COLOR, label="Probe-blocked EQLR (all = 0.00)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
plt.savefig("fig1_multimodel_eqlr.pdf", bbox_inches="tight")
plt.savefig("fig1_multimodel_eqlr.png", dpi=300, bbox_inches="tight")
print("Saved fig1_multimodel_eqlr.pdf + .png")
