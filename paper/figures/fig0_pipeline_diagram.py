"""
Figure 0 (conceptual): The EQL failure mode — pipeline diagram.

Illustrates how qualifier loss occurs: user states uncertain value →
context compression strips the qualifier → downstream model treats it as fact →
unverified constant ships to production.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

def rounded_box(ax, x, y, w, h, color, text, fontsize=8.5, text_color="white"):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, zorder=4, wrap=True,
            multialignment="center")

def arrow(ax, x1, y, x2, color="#555555"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14), zorder=5)

# ── Row 1: without Credence (failure path) ────────────────────────────────────
y_row1 = 2.4
label_y = 3.7

ax.text(0.05, label_y, "Without Credence  (EQLR = 0.46–0.75)", fontsize=9,
        color="#d73027", fontweight="bold", va="top")

# Boxes
rounded_box(ax, 0.05, y_row1, 1.8, 0.9, "#4292c6",
            '"I think the rate\nlimit is ~50 req/min\n— not confirmed"')
arrow(ax, 1.85, y_row1 + 0.45, 2.1)

rounded_box(ax, 2.1, y_row1, 2.0, 0.9, "#d73027",
            "Context compression\nstrips qualifier:\n\"rate limit: 50\"")
arrow(ax, 4.1, y_row1 + 0.45, 4.35)

rounded_box(ax, 4.35, y_row1, 2.0, 0.9, "#d73027",
            "Model treats value\nas confirmed fact\n(EQLR event)")
arrow(ax, 6.35, y_row1 + 0.45, 6.6)

rounded_box(ax, 6.6, y_row1, 2.0, 0.9, "#7f0000",
            "Code ships:\nRATE_LIMIT = 50\n# ← no flag")

# "Production failure" annotation
ax.annotate("Production failure\nat 2 am", xy=(9.0, y_row1 + 0.45),
            xytext=(9.6, y_row1 + 1.1),
            fontsize=7.5, color="#7f0000",
            arrowprops=dict(arrowstyle="-|>", color="#7f0000", lw=1.2))

# ── Divider ────────────────────────────────────────────────────────────────────
ax.axhline(2.2, xmin=0.005, xmax=0.995, color="#cccccc", linewidth=0.8, linestyle="--")

# ── Row 2: with Credence (success path) ───────────────────────────────────────
y_row2 = 0.85

ax.text(0.05, 2.15, "With Credence  (EQLR = 0.00)", fontsize=9,
        color="#276419", fontweight="bold", va="top")

rounded_box(ax, 0.05, y_row2, 1.8, 0.9, "#4292c6",
            '"I think the rate\nlimit is ~50 req/min\n— not confirmed"')

# Observer fires before model even responds
ax.annotate("Observer hook\nregisters immediately\n(before model)", xy=(0.95, y_row2 + 0.9),
            xytext=(0.95, y_row2 + 1.28),
            ha="center", fontsize=7, color="#4dac26",
            arrowprops=dict(arrowstyle="-|>", color="#4dac26", lw=1.0))

arrow(ax, 1.85, y_row2 + 0.45, 2.1)

rounded_box(ax, 2.1, y_row2, 2.0, 0.9, "#74c476",
            "Context compression\nrun; probe intercepts\nif qualifier present",
            text_color="#1a1a1a")
arrow(ax, 4.1, y_row2 + 0.45, 4.35)

rounded_box(ax, 4.35, y_row2, 2.0, 0.9, "#4dac26",
            "Constraint registry:\n\"rate_limit: 50\n[UNVERIFIED]\"")
arrow(ax, 6.35, y_row2 + 0.45, 6.6)

rounded_box(ax, 6.6, y_row2, 2.0, 0.9, "#238b45",
            "Write blocked until\nuser confirms:\n\"Verify first\"",
            text_color="white")

ax.annotate("Gate clears\nafter verification", xy=(9.0, y_row2 + 0.45),
            xytext=(9.6, y_row2 + 1.1),
            fontsize=7.5, color="#238b45",
            arrowprops=dict(arrowstyle="-|>", color="#238b45", lw=1.2))

fig.suptitle(
    "Figure 0.  The EQL failure mode (top) and Credence's intervention (bottom).\n"
    "The observer fires before the model responds; the gate fires before any file write.",
    fontsize=9, y=0.01, x=0.01, ha="left", va="bottom"
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("fig0_pipeline_diagram.pdf", bbox_inches="tight")
plt.savefig("fig0_pipeline_diagram.png", dpi=300, bbox_inches="tight")
print("Saved fig0_pipeline_diagram.pdf + .png")
