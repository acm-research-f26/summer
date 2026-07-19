
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: Colab scripts and pytest both render to file
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"   # primary series
AQUA = "#1baf7a"   # secondary series
RED = "#d03b3b"    # synthetic-data warning


def new_axes(figsize=(7.0, 5.0)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {path}")
