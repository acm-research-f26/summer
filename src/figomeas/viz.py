"""QC overlays and result figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABEL_COLORS = {
    1: (0.55, 0.55, 0.60),   # wall
    2: (0.20, 0.55, 0.90),   # cavity
    3: (0.90, 0.35, 0.25),   # fibroid
    4: (0.95, 0.80, 0.25),   # nabothian
}


def _outline(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return binary & ~ndimage.binary_erosion(binary)


def overlay_body(mask, model, out_path, title="", n_slices=4):
    """Contact sheet: labels underneath, reconstructed surfaces drawn on top.

    Gate 4 of Phase 2 is visual and cannot be automated away -- an over-inflated
    body is obvious in one glance and invisible in a summary statistic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = np.nonzero(((mask > 0)).any(axis=(0, 1)))[0]
    if present.size == 0:
        return None
    picks = present[np.linspace(0, present.size - 1, min(n_slices, present.size)).astype(int)]

    rows = np.nonzero((mask > 0).any(axis=(1, 2)))[0]
    cols = np.nonzero((mask > 0).any(axis=(0, 2)))[0]
    r0, r1 = max(rows.min() - 10, 0), min(rows.max() + 10, mask.shape[0])
    c0, c1 = max(cols.min() - 10, 0), min(cols.max() + 10, mask.shape[1])

    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, picks):
        sl = mask[r0:r1, c0:c1, k]
        rgb = np.ones((*sl.shape, 3))
        for lab, col in LABEL_COLORS.items():
            rgb[sl == lab] = col
        body_o = _outline(model.body[r0:r1, c0:c1, k])
        cav_o = _outline(model.cavity_ref[r0:r1, c0:c1, k])
        rgb[body_o] = (0.05, 0.05, 0.05)
        rgb[cav_o] = (0.10, 0.85, 0.45)
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(f"slice {k}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{title}   grey=wall blue=cavity red=fibroid | "
                 f"black=serosa ref, green=endometrial ref", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
