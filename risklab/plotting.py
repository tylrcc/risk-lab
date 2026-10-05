from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#1f2937", "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#9ca3af"]


def setup() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
