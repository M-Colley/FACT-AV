"""Shared publication-quality matplotlib style for FACT-AV figures.

Import and call ``apply_paper_style()`` at the top of any plotting script.
All figures use a colorblind-safe Okabe-Ito palette and consistent typography.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (8 colors).
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Two-condition palette used for INTRODUCTION (ambiguous vs boasting).
INTRO_COLORS = {
    "ambiguous": "#0072B2",
    "ambigious": "#0072B2",  # tolerate the typo present in raw data
    "boasting": "#D55E00",
}

# Four-scenario palette used for SCENARIO panels.
SCENARIO_COLORS = {
    "3Spurig": "#009E73",
    "NeueMitte": "#E69F00",
    "Spielstrasse": "#CC79A7",
    "Ueberland": "#56B4E9",
}

# Human-readable scenario labels.
SCENARIO_LABELS = {
    "3Spurig": "Highway",
    "NeueMitte": "City",
    "Spielstrasse": "Walking Zone",
    "Ueberland": "Cross-Country",
}


def apply_paper_style(font_scale: float = 1.0) -> None:
    """Set matplotlib rcParams for consistent publication-quality figures."""
    base_size = 11 * font_scale
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "DejaVu Sans",
            "font.size": base_size,
            "axes.titlesize": base_size * 1.1,
            "axes.labelsize": base_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.labelsize": base_size * 0.9,
            "ytick.labelsize": base_size * 0.9,
            "legend.fontsize": base_size * 0.9,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def palette(n: int) -> Sequence[str]:
    """Return ``n`` colors from the Okabe-Ito palette, cycling if needed."""
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def save_fig(fig: plt.Figure, path_stem, *, formats: Sequence[str] = ("pdf", "png")) -> None:
    """Save a figure to both PDF (vector, archival) and PNG (for previews)."""
    from pathlib import Path

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"))
