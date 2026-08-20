"""Shared scatter + fitted-curve figure for the PySR pipelines.

The four ``main_*_pysr_*.py`` scripts each contained their own copy of the same
~20-line seaborn block, differing only in line colour, scatter alpha, y-limits,
title, and whether a hue was applied. Those copies had already drifted apart in
small ways, which is how two figures of the same fit end up looking different.
Every parameter that actually varied between the copies is a keyword argument
here, so the rendered output is unchanged for every existing caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _as_1d(values) -> np.ndarray:
    """Accept a Series, a single-column DataFrame, or an array."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"Expected a single column, got shape {values.shape}.")
        values = values.iloc[:, 0]
    return np.asarray(values).ravel()


def save_relationship_plot(
    x_values,
    y_values,
    predictions,
    plot_path: Path | str,
    *,
    title: str = "Visualization of the Equation",
    xlabel: str = "mIoU",
    ylabel: str = "Trust",
    hue_series: Sequence | None = None,
    legend: bool = True,
    line_color: str = "black",
    scatter_alpha: float = 0.3,
    scatter_color: str | None = None,
    ylim: tuple[float, float] = (1, 6),
    sort_by_x: bool = False,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Scatter of the observed data with the fitted curve on top.

    Parameters
    ----------
    x_values, y_values, predictions
        Arrays (or Series / single-column DataFrames) of equal length.
    sort_by_x
        Sort the curve by ``x`` before drawing. Required whenever the model was
        fitted on more predictors than the single ``x`` being plotted: the rows
        then arrive in dataset order rather than x order, and an unsorted line
        plot draws a zigzag across the panel instead of a curve.
    """
    x = _as_1d(x_values)
    y = _as_1d(y_values)
    y_hat = _as_1d(predictions)
    if not (len(x) == len(y) == len(y_hat)):
        raise ValueError(f"Length mismatch: x={len(x)}, y={len(y)}, predictions={len(y_hat)}.")

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=figsize)

    scatter_kwargs = dict(alpha=scatter_alpha, s=50, edgecolor=None, ax=ax)
    if hue_series is not None:
        sns.scatterplot(
            x=x, y=y, hue=hue_series, palette="viridis", legend=legend, **scatter_kwargs
        )
    elif scatter_color is not None:
        sns.scatterplot(x=x, y=y, color=scatter_color, **scatter_kwargs)
    else:
        sns.scatterplot(x=x, y=y, **scatter_kwargs)

    if sort_by_x:
        order = np.argsort(x)
        ax.plot(x[order], y_hat[order], color=line_color, lw=2)
    else:
        sns.lineplot(x=x, y=y_hat, color=line_color, lw=2, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    sns.despine()

    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
