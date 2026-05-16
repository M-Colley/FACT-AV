#!/usr/bin/env python3
"""Publication-quality figures for the FACT-AV trust calibration study.

This script consolidates four reviewer-grade figures into one runnable module:

1. **Cross-model feature-importance forest plot** — all tree-based models on one
   normalized axis so reviewers can compare them visually.
2. **Per-scenario mIoU -> trust panel** — a 2x2 grid (one scenario per panel)
   with both INTRODUCTION conditions overlaid, including bootstrap 95% bands.
3. **Partial Dependence + ICE plots** — mIoU PDP per (INTRODUCTION, SCENARIO)
   cell with individual conditional expectation curves overlaid.
4. **Cross-model importance ranking heatmap** — companion to the forest plot,
   showing per-feature rank stability across models.

All figures use the Okabe-Ito colorblind-safe palette via ``plotting_style``.
Run with ``python publication_figures.py`` from the repository root.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from plotting_style import (
    INTRO_COLORS,
    OKABE_ITO,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    apply_paper_style,
    save_fig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class FigureConfig:
    """Paths and modeling settings for publication figures."""

    # Baseline file has BOTH demographics AND real ProlificIDs (134 participants),
    # which the demographics-only file lacks.
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
    results_path: Path = Path("results") / "publication"
    sheet_name: str = "Sheet1"
    random_state: int = 42
    test_size: float = 0.2
    bootstrap_n: int = 1000

    def __post_init__(self) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)


INTRODUCTION_NORMALIZER = {"ambigious": "ambiguous", "ambiguous": "ambiguous", "boasting": "boasting"}

NUM_FEATURES: tuple[str, ...] = ("mIoU", "Age", "License")
CAT_FEATURES: tuple[str, ...] = (
    "SCENARIO",
    "INTRODUCTION",
    "Gender",
    "Education",
    "Job",
    "DrivingFrequency",
    "Distance",
)


def load_data(config: FigureConfig) -> pd.DataFrame:
    """Load the demographic+baseline dataset and normalize categorical strings."""
    df = pd.read_excel(config.data_path, sheet_name=config.sheet_name)
    needed = list(NUM_FEATURES) + list(CAT_FEATURES) + ["trust", "ProlificID"]
    df = df.dropna(subset=needed).copy()
    df["INTRODUCTION"] = df["INTRODUCTION"].astype(str).str.lower().map(INTRODUCTION_NORMALIZER)
    df["SCENARIO"] = df["SCENARIO"].astype(str)
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)
    df["trust"] = pd.to_numeric(df["trust"], errors="raise")
    logger.info("Loaded %d rows, %d unique participants.", len(df), df["ProlificID"].nunique())
    return df


# ---------------------------------------------------------------------------
# 1. Forest plot: cross-model feature importance comparison
# ---------------------------------------------------------------------------


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative importance vector to sum to 1."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(arr < 0, 0.0, arr)
    total = arr.sum()
    return arr / total if total > 0 else arr


def collect_model_importances(df: pd.DataFrame, config: FigureConfig) -> pd.DataFrame:
    """Fit RF + (optional) gradient-boosting models and return tidy importances.

    Each model is trained on a participant-grouped train split. The returned
    DataFrame has columns ``feature``, ``model``, ``importance`` (normalized).
    """
    X = df[list(NUM_FEATURES) + list(CAT_FEATURES)].copy()
    y = df["trust"].copy()
    groups = df["ProlificID"].copy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.test_size, random_state=config.random_state
    )
    (train_idx, _), = splitter.split(X, y, groups=groups)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUM_FEATURES)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
                list(CAT_FEATURES),
            ),
        ]
    )

    importances: List[pd.DataFrame] = []

    # Random Forest -----------------------------------------------------------
    rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=config.random_state)),
        ]
    )
    rf.fit(X_train, y_train)
    rf_features = rf.named_steps["preprocessor"].get_feature_names_out()
    rf_features = [f.split("__", 1)[-1] for f in rf_features]
    importances.append(
        pd.DataFrame(
            {
                "feature": rf_features,
                "model": "Random Forest",
                "importance": _normalize_importance(rf.named_steps["regressor"].feature_importances_),
            }
        )
    )

    # XGBoost ----------------------------------------------------------------
    try:
        import xgboost as xgb

        X_train_xgb = X_train.copy()
        for c in CAT_FEATURES:
            X_train_xgb[c] = X_train_xgb[c].astype("category")
        xgb_model = xgb.XGBRegressor(
            random_state=config.random_state, enable_categorical=True, n_estimators=300
        )
        xgb_model.fit(X_train_xgb, y_train)
        importances.append(
            pd.DataFrame(
                {
                    "feature": list(X_train_xgb.columns),
                    "model": "XGBoost",
                    "importance": _normalize_importance(xgb_model.feature_importances_),
                }
            )
        )
    except ImportError:
        logger.warning("XGBoost not available - skipping in forest plot.")

    # LightGBM ---------------------------------------------------------------
    try:
        import lightgbm as lgb

        X_train_lgb = X_train.copy()
        for c in CAT_FEATURES:
            X_train_lgb[c] = X_train_lgb[c].astype("category")
        lgb_model = lgb.LGBMRegressor(random_state=config.random_state, verbose=-1, n_estimators=300)
        lgb_model.fit(X_train_lgb, y_train)
        importances.append(
            pd.DataFrame(
                {
                    "feature": list(X_train_lgb.columns),
                    "model": "LightGBM",
                    "importance": _normalize_importance(lgb_model.feature_importances_),
                }
            )
        )
    except ImportError:
        logger.warning("LightGBM not available - skipping in forest plot.")

    # CatBoost ---------------------------------------------------------------
    try:
        from catboost import CatBoostRegressor

        cb_model = CatBoostRegressor(
            random_state=config.random_state,
            verbose=False,
            cat_features=list(CAT_FEATURES),
            iterations=300,
        )
        cb_model.fit(X_train, y_train)
        importances.append(
            pd.DataFrame(
                {
                    "feature": list(X_train.columns),
                    "model": "CatBoost",
                    "importance": _normalize_importance(cb_model.get_feature_importance()),
                }
            )
        )
    except ImportError:
        logger.warning("CatBoost not available - skipping in forest plot.")

    return pd.concat(importances, ignore_index=True)


def _consolidate_one_hot(importances: pd.DataFrame) -> pd.DataFrame:
    """Sum importances across one-hot columns of the same parent feature."""
    def parent(name: str) -> str:
        for cat in CAT_FEATURES:
            if name == cat or name.startswith(f"{cat}_"):
                return cat
        return name

    out = importances.copy()
    out["feature"] = out["feature"].map(parent)
    return out.groupby(["feature", "model"], as_index=False)["importance"].sum()


def plot_forest_importances(importances: pd.DataFrame, config: FigureConfig) -> None:
    """Forest plot: one row per feature, one marker per model, normalized x-axis."""
    apply_paper_style()
    df = _consolidate_one_hot(importances)
    feature_order = (
        df.groupby("feature")["importance"].mean().sort_values(ascending=True).index.tolist()
    )
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(feature_order) + 1.5))
    y_positions = np.arange(len(feature_order))
    width = 0.7 / max(len(models), 1)
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("feature").reindex(feature_order)
        offset = (i - (len(models) - 1) / 2) * width
        ax.scatter(
            sub["importance"].values,
            y_positions + offset,
            label=model,
            color=OKABE_ITO[(i + 1) % len(OKABE_ITO)],
            s=55,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("Normalized feature importance (per model)")
    ax.set_title("Cross-model feature importance for predicting trust")
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.legend(loc="lower right", ncols=2)
    save_fig(fig, config.results_path / "forest_feature_importance")
    plt.close(fig)


def plot_importance_rank_heatmap(importances: pd.DataFrame, config: FigureConfig) -> None:
    """Companion heatmap showing per-feature rank stability across models."""
    apply_paper_style()
    df = _consolidate_one_hot(importances)
    wide = df.pivot(index="feature", columns="model", values="importance").fillna(0.0)
    ranks = wide.rank(axis=0, ascending=False, method="min").astype(int)
    order = wide.mean(axis=1).sort_values(ascending=False).index
    ranks = ranks.loc[order]

    fig, ax = plt.subplots(figsize=(1.6 * len(ranks.columns) + 2, 0.4 * len(ranks.index) + 1.5))
    im = ax.imshow(ranks.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(ranks.columns)))
    ax.set_xticklabels(ranks.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ranks.index)))
    ax.set_yticklabels(ranks.index)
    for (i, j), value in np.ndenumerate(ranks.values):
        ax.text(j, i, int(value), ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Feature importance rank by model (1 = most important)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Rank")
    save_fig(fig, config.results_path / "importance_rank_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-scenario mIoU -> trust panel with bootstrap CI bands
# ---------------------------------------------------------------------------


def _loess_like(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Simple kernel-smoothed mean (Gaussian kernel) - a robust LOESS stand-in."""
    weights = np.exp(-0.5 * ((x_grid[:, None] - x[None, :]) / bandwidth) ** 2)
    norm = weights.sum(axis=1)
    norm = np.where(norm == 0, 1, norm)
    return (weights * y[None, :]).sum(axis=1) / norm


def _bootstrap_smooth(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point estimate plus pointwise 2.5/97.5 percentile bands."""
    if len(x) < 5:
        flat = np.full_like(x_grid, np.mean(y) if len(y) else np.nan, dtype=float)
        return flat, flat, flat
    boots = np.empty((n_boot, len(x_grid)))
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _loess_like(x[idx], y[idx], x_grid, bandwidth)
    mean = _loess_like(x, y, x_grid, bandwidth)
    lower = np.nanpercentile(boots, 2.5, axis=0)
    upper = np.nanpercentile(boots, 97.5, axis=0)
    return mean, lower, upper


def plot_miou_trust_panel(df: pd.DataFrame, config: FigureConfig) -> None:
    """2x2 grid of mIoU -> trust curves, one panel per scenario, both intros overlaid."""
    apply_paper_style()
    rng = np.random.default_rng(config.random_state)

    scenarios = sorted(df["SCENARIO"].unique(), key=lambda s: SCENARIO_LABELS.get(s, s))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    x_lo, x_hi = float(df["mIoU"].min()), float(df["mIoU"].max())
    x_grid = np.linspace(x_lo, x_hi, 80)
    bandwidth = max((x_hi - x_lo) / 8.0, 0.5)

    for ax, scenario in zip(axes, scenarios):
        for intro in ("ambiguous", "boasting"):
            sub = df[(df["SCENARIO"] == scenario) & (df["INTRODUCTION"] == intro)]
            if len(sub) == 0:
                continue
            x = sub["mIoU"].to_numpy(dtype=float)
            y = sub["trust"].to_numpy(dtype=float)
            mean, lower, upper = _bootstrap_smooth(
                x, y, x_grid, bandwidth, config.bootstrap_n, rng
            )
            color = INTRO_COLORS.get(intro, "grey")
            ax.fill_between(x_grid, lower, upper, color=color, alpha=0.2)
            ax.plot(x_grid, mean, color=color, linewidth=2.0, label=intro.capitalize())
            ax.scatter(
                x,
                y + rng.normal(0, 0.02, size=len(y)),  # tiny jitter for visibility
                color=color,
                alpha=0.18,
                s=10,
                edgecolor="none",
            )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_ylim(0.8, 5.2)

    for ax in axes[2:]:
        ax.set_xlabel("mIoU")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Trust (1-5)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 1.02),
        title="Introduction",
        title_fontsize=11,
    )
    fig.suptitle("mIoU -> Trust by scenario (bootstrap 95% bands)", y=1.07, fontsize=13)
    fig.tight_layout()
    save_fig(fig, config.results_path / "miou_trust_panel")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Partial Dependence + ICE plots for mIoU
# ---------------------------------------------------------------------------


def plot_pdp_ice(df: pd.DataFrame, config: FigureConfig) -> None:
    """PDP + ICE for mIoU across the 8 (INTRODUCTION, SCENARIO) cells.

    Trains one CatBoost (or RF fallback) on the full data with participant-grouped
    train split, then computes PDP+ICE on each subset using sklearn's
    ``PartialDependenceDisplay`` with ``kind='both'``.
    """
    apply_paper_style()
    X = df[list(NUM_FEATURES) + list(CAT_FEATURES)].copy()
    y = df["trust"].copy()
    groups = df["ProlificID"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    (train_idx, _), = splitter.split(X, y, groups=groups)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    # Use a sklearn-style estimator so PartialDependenceDisplay just works.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUM_FEATURES)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
                list(CAT_FEATURES),
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=config.random_state)),
        ]
    )
    model.fit(X_train, y_train)

    scenarios = sorted(df["SCENARIO"].unique(), key=lambda s: SCENARIO_LABELS.get(s, s))
    intros = ("ambiguous", "boasting")
    fig, axes = plt.subplots(
        len(intros), len(scenarios), figsize=(3.0 * len(scenarios), 3.0 * len(intros)),
        sharex=True, sharey=True,
    )

    for i, intro in enumerate(intros):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            sub = X[(X["INTRODUCTION"] == intro) & (X["SCENARIO"] == scenario)]
            if len(sub) < 5:
                ax.set_axis_off()
                continue
            sample = sub.sample(min(len(sub), 200), random_state=config.random_state)
            try:
                PartialDependenceDisplay.from_estimator(
                    model,
                    sample,
                    features=["mIoU"],
                    kind="both",
                    ax=ax,
                    grid_resolution=30,
                    ice_lines_kw={"alpha": 0.08, "color": INTRO_COLORS[intro]},
                    pd_line_kw={"color": "black", "linewidth": 2.0},
                )
            except Exception as exc:
                logger.warning("PDP failed for %s/%s: %s", intro, scenario, exc)
                ax.set_axis_off()
                continue
            ax.set_title(f"{intro.capitalize()}\n{SCENARIO_LABELS.get(scenario, scenario)}")
            ax.set_xlabel("mIoU" if i == len(intros) - 1 else "")
            ax.set_ylabel("Predicted trust" if j == 0 else "")
            ax.legend().remove() if ax.get_legend() else None

    fig.suptitle("Partial Dependence + ICE for mIoU by (Introduction x Scenario)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, config.results_path / "pdp_ice_miou_by_cell")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    config = FigureConfig()
    df = load_data(config)

    logger.info("Computing cross-model importances...")
    importances = collect_model_importances(df, config)
    importances.to_csv(config.results_path / "model_importances_raw.csv", index=False)

    logger.info("Plotting forest plot + rank heatmap...")
    plot_forest_importances(importances, config)
    plot_importance_rank_heatmap(importances, config)

    logger.info("Plotting per-scenario mIoU -> trust panel...")
    plot_miou_trust_panel(df, config)

    logger.info("Plotting PDP+ICE per (Introduction x Scenario) cell...")
    plot_pdp_ice(df, config)

    logger.info("Done. Figures saved under %s", config.results_path)


if __name__ == "__main__":
    main()
