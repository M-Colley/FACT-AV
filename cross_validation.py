#!/usr/bin/env python3
"""Repeated participant-grouped cross-validation for the trust regression models.

Why this exists
---------------
``ML-approaches.py`` evaluates every model on **one** 80/20 participant-grouped
split. With 134 participants that leaves ~27 in test, and a single draw of 27
participants is not enough to separate "this model is worse" from "this split was
unlucky". The headline R^2 values in the README (TabPFN 0.16, CatBoost 0.03,
XGBoost -0.02, LightGBM -0.02, Random Forest -0.06) come from that one split, so
their ordering could easily be noise.

This script re-runs the same models under **repeated GroupKFold**: ``--folds``
folds x ``--repeats`` repeats, with the group-to-fold assignment reshuffled by a
different seed each repeat. Every model sees exactly the same folds. The output
is a mean +- SD (and a paired comparison against a mean-predicting baseline) per
model, which is the form a reviewer can actually interpret.

The mean baseline is included as a model rather than assumed: "no model beats
predicting the mean" is then a computed result under the identical protocol, not
an inference from an R^2 that happens to be negative.

Outputs
-------
``results/ML-Approaches/cv_metrics.json``
    Per-model aggregate (mean, SD, 95% CI of the mean) and every per-fold score.
``results/ML-Approaches/cv_r2_by_model.{png,pdf}``
    Per-fold R^2 distributions with the zero line (= predicting the test mean).

Run with::

    python cross_validation.py --folds 5 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, RandomizedSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from plotting_style import OKABE_ITO, apply_paper_style, save_fig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# matplotlib/fontTools emit hundreds of INFO lines per saved figure ("glyf
# subsetted", ...) which bury the actual results under root-level INFO logging.
for noisy in ("matplotlib", "PIL", "fontTools"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

NUM_FEATURES = ["mIoU", "Age", "License"]
CAT_FEATURES = [
    "SCENARIO",
    "INTRODUCTION",
    "Gender",
    "Education",
    "Job",
    "DrivingFrequency",
    "Distance",
]
TARGET = "trust"
GROUP_COLUMN = "ProlificID"


@dataclass
class CVConfig:
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
    results_path: Path = Path("results") / "ML-Approaches"
    sheet_name: str = "Sheet1"
    folds: int = 5
    repeats: int = 5
    random_state: int = 42
    include_tabpfn: bool = False
    tune: bool = False
    tune_iterations: int = 20
    tune_inner_folds: int = 3
    metrics: list[str] = field(default_factory=lambda: ["r2", "mae", "rmse"])

    def __post_init__(self) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)


def load_data(config: CVConfig) -> pd.DataFrame:
    df = pd.read_excel(config.data_path, sheet_name=config.sheet_name)
    needed = NUM_FEATURES + CAT_FEATURES + [TARGET, GROUP_COLUMN]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {config.data_path}: {missing}")
    df = df.dropna(subset=needed).copy()
    for column in CAT_FEATURES:
        df[column] = df[column].astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    logger.info("Loaded %d rows from %d participants.", len(df), df[GROUP_COLUMN].nunique())
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def build_model_factories(config: CVConfig) -> dict[str, Callable[[], object]]:
    """One factory per model, each returning a fresh unfitted estimator.

    Factories rather than instances so no fold can inherit another fold's fitted
    state. Optional dependencies are probed here and skipped with a log line
    rather than failing the whole run.
    """
    seed = config.random_state

    def sk(estimator) -> Pipeline:
        return Pipeline([("preprocessor", make_preprocessor()), ("regressor", estimator)])

    factories: dict[str, Callable[[], object]] = {
        # The bar every other model has to clear: predict the training mean.
        "Mean baseline": lambda: sk(DummyRegressor(strategy="mean")),
        "Ridge": lambda: sk(Ridge(alpha=1.0, random_state=None)),
        "Random Forest": lambda: sk(
            RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        ),
    }

    try:
        from xgboost import XGBRegressor

        factories["XGBoost"] = lambda: sk(XGBRegressor(random_state=seed, verbosity=0, n_jobs=-1))
    except ImportError:
        logger.warning("XGBoost not available — skipping.")

    try:
        from lightgbm import LGBMRegressor

        factories["LightGBM"] = lambda: sk(LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1))
    except ImportError:
        logger.warning("LightGBM not available — skipping.")

    try:
        from catboost import CatBoostRegressor

        factories["CatBoost"] = lambda: sk(
            CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False)
        )
    except ImportError:
        logger.warning("CatBoost not available — skipping.")

    if config.include_tabpfn:
        try:
            from tabpfn import TabPFNRegressor

            factories["TabPFN"] = lambda: sk(TabPFNRegressor())
        except ImportError:
            logger.warning("TabPFN not available — skipping.")

    return factories


# Search spaces for --tune. Deliberately small and centred on the defaults: the
# point is to show the comparison is not an artefact of leaving every model at its
# out-of-the-box settings, not to squeeze out the last decimal.
PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "Ridge": {
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "Random Forest": {
        "regressor__n_estimators": [100, 300, 600],
        "regressor__max_depth": [None, 3, 5, 10],
        "regressor__min_samples_leaf": [1, 5, 20, 50],
        "regressor__max_features": ["sqrt", 0.5, 1.0],
    },
    "XGBoost": {
        "regressor__n_estimators": [100, 300, 600],
        "regressor__max_depth": [2, 3, 5, 8],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.3],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__reg_lambda": [0.1, 1.0, 10.0],
    },
    "LightGBM": {
        "regressor__n_estimators": [100, 300, 600],
        "regressor__num_leaves": [7, 15, 31, 63],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__min_child_samples": [5, 20, 50],
    },
    "CatBoost": {
        "regressor__iterations": [200, 500, 1000],
        "regressor__depth": [2, 4, 6, 8],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__l2_leaf_reg": [1.0, 3.0, 10.0],
    },
}


def maybe_tune(name: str, estimator, config: CVConfig):
    """Wrap ``estimator`` in a nested, participant-grouped randomised search.

    Comparing an untuned Random Forest against an untuned XGBoost confounds "this
    model is worse" with "this model's defaults are worse for this data". With
    ``--tune`` each outer fold selects its own hyperparameters using an **inner**
    GroupKFold over the training participants only, so tuning never sees the outer
    test fold. That is what makes the reported score an honest generalisation
    estimate rather than the maximum over a grid.

    Models without a search space (the mean baseline, TabPFN — which has no
    hyperparameters to speak of) are returned unchanged.
    """
    if not config.tune or name not in PARAM_DISTRIBUTIONS:
        return estimator

    return RandomizedSearchCV(
        estimator,
        PARAM_DISTRIBUTIONS[name],
        n_iter=config.tune_iterations,
        cv=GroupKFold(n_splits=config.tune_inner_folds),
        scoring="r2",
        random_state=config.random_state,
        n_jobs=1,  # the estimators already use n_jobs=-1 internally
        refit=True,
        error_score="raise",
    )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mse)),
    }


def iter_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, config: CVConfig):
    """Yield ``(repeat, fold, train_idx, test_idx)`` for repeated GroupKFold.

    Each repeat reshuffles the participant-to-fold assignment with its own seed,
    so the repeats are genuinely different partitions rather than the same one
    re-scored. No participant is ever split across a fold boundary.
    """
    for repeat in range(config.repeats):
        splitter = GroupKFold(
            n_splits=config.folds, shuffle=True, random_state=config.random_state + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
            yield repeat, fold, train_idx, test_idx


def run_cross_validation(df: pd.DataFrame, config: CVConfig) -> dict[str, dict]:
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COLUMN]

    factories = build_model_factories(config)
    folds = list(iter_folds(X, y, groups, config))
    logger.info(
        "Evaluating %d models over %d folds (%d-fold x %d repeats)%s.",
        len(factories),
        len(folds),
        config.folds,
        config.repeats,
        (
            f", tuning each with {config.tune_iterations} candidates over "
            f"{config.tune_inner_folds} inner grouped folds"
            if config.tune
            else ""
        ),
    )

    per_fold: dict[str, list[dict]] = {name: [] for name in factories}
    for repeat, fold, train_idx, test_idx in folds:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Assertion, not a comment: a leaked participant would silently inflate
        # every number downstream, and this costs microseconds per fold.
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        assert not overlap, f"Participant leak across fold boundary: {sorted(overlap)[:5]}"

        groups_train = groups.iloc[train_idx]
        for name, factory in factories.items():
            model = maybe_tune(name, factory(), config)
            if isinstance(model, RandomizedSearchCV):
                # ``groups`` must reach the inner splitter, or the inner folds
                # would be row-wise and the selected hyperparameters would be
                # tuned against leaked participants.
                model.fit(X_train, y_train, groups=groups_train)
            else:
                model.fit(X_train, y_train)
            metrics = score(y_test.to_numpy(), model.predict(X_test))
            if isinstance(model, RandomizedSearchCV):
                metrics["best_params"] = {
                    key.replace("regressor__", ""): value
                    for key, value in model.best_params_.items()
                }
            metrics.update(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_test_rows": int(len(test_idx)),
                    "n_test_participants": int(groups.iloc[test_idx].nunique()),
                }
            )
            per_fold[name].append(metrics)
        logger.info("repeat %d fold %d done.", repeat, fold)

    return aggregate(per_fold, config)


def aggregate(per_fold: dict[str, list[dict]], config: CVConfig) -> dict[str, dict]:
    """Mean, SD and a 95% CI of the mean across folds, per model and metric.

    The CI uses the SD across folds. Folds within a repeat share training data,
    so they are not fully independent and this interval is mildly optimistic --
    it is reported as a spread indicator, not as a significance test. The paired
    per-fold differences against the baseline (below) are the more defensible
    comparison, because they cancel the fold-difficulty term.
    """
    baseline_scores = {
        (row["repeat"], row["fold"]): row for row in per_fold.get("Mean baseline", [])
    }

    summary: dict[str, dict] = {}
    for name, rows in per_fold.items():
        entry: dict[str, object] = {"n_folds": len(rows), "per_fold": rows}
        for metric in config.metrics:
            values = np.array([row[metric] for row in rows], dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            half_width = 1.96 * sd / sqrt(values.size) if values.size > 1 else float("nan")
            entry[metric] = {
                "mean": mean,
                "sd": sd,
                "ci_lo": mean - half_width,
                "ci_hi": mean + half_width,
                "min": float(values.min()),
                "max": float(values.max()),
            }

        if baseline_scores and name != "Mean baseline":
            # Paired per-fold difference vs the baseline on the SAME fold. This
            # removes the fold-to-fold difficulty variation, which is the single
            # largest source of spread here.
            deltas = np.array(
                [row["r2"] - baseline_scores[(row["repeat"], row["fold"])]["r2"] for row in rows],
                dtype=float,
            )
            mean_delta = float(deltas.mean())
            sd_delta = float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan")
            half = 1.96 * sd_delta / sqrt(deltas.size) if deltas.size > 1 else float("nan")
            entry["r2_delta_vs_baseline"] = {
                "mean": mean_delta,
                "sd": sd_delta,
                "ci_lo": mean_delta - half,
                "ci_hi": mean_delta + half,
                "folds_better_than_baseline": int((deltas > 0).sum()),
                "beats_baseline": bool(mean_delta - half > 0),
            }
        summary[name] = entry

    return summary


def output_stem(config: CVConfig) -> str:
    """Filename stem encoding the protocol.

    The canonical 5x5 untuned run keeps the bare name (`cv_metrics.json`) that the
    README quotes; any other configuration gets its shape appended. Without this a
    quick exploratory run (`--folds 3 --repeats 1 --tune`) silently overwrites the
    committed headline result with a much weaker one, and nothing in the file
    name would reveal it.
    """
    stem = "cv"
    if (config.folds, config.repeats) != (5, 5):
        stem = f"{stem}_{config.folds}x{config.repeats}"
    if config.tune:
        stem = f"{stem}_tuned{config.tune_iterations}x{config.tune_inner_folds}"
    return stem


def plot_cv_r2(summary: dict[str, dict], config: CVConfig) -> None:
    apply_paper_style()
    names = list(summary)
    data = [[row["r2"] for row in summary[name]["per_fold"]] for name in names]

    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 5.0))
    positions = np.arange(len(names))
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
    for position, values in zip(positions, data, strict=True):
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            position + jitter, values, s=16, alpha=0.65, color=OKABE_ITO[2], zorder=3, linewidths=0
        )

    # R^2 = 0 is exactly "predict the test-set mean". Anything below it is worse
    # than a constant, which is the point the figure has to make legible.
    ax.axhline(0.0, color=OKABE_ITO[6], linestyle="--", lw=1.2, label="$R^2=0$ (predict the mean)")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Held-out $R^2$")
    ax.set_title(
        f"Participant-grouped {config.folds}-fold CV x {config.repeats} repeats "
        f"({config.folds * config.repeats} folds)"
    )
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    save_fig(fig, config.results_path / f"{output_stem(config)}_r2_by_model")
    plt.close(fig)


def log_summary(summary: dict[str, dict]) -> None:
    logger.info("%-16s %-22s %s", "MODEL", "R^2 mean +- SD", "delta vs mean baseline [95% CI]")
    for name, entry in summary.items():
        r2 = entry["r2"]
        line = f"{name:<16} {r2['mean']:+.4f} +- {r2['sd']:.4f}     "
        delta = entry.get("r2_delta_vs_baseline")
        if delta:
            line += (
                f"{delta['mean']:+.4f} [{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}]"
                f"  {'BEATS' if delta['beats_baseline'] else 'no'}"
            )
        logger.info("%s", line)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5, help="Folds per repeat.")
    parser.add_argument("--repeats", type=int, default=5, help="Independent re-partitions.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-tabpfn",
        action="store_true",
        help="Also cross-validate TabPFN (substantially slower).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Select hyperparameters per outer fold with a nested, participant-grouped "
            "randomised search. Multiplies runtime by roughly --tune-iterations x "
            "--tune-inner-folds."
        ),
    )
    parser.add_argument("--tune-iterations", type=int, default=20)
    parser.add_argument("--tune-inner-folds", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = CVConfig(
        folds=args.folds,
        repeats=args.repeats,
        random_state=args.random_state,
        include_tabpfn=args.include_tabpfn,
        tune=args.tune,
        tune_iterations=args.tune_iterations,
        tune_inner_folds=args.tune_inner_folds,
    )

    df = load_data(config)
    summary = run_cross_validation(df, config)
    log_summary(summary)
    plot_cv_r2(summary, config)

    payload = {
        "protocol": {
            "scheme": "repeated GroupKFold (grouped by ProlificID)",
            "folds": config.folds,
            "repeats": config.repeats,
            "total_fits_per_model": config.folds * config.repeats,
            "random_state": config.random_state,
            "n_rows": int(len(df)),
            "n_participants": int(df[GROUP_COLUMN].nunique()),
            "tuned": config.tune,
            "tune_iterations": config.tune_iterations if config.tune else None,
            "tune_inner_folds": config.tune_inner_folds if config.tune else None,
        },
        "models": summary,
    }
    output_path = config.results_path / f"{output_stem(config)}_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
