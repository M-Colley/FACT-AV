#!/usr/bin/env python3
"""Advanced explainability outputs for the FACT-AV trust calibration paper.

Three additions on top of the existing SHAP bar/beeswarm plots in
``ML-approaches.py``:

1. **SHAP interaction values** — pairwise interaction matrix from XGBoost, with
   focused dependence plots for ``mIoU x INTRODUCTION`` and ``mIoU x SCENARIO``
   (these are the core moderators in the study's hypotheses).
2. **DiCE counterfactual explanations** — for a handful of misclassified or
   low-trust observations, what minimal change would push the prediction to
   high trust? Output as a tidy CSV plus a faceted figure.
3. **Anchors-style rule extraction** — local high-precision rules (decision-
   tree surrogate over feature neighbourhoods) summarising when the model
   predicts low vs high trust. Falls back gracefully if ``alibi`` is missing.

Run with ``python explainability_extras.py`` from the repository root.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

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
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ExplainConfig:
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
    results_path: Path = Path("results") / "publication" / "explainability"
    sheet_name: str = "Sheet1"
    random_state: int = 42
    test_size: float = 0.2
    n_counterfactuals: int = 8

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


def load_data(config: ExplainConfig) -> pd.DataFrame:
    df = pd.read_excel(config.data_path, sheet_name=config.sheet_name)
    needed = list(NUM_FEATURES) + list(CAT_FEATURES) + ["trust", "ProlificID"]
    df = df.dropna(subset=needed).copy()
    df["INTRODUCTION"] = df["INTRODUCTION"].astype(str).str.lower().map(INTRODUCTION_NORMALIZER)
    df["SCENARIO"] = df["SCENARIO"].astype(str)
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    df["trust"] = pd.to_numeric(df["trust"], errors="raise")
    return df


def split_data(df: pd.DataFrame, config: ExplainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Participant-grouped split (no leakage across train/test)."""
    X = df[list(NUM_FEATURES) + list(CAT_FEATURES)].copy()
    y = df["trust"].copy()
    groups = df["ProlificID"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    (train_idx, test_idx), = splitter.split(X, y, groups=groups)
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ---------------------------------------------------------------------------
# 1. SHAP interaction values
# ---------------------------------------------------------------------------


def _train_xgb_for_shap(X_train: pd.DataFrame, y_train: pd.Series, config: ExplainConfig):
    """Train an XGBoost regressor on integer-coded categoricals.

    SHAP's interaction-value path constructs an internal DMatrix that does not
    accept ``category`` dtypes; ordinal-encoding the categoricals as int sidesteps
    this while preserving model behaviour for tree splits.
    """
    import xgboost as xgb

    X = X_train.copy()
    category_codes: Dict[str, List[str]] = {}
    for c in CAT_FEATURES:
        codes = pd.Categorical(X[c])
        category_codes[c] = list(codes.categories)
        X[c] = codes.codes.astype(np.int32)
    model = xgb.XGBRegressor(
        random_state=config.random_state,
        n_estimators=400,
        max_depth=4,
    )
    model.fit(X, y_train)
    model._fact_category_codes = category_codes  # stash for downstream use
    return model, X


def shap_interactions(df: pd.DataFrame, config: ExplainConfig) -> None:
    """Compute pairwise SHAP interaction values and plot mIoU-focused panels."""
    apply_paper_style()
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available - skipping interaction plots.")
        return

    X_train, X_test, y_train, _ = split_data(df, config)
    model, X_train_enc = _train_xgb_for_shap(X_train, y_train, config)

    X_test_enc = X_test.copy()
    category_codes = getattr(model, "_fact_category_codes", {})
    for c in CAT_FEATURES:
        cats = category_codes.get(c)
        codes = pd.Categorical(X_test_enc[c], categories=cats) if cats is not None else pd.Categorical(X_test_enc[c])
        X_test_enc[c] = codes.codes.astype(np.int32)
    # Keep an unencoded copy so we can colour points by original category labels.
    X_test_raw = X_test.copy().loc[X_test_enc.index]
    if len(X_test_enc) > 400:
        keep_idx = X_test_enc.sample(400, random_state=config.random_state).index
        X_test_enc = X_test_enc.loc[keep_idx]
        X_test_raw = X_test_raw.loc[keep_idx]
    # ``shap`` and the rest of the function expect a frame called ``X_test_cat``.
    X_test_cat = X_test_enc

    explainer = shap.TreeExplainer(model)
    try:
        inter = explainer.shap_interaction_values(X_test_cat)
    except Exception as exc:
        logger.warning("shap_interaction_values failed (%s). Falling back to dependence plots only.", exc)
        inter = None

    feat_names = list(X_test_cat.columns)
    if inter is not None:
        # Summary heatmap of mean |interaction|.
        abs_inter = np.abs(inter).mean(axis=0)
        np.fill_diagonal(abs_inter, 0.0)  # diagonal is main effect, distracts
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(abs_inter, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels(feat_names, rotation=45, ha="right")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names)
        ax.set_title("Mean |SHAP interaction| across features")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Mean |interaction|")
        save_fig(fig, config.results_path / "shap_interaction_heatmap")
        plt.close(fig)

        # Top-10 interactions table.
        rows = []
        for i in range(len(feat_names)):
            for j in range(i + 1, len(feat_names)):
                rows.append({"feature_a": feat_names[i], "feature_b": feat_names[j], "mean_abs_interaction": float(abs_inter[i, j])})
        ranked = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).head(15)
        ranked.to_csv(config.results_path / "shap_top_interactions.csv", index=False)

    # mIoU x INTRODUCTION dependence plot (manual draw so we control style)
    explanation = explainer(X_test_cat)
    shap_vals = explanation.values
    miou_idx = feat_names.index("mIoU")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    miou = X_test_cat["mIoU"].values

    # Panel A: SHAP(mIoU) colored by INTRODUCTION
    intro_vals = X_test_raw["INTRODUCTION"].astype(str).values
    for intro in ("ambiguous", "boasting"):
        mask = intro_vals == intro
        axes[0].scatter(
            miou[mask],
            shap_vals[mask, miou_idx],
            label=intro.capitalize(),
            color=INTRO_COLORS[intro],
            alpha=0.55,
            s=20,
            edgecolor="none",
        )
    axes[0].axhline(0, color="grey", linewidth=0.6)
    axes[0].set_xlabel("mIoU")
    axes[0].set_ylabel("SHAP value for mIoU")
    axes[0].set_title("mIoU effect, split by Introduction")
    axes[0].legend(title="Introduction")

    # Panel B: SHAP(mIoU) colored by SCENARIO
    scenario_vals = X_test_raw["SCENARIO"].astype(str).values
    for scen in sorted(set(scenario_vals)):
        mask = scenario_vals == scen
        axes[1].scatter(
            miou[mask],
            shap_vals[mask, miou_idx],
            label=SCENARIO_LABELS.get(scen, scen),
            color=SCENARIO_COLORS.get(scen, "grey"),
            alpha=0.55,
            s=20,
            edgecolor="none",
        )
    axes[1].axhline(0, color="grey", linewidth=0.6)
    axes[1].set_xlabel("mIoU")
    axes[1].set_ylabel("SHAP value for mIoU")
    axes[1].set_title("mIoU effect, split by Scenario")
    axes[1].legend(title="Scenario")

    fig.suptitle("SHAP-based moderation analysis for mIoU", fontsize=13)
    fig.tight_layout()
    save_fig(fig, config.results_path / "shap_miou_moderation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. DiCE counterfactuals
# ---------------------------------------------------------------------------


def _trust_to_class(trust_value: float) -> int:
    """Binary classification: 0 = low/medium (<=3), 1 = high (>3)."""
    return 1 if float(trust_value) > 3.0 else 0


def dice_counterfactuals(df: pd.DataFrame, config: ExplainConfig) -> None:
    """Find minimal feature changes that would have flipped low-trust to high-trust."""
    try:
        import dice_ml
        from dice_ml import Dice
    except ImportError:
        logger.warning("dice-ml not available - skipping counterfactual analysis.")
        return

    X_train, X_test, y_train, y_test = split_data(df, config)
    y_train_bin = y_train.map(_trust_to_class)
    y_test_bin = y_test.map(_trust_to_class)

    train_for_dice = X_train.copy()
    train_for_dice["trust_high"] = y_train_bin.values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(NUM_FEATURES)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
                list(CAT_FEATURES),
            ),
        ]
    )
    rf = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=config.random_state)),
        ]
    )
    rf.fit(X_train, y_train_bin)

    backend = "sklearn"
    data_dice = dice_ml.Data(
        dataframe=train_for_dice,
        continuous_features=list(NUM_FEATURES),
        outcome_name="trust_high",
    )
    model_dice = dice_ml.Model(model=rf, backend=backend, model_type="classifier")
    exp = Dice(data_dice, model_dice, method="random")

    # Pick low-trust test points the model also predicts low.
    preds = rf.predict(X_test)
    low_idx = np.where((preds == 0) & (y_test_bin.values == 0))[0]
    if len(low_idx) == 0:
        logger.warning("No low-trust test rows to generate counterfactuals on.")
        return
    sample_idx = np.random.default_rng(config.random_state).choice(
        low_idx, size=min(config.n_counterfactuals, len(low_idx)), replace=False
    )

    records = []
    for k, idx in enumerate(sample_idx):
        query = X_test.iloc[[idx]]
        try:
            cf = exp.generate_counterfactuals(query, total_CFs=2, desired_class=1)
        except Exception as exc:
            logger.warning("DiCE failed on sample %d: %s", k, exc)
            continue
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            continue
        for cf_row_i, (_, cf_row) in enumerate(cf_df.iterrows()):
            for col in list(NUM_FEATURES) + list(CAT_FEATURES):
                orig = query.iloc[0][col]
                proposed = cf_row[col]
                if str(orig) != str(proposed):
                    records.append(
                        {
                            "sample_id": int(idx),
                            "cf_id": cf_row_i,
                            "feature": col,
                            "original": orig,
                            "counterfactual": proposed,
                        }
                    )

    if not records:
        logger.warning("DiCE produced no usable counterfactuals.")
        return

    cf_table = pd.DataFrame(records)
    cf_table.to_csv(config.results_path / "dice_counterfactuals.csv", index=False)

    # Bar chart of how often each feature is suggested for change.
    apply_paper_style()
    counts = cf_table["feature"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(counts) + 1.5))
    ax.barh(counts.index, counts.values, color=OKABE_ITO[2], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("# counterfactual examples where this feature was changed")
    ax.set_title("Features most often nudged to push trust from low -> high (DiCE)")
    save_fig(fig, config.results_path / "dice_feature_change_frequency")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Anchors-style rule extraction (with surrogate-tree fallback)
# ---------------------------------------------------------------------------


def _try_alibi_anchors(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, n_rules: int, config: ExplainConfig) -> Optional[List[str]]:
    """Try the canonical Alibi Anchors explainer; return None if unavailable."""
    try:
        from alibi.explainers import AnchorTabular
    except Exception:
        return None

    X_train_enc = pd.get_dummies(X_train, columns=list(CAT_FEATURES), drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=list(CAT_FEATURES), drop_first=False).reindex(
        columns=X_train_enc.columns, fill_value=0
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=config.random_state)
    rf.fit(X_train_enc, y_train.map(_trust_to_class))
    explainer = AnchorTabular(rf.predict, feature_names=list(X_train_enc.columns))
    explainer.fit(X_train_enc.values, disc_perc=(25, 50, 75))

    rng = np.random.default_rng(config.random_state)
    sample = rng.choice(len(X_test_enc), size=min(n_rules, len(X_test_enc)), replace=False)
    rules = []
    for idx in sample:
        try:
            expl = explainer.explain(X_test_enc.values[idx], threshold=0.9)
            rule = " AND ".join(expl.anchor) if expl.anchor else "(no rule)"
            pred_cls = int(rf.predict(X_test_enc.values[idx:idx + 1])[0])
            rules.append(
                f"IF {rule} THEN trust={'high' if pred_cls == 1 else 'low/med'} "
                f"(precision={expl.precision:.2f}, coverage={expl.coverage:.2f})"
            )
        except Exception as exc:
            logger.warning("Anchor explanation failed on sample %d: %s", idx, exc)
    return rules


def _surrogate_tree_rules(X_train: pd.DataFrame, y_train: pd.Series, config: ExplainConfig) -> List[str]:
    """Fallback: train a shallow surrogate tree over a hot-encoded feature space."""
    X_enc = pd.get_dummies(X_train, columns=list(CAT_FEATURES), drop_first=False)
    rf = RandomForestClassifier(n_estimators=300, random_state=config.random_state)
    rf.fit(X_enc.values, y_train.map(_trust_to_class).values)

    surrogate = DecisionTreeClassifier(max_depth=4, random_state=config.random_state)
    surrogate.fit(X_enc.values, rf.predict(X_enc.values))
    rules_text = export_text(surrogate, feature_names=list(X_enc.columns), max_depth=4)
    return rules_text.splitlines()


def anchors_rules(df: pd.DataFrame, config: ExplainConfig) -> None:
    """Generate Anchor rules (or surrogate-tree fallback)."""
    X_train, X_test, y_train, _ = split_data(df, config)

    anchors_lines = _try_alibi_anchors(X_train, y_train, X_test, n_rules=12, config=config)
    out_path = config.results_path / "anchors_rules.txt"
    header = "# Local high-precision rules for trust classification (low/med vs high)\n"

    if anchors_lines is not None:
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(header + "# Generated via alibi.AnchorTabular\n\n")
            handle.write("\n".join(anchors_lines))
        logger.info("Wrote %d Alibi anchor rules.", len(anchors_lines))
    else:
        logger.info("alibi not available; falling back to a surrogate decision tree.")
        rules_lines = _surrogate_tree_rules(X_train, y_train, config)
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(header + "# Surrogate-tree fallback (alibi not installed)\n\n")
            handle.write("\n".join(rules_lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    config = ExplainConfig()
    df = load_data(config)
    logger.info("Loaded %d rows for explainability extras.", len(df))

    logger.info("Computing SHAP interactions ...")
    shap_interactions(df, config)

    logger.info("Computing DiCE counterfactuals ...")
    dice_counterfactuals(df, config)

    logger.info("Extracting Anchor rules ...")
    anchors_rules(df, config)

    logger.info("Explainability extras saved to %s", config.results_path)


if __name__ == "__main__":
    main()
