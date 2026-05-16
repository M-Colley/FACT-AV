#!/usr/bin/env python3
"""Linear mixed-effects baseline for trust calibration analysis.

Why this script exists
----------------------
The study uses a within-subjects design: each of ~130 participants rated trust
across multiple mIoU values, two INTRODUCTION conditions, and four SCENARIOs.
The existing ML baselines treat every row as i.i.d., which violates the
repeated-measures structure and inflates effective sample size. A linear
mixed-effects (LME) model:

* gives a *principled* statistical baseline that HRI / human-factors reviewers
  recognise immediately,
* partitions variance into between-participant (random intercept) and
  within-participant residual components, so we can report intraclass
  correlation (ICC) and an honest R^2,
* tests the moderation hypotheses (mIoU x INTRODUCTION, mIoU x SCENARIO)
  directly with p-values and confidence intervals.

Three nested models are fit and compared (likelihood-ratio + AIC + BIC):
    M0  random-intercept only (null variance decomposition)
    M1  + fixed effects for mIoU, INTRODUCTION, SCENARIO
    M2  + mIoU x INTRODUCTION and mIoU x SCENARIO interactions

Outputs
-------
``results/publication/mixed_effects/``
    summary_M0.txt / M1.txt / M2.txt        (statsmodels textual summaries)
    fixed_effects_M2.csv                    (coefficients + CI + p-values)
    model_comparison.csv                    (AIC, BIC, LL, df, LR test)
    icc.json                                (ICC + variance decomposition)
    fixed_effects_forest.{pdf,png}          (coefficient forest plot)
    interaction_marginal_effects.{pdf,png}  (predicted trust curves)

Run with: ``python mixed_effects_baseline.py``
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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class MixedEffectsConfig:
    # Use the file that has BOTH demographics AND real ProlificIDs.
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
    results_path: Path = Path("results") / "publication" / "mixed_effects"
    sheet_name: str = "Sheet1"

    def __post_init__(self) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)


INTRODUCTION_NORMALIZER = {"ambigious": "ambiguous", "ambiguous": "ambiguous", "boasting": "boasting"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(config: MixedEffectsConfig) -> pd.DataFrame:
    df = pd.read_excel(config.data_path, sheet_name=config.sheet_name)
    needed = ["mIoU", "INTRODUCTION", "SCENARIO", "trust", "ProlificID"]
    df = df.dropna(subset=needed).copy()
    df["INTRODUCTION"] = df["INTRODUCTION"].astype(str).str.lower().map(INTRODUCTION_NORMALIZER)
    df["SCENARIO"] = df["SCENARIO"].astype(str)
    df["trust"] = pd.to_numeric(df["trust"], errors="raise")
    df["mIoU"] = pd.to_numeric(df["mIoU"], errors="raise")

    # Center mIoU so the intercept reflects "average mIoU" trust, and interactions
    # are interpretable as deviations from the grand mean.
    df["mIoU_c"] = df["mIoU"] - df["mIoU"].mean()

    # Use Treatment coding: ambiguous and 3Spurig (Highway) as reference levels.
    df["INTRODUCTION"] = pd.Categorical(df["INTRODUCTION"], categories=["ambiguous", "boasting"])
    df["SCENARIO"] = pd.Categorical(
        df["SCENARIO"], categories=["3Spurig", "NeueMitte", "Spielstrasse", "Ueberland"]
    )

    logger.info(
        "Loaded %d observations from %d participants. mIoU center=%.3f",
        len(df),
        df["ProlificID"].nunique(),
        df["mIoU"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_models(df: pd.DataFrame, config: MixedEffectsConfig) -> Dict[str, "MixedLMResults"]:
    """Fit M0 (random intercept), M1 (main effects), M2 (interactions)."""
    import statsmodels.formula.api as smf

    fits: Dict[str, "MixedLMResults"] = {}

    # M0: null model -- random intercept only, fixed intercept only
    logger.info("Fitting M0 (random intercept only) ...")
    fits["M0"] = smf.mixedlm("trust ~ 1", df, groups=df["ProlificID"]).fit(reml=False)

    # M1: + main effects
    logger.info("Fitting M1 (main effects) ...")
    fits["M1"] = smf.mixedlm(
        "trust ~ mIoU_c + C(INTRODUCTION) + C(SCENARIO)", df, groups=df["ProlificID"]
    ).fit(reml=False)

    # M2: + moderating interactions
    logger.info("Fitting M2 (with mIoU x INTRO + mIoU x SCENARIO interactions) ...")
    fits["M2"] = smf.mixedlm(
        "trust ~ mIoU_c * (C(INTRODUCTION) + C(SCENARIO))", df, groups=df["ProlificID"]
    ).fit(reml=False)

    for name, fit in fits.items():
        (config.results_path / f"summary_{name}.txt").write_text(fit.summary().as_text(), encoding="utf-8")

    return fits


def compare_models(fits: Dict[str, "MixedLMResults"], config: MixedEffectsConfig) -> pd.DataFrame:
    """Likelihood-ratio + AIC/BIC comparison for nested LME models."""
    from scipy.stats import chi2

    rows = []
    ordered = ["M0", "M1", "M2"]
    for i, name in enumerate(ordered):
        fit = fits[name]
        row = {
            "model": name,
            "logL": float(fit.llf),
            "AIC": float(fit.aic),
            "BIC": float(fit.bic),
            "df_resid": int(getattr(fit, "df_resid", np.nan)) if hasattr(fit, "df_resid") else np.nan,
            "n_params": int(fit.df_modelwc) if hasattr(fit, "df_modelwc") else len(fit.fe_params),
        }
        if i > 0:
            prev = fits[ordered[i - 1]]
            d_logL = 2 * (fit.llf - prev.llf)
            d_df = len(fit.fe_params) - len(prev.fe_params)
            row["LR_vs_prev"] = float(d_logL)
            row["d_df"] = int(d_df)
            row["p_LR"] = float(chi2.sf(d_logL, df=max(d_df, 1)))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(config.results_path / "model_comparison.csv", index=False, float_format="%.4f")
    logger.info("Model comparison:\n%s", table.to_string(index=False))
    return table


def write_fixed_effects(fit, config: MixedEffectsConfig) -> pd.DataFrame:
    """Tidy fixed-effects table with 95% CIs from the final model."""
    params = fit.params
    conf = fit.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]
    pvalues = fit.pvalues
    se = fit.bse

    table = pd.DataFrame(
        {
            "term": params.index,
            "estimate": params.values,
            "std_err": se.reindex(params.index).values,
            "ci_lower": conf["ci_lower"].reindex(params.index).values,
            "ci_upper": conf["ci_upper"].reindex(params.index).values,
            "p_value": pvalues.reindex(params.index).values,
        }
    )
    # Drop the participant-level variance row from the fixed-effects table.
    table = table[~table["term"].str.contains("Group Var", case=False)].reset_index(drop=True)
    table.to_csv(config.results_path / "fixed_effects_M2.csv", index=False, float_format="%.4f")
    return table


def compute_icc(fit_m0, config: MixedEffectsConfig) -> Dict[str, float]:
    """Intraclass correlation from the random-intercept null model.

    ICC = sigma_u^2 / (sigma_u^2 + sigma_e^2) -- proportion of variance
    attributable to between-participant differences. Higher ICC means more
    of the variation is participant-driven rather than within-participant
    response variability.
    """
    # MixedLMResults stores group variance in .cov_re and residual variance in .scale.
    sigma_u2 = float(fit_m0.cov_re.values[0, 0])
    sigma_e2 = float(fit_m0.scale)
    icc = sigma_u2 / (sigma_u2 + sigma_e2)
    info = {
        "between_participant_var": sigma_u2,
        "residual_var": sigma_e2,
        "ICC": icc,
        "interpretation": (
            f"{icc * 100:.1f}% of variance in trust ratings is between participants; "
            f"{(1 - icc) * 100:.1f}% is within participants."
        ),
    }
    (config.results_path / "icc.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("ICC = %.3f (between-participant share of total variance)", icc)
    return info


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fixed_effects_forest(table: pd.DataFrame, config: MixedEffectsConfig) -> None:
    """Coefficient forest plot with 95% CIs for the M2 model."""
    apply_paper_style()
    # Drop intercept for plotting -- coefficients are interpretable as effects.
    rows = table[table["term"] != "Intercept"].copy()
    rows = rows.iloc[::-1].reset_index(drop=True)  # show first term on top

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.5))
    y = np.arange(len(rows))
    colors = [
        OKABE_ITO[6] if (lo > 0 or hi < 0) else "grey"
        for lo, hi in zip(rows["ci_lower"], rows["ci_upper"])
    ]
    ax.errorbar(
        rows["estimate"],
        y,
        xerr=[rows["estimate"] - rows["ci_lower"], rows["ci_upper"] - rows["estimate"]],
        fmt="o",
        color="black",
        ecolor="grey",
        markersize=5,
        capsize=3,
    )
    for yi, color, est in zip(y, colors, rows["estimate"]):
        ax.plot([est], [yi], "o", color=color, markersize=7, zorder=3)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["term"])
    ax.set_xlabel("Coefficient estimate (trust units)")
    ax.set_title("Mixed-effects model M2: fixed effects (95% CI)")
    save_fig(fig, config.results_path / "fixed_effects_forest")
    plt.close(fig)


def plot_marginal_effects(fit_m2, df: pd.DataFrame, config: MixedEffectsConfig) -> None:
    """Predicted trust as a function of mIoU, by INTRODUCTION x SCENARIO cell.

    Uses the fitted M2 fixed-effect coefficients (random intercept set to its
    expected value of zero) to draw the population-average response curves.
    """
    apply_paper_style()
    mIoU_grid = np.linspace(df["mIoU"].min(), df["mIoU"].max(), 60)
    miou_c_grid = mIoU_grid - df["mIoU"].mean()

    scenarios = list(df["SCENARIO"].cat.categories)
    intros = list(df["INTRODUCTION"].cat.categories)

    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.6), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        for intro in intros:
            pred_df = pd.DataFrame(
                {
                    "mIoU_c": miou_c_grid,
                    "INTRODUCTION": pd.Categorical([intro] * len(miou_c_grid), categories=intros),
                    "SCENARIO": pd.Categorical([scenario] * len(miou_c_grid), categories=scenarios),
                }
            )
            # ``predict`` on a MixedLMResults returns the fixed-effects portion.
            mu = fit_m2.predict(pred_df)
            ax.plot(
                mIoU_grid,
                mu,
                color=INTRO_COLORS[intro],
                linewidth=2,
                label=intro.capitalize(),
            )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel("mIoU")
        ax.set_ylim(1, 5)
    axes[0].set_ylabel("Predicted trust (1-5)")
    axes[-1].legend(title="Introduction", loc="lower right")
    fig.suptitle("Marginal predicted trust from M2 (mIoU x INTRO + mIoU x SCEN)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, config.results_path / "interaction_marginal_effects")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    config = MixedEffectsConfig()
    df = load_data(config)

    if df["ProlificID"].nunique() < 5:
        logger.error(
            "Dataset has only %d unique participants -- LME is not meaningful. "
            "Make sure the data file with real ProlificIDs is being used.",
            df["ProlificID"].nunique(),
        )
        raise SystemExit(1)

    fits = fit_models(df, config)
    compare_models(fits, config)
    icc = compute_icc(fits["M0"], config)
    logger.info(icc["interpretation"])

    fe_table = write_fixed_effects(fits["M2"], config)
    plot_fixed_effects_forest(fe_table, config)
    plot_marginal_effects(fits["M2"], df, config)

    logger.info("Mixed-effects baseline complete. Outputs in %s", config.results_path)


if __name__ == "__main__":
    main()
