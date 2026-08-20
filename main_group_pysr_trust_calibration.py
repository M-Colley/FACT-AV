#!/usr/bin/env python3
"""Symbolic regression on the equal-trust / variable-trust split of the sample.

Participants whose ratings barely vary are separated from the rest (see
``trust_groups``) and each subset is searched separately, then every participant
in the variable-trust subset gets a personalised fit.

Run a reproducible (publication) search with::

    python main_group_pysr_trust_calibration.py --seed 0 --deterministic

**Read the caveat in ``trust_groups``**: the split conditions on the outcome's
own variance, so any difference in fit quality between the two subsets is partly
mechanical and must be reported as exploratory.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from pysr_config import add_search_args, model_factory, write_model_info
from pysr_plots import save_relationship_plot
from trust_groups import find_equal_groups, split_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

results_path_split_groups = Path("results") / "PySR" / "split_groups"
results_path_split_groups.mkdir(parents=True, exist_ok=True)

results_path_split_groups_personalized = Path("results") / "PySR" / "split_groups_personalized"
results_path_split_groups_personalized.mkdir(parents=True, exist_ok=True)

_DATA_FILE = Path("data") / "all_combined_prepared.xlsx"
_DATA_FILE_REMOVED_DEI = Path("data") / "all_combined_prepared_removed_REI.xlsx"
_SHEET_NAME = "Sheet1"

# Re-exported so existing imports of this module keep resolving; the canonical
# implementation now lives in ``trust_groups`` and is shared with
# ``main_group_pysr_trust_calibration_more_predictors.py``.
_find_equal_groups = find_equal_groups


def _fit_and_save(model, x_values, y_values, info_path, plot_path, hue_series=None, legend=True):
    """Fit model, write equation text, save scatter+line plot, return the model.

    The fitted model is returned so a caller can render additional figures from
    the same search instead of re-fitting.
    """
    model.fit(x_values, y_values)

    logger.info("SYMPY: %s", model.sympy())
    logger.info("LATEX: %s", model.latex())

    write_model_info(model, info_path)
    save_relationship_plot(
        x_values,
        y_values,
        model.predict(x_values),
        plot_path,
        hue_series=hue_series,
        legend=legend,
    )
    return model


def _fit_personalized(df, participant_id, model, name):
    """Fit a per-participant model and write its outputs.

    ``name`` (the dataset stem) is part of both output filenames. Without it the
    second dataset's equations overwrite the first's, since participant IDs are
    shared between the two files.
    """
    logger.info("Working with ProlificID: %s", participant_id)

    filtered_df = df[df["ProlificID"] == participant_id]
    x_values = filtered_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
    y_values = filtered_df[["trust"]].dropna()

    if len(x_values) < 3:
        logger.warning(
            "Skipping ProlificID=%s: insufficient rows (%d)", participant_id, len(x_values)
        )
        return

    _fit_and_save(
        model,
        x_values,
        y_values,
        info_path=results_path_split_groups_personalized
        / f"model_info_{participant_id}_{name}.txt",
        plot_path=results_path_split_groups_personalized
        / f"relationship_pysr_{participant_id}_{name}.png",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_search_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # NOTE: a *fresh* PySRRegressor is used for every fit. Reusing one instance
    # lets Julia-side search state (the equation hall-of-fame) bleed from one
    # dataset into the next. ``model_factory`` returns a builder, not a model.
    create = model_factory(args)

    file_paths = [_DATA_FILE, _DATA_FILE_REMOVED_DEI]

    for path in file_paths:
        name = path.stem

        df = pd.read_excel(path, sheet_name=_SHEET_NAME)
        df.dropna(inplace=True)
        df["intro_scenario_combo"] = (
            df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)
        )

        logger.info("%s: df shape %s", name, df.shape)

        # Identify "equal-trust" groups (>=14 identical ratings, or 2 groups of >=7).
        all_equal_df, other_rows_df = split_groups(df)
        logger.info(
            "%s: all_equal_df %s, other_rows_df %s", name, all_equal_df.shape, other_rows_df.shape
        )

        # --- Fit 1: other_rows_df (mIoU only) ------------------------------------
        # This used to be run twice: "Fit 1" and "Fit 3" were byte-identical
        # searches of the same x, y and hue, differing only in output filename.
        # That doubled runtime and produced two files that could disagree purely
        # because the search is stochastic. One search now backs both figures.
        x = other_rows_df["mIoU"].to_numpy().reshape(-1, 1)
        y = other_rows_df[["trust"]].dropna()
        other_rows_model = _fit_and_save(
            create(),
            x,
            y,
            info_path=results_path_split_groups / f"model_info_other_rows_df_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_other_rows_df_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
        )
        save_relationship_plot(
            x,
            y,
            other_rows_model.predict(x),
            results_path_split_groups / f"relationship_pysr_other_rows_df_stacked_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
            legend=False,
        )

        # --- Fit 2: all_equal_df (mIoU only) ------------------------------------
        if all_equal_df.empty:
            logger.warning("%s: no equal-trust groups found, skipping that fit.", name)
        else:
            x = all_equal_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
            y = all_equal_df[["trust"]].dropna()
            _fit_and_save(
                create(),
                x,
                y,
                info_path=results_path_split_groups / f"model_info_all_equal_df_{name}.txt",
                plot_path=results_path_split_groups / f"relationship_pysr_all_equal_df_{name}.png",
                hue_series=all_equal_df["intro_scenario_combo"],
            )

        # --- Personalized fits for each participant in other_rows_df -----------
        # Inside the per-file loop. Previously this sat after the loop and read
        # the leaked ``other_rows_df``, so it silently ran on the last dataset
        # only -- and on just its 43 participants, not all 130.
        for participant_id in other_rows_df["ProlificID"].unique():
            _fit_personalized(other_rows_df, participant_id, create(), name)


if __name__ == "__main__":
    main()
