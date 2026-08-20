#!/usr/bin/env python3
"""Symbolic regression of trust on mIoU, per INTRODUCTION x SCENARIO cell.

Run a reproducible (publication) search with::

    python main_pysr_trust_calibration.py --seed 0 --deterministic

See ``pysr_config`` for why ``--deterministic`` is required for reproducibility.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from pysr_config import add_search_args, model_factory, write_model_info
from pysr_plots import save_relationship_plot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

results_path = Path("results") / "PySR"
results_path.mkdir(parents=True, exist_ok=True)

file_path = Path("data") / "all_combined_prepared.xlsx"
file_path_removed_dei = Path("data") / "all_combined_prepared_removed_REI.xlsx"
sheet_name = "Sheet1"


# This script's searches are shorter than the other three PySR pipelines'.
NITERATIONS = 300


def fit_and_plot_subset(df, intro, scenario, name_without_extension, make_model):
    logger.info("Working with Introduction: %s, Scenario: %s", intro, scenario)
    filtered_df = df[(df["INTRODUCTION"] == intro) & (df["SCENARIO"] == scenario)]

    if len(filtered_df) < 3:
        logger.warning(
            "Skipping Introduction=%s, Scenario=%s due to insufficient rows: %d",
            intro,
            scenario,
            len(filtered_df),
        )
        return

    x_values = filtered_df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = filtered_df["trust"].to_numpy()

    model = make_model()
    model.fit(x_values, y_values)

    info_path = results_path / f"model_info_{intro}_{scenario}_{name_without_extension}.txt"
    write_model_info(model, info_path)

    save_relationship_plot(
        x_values,
        y_values,
        model.predict(x_values),
        results_path / f"relationship_pysr_{intro}_{scenario}_{name_without_extension}.png",
        title=f"Visualization of the Equation for {intro} and {scenario}",
        scatter_color="grey",
        scatter_alpha=0.5,
        line_color="green",
    )


def run_all_data(df, name_without_extension, make_model):
    if len(df) < 3:
        logger.warning(
            "Skipping all-data fit for %s: insufficient rows (%d)", name_without_extension, len(df)
        )
        return

    x_values = df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = df["trust"].to_numpy()

    model = make_model()
    model.fit(x_values, y_values)

    # The all-data equation used to be fitted, plotted, and then discarded --
    # only the figure was written, so the equation behind it was unrecoverable.
    info_path = results_path / f"model_info_all_data_{name_without_extension}.txt"
    write_model_info(model, info_path)

    save_relationship_plot(
        x_values,
        y_values,
        model.predict(x_values),
        results_path / f"relationship_pysr_all_data_{name_without_extension}.png",
        scatter_color="grey",
        scatter_alpha=0.5,
        line_color="green",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_search_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    make_model = model_factory(args, niterations=NITERATIONS)

    file_paths = [file_path, file_path_removed_dei]
    required_cols = ["mIoU", "trust", "INTRODUCTION", "SCENARIO"]

    for path in file_paths:
        name_without_extension = path.stem
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(subset=required_cols)

        observed_pairs = df[["INTRODUCTION", "SCENARIO"]].drop_duplicates()
        for intro, scenario in observed_pairs.itertuples(index=False, name=None):
            fit_and_plot_subset(df, intro, scenario, name_without_extension, make_model)

        run_all_data(df, name_without_extension, make_model)


if __name__ == "__main__":
    main()
