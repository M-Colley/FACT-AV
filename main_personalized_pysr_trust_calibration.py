#!/usr/bin/env python3
"""Per-participant symbolic regression of trust on mIoU.

Run a reproducible (publication) search with::

    python main_personalized_pysr_trust_calibration.py --seed 0 --deterministic
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from pysr_config import add_search_args, model_factory, write_model_info
from pysr_plots import save_relationship_plot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

results_path_personalized = Path("results") / "PySR" / "personalized_plots"
results_path_personalized.mkdir(parents=True, exist_ok=True)

file_path = Path("data") / "all_combined_prepared.xlsx"
file_path_removed_dei = Path("data") / "all_combined_prepared_removed_REI.xlsx"
sheet_name = "Sheet1"


def fit_personalized(df, participant_id, name_without_extension, make_model):
    logger.info("Working with ProlificID: %s", participant_id)

    filtered_df = df[df["ProlificID"] == participant_id]
    if len(filtered_df) < 3:
        logger.warning(
            "Skipping ProlificID=%s: insufficient rows (%d)", participant_id, len(filtered_df)
        )
        return

    x_values = filtered_df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = filtered_df["trust"].to_numpy()

    model = make_model()
    model.fit(x_values, y_values)

    # The dataset stem is part of the filename. Without it, the second dataset's
    # equations overwrite the first's, since both files share participant IDs --
    # the .png path already included it, so the two artifacts disagreed.
    info_path = (
        results_path_personalized / f"model_info_{participant_id}_{name_without_extension}.txt"
    )
    write_model_info(model, info_path)

    save_relationship_plot(
        x_values,
        y_values,
        model.predict(x_values),
        results_path_personalized
        / f"relationship_pysr_{participant_id}_{name_without_extension}.png",
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
    make_model = model_factory(args)

    file_paths = [file_path, file_path_removed_dei]
    required_cols = ["mIoU", "trust", "ProlificID"]

    for path in file_paths:
        name_without_extension = path.stem
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(subset=required_cols)

        for participant_id in df["ProlificID"].dropna().unique():
            fit_personalized(df, participant_id, name_without_extension, make_model)


if __name__ == "__main__":
    main()
