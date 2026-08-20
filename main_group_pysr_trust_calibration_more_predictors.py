#!/usr/bin/env python3
"""Symbolic regression on the variable-trust subset using the full predictor set.

Run a reproducible (publication) search with::

    python main_group_pysr_trust_calibration_more_predictors.py --seed 0 --deterministic
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from pysr_config import add_search_args, model_factory, write_model_info
from pysr_plots import save_relationship_plot
from trust_groups import find_equal_groups, split_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

results_path_more_predictors = Path("results") / "PySR" / "more_predictors"
results_path_more_predictors.mkdir(parents=True, exist_ok=True)

# Use the file that has BOTH demographics AND real ProlificIDs.
# ``all_combined_prepared_with_demographics.xlsx`` stores ProlificID as the
# constant 1 for all 2656 rows, so every (ProlificID, INTRODUCTION, SCENARIO)
# cell qualified as an "equal group", ``other_rows_df`` came out empty, and
# ``fit_and_plot`` returned at its ``len(df) < 3`` guard without writing
# anything -- which is why ``results/PySR/more_predictors/`` never existed.
file_path = Path("data") / "all_combined_prepared_with_demographics_with_baseline.xlsx"
sheet_name = "Sheet1"

# Columns required by build_feature_matrix() plus the split and target columns.
REQUIRED_COLS = [
    "mIoU",
    "Age",
    "SCENARIO",
    "Gender",
    "Education",
    "Job",
    "INTRODUCTION",
    "License",
    "DrivingFrequency",
    "Distance",
    "trust",
    "ProlificID",
]


# ``find_equal_groups`` / ``split_groups`` used to be defined here AND, in a
# slightly different form, in ``main_group_pysr_trust_calibration.py`` -- one
# returned a list and the other a set. The canonical pair now lives in
# ``trust_groups`` and is imported above; they stay in this module's public
# surface (tests and the README both reference them from here).
__all__ = ["find_equal_groups", "split_groups", "build_feature_matrix", "fit_and_plot", "main"]


def build_feature_matrix(df):
    categorical_features = ["SCENARIO", "Gender", "Education", "Job", "INTRODUCTION"]
    ordinal_features = ["License", "DrivingFrequency", "Distance"]

    x_numeric = df[["mIoU", "Age"]].to_numpy(dtype=float)
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_categorical = one_hot_encoder.fit_transform(df[categorical_features])

    encoded_ordinals = []
    for column in ordinal_features:
        encoder = LabelEncoder()
        encoded_ordinals.append(encoder.fit_transform(df[column]).reshape(-1, 1).astype(float))

    return np.hstack([x_numeric, x_categorical, *encoded_ordinals])


def fit_and_plot(df, name_without_extension, make_model):
    if len(df) < 3:
        logger.warning("Skipping %s: insufficient rows (%d)", name_without_extension, len(df))
        return

    x_values = df["mIoU"].to_numpy(dtype=float).reshape(-1, 1)
    x_values_extended = build_feature_matrix(df)
    y_values = df["trust"].to_numpy(dtype=float)

    model = make_model()
    model.fit(x_values_extended, y_values)

    info_path = (
        results_path_more_predictors
        / f"model_info_other_rows_df_stacked_MULTIPLE_{name_without_extension}.txt"
    )
    write_model_info(model, info_path)

    # ``sort_by_x`` matters here and only here: the model is fitted on the full
    # predictor matrix, so predictions arrive in dataset order rather than mIoU
    # order. Drawing them unsorted produces a zigzag across the panel.
    save_relationship_plot(
        x_values,
        y_values,
        model.predict(x_values_extended),
        results_path_more_predictors
        / f"relationship_pysr_other_rows_df_stacked_MULTIPLE_{name_without_extension}.png",
        title="Visualization of the Equation with Additional Predictors",
        hue_series=df["intro_scenario_combo"],
        ylim=(1, 5),
        sort_by_x=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_search_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    make_model = model_factory(args)

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")

    # Drop only on the columns this pipeline actually reads. A blanket
    # ``dropna()`` also discards rows for unrelated columns elsewhere in the
    # 64-column sheet.
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["intro_scenario_combo"] = df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)

    if df["ProlificID"].nunique() < 2:
        raise ValueError(
            f"{file_path} has only {df['ProlificID'].nunique()} distinct ProlificID(s). "
            "The equal-group split needs real participant identifiers; with a constant "
            "ID every cell qualifies and other_rows_df comes out empty."
        )

    all_equal_df, other_rows_df = split_groups(df)
    logger.info(
        "rows=%d  all_equal_df=%d  other_rows_df=%d", len(df), len(all_equal_df), len(other_rows_df)
    )
    fit_and_plot(other_rows_df, file_path.stem, make_model)


if __name__ == "__main__":
    main()
