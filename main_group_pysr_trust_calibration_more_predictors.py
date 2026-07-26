#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from pysr_config import create_model, write_model_info

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


def find_equal_groups(df):
    """(ProlificID, INTRODUCTION, SCENARIO) cells with clustered trust ratings.

    A cell qualifies if either:

    * its most common trust value occurs >=14 times -- the participant answered
      almost flatly; or
    * its two most common trust values each occur >=7 times **and those two
      frequencies differ by at most 1** -- the participant split roughly evenly
      between two ratings.

    Note the ``<= 1`` compares *frequencies*, not the trust values themselves:
    a 10/10 split between trust 1 and trust 5 qualifies, because the criterion is
    about how concentrated the responses are, not how close the two ratings are.
    The earlier wording ("within 1 of each other") read ambiguously as if it
    constrained the values; it does not, and the behaviour here is unchanged.

    Computed with a single groupby pass. The original implementation re-scanned
    the whole frame once per row (O(N^2)) and tracked the ">=7" comparison in a
    dict keyed across *different* cells, so a count from one cell could spuriously
    qualify another.
    """
    trust_counts = (
        df.groupby(["ProlificID", "INTRODUCTION", "SCENARIO"])["trust"]
        .value_counts()
        .unstack(fill_value=0)
    )

    combinations = set()
    for key, row in trust_counts.iterrows():
        counts = sorted(row.to_numpy(), reverse=True)
        if counts and counts[0] >= 14:
            combinations.add(key)
        elif (
            len(counts) >= 2
            and counts[0] >= 7
            and counts[1] >= 7
            and abs(counts[0] - counts[1]) <= 1
        ):
            combinations.add(key)

    return combinations


def split_groups(df):
    combinations = find_equal_groups(df)
    if not combinations:
        return pd.DataFrame(columns=df.columns), df.copy()

    equal_frames = [
        df[
            (df["ProlificID"] == combination[0])
            & (df["INTRODUCTION"] == combination[1])
            & (df["SCENARIO"] == combination[2])
        ]
        for combination in combinations
    ]
    all_equal_df = pd.concat(equal_frames).sort_index()
    other_rows_df = df.drop(index=all_equal_df.index).sort_index()
    return all_equal_df, other_rows_df


def build_feature_matrix(df):
    categorical_features = ["SCENARIO", "Gender", "Education", "Job", "INTRODUCTION"]
    ordinal_features = ["License", "DrivingFrequency", "Distance"]

    x_numeric = df[["mIoU", "Age"]].to_numpy(dtype=float)
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_categorical = one_hot_encoder.fit_transform(df[categorical_features])

    encoded_ordinals = []
    for column in ordinal_features:
        encoder = LabelEncoder()
        encoded_ordinals.append(
            encoder.fit_transform(df[column]).reshape(-1, 1).astype(float)
        )

    return np.hstack([x_numeric, x_categorical, *encoded_ordinals])


def fit_and_plot(df, name_without_extension):
    if len(df) < 3:
        print(f"Skipping {name_without_extension}: insufficient rows ({len(df)})")
        return

    x_values = df["mIoU"].to_numpy(dtype=float).reshape(-1, 1)
    x_values_extended = build_feature_matrix(df)
    y_values = df["trust"].to_numpy(dtype=float)

    model = create_model()
    model.fit(x_values_extended, y_values)

    info_path = (
        results_path_more_predictors
        / f"model_info_other_rows_df_stacked_MULTIPLE_{name_without_extension}.txt"
    )
    write_model_info(model, info_path)

    predictions = model.predict(x_values_extended)
    sort_idx = np.argsort(x_values.ravel())
    sorted_x = x_values.ravel()[sort_idx]
    sorted_predictions = predictions[sort_idx]

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=x_values.ravel(),
        y=y_values,
        hue=df["intro_scenario_combo"],
        palette="viridis",
        alpha=0.3,
        s=50,
        edgecolor=None,
        ax=ax,
    )
    ax.plot(sorted_x, sorted_predictions, color="black", lw=2)

    ax.set_xlabel("mIoU")
    ax.set_ylabel("Trust")
    ax.set_title("Visualization of the Equation with Additional Predictors")
    ax.set_ylim(1, 5)

    sns.despine()

    output_path = (
        results_path_more_predictors
        / f"relationship_pysr_other_rows_df_stacked_MULTIPLE_{name_without_extension}.png"
    )
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")

    # Drop only on the columns this pipeline actually reads. A blanket
    # ``dropna()`` also discards rows for unrelated columns elsewhere in the
    # 64-column sheet.
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["intro_scenario_combo"] = (
        df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)
    )

    if df["ProlificID"].nunique() < 2:
        raise ValueError(
            f"{file_path} has only {df['ProlificID'].nunique()} distinct ProlificID(s). "
            "The equal-group split needs real participant identifiers; with a constant "
            "ID every cell qualifies and other_rows_df comes out empty."
        )

    all_equal_df, other_rows_df = split_groups(df)
    print(f"rows={len(df)}  all_equal_df={len(all_equal_df)}  other_rows_df={len(other_rows_df)}")
    fit_and_plot(other_rows_df, file_path.stem)


if __name__ == "__main__":
    main()
