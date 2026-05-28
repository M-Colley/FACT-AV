#!/usr/bin/env python3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from pysr import PySRRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

results_path_more_predictors = Path("results") / "PySR" / "more_predictors"
results_path_more_predictors.mkdir(parents=True, exist_ok=True)

file_path = Path("data") / "all_combined_prepared_with_demographics.xlsx"
sheet_name = "Sheet1"


def create_model():
    return PySRRegressor(
        niterations=500,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "sin",
            "square",
            "tan",
            "cos",
            "cube",
            "tanh",
            "sqrt",
            "abs",
            "log",
            "exp",
            "cos2(x)=cos(x)^2",
            "quart(x) = x^4",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={
            "cos2": lambda x: sympy.cos(x) ** 2,
            "inv": lambda x: 1 / x,
            "quart": lambda x: x**4,
        },
        constraints={"^": (-1, 1)},
        ncyclesperiteration=2500,
        maxsize=10,
        precision=32,
        turbo=True,
    )


def write_model_info(model, output_path):
    with output_path.open("w") as handle:
        handle.write("SYMPY\n")
        handle.write(str(model.sympy()))
        handle.write("\n\nLATEX\n")
        handle.write(str(model.latex()))
        handle.write("\n\nLATEX TABLE\n")
        handle.write(str(model.latex_table()))


def find_equal_groups(df):
    """(ProlificID, INTRODUCTION, SCENARIO) cells with clustered trust ratings.

    A cell qualifies if its most common trust value occurs >=14 times, or its
    two most common values each occur >=7 times and are within 1 of each other.

    Computed with a single groupby pass. The previous implementation re-scanned
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
    df = df.dropna().copy()
    df["intro_scenario_combo"] = (
        df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)
    )

    _, other_rows_df = split_groups(df)
    fit_and_plot(other_rows_df, file_path.stem)


if __name__ == "__main__":
    main()
