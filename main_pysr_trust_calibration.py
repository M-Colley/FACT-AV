#!/usr/bin/env python3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sympy
from pysr import PySRRegressor

warnings.filterwarnings("ignore")

results_path = Path("results") / "PySR"
results_path.mkdir(parents=True, exist_ok=True)

file_path = Path("data") / "all_combined_prepared.xlsx"
file_path_removed_dei = Path("data") / "all_combined_prepared_removed_REI.xlsx"
sheet_name = "Sheet1"


def create_model():
    return PySRRegressor(
        niterations=300,
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


def fit_and_plot_subset(df, intro, scenario, name_without_extension):
    print(f"Working with Introduction: {intro}, Scenario: {scenario}")
    filtered_df = df[(df["INTRODUCTION"] == intro) & (df["SCENARIO"] == scenario)]

    if len(filtered_df) < 3:
        print(
            f"Skipping Introduction={intro}, Scenario={scenario} due to insufficient rows: {len(filtered_df)}"
        )
        return

    x_values = filtered_df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = filtered_df["trust"].to_numpy()

    model = create_model()
    model.fit(x_values, y_values)

    info_path = results_path / f"model_info_{intro}_{scenario}_{name_without_extension}.txt"
    write_model_info(model, info_path)

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_values.ravel(), y=y_values, color="grey", alpha=0.5, s=50, edgecolor=None)
    sns.lineplot(x=x_values.ravel(), y=model.predict(x_values), color="green", lw=2)

    ax.set_xlabel("mIoU")
    ax.set_ylabel("Trust")
    ax.set_title(f"Visualization of the Equation for {intro} and {scenario}")
    ax.set_ylim(1, 6)

    sns.despine()
    plot_path = results_path / f"relationship_pysr_{intro}_{scenario}_{name_without_extension}.png"
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def run_all_data(df, name_without_extension):
    if len(df) < 3:
        print(f"Skipping all-data fit for {name_without_extension}: insufficient rows ({len(df)})")
        return

    x_values = df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = df["trust"].to_numpy()

    model = create_model()
    model.fit(x_values, y_values)

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_values.ravel(), y=y_values, color="grey", alpha=0.5, s=50, edgecolor=None)
    sns.lineplot(x=x_values.ravel(), y=model.predict(x_values), color="green", lw=2)

    ax.set_xlabel("mIoU")
    ax.set_ylabel("Trust")
    ax.set_title("Visualization of the Equation")
    ax.set_ylim(1, 6)

    sns.despine()
    plot_path = results_path / f"relationship_pysr_all_data_{name_without_extension}.png"
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    file_paths = [file_path, file_path_removed_dei]
    required_cols = ["mIoU", "trust", "INTRODUCTION", "SCENARIO"]

    for path in file_paths:
        name_without_extension = path.stem
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(subset=required_cols)

        observed_pairs = df[["INTRODUCTION", "SCENARIO"]].drop_duplicates()
        for intro, scenario in observed_pairs.itertuples(index=False, name=None):
            fit_and_plot_subset(df, intro, scenario, name_without_extension)

        run_all_data(df, name_without_extension)


if __name__ == "__main__":
    main()
