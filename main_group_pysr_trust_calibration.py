#!/usr/bin/env python3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from pathlib import Path
from pysr import PySRRegressor

warnings.filterwarnings("ignore")

results_path_split_groups = Path("results") / "PySR" / "split_groups"
results_path_split_groups.mkdir(parents=True, exist_ok=True)

results_path_split_groups_personalized = Path("results") / "PySR" / "split_groups_personalized"
results_path_split_groups_personalized.mkdir(parents=True, exist_ok=True)

_DATA_FILE = Path("data") / "all_combined_prepared.xlsx"
_DATA_FILE_REMOVED_DEI = Path("data") / "all_combined_prepared_removed_REI.xlsx"
_SHEET_NAME = "Sheet1"


def _create_model():
    return PySRRegressor(
        niterations=500,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "sin", "square", "tan", "cos", "cube", "tanh",
            "sqrt", "abs", "log", "exp", "cos2(x)=cos(x)^2",
            "quart(x) = x^4", "inv(x) = 1/x",
        ],
        extra_sympy_mappings={
            "cos2": lambda x: sympy.cos(x) ** 2,
            "inv": lambda x: 1 / x,
            "quart": lambda x: x ** 4,
        },
        constraints={"^": (-1, 1)},
        ncyclesperiteration=2500,
        maxsize=10,
        precision=32,
        turbo=True,
    )


def _fit_and_save(model, x_values, y_values, info_path, plot_path, hue_series=None):
    """Fit model, write equation text, and save scatter+line plot."""
    model.fit(x_values, y_values)

    print("SYMPY")
    print(model.sympy())
    print("\nLATEX")
    print(model.latex())
    print(model.latex_table())

    with info_path.open("w") as f:
        f.write("SYMPY\n")
        f.write(str(model.sympy()))
        f.write("\n\nLATEX\n")
        f.write(str(model.latex()))
        f.write("\n\nLATEX TABLE\n")
        f.write(str(model.latex_table()))

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_kwargs = dict(alpha=0.3, s=50, edgecolor=None)
    if hue_series is not None:
        sns.scatterplot(
            x=x_values.ravel(), y=y_values["trust"] if hasattr(y_values, "columns") else y_values,
            hue=hue_series, palette="viridis", ax=ax, **scatter_kwargs,
        )
    else:
        sns.scatterplot(
            x=x_values.ravel(), y=y_values["trust"].values if hasattr(y_values, "columns") else y_values,
            ax=ax, **scatter_kwargs,
        )

    sns.lineplot(x=x_values.ravel(), y=model.predict(x_values), color="black", lw=2, ax=ax)
    ax.set_xlabel("mIoU")
    ax.set_ylabel("Trust")
    ax.set_title("Visualization of the Equation")
    ax.set_ylim(1, 6)
    sns.despine()

    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _fit_personalized(df, participant_id, model):
    """Fit a per-participant model and write its outputs."""
    print(f"Working with ProlificID: {participant_id}")

    filtered_df = df[df["ProlificID"] == participant_id]
    x_values = filtered_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
    y_values = filtered_df[["trust"]].dropna()

    _fit_and_save(
        model,
        x_values,
        y_values,
        info_path=results_path_split_groups_personalized / f"model_info_{participant_id}.txt",
        plot_path=results_path_split_groups_personalized / f"relationship_pysr_{participant_id}.png",
    )


def main():
    model = _create_model()
    file_paths = [_DATA_FILE, _DATA_FILE_REMOVED_DEI]
    other_rows_df = None

    for path in file_paths:
        name = path.stem

        df = pd.read_excel(path, sheet_name=_SHEET_NAME)
        df.dropna(inplace=True)
        df["intro_scenario_combo"] = (
            df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)
        )

        print(df.head())
        print("df shape:", df.shape)

        # Build trust-count dict: (ProlificID, INTRO, SCENARIO) → value_counts
        trust_counts = {}
        for pid, intro, scenario in df[["ProlificID", "INTRODUCTION", "SCENARIO"]].values:
            trust_counts[(pid, intro, scenario)] = df[
                (df["ProlificID"] == pid)
                & (df["INTRODUCTION"] == intro)
                & (df["SCENARIO"] == scenario)
            ]["trust"].value_counts()

        # Identify "equal-trust" groups (≥14 identical ratings, or 2 groups of ≥7)
        combinations = []
        last_value2_dict = {}
        for key, value in trust_counts.items():
            one_was_eight = False
            for _, value2 in value.items():
                if value2 >= 14:
                    combinations.append(key)
                elif value2 >= 7:
                    if one_was_eight and abs(last_value2_dict.get(key, 0) - value2) <= 1:
                        combinations.append(key)
                    else:
                        one_was_eight = True
                        last_value2_dict[key] = value2

        all_equal_df = pd.DataFrame()
        for combination in combinations:
            filtered_df = df[
                (df["ProlificID"] == combination[0])
                & (df["INTRODUCTION"] == combination[1])
                & (df["SCENARIO"] == combination[2])
            ]
            all_equal_df = pd.concat([all_equal_df, filtered_df])

        other_rows_df = df[~df.isin(all_equal_df)].dropna()
        print("other_rows_df shape:", other_rows_df.shape)

        # --- Fit 1: other_rows_df (mIoU only) ------------------------------------
        x = other_rows_df["mIoU"].to_numpy().reshape(-1, 1)
        y = other_rows_df[["trust"]].dropna()
        _fit_and_save(
            model, x, y,
            info_path=results_path_split_groups / f"model_info_other_rows_df_stacked_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_other_rows_df_stacked_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
        )

        # --- Fit 2: all_equal_df (mIoU only) ------------------------------------
        x = all_equal_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
        y = all_equal_df[["trust"]].dropna()
        _fit_and_save(
            model, x, y,
            info_path=results_path_split_groups / f"model_info_all_equal_df_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_all_equal_df_{name}.png",
            hue_series=all_equal_df["intro_scenario_combo"],
        )

        # --- Fit 3: other_rows_df repeated (mIoU only, no legend) ---------------
        x = other_rows_df["mIoU"].to_numpy().reshape(-1, 1)
        y = other_rows_df[["trust"]].dropna()
        _fit_and_save(
            model, x, y,
            info_path=results_path_split_groups / f"model_info_other_rows_df_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_other_rows_df_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
        )

    # --- Personalized fits for each participant in other_rows_df (last file) ---
    if other_rows_df is not None:
        for participant_id in other_rows_df["ProlificID"].unique():
            _fit_personalized(other_rows_df, participant_id, model)


if __name__ == "__main__":
    main()
