#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from pysr_config import create_model, write_model_info

results_path_split_groups = Path("results") / "PySR" / "split_groups"
results_path_split_groups.mkdir(parents=True, exist_ok=True)

results_path_split_groups_personalized = Path("results") / "PySR" / "split_groups_personalized"
results_path_split_groups_personalized.mkdir(parents=True, exist_ok=True)

_DATA_FILE = Path("data") / "all_combined_prepared.xlsx"
_DATA_FILE_REMOVED_DEI = Path("data") / "all_combined_prepared_removed_REI.xlsx"
_SHEET_NAME = "Sheet1"


def _create_model():
    return create_model()


def _save_plot(model, x_values, y_values, plot_path, hue_series=None, legend=True):
    """Save a scatter + fitted-curve plot for an already-fitted model."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_kwargs = dict(alpha=0.3, s=50, edgecolor=None)
    if hue_series is not None:
        sns.scatterplot(
            x=x_values.ravel(), y=y_values["trust"] if hasattr(y_values, "columns") else y_values,
            hue=hue_series, palette="viridis", ax=ax, legend=legend, **scatter_kwargs,
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


def _fit_and_save(model, x_values, y_values, info_path, plot_path, hue_series=None, legend=True):
    """Fit model, write equation text, save scatter+line plot, return the model.

    The fitted model is returned so a caller can render additional figures from
    the same search instead of re-fitting.
    """
    model.fit(x_values, y_values)

    print("SYMPY")
    print(model.sympy())
    print("\nLATEX")
    print(model.latex())
    print(model.latex_table())

    write_model_info(model, info_path)
    _save_plot(model, x_values, y_values, plot_path, hue_series=hue_series, legend=legend)
    return model


def _find_equal_groups(df):
    """(ProlificID, INTRODUCTION, SCENARIO) cells where trust ratings cluster.

    A cell qualifies if either:

    * its single most common trust value occurs >=14 times -- the participant
      answered almost flatly; or
    * its two most common trust values each occur >=7 times **and those two
      frequencies differ by at most 1** -- the participant split roughly evenly
      between two ratings.

    The ``<= 1`` compares *frequencies*, not the trust values themselves, so a
    10/10 split between trust 1 and trust 5 qualifies. The criterion is about how
    concentrated a participant's responses are, not how close the two ratings are.

    Computed with a single groupby pass instead of re-scanning the whole frame
    once per row.
    """
    trust_counts = (
        df.groupby(["ProlificID", "INTRODUCTION", "SCENARIO"])["trust"]
        .value_counts()
        .unstack(fill_value=0)
    )
    combinations = []
    for key, row in trust_counts.iterrows():
        counts = sorted(row.to_numpy(), reverse=True)
        if counts and counts[0] >= 14:
            combinations.append(key)
        elif (
            len(counts) >= 2
            and counts[0] >= 7
            and counts[1] >= 7
            and abs(counts[0] - counts[1]) <= 1
        ):
            combinations.append(key)
    return combinations


def _fit_personalized(df, participant_id, model, name):
    """Fit a per-participant model and write its outputs.

    ``name`` (the dataset stem) is part of both output filenames. Without it the
    second dataset's equations overwrite the first's, since participant IDs are
    shared between the two files.
    """
    print(f"Working with ProlificID: {participant_id}")

    filtered_df = df[df["ProlificID"] == participant_id]
    x_values = filtered_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
    y_values = filtered_df[["trust"]].dropna()

    if len(x_values) < 3:
        print(f"Skipping ProlificID={participant_id}: insufficient rows ({len(x_values)})")
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


def main():
    file_paths = [_DATA_FILE, _DATA_FILE_REMOVED_DEI]

    for path in file_paths:
        name = path.stem

        df = pd.read_excel(path, sheet_name=_SHEET_NAME)
        df.dropna(inplace=True)
        df["intro_scenario_combo"] = (
            df["INTRODUCTION"].astype(str) + "_" + df["SCENARIO"].astype(str)
        )

        print(df.head())
        print("df shape:", df.shape)

        # Identify "equal-trust" groups (>=14 identical ratings, or 2 groups of >=7).
        combinations = _find_equal_groups(df)

        if combinations:
            equal_frames = [
                df[
                    (df["ProlificID"] == pid)
                    & (df["INTRODUCTION"] == intro)
                    & (df["SCENARIO"] == scenario)
                ]
                for pid, intro, scenario in combinations
            ]
            all_equal_df = pd.concat(equal_frames)
        else:
            all_equal_df = df.iloc[0:0]

        # Set difference by index. The previous ``df[~df.isin(all_equal_df)]``
        # did an element-wise (value+index aligned) comparison, then dropna(),
        # which silently discarded far more rows than intended.
        other_rows_df = df.drop(index=all_equal_df.index)
        print("other_rows_df shape:", other_rows_df.shape)

        # NOTE: a *fresh* PySRRegressor is used for every fit. Reusing one
        # instance lets Julia-side search state (the equation hall-of-fame)
        # bleed from one dataset into the next.

        # --- Fit 1: other_rows_df (mIoU only) ------------------------------------
        # This used to be run twice: "Fit 1" and "Fit 3" were byte-identical
        # searches of the same x, y and hue, differing only in output filename.
        # That doubled runtime and produced two files that could disagree purely
        # because the search is stochastic. One search now backs both figures.
        x = other_rows_df["mIoU"].to_numpy().reshape(-1, 1)
        y = other_rows_df[["trust"]].dropna()
        other_rows_model = _fit_and_save(
            _create_model(), x, y,
            info_path=results_path_split_groups / f"model_info_other_rows_df_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_other_rows_df_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
        )
        _save_plot(
            other_rows_model, x, y,
            plot_path=results_path_split_groups / f"relationship_pysr_other_rows_df_stacked_{name}.png",
            hue_series=other_rows_df["intro_scenario_combo"],
            legend=False,
        )

        # --- Fit 2: all_equal_df (mIoU only) ------------------------------------
        x = all_equal_df["mIoU"].dropna().to_numpy().reshape(-1, 1)
        y = all_equal_df[["trust"]].dropna()
        _fit_and_save(
            _create_model(), x, y,
            info_path=results_path_split_groups / f"model_info_all_equal_df_{name}.txt",
            plot_path=results_path_split_groups / f"relationship_pysr_all_equal_df_{name}.png",
            hue_series=all_equal_df["intro_scenario_combo"],
        )

        # --- Personalized fits for each participant in other_rows_df -----------
        # Inside the per-file loop. Previously this sat after the loop and read
        # the leaked ``other_rows_df``, so it silently ran on the last dataset
        # only -- and on just its 43 participants, not all 130.
        for participant_id in other_rows_df["ProlificID"].unique():
            _fit_personalized(other_rows_df, participant_id, _create_model(), name)


if __name__ == "__main__":
    main()
