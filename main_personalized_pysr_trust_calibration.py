#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pysr_config import create_model, write_model_info

results_path_personalized = Path("results") / "PySR" / "personalized_plots"
results_path_personalized.mkdir(parents=True, exist_ok=True)

file_path = Path("data") / "all_combined_prepared.xlsx"
file_path_removed_dei = Path("data") / "all_combined_prepared_removed_REI.xlsx"
sheet_name = "Sheet1"


def fit_personalized(df, participant_id, name_without_extension):
    print(f"Working with ProlificID: {participant_id}")

    filtered_df = df[df["ProlificID"] == participant_id]
    if len(filtered_df) < 3:
        print(f"Skipping ProlificID={participant_id}: insufficient rows ({len(filtered_df)})")
        return

    x_values = filtered_df["mIoU"].to_numpy().reshape(-1, 1)
    y_values = filtered_df["trust"].to_numpy()

    model = create_model()
    model.fit(x_values, y_values)

    # The dataset stem is part of the filename. Without it, the second dataset's
    # equations overwrite the first's, since both files share participant IDs --
    # the .png path already included it, so the two artifacts disagreed.
    info_path = results_path_personalized / f"model_info_{participant_id}_{name_without_extension}.txt"
    write_model_info(model, info_path)

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
    plot_path = results_path_personalized / f"relationship_pysr_{participant_id}_{name_without_extension}.png"
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    file_paths = [file_path, file_path_removed_dei]
    required_cols = ["mIoU", "trust", "ProlificID"]

    for path in file_paths:
        name_without_extension = path.stem
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(subset=required_cols)

        for participant_id in df["ProlificID"].dropna().unique():
            fit_personalized(df, participant_id, name_without_extension)


if __name__ == "__main__":
    main()
