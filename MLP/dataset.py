import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

TRUST_LABEL_MODES = ("floor", "separate_fractional")


def _encode_one_hot(value, classes, feature_name):
    value_str = str(value).strip()
    try:
        class_index = classes.index(value_str)
    except ValueError as exc:
        raise ValueError(
            f"Unknown value for {feature_name}: {value_str!r}. Expected one of {classes}."
        ) from exc
    return torch.nn.functional.one_hot(
        torch.tensor(class_index),
        num_classes=len(classes),
    ).to(torch.float32)


def encode_scenario(scenario):
    classes = ["3Spurig", "Spielstrasse", "Ueberland", "NeueMitte"]
    return _encode_one_hot(scenario, classes, "SCENARIO")


def encode_intro(intro):
    # Support both spellings observed in project docs/data.
    normalized = str(intro).strip().lower()
    if normalized in {"ambiguous", "ambigious"}:
        return torch.tensor(0.0, dtype=torch.float32)
    if normalized == "boasting":
        return torch.tensor(1.0, dtype=torch.float32)
    raise ValueError(
        f"Unknown value for INTRODUCTION: {intro!r}. Expected one of ['ambiguous', 'ambigious', 'boasting']."
    )


def encode_cat(string, num_classes):
    classes = ["A1", "A2", "A3", "A4", "A5", "A6"][:num_classes]
    return _encode_one_hot(string, classes, "categorical")


def encode_gender(string):
    return encode_cat(string, 4)


def encode_education(string):
    return encode_cat(string, 5)


def encode_job(string):
    return encode_cat(string, 6)


def encode_driving(string):
    return encode_cat(string, 6)


def encode_distance(string):
    return encode_cat(string, 5)


def encode_license(value):
    # ``License`` is the number of years the participant has held a driving
    # licence (integer), not a Y/N flag. Pass it through as a float scalar so
    # the feature actually reaches the model (previously ``value == "Y"`` was
    # always False, silently feeding the network a constant 0).
    return torch.tensor([float(value)], dtype=torch.float32)


def resolve_trust_class_values(trust_values, trust_label_mode):
    if trust_label_mode == "floor":
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    if trust_label_mode == "separate_fractional":
        return sorted({float(value) for value in trust_values.dropna()})

    raise ValueError(
        f"Unknown trust_label_mode: {trust_label_mode!r}. Expected one of {TRUST_LABEL_MODES}."
    )


def encode_trust_value(value, trust_label_mode, class_values):
    numeric_value = float(value)

    if trust_label_mode == "floor":
        floored_value = min(5, max(1, math.floor(numeric_value)))
        return floored_value - 1

    if trust_label_mode == "separate_fractional":
        for index, class_value in enumerate(class_values):
            if np.isclose(numeric_value, class_value):
                return index
        raise ValueError(
            f"Unknown trust value {numeric_value!r} for mode {trust_label_mode!r}. "
            f"Expected one of {class_values}."
        )

    raise ValueError(
        f"Unknown trust_label_mode: {trust_label_mode!r}. Expected one of {TRUST_LABEL_MODES}."
    )


def addlabels(x_positions, y_values):
    total = np.sum(y_values)
    for x_position, y_value in zip(x_positions, y_values):
        if y_value == 0:
            continue
        plt.text(
            x_position,
            y_value,
            f"{int(y_value)}({(y_value / total) * 100:0.1f}%)",
            ha="center",
        )


class TrustDataset(Dataset):
    def __init__(self, file_path, split, trust_label_mode="floor") -> None:
        super().__init__()
        self.split = split
        self.trust_label_mode = trust_label_mode
        # read data

        # Specify the sheet name (optional)
        sheet_name = "Sheet1"

        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.dropna().reset_index(drop=True)

        self.class_values = resolve_trust_class_values(df["trust"], trust_label_mode)
        self.num_classes = len(self.class_values)

        # Assemble the feature matrix in a fixed column order. The trailing
        # column holds the raw trust target, decoded lazily in __getitem__.
        # The order here MUST stay in sync with the positional indexing in
        # __getitem__ (data[0]..data[10]).
        feature_columns = [
            "mIoU",
            "SCENARIO",
            "INTRODUCTION",
            "Gender",
            "Age",
            "Education",
            "Job",
            "License",
            "DrivingFrequency",
            "Distance",
        ]
        x_values_extended = np.column_stack(
            [df[column].to_numpy() for column in feature_columns]
            + [df["trust"].to_numpy()]
        )

        # Participant-grouped split: the study is within-subjects (each
        # ProlificID rated trust many times), so a plain row shuffle leaks the
        # same participant across train/valid/test and inflates test metrics.
        participant_ids = df["ProlificID"].to_numpy()
        train_idx, valid_idx, test_idx = self._participant_grouped_indices(participant_ids)

        if self.split == "train":
            self.datapoints = x_values_extended[train_idx]
        elif self.split == "valid":
            self.datapoints = x_values_extended[valid_idx]
        elif self.split == "test":
            self.datapoints = x_values_extended[test_idx]
        else:
            # all datapoints
            self.datapoints = x_values_extended

        encoded_labels = [
            encode_trust_value(d[-1], self.trust_label_mode, self.class_values)
            for d in self.datapoints
        ]
        self.labels, self.counts = np.unique(encoded_labels, return_counts=True)
        print(f"Found {len(self.datapoints)} for split {self.split}")
        print(f"Labels: {self.labels} counts {self.counts}")
        print(f"Labels: {self.labels} weights {np.sum(self.counts) / self.counts}")

    @staticmethod
    def _participant_grouped_indices(participant_ids, seed: int = 1337):
        """Deterministic 80/10/10 split that never splits a participant.

        Uses two nested GroupShuffleSplit passes (80% train, then a 50/50
        split of the 20% holdout into valid/test). Returns integer index
        arrays into the row order of ``participant_ids``.
        """
        from sklearn.model_selection import GroupShuffleSplit

        indices = np.arange(len(participant_ids))
        train_idx, holdout_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                indices, groups=participant_ids
            )
        )
        valid_rel, test_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
                holdout_idx, groups=participant_ids[holdout_idx]
            )
        )
        return train_idx, holdout_idx[valid_rel], holdout_idx[test_rel]

    @property
    def input_size(self) -> int:
        """Feature dimension, probed from a real sample so it always tracks the
        encoders (avoids the previously hard-coded ``input_size=34``)."""
        x, _ = self[0]
        return int(x.shape[0])

    def plot_label_distribution(self, results_dir: Path = results_folder) -> None:
        """Save the per-split label-distribution bar chart.

        Kept out of ``__init__`` so that merely constructing a dataset has no
        filesystem / global-pyplot side effects. Call explicitly when wanted.
        """
        fig, ax = plt.subplots()
        positions = np.arange(self.num_classes)
        full_counts = np.zeros(self.num_classes, dtype=int)
        full_counts[self.labels] = self.counts
        bar_labels = [f"Trust {value:g}" for value in self.class_values]
        bar_colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        ax.bar(positions, full_counts, width=0.8, color=bar_colors)
        ax.set_title(f"{str(self.split).capitalize()} Dataset Label Distribution")
        ax.set_xlabel("Labels")
        ax.set_xticks(positions)
        ax.set_xticklabels(bar_labels, rotation=45 if self.num_classes > 5 else 0)
        ax.set_yticks([])

        total = full_counts.sum()
        for x_pos, y_val in zip(positions, full_counts):
            if y_val:
                ax.text(x_pos, y_val, f"{int(y_val)}({y_val / total * 100:0.1f}%)", ha="center")

        mode_suffix = "" if self.trust_label_mode == "floor" else f".{self.trust_label_mode}"
        for ext in ("pdf", "jpg"):
            fig.savefig(
                results_dir / f"{self.split}{mode_suffix}.labels.{ext}",
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close(fig)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        data = self.datapoints[index]
        x_miou = torch.tensor([data[0]], dtype=torch.float32)
        x_scenario = encode_scenario(data[1])
        x_intro = encode_intro(data[2])
        x_gender = encode_gender(data[3])
        x_age = torch.tensor([data[4]], dtype=torch.float32)
        x_education = encode_education(data[5])
        x_job = encode_job(data[6])
        x_license = encode_license(data[7])
        x_drivingfreq = encode_driving(data[8])
        x_distance = encode_distance(data[9])
        y = data[10]

        x = torch.concatenate(
            [
                x_miou.view(1, -1),
                x_scenario.view(1, -1),
                x_intro.view(1, -1),
                x_gender.view(1, -1),
                x_age.view(1, -1),
                x_education.view(1, -1),
                x_job.view(1, -1),
                x_license.view(1, -1),
                x_drivingfreq.view(1, -1),
                x_distance.view(1, -1),
            ],
            dim=1,
        )
        y = torch.tensor(
            encode_trust_value(y, self.trust_label_mode, self.class_values)
        ).long()
        return x.flatten(), y.flatten()
