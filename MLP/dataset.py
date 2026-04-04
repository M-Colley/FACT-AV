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


def encode_license(string):
    return torch.tensor([string == "Y"]).float()


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

        df.dropna(inplace=True)
        self.class_values = resolve_trust_class_values(df["trust"], trust_label_mode)
        self.num_classes = len(self.class_values)

        ### ONCE FOR ALL DATA

        y_values = df[["trust"]].dropna()

        x_values = df["mIoU"].dropna().to_numpy()
        x_values = x_values.reshape(-1, 1)

        # New dimension based on the 'SCENARIO' column
        x_scenario = df["SCENARIO"].dropna().to_numpy()
        x_scenario = x_scenario.reshape(-1, 1)

        # # New dimension based on the 'GENDER' column
        x_gender = df["Gender"].dropna().to_numpy()
        x_gender = x_gender.reshape(-1, 1)

        #     # New dimension based on the 'AGE' column
        x_age = df["Age"].dropna().to_numpy()
        x_age = x_age.reshape(-1, 1)

        #     # New dimension based on the 'Education' column
        x_education = df["Education"].dropna().to_numpy()
        x_education = x_education.reshape(-1, 1)

        #     # New dimension based on the 'Job' column
        x_job = df["Job"].dropna().to_numpy()
        x_job = x_job.reshape(-1, 1)

        #     # New dimension based on the 'License' column
        x_license = df["License"].dropna().to_numpy()
        x_license = x_license.reshape(-1, 1)

        #     # New dimension based on the 'DrivingFrequency' column
        x_drivingfreq = df["DrivingFrequency"].dropna().to_numpy()
        x_drivingfreq = x_drivingfreq.reshape(-1, 1)

        #     # New dimension based on the 'Distance' column
        x_distance = df["Distance"].dropna().to_numpy()
        x_distance = x_distance.reshape(-1, 1)

        # New dimension based on the 'INTRODUCTION' column
        x_intro = df["INTRODUCTION"].dropna().to_numpy()
        x_intro = x_intro.reshape(-1, 1)

        # Adding new dimension to x_values
        x_values_extended = np.hstack(
            [
                x_values,
                x_scenario,
                x_intro,
                x_gender,
                x_age,
                x_education,
                x_job,
                x_license,
                x_drivingfreq,
                x_distance,
                y_values,
            ]
        )

        N = len(x_values_extended)
        N_Train = int(0.8 * N)
        N_Valid = int(0.1 * N)
        N_Test = N - N_Train - N_Valid

        np.random.seed(1337)
        np.random.shuffle(x_values_extended)

        if self.split == "train":
            self.datapoints = x_values_extended[0:N_Train]
        elif self.split == "valid":
            self.datapoints = x_values_extended[N_Train : N_Train + N_Valid]
        elif self.split == "test":
            self.datapoints = x_values_extended[N_Train + N_Valid :]
            assert len(self.datapoints) == N_Test, f"{len(self.datapoints)} {N_Test}"
        else:
            # all datapoints
            self.datapoints = x_values_extended[0:]

        encoded_labels = [
            encode_trust_value(d[-1], self.trust_label_mode, self.class_values)
            for d in self.datapoints
        ]
        self.labels, self.counts = np.unique(encoded_labels, return_counts=True)
        print(f"Found {len(self.datapoints)} for split {self.split}")
        print(f"Labels: {self.labels} counts {self.counts}")
        print(f"Labels: {self.labels} weights {np.sum(self.counts) / self.counts}")

        positions = np.arange(self.num_classes)
        full_counts = np.zeros(self.num_classes, dtype=int)
        full_counts[self.labels] = self.counts
        bar_labels = [f"Trust {value:g}" for value in self.class_values]
        bar_colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        plt.bar(positions, full_counts, width=0.8, color=bar_colors)
        # Add in a title and axes labels
        plt.title(f"{str(self.split).capitalize()} Dataset Label Distribution")
        plt.xlabel("Labels")
        plt.xticks(positions, bar_labels, rotation=45 if self.num_classes > 5 else 0)
        addlabels(positions, full_counts)
        # Set the tick locations
        plt.yticks([])

        mode_suffix = "" if self.trust_label_mode == "floor" else f".{self.trust_label_mode}"
        file_path = results_folder / f"{self.split}{mode_suffix}.labels.pdf"
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
        file_path = results_folder / f"{self.split}{mode_suffix}.labels.jpg"
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
        plt.close()

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
