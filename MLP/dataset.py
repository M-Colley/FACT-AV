import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataset import Dataset

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

TRUST_LABEL_MODES = ("floor", "separate_fractional")

# Single source of truth for the categorical class orders. The scalar encoders
# below and the vectorised pre-encoder in ``TrustDataset`` both use these, so
# the two code paths can never drift out of sync.
SCENARIO_CLASSES = ["3Spurig", "Spielstrasse", "Ueberland", "NeueMitte"]
GENDER_CLASSES = ["A1", "A2", "A3", "A4"]
EDUCATION_CLASSES = ["A1", "A2", "A3", "A4", "A5"]
JOB_CLASSES = ["A1", "A2", "A3", "A4", "A5", "A6"]
DRIVING_CLASSES = ["A1", "A2", "A3", "A4", "A5", "A6"]
DISTANCE_CLASSES = ["A1", "A2", "A3", "A4", "A5"]

# Numeric features that get standardised (zero mean / unit variance). The scaler
# is fit on the TRAIN split only, mirroring the StandardScaler used in
# ML-approaches.py so the MLP sees features on the same footing as the tree
# models. Order here defines the columns handed to the scaler.
NUMERIC_FEATURES = ["mIoU", "Age", "License"]


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
    return _encode_one_hot(scenario, SCENARIO_CLASSES, "SCENARIO")


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
    # always False, silently feeding the network a constant 0). NOTE: the
    # production path standardises this in ``TrustDataset``; this scalar encoder
    # is kept for unit tests and returns the raw (unscaled) value.
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


def _one_hot_block(values, classes, feature_name):
    """Vectorised one-hot for a whole column (mirror of ``_encode_one_hot``)."""
    index = {cls: i for i, cls in enumerate(classes)}
    block = np.zeros((len(values), len(classes)), dtype=np.float32)
    for row, raw in enumerate(values):
        key = str(raw).strip()
        position = index.get(key)
        if position is None:
            raise ValueError(
                f"Unknown value for {feature_name}: {key!r}. Expected one of {classes}."
            )
        block[row, position] = 1.0
    return block


def _intro_block(values):
    """Vectorised INTRODUCTION encoder (mirror of ``encode_intro``)."""
    block = np.zeros((len(values), 1), dtype=np.float32)
    for row, raw in enumerate(values):
        normalized = str(raw).strip().lower()
        if normalized == "boasting":
            block[row, 0] = 1.0
        elif normalized not in {"ambiguous", "ambigious"}:
            raise ValueError(
                f"Unknown value for INTRODUCTION: {raw!r}. "
                "Expected one of ['ambiguous', 'ambigious', 'boasting']."
            )
    return block


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

        df = pd.read_excel(file_path, sheet_name="Sheet1")
        df = df.dropna().reset_index(drop=True)

        self.class_values = resolve_trust_class_values(df["trust"], trust_label_mode)
        self.num_classes = len(self.class_values)

        # Participant-grouped split: the study is within-subjects (each
        # ProlificID rated trust many times), so a plain row shuffle leaks the
        # same participant across train/valid/test and inflates test metrics.
        participant_ids = df["ProlificID"].to_numpy()
        train_idx, valid_idx, test_idx = self._participant_grouped_indices(participant_ids)

        # Standardise numeric features using TRAIN ROWS ONLY (no leakage),
        # matching the StandardScaler in ML-approaches.py. The scaler is refit
        # here deterministically, so train.py and eval.py derive identical
        # statistics from the same train participants.
        numeric_raw = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(numeric_raw[train_idx])
        numeric_scaled = self.scaler.transform(numeric_raw).astype(np.float32)

        # Build the full encoded feature matrix ONCE (vectorised). Column order
        # must match the model's expected input layout; total width = 34.
        scenario_block = _one_hot_block(df["SCENARIO"], SCENARIO_CLASSES, "SCENARIO")
        intro_block = _intro_block(df["INTRODUCTION"])
        gender_block = _one_hot_block(df["Gender"], GENDER_CLASSES, "Gender")
        education_block = _one_hot_block(df["Education"], EDUCATION_CLASSES, "Education")
        job_block = _one_hot_block(df["Job"], JOB_CLASSES, "Job")
        driving_block = _one_hot_block(df["DrivingFrequency"], DRIVING_CLASSES, "DrivingFrequency")
        distance_block = _one_hot_block(df["Distance"], DISTANCE_CLASSES, "Distance")

        miou_col = numeric_scaled[:, 0:1]
        age_col = numeric_scaled[:, 1:2]
        license_col = numeric_scaled[:, 2:3]

        features = np.hstack(
            [
                miou_col,
                scenario_block,
                intro_block,
                gender_block,
                age_col,
                education_block,
                job_block,
                license_col,
                driving_block,
                distance_block,
            ]
        ).astype(np.float32)

        labels = np.array(
            [
                encode_trust_value(value, self.trust_label_mode, self.class_values)
                for value in df["trust"].to_numpy()
            ],
            dtype=np.int64,
        )

        if self.split == "train":
            selection = train_idx
        elif self.split == "valid":
            selection = valid_idx
        elif self.split == "test":
            selection = test_idx
        else:
            selection = np.arange(len(df))  # all datapoints

        self.X = torch.from_numpy(features[selection])
        self.y = torch.from_numpy(labels[selection])

        self.labels, self.counts = np.unique(self.y.numpy(), return_counts=True)
        print(f"Found {len(self.X)} for split {self.split}")
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
        """Feature dimension of the pre-encoded matrix (tracks the encoders, so
        no hard-coded ``input_size=34`` is needed)."""
        return int(self.X.shape[1])

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights (sklearn 'balanced' form) over the
        full class range, for ``CrossEntropyLoss(weight=...)``.

        Trust labels are heavily skewed toward high trust; without weighting the
        model collapses onto the majority classes and never predicts the rare
        ones. Weights are ``N / (num_classes * count_c)`` so they average ~1.
        Classes absent from this split get weight 0 (cannot be learned anyway).
        Compute this from the TRAIN dataset and pass it to the loss.
        """
        full_counts = np.zeros(self.num_classes, dtype=np.float64)
        full_counts[self.labels] = self.counts
        total = full_counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = total / (self.num_classes * full_counts)
        weights[~np.isfinite(weights)] = 0.0
        return torch.tensor(weights, dtype=torch.float32)

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
        return len(self.X)

    def __getitem__(self, index):
        # Pre-encoded: indexing is a cheap tensor slice (no per-row Python
        # encoding), which is what makes large batch sizes and the GPU actually
        # pay off.
        return self.X[index], self.y[index].reshape(1)
