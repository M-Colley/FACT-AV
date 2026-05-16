import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    mean_absolute_error,
)
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score

from dataset import TRUST_LABEL_MODES, TrustDataset
from network import Model

# Allow ``import plotting_style`` when running from the MLP/ subdirectory.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from plotting_style import OKABE_ITO, apply_paper_style, save_fig  # noqa: E402

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

# Get the parent directory and construct the path to the data folder
data_folder = Path(__file__).parent.parent / "data"

# Construct the full file path
data_file = data_folder / "all_combined_prepared_with_demographics.xlsx"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trust-label-mode",
        choices=TRUST_LABEL_MODES,
        default="floor",
        help="How to map trust labels into classification classes.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    return parser.parse_args()


def default_checkpoint_path(trust_label_mode):
    return results_folder / f"best_valid_{trust_label_mode}.pt"


def resolve_checkpoint(trust_label_mode, checkpoint_path=None):
    path = checkpoint_path or default_checkpoint_path(trust_label_mode)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def plot_calibration(y_true, y_probs, n_bins, save_path, title):
    """Reliability diagram for a multiclass classifier (max-prob calibration).

    For each test point we take the model's predicted class and its confidence
    (max softmax probability), bin those confidences, and compare the per-bin
    accuracy to the per-bin mean confidence. A well-calibrated model lies on
    y=x; below the diagonal means overconfident, above means underconfident.
    The Expected Calibration Error (ECE) summarises the gap as a scalar.
    """
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accuracies = np.zeros(n_bins)
    confidences_bin = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            accuracies[b] = correct[mask].mean()
            confidences_bin[b] = confidences[mask].mean()
            counts[b] = int(mask.sum())

    nonempty = counts > 0
    ece = float(
        np.sum(
            (counts[nonempty] / counts.sum())
            * np.abs(accuracies[nonempty] - confidences_bin[nonempty])
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.0, 0.7]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.bar(
        bin_centers[nonempty],
        accuracies[nonempty],
        width=1.0 / n_bins * 0.9,
        color=OKABE_ITO[2],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        label="Empirical accuracy",
    )
    ax.scatter(
        bin_centers[nonempty],
        confidences_bin[nonempty],
        color=OKABE_ITO[6],
        marker="x",
        s=55,
        label="Mean confidence",
        zorder=5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted confidence (max softmax)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram - ECE = {ece:.3f}")
    ax.legend(loc="upper left")

    ax2 = axes[1]
    ax2.bar(
        bin_centers,
        counts,
        width=1.0 / n_bins * 0.9,
        color=OKABE_ITO[1],
        edgecolor="black",
        linewidth=0.4,
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("# test samples in bin")
    ax2.set_title("Confidence histogram")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)


def write_per_class_report(y_true, y_pred, class_values, csv_path):
    """Per-class precision/recall/F1/support as a publication-ready CSV."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_values))), zero_division=0
    )
    import pandas as pd

    table = pd.DataFrame(
        {
            "class_index": list(range(len(class_values))),
            "trust_value": class_values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    table.to_csv(csv_path, index=False, float_format="%.4f")


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Trust label mode: {args.trust_label_mode}")

    test_dataset = TrustDataset(
        data_file,
        split="test",
        trust_label_mode=args.trust_label_mode,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = resolve_checkpoint(args.trust_label_mode, args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", test_dataset.num_classes)
        class_values = checkpoint.get("class_values", test_dataset.class_values)
    else:
        model_state_dict = checkpoint
        num_classes = test_dataset.num_classes
        class_values = test_dataset.class_values

    if num_classes != test_dataset.num_classes:
        raise ValueError(
            f"Checkpoint expects {num_classes} classes but dataset mode "
            f"{args.trust_label_mode!r} resolves to {test_dataset.num_classes} classes."
        )

    model = Model(input_size=34, num_classes=num_classes).to(device)
    model.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
            batch_size_current = y.size(0)
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current

            y_true.append(y.cpu())
            y_pred.append(predictions.cpu())
            y_probs.append(probs.cpu())

    y_true_tensor = torch.cat(y_true)
    y_pred_tensor = torch.cat(y_pred)
    y_probs_array = torch.cat(y_probs).numpy()
    y_true_array = y_true_tensor.numpy()
    y_pred_array = y_pred_tensor.numpy()

    test_loss = total_loss / total_samples
    test_acc = float(multiclass_accuracy(y_pred_tensor, y_true_tensor, num_classes=num_classes).item())
    test_f1 = float(multiclass_f1_score(y_pred_tensor, y_true_tensor, num_classes=num_classes).item())

    # Ordinal-aware metrics: trust is on an ordered scale, so reward "almost right".
    qwk = float(cohen_kappa_score(y_true_array, y_pred_array, weights="quadratic"))
    class_values_array = np.asarray(class_values, dtype=float)
    true_trust = class_values_array[y_true_array]
    pred_trust = class_values_array[y_pred_array]
    mae_trust = float(mean_absolute_error(true_trust, pred_trust))

    print(f"Test Loss: {test_loss}")
    print(f"Test Acc: {test_acc}")
    print(f"Test F1: {test_f1}")
    print(f"Test Quadratic-Weighted Kappa: {qwk:.4f}")
    print(f"Test MAE in trust units: {mae_trust:.4f}")

    apply_paper_style()
    suffix = "" if args.trust_label_mode == "floor" else f"_{args.trust_label_mode}"

    # -------- Ordinal-aware confusion matrix (QWK + MAE annotation) ---------
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true_array,
        y_pred_array,
        display_labels=[f"Trust {value:g}" for value in class_values],
        cmap=plt.cm.Blues,
        normalize="pred",
        ax=ax,
        colorbar=True,
    )
    ax.set_title(f"Trust estimation - QWK={qwk:.3f}, MAE={mae_trust:.2f}, F1={test_f1:.3f}")
    ax.grid(False)
    save_fig(fig, results_folder / f"confusion_matrix{suffix}")
    plt.close(fig)

    # ------------------------- Reliability diagram --------------------------
    plot_calibration(
        y_true_array,
        y_probs_array,
        n_bins=10,
        save_path=results_folder / f"calibration{suffix}",
        title=f"MLP reliability diagram (mode={args.trust_label_mode})",
    )

    # ------------ Per-class precision/recall/F1 + support table -------------
    write_per_class_report(
        y_true_array,
        y_pred_array,
        class_values,
        results_folder / f"per_class_metrics{suffix}.csv",
    )


if __name__ == "__main__":
    main()
