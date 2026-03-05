import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score

from dataset import TrustDataset
from network import Model

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

# Get the parent directory and construct the path to the data folder
data_folder = Path(__file__).parent.parent / "data"

# Construct the full file path
data_file = data_folder / "all_combined_prepared_with_demographics.xlsx"

num_classes = 5


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(path):
    candidates = sorted(path.glob("test_f1_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching 'test_f1_*.pt' found in {path}")

    def checkpoint_score(checkpoint_path):
        match = re.search(r"test_f1_([0-9]*\.?[0-9]+)", checkpoint_path.stem)
        if not match:
            return float("-inf")
        return float(match.group(1))

    return max(candidates, key=checkpoint_score)


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_dataset = TrustDataset(data_file, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = Model(input_size=34).to(device)

    checkpoint_path = resolve_checkpoint(results_folder)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
            acc = multiclass_accuracy(predictions, y, num_classes=num_classes)
            f1 = multiclass_f1_score(predictions, y, num_classes=num_classes)

            test_losses.append(loss.item())
            test_accuracies.append(acc.item())
            test_f1_scores.append(f1.item())

            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    print(f"Test Loss: {np.mean(test_losses)}")
    print(f"Test Acc: {np.mean(test_accuracies)}")
    print(f"Test F1: {np.mean(test_f1_scores)}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[f"Trust Level {i + 1}" for i in range(num_classes)],
        cmap=plt.cm.Blues,
        normalize="pred",
        ax=ax,
    )
    ax.set_title("Trust Estimation")

    pdf_path = results_folder / "confusion_matrix.pdf"
    jpg_path = results_folder / "confusion_matrix.jpg"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(jpg_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
