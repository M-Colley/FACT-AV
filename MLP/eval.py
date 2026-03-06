import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score

from dataset import TRUST_LABEL_MODES, TrustDataset
from network import Model

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

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
            batch_size_current = y.size(0)
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current

            y_true.append(y.cpu())
            y_pred.append(predictions.cpu())

    y_true_tensor = torch.cat(y_true)
    y_pred_tensor = torch.cat(y_pred)

    test_loss = total_loss / total_samples
    test_acc = float(multiclass_accuracy(y_pred_tensor, y_true_tensor, num_classes=num_classes).item())
    test_f1 = float(multiclass_f1_score(y_pred_tensor, y_true_tensor, num_classes=num_classes).item())

    print(f"Test Loss: {test_loss}")
    print(f"Test Acc: {test_acc}")
    print(f"Test F1: {test_f1}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true_tensor.numpy(),
        y_pred_tensor.numpy(),
        display_labels=[f"Trust {value:g}" for value in class_values],
        cmap=plt.cm.Blues,
        normalize="pred",
        ax=ax,
    )
    ax.set_title("Trust Estimation")

    suffix = "" if args.trust_label_mode == "floor" else f"_{args.trust_label_mode}"
    pdf_path = results_folder / f"confusion_matrix{suffix}.pdf"
    jpg_path = results_folder / f"confusion_matrix{suffix}.jpg"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(jpg_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
