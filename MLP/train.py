import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from tqdm import tqdm

from dataset import TRUST_LABEL_MODES, TrustDataset
from network import Model

# Get the parent directory and construct the path to the data folder
data_folder = Path(__file__).parent.parent / "data"

# Construct the full file path
data_file = data_folder / "all_combined_prepared_with_demographics.xlsx"

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)

epochs_dir = Path(__file__).parent / "epochs"
epochs_dir.mkdir(parents=True, exist_ok=True)

epochs = 2000
batch_size = 16
learning_rate = 1e-4


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
    return parser.parse_args()


def evaluate_split(model, data_loader, criterion, device, num_classes):
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predicted = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
            batch_size_current = y.size(0)
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current

            predictions.append(predicted.cpu())
            targets.append(y.cpu())

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)

    return {
        "loss": total_loss / total_samples,
        "acc": float(multiclass_accuracy(y_pred, y_true, num_classes=num_classes).item()),
        "f1": float(multiclass_f1_score(y_pred, y_true, num_classes=num_classes).item()),
    }


def get_checkpoint_path(trust_label_mode):
    return results_folder / f"best_valid_{trust_label_mode}.pt"


def get_report_path(trust_label_mode):
    return results_folder / f"best_valid_{trust_label_mode}.json"


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Trust label mode: {args.trust_label_mode}")

    train_dataset = TrustDataset(
        data_file,
        split="train",
        trust_label_mode=args.trust_label_mode,
    )
    valid_dataset = TrustDataset(
        data_file,
        split="valid",
        trust_label_mode=args.trust_label_mode,
    )
    test_dataset = TrustDataset(
        data_file,
        split="test",
        trust_label_mode=args.trust_label_mode,
    )

    num_classes = train_dataset.num_classes

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    model = Model(input_size=34, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"N steps: {len(train_loader) * epochs}")

    history = []
    best_valid_f1 = float("-inf")
    checkpoint_path = get_checkpoint_path(args.trust_label_mode)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        train_predictions = []
        train_targets = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_current = y.size(0)
            total_train_loss += loss.item() * batch_size_current
            total_train_samples += batch_size_current
            train_predictions.append(predictions.detach().cpu())
            train_targets.append(y.detach().cpu())

        train_predictions_tensor = torch.cat(train_predictions)
        train_targets_tensor = torch.cat(train_targets)
        valid_metrics = evaluate_split(model, valid_loader, criterion, device, num_classes)

        history.append(
            {
                "Train_Loss": total_train_loss / total_train_samples,
                "Train_Acc": float(
                    multiclass_accuracy(
                        train_predictions_tensor,
                        train_targets_tensor,
                        num_classes=num_classes,
                    ).item()
                ),
                "Train_F1": float(
                    multiclass_f1_score(
                        train_predictions_tensor,
                        train_targets_tensor,
                        num_classes=num_classes,
                    ).item()
                ),
                "Valid_Loss": valid_metrics["loss"],
                "Valid_Acc": valid_metrics["acc"],
                "Valid_F1": valid_metrics["f1"],
            }
        )

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "trust_label_mode": args.trust_label_mode,
                    "num_classes": num_classes,
                    "class_values": train_dataset.class_values,
                    "best_epoch": epoch,
                    "valid_metrics": valid_metrics,
                },
                checkpoint_path,
            )

        if epoch % 10 == 0 and epoch > 0:
            epoch_numbers = range(1, epoch + 2)

            plt.plot(epoch_numbers, [h["Train_Acc"] for h in history], label="Train Accuracy")
            plt.plot(epoch_numbers, [h["Valid_Acc"] for h in history], label="Valid Accuracy")

            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")

            plt.xticks(np.arange(0, epochs + 1, 200))
            plt.yticks(np.arange(0, 1.1, 0.2))

            plt.legend(loc="best")
            output_name = (
                f"epoch{epoch}.jpg"
                if args.trust_label_mode == "floor"
                else f"epoch{epoch}.{args.trust_label_mode}.jpg"
            )
            plt.savefig(epochs_dir / output_name)
            plt.close()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_split(model, test_loader, criterion, device, num_classes)
    print(f"Best Validation F1: {checkpoint['valid_metrics']['f1']}")
    print(f"Test Loss: {test_metrics['loss']}")
    print(f"Test Acc: {test_metrics['acc']}")
    print(f"Test F1: {test_metrics['f1']}")
    print(f"Saved model: {checkpoint_path}")

    report = {
        "trust_label_mode": args.trust_label_mode,
        "num_classes": num_classes,
        "class_values": train_dataset.class_values,
        "best_epoch": checkpoint["best_epoch"],
        "validation_metrics": checkpoint["valid_metrics"],
        "test_metrics": test_metrics,
    }
    with get_report_path(args.trust_label_mode).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
