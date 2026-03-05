from pathlib import Path

import numpy as np
import torch
from matplotlib.pylab import plt
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from tqdm import tqdm

from dataset import TrustDataset
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
num_classes = 5


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_split(model, data_loader, criterion, device):
    losses = []
    accuracies = []
    f1_scores = []

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
            acc = multiclass_accuracy(predictions, y, num_classes=num_classes)
            f1 = multiclass_f1_score(predictions, y, num_classes=num_classes)

            losses.append(loss.item())
            accuracies.append(acc.item())
            f1_scores.append(f1.item())

    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(accuracies)),
        "f1": float(np.mean(f1_scores)),
    }


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = TrustDataset(data_file, split="train")
    valid_dataset = TrustDataset(data_file, split="valid")
    test_dataset = TrustDataset(data_file, split="test")

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

    model = Model(input_size=34).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"N steps: {len(train_loader) * epochs}")

    history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_losses = []
        train_accuracies = []
        train_f1_scores = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = multiclass_accuracy(predictions, y, num_classes=num_classes)
            f1 = multiclass_f1_score(predictions, y, num_classes=num_classes)

            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_f1_scores.append(f1.item())

        valid_metrics = evaluate_split(model, valid_loader, criterion, device)

        history.append(
            {
                "Train_Loss": float(np.mean(train_losses)),
                "Train_Acc": float(np.mean(train_accuracies)),
                "Train_F1": float(np.mean(train_f1_scores)),
                "Valid_Loss": valid_metrics["loss"],
                "Valid_Acc": valid_metrics["acc"],
                "Valid_F1": valid_metrics["f1"],
            }
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
            plt.savefig(epochs_dir / f"epoch{epoch}.jpg")
            plt.close()

    test_metrics = evaluate_split(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']}")
    print(f"Test Acc: {test_metrics['acc']}")
    print(f"Test F1: {test_metrics['f1']}")

    model_path = results_folder / f"test_f1_{test_metrics['f1']:0.5f}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
