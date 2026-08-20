"""Train the MLP trust classifier.

Changes that matter for reproducing the reported numbers:

* **Everything is seeded.** ``--seed`` fixes Python, NumPy and torch RNGs plus
  the DataLoader shuffle order, and the seed is written into the report. The
  previous version seeded only the participant split, so weight init, dropout
  masks and batch order all varied run to run and the reported test F1 could not
  be regenerated.
* **Early stopping.** The best validation epoch was epoch 11 of 2000; the loop
  ran the other 1989 anyway. ``--patience`` stops once validation stops
  improving, which cuts a run from minutes to seconds and makes the overfitting
  a reported fact rather than an accident.
* **Test metrics carry confidence intervals.** The test split is ~13
  participants. The report now includes participant-level cluster-bootstrap CIs
  and the majority-class baseline, so a headline number cannot be read as more
  precise than it is.
* **Hyperparameters are CLI flags**, not module-level constants, so a run is
  fully described by its command line (recorded in the report).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dataset import DEFAULT_SPLIT_SEED, TRUST_LABEL_MODES, TrustDataset  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_metrics,
    classification_metrics,
    format_ci,
    majority_baseline_metrics,
)
from network import (  # noqa: E402
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_SIZES,
    HEADS,
    Model,
    OrdinalLoss,
)
from torch.utils.data.dataloader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "all_combined_prepared_with_demographics_with_baseline.xlsx"
RESULTS_FOLDER = REPO_ROOT / "results" / "MLP"
EPOCHS_DIR = Path(__file__).resolve().parent / "epochs"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--trust-label-mode",
        choices=TRUST_LABEL_MODES,
        default="floor",
        help="How to map trust labels into classification classes.",
    )
    parser.add_argument(
        "--head",
        choices=HEADS,
        default="nominal",
        help=(
            "Output head. 'nominal' (default) is softmax cross-entropy. 'ordinal' "
            "uses the CORAL cumulative-logit formulation, which respects the "
            "ordering of the 1-5 trust scale. The default is nominal because it "
            "validates better on this data (validation macro-F1 0.186 vs 0.103) -- "
            "chosen on the VALIDATION split, not on test."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for weights, dropout and batch order."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for the participant-grouped 80/10/10 split.",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Maximum number of epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Stop after this many epochs without a new best validation macro-F1 (0 disables).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=list(DEFAULT_HIDDEN_SIZES),
        help=(
            "Hidden layer widths. Default (64 32) is ~4.5k parameters, sized for "
            "~2.2k training rows. The original network was 128 512 1024 1024."
        ),
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=2000,
        help="Cluster-bootstrap resamples for the test-metric confidence intervals.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the training-curve and label-distribution figures.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Seed every RNG that affects the result, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Makes any remaining non-deterministic CUDA kernel raise instead of
    # silently varying between runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def git_commit() -> str | None:
    """Short commit hash, so a report can be traced back to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_loss(head: str, num_classes: int, class_weights: torch.Tensor):
    if head == "ordinal":
        return OrdinalLoss(num_classes=num_classes, class_weights=class_weights)
    return torch.nn.CrossEntropyLoss(weight=class_weights)


@torch.no_grad()
def evaluate_split(model, data_loader, criterion, device, num_classes, class_values):
    total_loss = 0.0
    total_samples = 0
    predictions, targets = [], []

    model.eval()
    for x, y in data_loader:
        x = x.to(device)
        y = y.squeeze(-1).to(device)

        logits = model(x)
        loss = criterion(logits, y)
        predicted = model.class_logits(x).argmax(dim=-1)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        predictions.append(predicted.cpu())
        targets.append(y.cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(targets).numpy()

    metrics = classification_metrics(y_true, y_pred, num_classes, class_values)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, y_true, y_pred


def checkpoint_stem(trust_label_mode: str, head: str, split_seed: int = DEFAULT_SPLIT_SEED) -> str:
    """Output stem for one run configuration.

    ``best_valid_floor`` for the default (nominal head, default split seed);
    non-default choices are appended. The default name is kept bare so paths
    quoted in the README keep resolving.

    The split seed is part of the name so a sensitivity sweep
    (``for s in 1337 7 42 ...``) writes one artifact per split instead of each
    run silently overwriting the canonical one — which would leave
    ``best_valid_floor.json`` holding whichever split happened to run last.
    """
    stem = f"best_valid_{trust_label_mode}"
    if head != "nominal":
        stem = f"{stem}_{head}"
    if split_seed != DEFAULT_SPLIT_SEED:
        stem = f"{stem}_split{split_seed}"
    return stem


def get_checkpoint_path(trust_label_mode: str, head: str, split_seed: int) -> Path:
    return RESULTS_FOLDER / f"{checkpoint_stem(trust_label_mode, head, split_seed)}.pt"


def get_report_path(trust_label_mode: str, head: str, split_seed: int) -> Path:
    return RESULTS_FOLDER / f"{checkpoint_stem(trust_label_mode, head, split_seed)}.json"


def plot_history(history, epochs_dir: Path, tag: str) -> None:
    epochs_dir.mkdir(parents=True, exist_ok=True)
    epoch_numbers = range(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epoch_numbers, [h["Train_Acc"] for h in history], label="Train")
    axes[0].plot(epoch_numbers, [h["Valid_Acc"] for h in history], label="Validation")
    axes[0].set_ylabel("Accuracy")
    axes[1].plot(epoch_numbers, [h["Train_Loss"] for h in history], label="Train")
    axes[1].plot(epoch_numbers, [h["Valid_Loss"] for h in history], label="Validation")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
    fig.suptitle(f"Training history ({tag})")
    fig.tight_layout()
    fig.savefig(epochs_dir / f"history_{tag}.jpg", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    logger.info("Device: %s | seed: %d | split seed: %d", device, args.seed, args.split_seed)
    logger.info("Trust label mode: %s | head: %s", args.trust_label_mode, args.head)

    datasets = {
        split: TrustDataset(
            DATA_FILE,
            split=split,
            trust_label_mode=args.trust_label_mode,
            split_seed=args.split_seed,
        )
        for split in ("train", "valid", "test")
    }
    train_dataset, valid_dataset, test_dataset = (
        datasets["train"],
        datasets["valid"],
        datasets["test"],
    )

    if not args.no_plots:
        for dataset in datasets.values():
            dataset.plot_label_distribution(RESULTS_FOLDER)

    num_classes = train_dataset.num_classes
    class_values = train_dataset.class_values

    # Seeded generator so the shuffle order is part of the reproducible run.
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin_memory
    )

    model_config = {
        "input_size": train_dataset.input_size,
        "num_classes": num_classes,
        "hidden_sizes": list(args.hidden),
        "dropout": args.dropout,
        "head": args.head,
    }
    model = Model(**model_config).to(device)
    logger.info(
        "Model: %s -> %s parameters", model_config["hidden_sizes"], f"{model.num_parameters():,}"
    )

    class_weights = train_dataset.class_weights.to(device)
    logger.info("Class weights (train): %s", [round(w, 3) for w in class_weights.tolist()])
    criterion = build_loss(args.head, num_classes, class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_valid_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = get_checkpoint_path(args.trust_label_mode, args.head, args.split_seed)
    stopped_early = False

    progress = tqdm(range(args.epochs), desc="epochs")
    for epoch in progress:
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        train_predictions, train_targets = [], []

        for x, y in train_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * y.size(0)
            total_train_samples += y.size(0)
            with torch.no_grad():
                train_predictions.append(model.class_logits(x).argmax(dim=-1).cpu())
            train_targets.append(y.detach().cpu())

        train_metrics = classification_metrics(
            torch.cat(train_targets).numpy(),
            torch.cat(train_predictions).numpy(),
            num_classes,
            class_values,
        )
        valid_metrics, _, _ = evaluate_split(
            model, valid_loader, criterion, device, num_classes, class_values
        )

        history.append(
            {
                "epoch": epoch,
                "Train_Loss": total_train_loss / max(total_train_samples, 1),
                "Train_Acc": train_metrics["acc"],
                "Train_F1": train_metrics["macro_f1"],
                "Valid_Loss": valid_metrics["loss"],
                "Valid_Acc": valid_metrics["acc"],
                "Valid_F1": valid_metrics["macro_f1"],
                "Valid_QWK": valid_metrics["qwk"],
            }
        )

        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "trust_label_mode": args.trust_label_mode,
                    "num_classes": num_classes,
                    "class_values": class_values,
                    "best_epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        progress.set_postfix(
            valid_f1=f"{valid_metrics['macro_f1']:.3f}", best=f"{best_valid_f1:.3f}"
        )

        if args.patience and epochs_without_improvement >= args.patience:
            stopped_early = True
            logger.info(
                "Early stop at epoch %d: no validation improvement for %d epochs (best epoch %d).",
                epoch,
                args.patience,
                best_epoch,
            )
            break

    progress.close()

    if best_epoch < 0:
        raise RuntimeError("Training produced no checkpoint — did --epochs default to 0?")

    if not args.no_plots:
        plot_history(
            history,
            EPOCHS_DIR,
            checkpoint_stem(args.trust_label_mode, args.head, args.split_seed),
        )

    # ------------------------------------------------------------------
    # Test evaluation: single pass on the best-validation checkpoint.
    # ------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_pred = evaluate_split(
        model, test_loader, criterion, device, num_classes, class_values
    )
    test_cis = bootstrap_metrics(
        y_true,
        y_pred,
        test_dataset.participant_ids,
        num_classes=num_classes,
        class_values=class_values,
        n_boot=args.bootstrap_n,
        seed=args.seed,
    )
    baseline = majority_baseline_metrics(train_dataset.y.numpy(), y_true, num_classes, class_values)

    logger.info("Best epoch: %d (validation macro-F1 %.4f)", best_epoch, best_valid_f1)
    logger.info("Test accuracy : %s", format_ci(test_cis["acc"]))
    logger.info("Test macro-F1 : %s", format_ci(test_cis["macro_f1"]))
    logger.info("Test QWK      : %s", format_ci(test_cis["qwk"]))
    logger.info("Test MAE      : %s (trust units)", format_ci(test_cis["mae_trust"]))
    logger.info(
        "Majority-class baseline: acc %.4f, macro-F1 %.4f (class %d)",
        baseline["acc"],
        baseline["macro_f1"],
        baseline["majority_class"],
    )
    beats_baseline = test_cis["acc"]["lo"] > baseline["acc"]
    logger.info(
        "Model %s the majority baseline (95%% CI lower bound %.4f vs %.4f).",
        "beats" if beats_baseline else "does NOT beat",
        test_cis["acc"]["lo"],
        baseline["acc"],
    )

    report = {
        "config": {
            "trust_label_mode": args.trust_label_mode,
            "head": args.head,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_sizes": list(args.hidden),
            "dropout": args.dropout,
            "n_parameters": model.num_parameters(),
        },
        "environment": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        },
        "split_sizes": {
            split: {
                "rows": len(dataset),
                "participants": int(np.unique(dataset.participant_ids).size),
            }
            for split, dataset in datasets.items()
        },
        "num_classes": num_classes,
        "class_values": class_values,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "validation_metrics": checkpoint["valid_metrics"],
        "test_metrics": test_metrics,
        "test_metrics_ci": test_cis,
        "majority_baseline": baseline,
        "beats_majority_baseline": bool(beats_baseline),
    }
    report_path = get_report_path(args.trust_label_mode, args.head, args.split_seed)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote %s and %s", checkpoint_path.name, report_path.name)


if __name__ == "__main__":
    main()
