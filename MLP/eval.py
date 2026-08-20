import argparse
import json
import os
import sys
from pathlib import Path

# IMPORTANT: pyarrow must be imported before matplotlib on Windows -- otherwise
# the matplotlib DLLs change loader state such that pyarrow's later import
# (triggered transitively through sklearn -> pandas) segfaults during
# ``from sklearn.metrics import ...``.
try:  # pragma: no cover - environment-dependent
    import pyarrow  # noqa: F401
except ImportError:
    pass

# Force a non-GUI matplotlib backend before anything imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

# Allow ``import plotting_style`` when running from the MLP/ subdirectory.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dataset import DEFAULT_SPLIT_SEED, TRUST_LABEL_MODES, TrustDataset  # noqa: E402
from metrics import bootstrap_metrics, format_ci, majority_baseline_metrics  # noqa: E402
from network import HEADS, Model  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    mean_absolute_error,
)
from torch.utils.data.dataloader import DataLoader  # noqa: E402
from torchmetrics.functional.classification import (  # noqa: E402
    multiclass_accuracy,
    multiclass_f1_score,
)

from plotting_style import OKABE_ITO, apply_paper_style, save_fig  # noqa: E402

results_folder = Path(__file__).parent.parent / "results" / "MLP"
results_folder.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

# Get the parent directory and construct the path to the data folder
data_folder = Path(__file__).parent.parent / "data"

# Construct the full file path. Mirror train.py: the *_with_baseline file is
# the one with real ProlificIDs needed for the participant-grouped split.
data_file = data_folder / "all_combined_prepared_with_demographics_with_baseline.xlsx"


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
        "--head",
        choices=HEADS,
        default="nominal",
        help="Output head of the checkpoint to load (must match how it was trained).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Must match the --split-seed used at training time.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=2000,
        help="Cluster-bootstrap resamples for the confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the bootstrap resampling.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    return parser.parse_args()


def default_checkpoint_path(trust_label_mode, head="nominal", split_seed=DEFAULT_SPLIT_SEED):
    """Mirrors ``train.checkpoint_stem``: a non-default head or split seed is part
    of the filename, so a split-sensitivity sweep does not overwrite the canonical
    run's checkpoint."""
    stem = f"best_valid_{trust_label_mode}"
    if head != "nominal":
        stem = f"{stem}_{head}"
    if split_seed != DEFAULT_SPLIT_SEED:
        stem = f"{stem}_split{split_seed}"
    return results_folder / f"{stem}.pt"


def resolve_checkpoint(
    trust_label_mode, checkpoint_path=None, head="nominal", split_seed=DEFAULT_SPLIT_SEED
):
    path = checkpoint_path or default_checkpoint_path(trust_label_mode, head, split_seed)
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

    # Returned so the caller can persist it. ECE was previously computed here,
    # rendered into the figure title, and then thrown away.
    return ece


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
        split_seed=args.split_seed,
    )
    # The majority-class baseline has to come from the TRAIN split: taking the
    # modal class from test would be choosing the baseline with the test labels.
    train_dataset = TrustDataset(
        data_file,
        split="train",
        trust_label_mode=args.trust_label_mode,
        split_seed=args.split_seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = resolve_checkpoint(
        args.trust_label_mode, args.checkpoint_path, args.head, args.split_seed
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", test_dataset.num_classes)
        class_values = checkpoint.get("class_values", test_dataset.class_values)
    else:
        model_state_dict = checkpoint
        num_classes = test_dataset.num_classes
        class_values = test_dataset.class_values

    # Architecture is read back from the checkpoint rather than assumed, so a run
    # trained with non-default --hidden/--dropout/--head reloads correctly. Older
    # checkpoints predate model_config; fall back to the original 128/512/1024/1024
    # nominal network those were trained with.
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_config is None:
        model_config = {
            "input_size": test_dataset.input_size,
            "num_classes": num_classes,
            "hidden_sizes": [128, 512, 1024, 1024],
            "dropout": 0.5,
            "head": "nominal",
        }
        print("Checkpoint has no model_config — assuming the original architecture.")

    if num_classes != test_dataset.num_classes:
        raise ValueError(
            f"Checkpoint expects {num_classes} classes but dataset mode "
            f"{args.trust_label_mode!r} resolves to {test_dataset.num_classes} classes."
        )

    model = Model(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint: {checkpoint_path.name} (head={model_config['head']})")

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

            # ``class_logits`` normalises the two heads onto the same (N, K)
            # scale, so the softmax, argmax and calibration code below is
            # identical for the nominal and ordinal models.
            logits = model.class_logits(x)
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
    # NOTE: torchmetrics defaults to macro averaging (mean over per-class
    # scores). With skewed trust labels this differs a lot from overall
    # accuracy, so we report BOTH plus the majority-class baseline to keep the
    # numbers from being misread.
    test_acc = float(
        multiclass_accuracy(y_pred_tensor, y_true_tensor, num_classes=num_classes).item()
    )
    test_f1 = float(
        multiclass_f1_score(y_pred_tensor, y_true_tensor, num_classes=num_classes).item()
    )
    micro_acc = float((y_pred_array == y_true_array).mean())
    class_counts = np.bincount(y_true_array, minlength=num_classes)
    # Baseline = always predict the modal TRAIN class. The previous version used
    # ``class_counts.max()`` from the TEST labels, which both peeks at test and
    # flatters the baseline by definition (it is the best constant *in hindsight*).
    baseline = majority_baseline_metrics(
        train_dataset.y.numpy(), y_true_array, num_classes, class_values
    )
    majority_baseline = baseline["acc"]

    # Participant-level cluster bootstrap. The test split is ~13 participants of
    # ~21 correlated ratings each, so every point estimate below needs an
    # interval attached before it can be quoted.
    test_cis = bootstrap_metrics(
        y_true_array,
        y_pred_array,
        test_dataset.participant_ids,
        num_classes=num_classes,
        class_values=class_values,
        n_boot=args.bootstrap_n,
        seed=args.seed,
    )

    # Ordinal-aware metrics: trust is on an ordered scale, so reward "almost right".
    # ``labels`` is passed explicitly: without it, sklearn infers the label set
    # from the data, so a class absent from both y_true and y_pred silently
    # shrinks the quadratic weight matrix and changes the QWK scale between runs.
    qwk = float(
        cohen_kappa_score(
            y_true_array,
            y_pred_array,
            labels=list(range(num_classes)),
            weights="quadratic",
        )
    )
    class_values_array = np.asarray(class_values, dtype=float)
    true_trust = class_values_array[y_true_array]
    pred_trust = class_values_array[y_pred_array]
    mae_trust = float(mean_absolute_error(true_trust, pred_trust))

    print(f"Test Loss: {test_loss}")
    print(f"Test Acc (macro-recall): {test_acc}")
    print(f"Test F1 (macro): {test_f1}")
    print(f"Test Accuracy (micro/overall): {format_ci(test_cis['acc'])}")
    print(f"Test macro-F1 with CI: {format_ci(test_cis['macro_f1'])}")
    print(
        f"Majority-class baseline (modal train class {baseline['majority_class']}): {majority_baseline:.4f}"
    )
    print(
        f"Predicted-class distribution: {np.bincount(y_pred_array, minlength=num_classes).tolist()}"
    )
    print(f"True-class distribution: {class_counts.tolist()}")
    print(f"Test Quadratic-Weighted Kappa: {format_ci(test_cis['qwk'])}")
    print(f"Test MAE in trust units: {format_ci(test_cis['mae_trust'])}")
    beats_baseline = test_cis["acc"]["lo"] > majority_baseline
    print(
        f"Model {'beats' if beats_baseline else 'does NOT beat'} the majority baseline "
        f"(accuracy CI lower bound {test_cis['acc']['lo']:.4f} vs {majority_baseline:.4f})."
    )

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
    ece = plot_calibration(
        y_true_array,
        y_probs_array,
        n_bins=10,
        save_path=results_folder / f"calibration{suffix}",
        title=f"MLP reliability diagram (mode={args.trust_label_mode})",
    )
    print(f"Test Expected Calibration Error: {ece:.4f}")

    # ------------ Per-class precision/recall/F1 + support table -------------
    write_per_class_report(
        y_true_array,
        y_pred_array,
        class_values,
        results_folder / f"per_class_metrics{suffix}.csv",
    )

    # --------------------- Machine-readable metric dump ---------------------
    # QWK, MAE-in-trust-units and ECE are the metrics the README calls primary
    # for this ordinal task, but they previously existed only as stdout lines and
    # figure titles -- nothing downstream could read them, and reruns could not be
    # compared. Persist everything alongside the CSV and figures.
    metrics_path = results_folder / f"eval_metrics{suffix}.json"
    metrics_payload = {
        "trust_label_mode": args.trust_label_mode,
        "checkpoint": str(checkpoint_path),
        "num_classes": num_classes,
        "class_values": [float(v) for v in class_values],
        "n_test_samples": int(total_samples),
        "test_loss_unweighted": float(test_loss),
        "accuracy_macro_recall": test_acc,
        "accuracy_micro": micro_acc,
        "f1_macro": test_f1,
        "majority_class_baseline": majority_baseline,
        "majority_baseline_detail": baseline,
        # Judged on the CI lower bound, not the point estimate: with 13 test
        # participants a point estimate a hair above the baseline is noise.
        "beats_majority_baseline": bool(test_cis["acc"]["lo"] > majority_baseline),
        "test_metrics_ci": test_cis,
        "bootstrap_n": int(args.bootstrap_n),
        "head": model_config["head"],
        "split_seed": int(args.split_seed),
        "quadratic_weighted_kappa": qwk,
        "mae_trust_units": mae_trust,
        "expected_calibration_error": ece,
        "true_class_distribution": class_counts.tolist(),
        "predicted_class_distribution": np.bincount(y_pred_array, minlength=num_classes).tolist(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
