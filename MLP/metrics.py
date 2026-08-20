"""Metrics for the trust classifier, shared by ``train.py`` and ``eval.py``.

Three things this module exists to enforce:

1. **Ordinal-aware reporting.** ``trust`` is an ordered 1-5 response. Accuracy
   and macro-F1 alone treat "predicted 1, truth 5" the same as "predicted 4,
   truth 5", so they under-report how a model actually fails on a Likert
   outcome. Every report therefore also carries MAE in trust units and
   quadratic-weighted kappa.

2. **Honest uncertainty.** The test split holds ~13 participants. A point
   estimate from 13 participants is close to meaningless on its own, so
   :func:`bootstrap_metrics` attaches percentile confidence intervals -- and it
   resamples *participants*, not rows. Rows within a participant are strongly
   correlated (ICC ~ 0.69 from the mixed-effects baseline); a row-wise bootstrap
   would treat ~279 correlated rows as 279 independent draws and produce an
   interval several times too narrow.

3. **A baseline to beat.** :func:`majority_baseline_metrics` scores the constant
   predictor that always returns the most frequent training class. Reporting it
   next to the model turns "34.4% accuracy" into a statement that can be
   evaluated rather than a number the reader has to contextualise themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

METRIC_KEYS = ("acc", "macro_f1", "mae_class", "mae_trust", "qwk")


def _as_int_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).ravel()


def classification_metrics(
    y_true,
    y_pred,
    num_classes: int,
    class_values: Sequence[float] | None = None,
) -> dict[str, float]:
    """Accuracy, macro-F1 and the two ordinal metrics for one set of predictions.

    ``class_values`` maps class indices back to the trust values they stand for,
    which matters in ``separate_fractional`` mode where the classes are not
    evenly spaced. When it is omitted ``mae_trust`` falls back to ``mae_class``.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if y_true.size == 0:
        return {key: float("nan") for key in METRIC_KEYS}

    labels = list(range(num_classes))
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    mae_class = float(np.abs(y_true - y_pred).mean())

    if class_values is not None:
        lookup = np.asarray(class_values, dtype=float)
        mae_trust = float(np.abs(lookup[y_true] - lookup[y_pred]).mean())
    else:
        mae_trust = mae_class

    # Quadratic weights are the right choice for a Likert outcome: the penalty
    # grows with the square of the distance between predicted and true class.
    # kappa is undefined when both vectors are constant and identical -- that is
    # a degenerate perfect agreement, so report 0 rather than propagating a nan.
    if y_true.size < 2 or (np.unique(y_true).size == 1 and np.unique(y_pred).size == 1):
        qwk = 0.0
    else:
        qwk = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))
        if not np.isfinite(qwk):
            qwk = 0.0

    return {
        "acc": accuracy,
        "macro_f1": macro_f1,
        "mae_class": mae_class,
        "mae_trust": mae_trust,
        "qwk": qwk,
    }


def bootstrap_metrics(
    y_true,
    y_pred,
    participant_ids,
    num_classes: int,
    class_values: Sequence[float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Cluster-bootstrap percentile CIs, resampling participants with replacement.

    Returns ``{metric: {"estimate", "lo", "hi", "se"}}``. ``estimate`` is the
    point estimate on the real sample, not the bootstrap mean.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    participant_ids = np.asarray(participant_ids).ravel()
    if len(participant_ids) != len(y_true):
        raise ValueError(
            f"participant_ids has length {len(participant_ids)} but y_true has {len(y_true)}."
        )

    point = classification_metrics(y_true, y_pred, num_classes, class_values)

    unique_participants = np.unique(participant_ids)
    rows_by_participant = {p: np.flatnonzero(participant_ids == p) for p in unique_participants}
    rng = np.random.default_rng(seed)

    draws: dict[str, list] = {key: [] for key in METRIC_KEYS}
    for _ in range(n_boot):
        sampled = rng.choice(unique_participants, size=len(unique_participants), replace=True)
        idx = np.concatenate([rows_by_participant[p] for p in sampled])
        sample = classification_metrics(y_true[idx], y_pred[idx], num_classes, class_values)
        for key in METRIC_KEYS:
            draws[key].append(sample[key])

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.asarray(draws[key], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[key] = {
                "estimate": point[key],
                "lo": float("nan"),
                "hi": float("nan"),
                "se": float("nan"),
            }
            continue
        out[key] = {
            "estimate": point[key],
            "lo": float(np.percentile(values, lo_q)),
            "hi": float(np.percentile(values, hi_q)),
            "se": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        }
    return out


def majority_baseline_metrics(
    y_train,
    y_true,
    num_classes: int,
    class_values: Sequence[float] | None = None,
) -> dict[str, float]:
    """Score the constant predictor that always returns the modal TRAIN class.

    This is the bar a 5-class classifier has to clear to be worth reporting at
    all. The modal class is taken from train (not test), because picking it from
    test would be using the test labels.
    """
    y_train = _as_int_array(y_train)
    y_true = _as_int_array(y_true)
    counts = np.bincount(y_train, minlength=num_classes)
    majority_class = int(counts.argmax())
    y_pred = np.full_like(y_true, majority_class)
    metrics = classification_metrics(y_true, y_pred, num_classes, class_values)
    metrics["majority_class"] = majority_class
    if class_values is not None:
        metrics["majority_trust_value"] = float(
            np.asarray(class_values, dtype=float)[majority_class]
        )
    return metrics


def format_ci(entry: dict[str, float], digits: int = 3) -> str:
    """``0.344 [0.221, 0.462]`` -- for log lines and README tables."""
    return f"{entry['estimate']:.{digits}f} [{entry['lo']:.{digits}f}, {entry['hi']:.{digits}f}]"
