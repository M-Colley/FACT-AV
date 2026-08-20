"""Tests for MLP/metrics.py -- ordinal metrics, cluster bootstrap, baseline.

The property worth guarding here is that :func:`bootstrap_metrics` resamples
*participants*, not rows. Rows within a participant are strongly correlated
(ICC ~ 0.69), so a row-wise bootstrap would report an interval several times too
narrow -- and it would look completely normal in the output. The width tests
below are what catch a regression to row-wise resampling.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MLP"))

from MLP.metrics import (  # noqa: E402
    bootstrap_metrics,
    classification_metrics,
    format_ci,
    majority_baseline_metrics,
)

# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_prediction_scores_one(self):
        y = np.array([0, 1, 2, 3, 4])
        metrics = classification_metrics(y, y, num_classes=5)
        assert metrics["acc"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["mae_class"] == pytest.approx(0.0)

    def test_mae_class_counts_index_distance(self):
        y_true = np.array([0, 0, 4])
        y_pred = np.array([1, 0, 2])  # errors of 1, 0 and 2 classes
        metrics = classification_metrics(y_true, y_pred, num_classes=5)
        assert metrics["mae_class"] == pytest.approx(1.0)

    def test_mae_trust_uses_class_values_not_indices(self):
        """In separate_fractional mode the classes are half-steps, so a
        one-class error is 0.5 trust units, not 1.0."""
        class_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 3])
        metrics = classification_metrics(y_true, y_pred, num_classes=5, class_values=class_values)
        assert metrics["mae_class"] == pytest.approx(1.0)
        assert metrics["mae_trust"] == pytest.approx(0.5)

    def test_mae_trust_falls_back_to_class_units_without_class_values(self):
        y_true = np.array([0, 1])
        y_pred = np.array([1, 2])
        metrics = classification_metrics(y_true, y_pred, num_classes=5)
        assert metrics["mae_trust"] == metrics["mae_class"]

    def test_qwk_penalises_distant_errors_more(self):
        """The whole reason for using QWK on a Likert outcome."""
        y_true = np.array([0, 1, 2, 3, 4] * 4)
        near = (y_true + 1).clip(0, 4)
        far = (y_true + 3) % 5
        near_qwk = classification_metrics(y_true, near, num_classes=5)["qwk"]
        far_qwk = classification_metrics(y_true, far, num_classes=5)["qwk"]
        assert near_qwk > far_qwk

    def test_qwk_uses_the_full_label_set(self):
        """Passing ``labels`` explicitly keeps the quadratic weight matrix at
        K x K even when a class is missing from both vectors; otherwise QWK
        silently changes scale between runs."""
        y_true = np.array([1, 2, 1, 2])
        y_pred = np.array([1, 1, 2, 2])
        five = classification_metrics(y_true, y_pred, num_classes=5)["qwk"]
        three = classification_metrics(y_true, y_pred, num_classes=3)["qwk"]
        assert five == pytest.approx(three)  # same weights over the observed range
        assert np.isfinite(five)

    def test_empty_input_returns_nan_rather_than_raising(self):
        metrics = classification_metrics([], [], num_classes=5)
        assert all(np.isnan(value) for value in metrics.values())

    def test_degenerate_constant_agreement_does_not_produce_nan(self):
        y = np.array([2, 2, 2, 2])
        metrics = classification_metrics(y, y, num_classes=5)
        assert np.isfinite(metrics["qwk"])


# ---------------------------------------------------------------------------
# bootstrap_metrics
# ---------------------------------------------------------------------------


def _clustered_case(n_participants=12, per_participant=21, seed=0):
    """Every participant answers one constant value — maximal within-cluster
    correlation, which is the regime where row vs cluster bootstrap diverge."""
    rng = np.random.default_rng(seed)
    participants, y_true, y_pred = [], [], []
    for index in range(n_participants):
        value = int(rng.integers(0, 5))
        prediction = value if index % 2 == 0 else (value + 1) % 5
        participants.extend([f"P{index}"] * per_participant)
        y_true.extend([value] * per_participant)
        y_pred.extend([prediction] * per_participant)
    return np.array(y_true), np.array(y_pred), np.array(participants)


class TestBootstrapMetrics:
    def test_estimate_matches_the_point_estimate(self):
        y_true, y_pred, participants = _clustered_case()
        point = classification_metrics(y_true, y_pred, num_classes=5)
        cis = bootstrap_metrics(y_true, y_pred, participants, num_classes=5, n_boot=200, seed=0)
        assert cis["acc"]["estimate"] == pytest.approx(point["acc"])

    def test_interval_brackets_the_estimate(self):
        y_true, y_pred, participants = _clustered_case()
        cis = bootstrap_metrics(y_true, y_pred, participants, num_classes=5, n_boot=500, seed=0)
        entry = cis["acc"]
        assert entry["lo"] <= entry["estimate"] <= entry["hi"]

    def test_resamples_participants_not_rows(self):
        """The regression guard. With perfectly correlated within-participant
        rows, a row-wise bootstrap on 252 rows would give a very narrow interval;
        resampling 12 participants must give a visibly wide one."""
        y_true, y_pred, participants = _clustered_case()
        cis = bootstrap_metrics(y_true, y_pred, participants, num_classes=5, n_boot=1000, seed=0)
        width = cis["acc"]["hi"] - cis["acc"]["lo"]
        assert width > 0.3, (
            f"Accuracy CI width {width:.3f} is too narrow for 12 clusters — "
            "this is what row-wise resampling would look like."
        )

    def test_more_participants_narrows_the_interval(self):
        narrow = bootstrap_metrics(
            *_clustered_case(n_participants=80, seed=1), num_classes=5, n_boot=500, seed=0
        )
        wide = bootstrap_metrics(
            *_clustered_case(n_participants=8, seed=1), num_classes=5, n_boot=500, seed=0
        )
        narrow_width = narrow["acc"]["hi"] - narrow["acc"]["lo"]
        wide_width = wide["acc"]["hi"] - wide["acc"]["lo"]
        assert narrow_width < wide_width

    def test_is_reproducible_for_a_fixed_seed(self):
        y_true, y_pred, participants = _clustered_case()
        kwargs = dict(num_classes=5, n_boot=200)
        first = bootstrap_metrics(y_true, y_pred, participants, seed=7, **kwargs)
        second = bootstrap_metrics(y_true, y_pred, participants, seed=7, **kwargs)
        assert first["acc"] == second["acc"]

    def test_mismatched_participant_ids_raise(self):
        y_true, y_pred, participants = _clustered_case()
        with pytest.raises(ValueError, match="participant_ids"):
            bootstrap_metrics(y_true, y_pred, participants[:-1], num_classes=5, n_boot=10)


# ---------------------------------------------------------------------------
# majority_baseline_metrics
# ---------------------------------------------------------------------------


class TestMajorityBaseline:
    def test_uses_the_modal_train_class_not_the_test_one(self):
        """Choosing the constant from test would be picking the best baseline in
        hindsight, using labels the model is not allowed to see."""
        y_train = np.array([0] * 50 + [4] * 10)  # modal train class is 0
        y_test = np.array([4] * 9 + [0])  # modal test class is 4
        baseline = majority_baseline_metrics(y_train, y_test, num_classes=5)
        assert baseline["majority_class"] == 0
        assert baseline["acc"] == pytest.approx(0.1)

    def test_reports_the_trust_value_of_the_majority_class(self):
        y_train = np.array([2] * 10)
        y_test = np.array([2, 3])
        baseline = majority_baseline_metrics(
            y_train, y_test, num_classes=5, class_values=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert baseline["majority_trust_value"] == pytest.approx(3.0)

    def test_macro_f1_of_a_constant_predictor_is_low(self):
        y_train = np.array([3] * 40)
        y_test = np.array([0, 1, 2, 3, 4] * 4)
        baseline = majority_baseline_metrics(y_train, y_test, num_classes=5)
        assert baseline["macro_f1"] < 0.25


def test_format_ci_is_readable():
    entry = {"estimate": 0.3442, "lo": 0.2211, "hi": 0.4623}
    assert format_ci(entry) == "0.344 [0.221, 0.462]"
