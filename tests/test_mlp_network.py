"""Tests for MLP/network.py -- capacity, and the CORAL ordinal head.

The ordinal head exists because ``trust`` is an ordered 1-5 response, and its
whole correctness argument is that the ``K-1`` cumulative logits share one
projection so ``P(y > k)`` cannot increase with ``k``. If someone "simplifies"
that into ``K-1`` independent outputs the model still trains and the metrics
still look plausible -- it just quietly stops being an ordinal model.
:meth:`test_cumulative_probabilities_are_monotone` is the guard for that.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MLP"))

from MLP.network import (  # noqa: E402
    DEFAULT_HIDDEN_SIZES,
    HEADS,
    Model,
    OrdinalLoss,
    ordinal_targets,
)

INPUT_SIZE = 34
NUM_CLASSES = 5


def make_batch(n=16, input_size=INPUT_SIZE, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, input_size, generator=generator)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("head", HEADS)
    def test_both_heads_build(self, head):
        model = Model(INPUT_SIZE, NUM_CLASSES, head=head)
        assert model.head == head

    def test_rejects_unknown_head(self):
        with pytest.raises(ValueError, match="Unknown head"):
            Model(INPUT_SIZE, NUM_CLASSES, head="regression")

    def test_rejects_fewer_than_two_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            Model(INPUT_SIZE, num_classes=1)

    def test_default_is_small_enough_for_the_sample_size(self):
        """~2.1k training rows. The original 128/512/1024/1024 network carried
        ~1.65M parameters (~750 per row) and hit its best validation epoch at
        epoch 11. Anything back in that range should fail this test."""
        model = Model(INPUT_SIZE, NUM_CLASSES, hidden_sizes=DEFAULT_HIDDEN_SIZES)
        assert model.num_parameters() < 20_000

    def test_hidden_sizes_are_respected(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, hidden_sizes=(8, 4), dropout=0.0)
        widths = [m.out_features for m in model.backbone if isinstance(m, torch.nn.Linear)]
        assert widths == [8, 4]

    def test_zero_dropout_omits_dropout_layers(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, hidden_sizes=(8,), dropout=0.0)
        assert not any(isinstance(m, torch.nn.Dropout) for m in model.backbone)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_nominal_forward_emits_k_logits(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, head="nominal")
        assert model(make_batch()).shape == (16, NUM_CLASSES)

    def test_ordinal_forward_emits_k_minus_one_logits(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, head="ordinal")
        assert model(make_batch()).shape == (16, NUM_CLASSES - 1)

    @pytest.mark.parametrize("head", HEADS)
    def test_class_logits_is_k_wide_for_both_heads(self, head):
        """Downstream code (argmax, softmax, calibration) must not need to know
        which head produced the scores."""
        model = Model(INPUT_SIZE, NUM_CLASSES, head=head)
        assert model.class_logits(make_batch()).shape == (16, NUM_CLASSES)


# ---------------------------------------------------------------------------
# CORAL semantics
# ---------------------------------------------------------------------------


class TestOrdinalHead:
    def test_cumulative_probabilities_are_monotone(self):
        """P(y > 0) >= P(y > 1) >= ... The shared projection guarantees this;
        independent per-threshold weights would not."""
        model = Model(INPUT_SIZE, NUM_CLASSES, head="ordinal", dropout=0.0)
        model.eval()
        with torch.no_grad():
            greater = torch.sigmoid(model(make_batch(64)))
        differences = greater[:, :-1] - greater[:, 1:]
        assert (differences >= -1e-6).all(), "Cumulative probabilities are not monotone."

    def test_class_probabilities_are_valid(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, head="ordinal", dropout=0.0)
        model.eval()
        with torch.no_grad():
            probabilities = model.class_logits(make_batch(32)).exp()
        assert (probabilities >= 0).all()
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(32), atol=1e-5)

    def test_ordinal_targets_are_cumulative_indicators(self):
        y = torch.tensor([0, 2, 4])
        targets = ordinal_targets(y, NUM_CLASSES)
        assert targets.shape == (3, NUM_CLASSES - 1)
        assert targets[0].tolist() == [0.0, 0.0, 0.0, 0.0]  # y=0 is > nothing
        assert targets[1].tolist() == [1.0, 1.0, 0.0, 0.0]  # y=2 is > 0 and > 1
        assert targets[2].tolist() == [1.0, 1.0, 1.0, 1.0]  # y=4 is > everything

    def test_higher_thresholds_shift_predictions_upward(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, head="ordinal", dropout=0.0)
        model.eval()
        x = make_batch(32)
        with torch.no_grad():
            low = model.class_logits(x).argmax(dim=1)
            model.thresholds.add_(10.0)  # P(y > k) -> 1 for every k
            high = model.class_logits(x).argmax(dim=1)
        assert (high >= low).all()
        assert high.max().item() == NUM_CLASSES - 1


class TestOrdinalLoss:
    def test_perfect_ordering_scores_lower_than_reversed(self):
        criterion = OrdinalLoss(NUM_CLASSES)
        y = torch.tensor([0, 4])
        confident_correct = torch.tensor([[-9.0, -9.0, -9.0, -9.0], [9.0, 9.0, 9.0, 9.0]])
        confident_wrong = torch.tensor([[9.0, 9.0, 9.0, 9.0], [-9.0, -9.0, -9.0, -9.0]])
        assert criterion(confident_correct, y) < criterion(confident_wrong, y)

    def test_near_miss_costs_less_than_far_miss(self):
        """The property plain cross-entropy does not have."""
        criterion = OrdinalLoss(NUM_CLASSES)
        y = torch.tensor([4])
        near = torch.tensor([[6.0, 6.0, 6.0, -6.0]])  # predicts class 3
        far = torch.tensor([[6.0, -6.0, -6.0, -6.0]])  # predicts class 1
        assert criterion(near, y) < criterion(far, y)

    def test_class_weights_raise_the_cost_of_rare_class_errors(self):
        """Upweighting a rare class must raise the loss when that class is the
        one being got wrong. Note the two samples need *different* per-sample
        losses for weighting to do anything at all -- with identical losses any
        weighted mean equals the unweighted one."""
        y = torch.tensor([0, 4])
        # Both rows predict class 4: badly wrong for the rare y=0, correct for y=4.
        logits = torch.full((2, NUM_CLASSES - 1), 9.0)
        unweighted = OrdinalLoss(NUM_CLASSES)(logits, y)
        weighted = OrdinalLoss(NUM_CLASSES, class_weights=torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0]))(
            logits, y
        )
        assert weighted > unweighted

    def test_class_weights_are_ignored_when_none(self):
        y = torch.tensor([0, 4])
        logits = torch.full((2, NUM_CLASSES - 1), 9.0)
        flat = OrdinalLoss(NUM_CLASSES, class_weights=torch.ones(NUM_CLASSES))(logits, y)
        none = OrdinalLoss(NUM_CLASSES)(logits, y)
        assert flat == pytest.approx(none.item())

    def test_gradients_flow_through_the_thresholds(self):
        model = Model(INPUT_SIZE, NUM_CLASSES, head="ordinal", dropout=0.0)
        criterion = OrdinalLoss(NUM_CLASSES)
        loss = criterion(model(make_batch(8)), torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]))
        loss.backward()
        assert model.thresholds.grad is not None
        assert torch.isfinite(model.thresholds.grad).all()
