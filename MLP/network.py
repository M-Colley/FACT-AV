"""Trust classifier architectures.

Two things changed here versus the original 34-128-512-1024-1024-5 network:

1. **Capacity.** That network carried ~1.65 M parameters and was trained on
   ~2 200 rows from 107 participants -- roughly 750 parameters per training row.
   It reached its best validation epoch at epoch 11 of 2000 and then memorised
   the train split, which is why every hidden layer needed ``Dropout(0.5)`` to
   stay usable. The default is now ``(64, 32)`` (~4.5 k parameters), which is in
   the right order of magnitude for this sample size. The old shape is still
   reachable with ``--hidden 128 512 1024 1024 --dropout 0.5`` so the two can be
   compared directly.

2. **Ordinal head.** ``trust`` is a 1-5 Likert response: the classes are
   *ordered*, so predicting 1 when the truth is 5 is a worse error than
   predicting 4. Plain softmax cross-entropy treats all confusions as equally
   wrong and throws that structure away. ``head="ordinal"`` implements the CORAL
   scheme (Cao, Mirjalili & Raschka, 2020): the backbone emits one scalar and
   the ``K-1`` cumulative logits share it, differing only by a learned bias per
   threshold. Sharing the weight vector is what guarantees the predicted
   cumulative probabilities stay monotone, so the head cannot produce an
   incoherent "P(y>2) < P(y>3)".
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import Module

HEADS = ("nominal", "ordinal")

DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (64, 32)
DEFAULT_DROPOUT = 0.2


def _backbone(input_size: int, hidden_sizes: Sequence[int], dropout: float) -> torch.nn.Sequential:
    layers: list[Module] = []
    previous = input_size
    for width in hidden_sizes:
        layers.append(torch.nn.Linear(previous, width))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        previous = width
    return torch.nn.Sequential(*layers)


class Model(Module):
    """MLP trust classifier with a selectable nominal or ordinal output head.

    Parameters
    ----------
    input_size
        Width of the encoded feature matrix (see ``TrustDataset.input_size``).
    num_classes
        Number of trust classes ``K``.
    hidden_sizes
        Widths of the hidden layers.
    dropout
        Dropout probability applied after each hidden layer (0 disables it).
    head
        ``"nominal"`` emits ``K`` softmax logits; ``"ordinal"`` emits ``K-1``
        CORAL cumulative logits.

    ``forward`` returns the head's raw logits. Use :meth:`class_logits` when you
    need something that is directly comparable across heads -- for the ordinal
    head it converts the cumulative probabilities into per-class probabilities in
    log space, so downstream ``argmax`` / softmax / calibration code does not
    need to know which head produced them.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 5,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        dropout: float = DEFAULT_DROPOUT,
        head: str = "nominal",
    ) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"Unknown head: {head!r}. Expected one of {HEADS}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.dropout = float(dropout)
        self.head = head

        self.backbone = _backbone(self.input_size, self.hidden_sizes, self.dropout)
        final_width = self.hidden_sizes[-1] if self.hidden_sizes else self.input_size

        if head == "nominal":
            self.output = torch.nn.Linear(final_width, self.num_classes)
        else:
            # CORAL: one shared projection, K-1 independent thresholds.
            self.output = torch.nn.Linear(final_width, 1, bias=False)
            self.thresholds = torch.nn.Parameter(torch.zeros(self.num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.head == "nominal":
            return self.output(features)
        # (N, 1) + (K-1,) -> (N, K-1) cumulative logits for P(y > k).
        return self.output(features) + self.thresholds

    def class_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class scores of shape ``(N, K)``, whatever the head.

        For the nominal head these are the raw logits. For the ordinal head the
        cumulative probabilities ``P(y > k)`` are differenced into class
        probabilities and returned as logs, so ``softmax`` over the result
        reproduces those probabilities and ``argmax`` picks the modal class.
        """
        logits = self.forward(x)
        if self.head == "nominal":
            return logits

        # P(y > k) for k = 0 .. K-2, bracketed by P(y > -1) = 1 and P(y > K-1) = 0.
        greater = torch.sigmoid(logits)
        ones = torch.ones_like(greater[:, :1])
        zeros = torch.zeros_like(greater[:, :1])
        upper = torch.cat([ones, greater], dim=1)  # P(y > k-1)
        lower = torch.cat([greater, zeros], dim=1)  # P(y > k)
        probabilities = (upper - lower).clamp_min(1e-12)
        return probabilities.log()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ordinal_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Binary CORAL targets: ``target[n, k] = 1`` iff ``y[n] > k``."""
    thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


class OrdinalLoss(Module):
    """CORAL loss: mean over the ``K-1`` binary cumulative tasks.

    ``class_weights`` (the same inverse-frequency vector the nominal head uses)
    is folded in per *sample* via the weight of that sample's true class, so the
    rare low-trust participants keep the same influence they have under
    ``CrossEntropyLoss(weight=...)``. Without it the ordinal head would quietly
    revert to chasing the majority classes.
    """

    def __init__(self, num_classes: int, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        targets = ordinal_targets(y, self.num_classes)
        per_task = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        per_sample = per_task.mean(dim=1)
        if self.class_weights is None:
            return per_sample.mean()
        weights = self.class_weights[y]
        total = weights.sum()
        if total <= 0:
            return per_sample.mean()
        return (per_sample * weights).sum() / total
