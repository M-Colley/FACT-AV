"""No participant may appear on both sides of any train/test boundary.

This is the property the whole analysis rests on and nothing tested it. The study
is repeated-measures -- every participant contributes ~21 trust ratings -- so a
row-wise split lets a model memorise a participant's rating level from their
train rows and reuse it on their test rows. Every reported held-out metric would
then be optimistically biased, and nothing in the output would look wrong.

The repo has four independent splitters. All four are covered here:

  * ``ML-approaches.participant_grouped_split``     (feature-importance pipeline)
  * ``MLP.dataset.TrustDataset._participant_grouped_indices``  (3-way MLP split)
  * ``explainability_extras.split_data``            (SHAP / DiCE / Anchors)
  * ``cross_validation.iter_folds``                 (repeated GroupKFold)

Tests run against synthetic frames rather than the real spreadsheets so they stay
fast and keep failing for the right reason if the data file changes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ``ml_approaches`` is a session-scoped fixture from the root conftest, so the
# module (and its shap / catboost / xgboost / torch imports) is loaded once for
# the whole run rather than once per test module.


def make_frame(n_participants: int = 20, per_participant: int = 21, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the columns every splitter needs."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_participants):
        participant = f"P{index:03d}"
        # INTRODUCTION and SCENARIO are between-subject: constant per participant.
        intro = ["ambiguous", "boasting"][index % 2]
        scenario = ["3Spurig", "Spielstrasse", "Ueberland", "NeueMitte"][index % 4]
        for step in range(per_participant):
            rows.append(
                {
                    "ProlificID": participant,
                    "mIoU": float(step * 5),
                    "trust": float(rng.integers(1, 6)),
                    "INTRODUCTION": intro,
                    "SCENARIO": scenario,
                    "Age": float(20 + index),
                    "License": float(index % 10),
                    "Gender": f"A{1 + index % 4}",
                    "Education": f"A{1 + index % 5}",
                    "Job": f"A{1 + index % 6}",
                    "DrivingFrequency": f"A{1 + index % 6}",
                    "Distance": f"A{1 + index % 5}",
                }
            )
    return pd.DataFrame(rows)


def assert_disjoint(left_groups, right_groups, label: str) -> None:
    overlap = set(left_groups) & set(right_groups)
    assert not overlap, f"{label}: participants on both sides of the split: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# 1. ML-approaches.py
# ---------------------------------------------------------------------------


class TestMLApproachesSplit:
    def _split(self, ml_approaches, df, random_state=42):
        X = df[["mIoU", "Age", "License", "SCENARIO", "INTRODUCTION"]]
        y = df["trust"]
        return ml_approaches.participant_grouped_split(
            X, y, groups=df["ProlificID"], test_size=0.2, random_state=random_state
        )

    def test_train_and_test_participants_are_disjoint(self, ml_approaches):
        df = make_frame()
        X_train, X_test, _, _, _ = self._split(ml_approaches, df)
        assert_disjoint(
            df.loc[X_train.index, "ProlificID"],
            df.loc[X_test.index, "ProlificID"],
            "ML-approaches",
        )

    def test_split_covers_every_row_exactly_once(self, ml_approaches):
        df = make_frame()
        X_train, X_test, y_train, y_test, _ = self._split(ml_approaches, df)
        assert len(X_train) + len(X_test) == len(df)
        assert set(X_train.index).isdisjoint(X_test.index)
        assert len(y_train) == len(X_train) and len(y_test) == len(X_test)

    def test_groups_train_aligns_with_x_train(self, ml_approaches):
        """``groups_train`` feeds the CatBoost cluster bootstrap; a misalignment
        there would resample the wrong rows and silently mis-state the error bars."""
        df = make_frame()
        X_train, _, _, _, groups_train = self._split(ml_approaches, df)
        assert groups_train is not None
        assert len(groups_train) == len(X_train)
        assert list(groups_train) == list(df.loc[X_train.index, "ProlificID"])

    def test_is_deterministic_for_a_fixed_random_state(self, ml_approaches):
        df = make_frame()
        first = self._split(ml_approaches, df)[0].index.tolist()
        second = self._split(ml_approaches, df)[0].index.tolist()
        assert first == second

    def test_falls_back_to_row_split_without_groups(self, ml_approaches):
        """The no-group path is a documented fallback, not silent behaviour."""
        df = make_frame()
        X = df[["mIoU", "Age"]]
        X_train, X_test, _, _, groups_train = ml_approaches.participant_grouped_split(
            X, df["trust"], groups=None, test_size=0.2, random_state=42
        )
        assert groups_train is None
        assert len(X_train) + len(X_test) == len(df)


# ---------------------------------------------------------------------------
# 2. MLP three-way split
# ---------------------------------------------------------------------------


class TestMLPSplit:
    @staticmethod
    def _indices(participant_ids, seed=1337):
        from MLP.dataset import TrustDataset

        return TrustDataset._participant_grouped_indices(participant_ids, seed=seed)

    def test_all_three_splits_are_participant_disjoint(self):
        df = make_frame(n_participants=30)
        participant_ids = df["ProlificID"].to_numpy()
        train_idx, valid_idx, test_idx = self._indices(participant_ids)

        train_p = set(participant_ids[train_idx])
        valid_p = set(participant_ids[valid_idx])
        test_p = set(participant_ids[test_idx])

        assert_disjoint(train_p, valid_p, "MLP train/valid")
        assert_disjoint(train_p, test_p, "MLP train/test")
        assert_disjoint(valid_p, test_p, "MLP valid/test")

    def test_splits_partition_every_row(self):
        df = make_frame(n_participants=30)
        participant_ids = df["ProlificID"].to_numpy()
        train_idx, valid_idx, test_idx = self._indices(participant_ids)
        combined = np.concatenate([train_idx, valid_idx, test_idx])
        assert sorted(combined.tolist()) == list(range(len(participant_ids)))

    def test_a_participants_rows_are_never_divided(self):
        df = make_frame(n_participants=30)
        participant_ids = df["ProlificID"].to_numpy()
        splits = self._indices(participant_ids)
        membership = {}
        for name, idx in zip(("train", "valid", "test"), splits, strict=True):
            for participant in participant_ids[idx]:
                membership.setdefault(participant, set()).add(name)
        divided = {p: s for p, s in membership.items() if len(s) > 1}
        assert not divided, f"Participants split across sets: {divided}"

    def test_different_seeds_give_different_splits(self):
        """Guards against the seed being ignored, which would make the
        multi-seed sensitivity check in the README meaningless."""
        participant_ids = make_frame(n_participants=30)["ProlificID"].to_numpy()
        a = self._indices(participant_ids, seed=1337)[2]
        b = self._indices(participant_ids, seed=7)[2]
        assert not np.array_equal(np.sort(a), np.sort(b))

    def test_same_seed_reproduces_the_split(self):
        participant_ids = make_frame(n_participants=30)["ProlificID"].to_numpy()
        a = self._indices(participant_ids, seed=99)
        b = self._indices(participant_ids, seed=99)
        for left, right in zip(a, b, strict=True):
            assert np.array_equal(left, right)


# ---------------------------------------------------------------------------
# 3. explainability_extras
# ---------------------------------------------------------------------------


class TestExplainabilitySplit:
    def test_train_and_test_participants_are_disjoint(self):
        explainability_extras = pytest.importorskip(
            "explainability_extras", reason="explainability dependencies unavailable"
        )
        df = make_frame(n_participants=25)
        config = explainability_extras.ExplainConfig()
        X_train, X_test, _, _ = explainability_extras.split_data(df, config)
        assert_disjoint(
            df.loc[X_train.index, "ProlificID"],
            df.loc[X_test.index, "ProlificID"],
            "explainability_extras",
        )


# ---------------------------------------------------------------------------
# 4. Repeated GroupKFold
# ---------------------------------------------------------------------------


class TestCrossValidationFolds:
    @staticmethod
    def _config(**kwargs):
        import cross_validation

        return cross_validation.CVConfig(**kwargs)

    def test_every_fold_is_participant_disjoint(self):
        import cross_validation

        df = make_frame(n_participants=25)
        config = self._config(folds=5, repeats=3)
        groups = df["ProlificID"]
        folds = list(cross_validation.iter_folds(df, df["trust"], groups, config))

        assert len(folds) == 15
        for repeat, fold, train_idx, test_idx in folds:
            assert_disjoint(
                groups.iloc[train_idx],
                groups.iloc[test_idx],
                f"CV repeat {repeat} fold {fold}",
            )

    def test_each_repeat_partitions_all_rows(self):
        import cross_validation

        df = make_frame(n_participants=25)
        config = self._config(folds=5, repeats=2)
        by_repeat = {}
        for repeat, _, _, test_idx in cross_validation.iter_folds(
            df, df["trust"], df["ProlificID"], config
        ):
            by_repeat.setdefault(repeat, []).append(test_idx)

        for repeat, test_indices in by_repeat.items():
            combined = np.concatenate(test_indices)
            assert sorted(combined.tolist()) == list(range(len(df))), (
                f"Repeat {repeat} test folds do not partition the rows."
            )

    def test_repeats_produce_different_partitions(self):
        """If every repeat produced the same folds, the SD across folds would
        understate the real split-to-split variability."""
        import cross_validation

        df = make_frame(n_participants=25)
        config = self._config(folds=5, repeats=2)
        first_folds = {}
        for repeat, fold, _, test_idx in cross_validation.iter_folds(
            df, df["trust"], df["ProlificID"], config
        ):
            first_folds.setdefault(repeat, {})[fold] = tuple(sorted(test_idx.tolist()))

        assert first_folds[0] != first_folds[1]


# ---------------------------------------------------------------------------
# 5. Nested hyperparameter tuning must also be participant-grouped
# ---------------------------------------------------------------------------


class TestNestedTuning:
    """``--tune`` selects hyperparameters per outer fold. If the *inner* split
    were row-wise, selection would see participants from its own validation rows
    and the chosen settings would be tuned against leaked data — invisible in the
    output, and it would make the tuned scores look better than they are."""

    @staticmethod
    def _config(**kwargs):
        import cross_validation

        return cross_validation.CVConfig(**kwargs)

    def test_tuning_off_returns_the_estimator_unchanged(self):
        import cross_validation

        estimator = object()
        assert (
            cross_validation.maybe_tune("Random Forest", estimator, self._config(tune=False))
            is estimator
        )

    def test_models_without_a_search_space_are_untouched(self):
        import cross_validation

        estimator = object()
        config = self._config(tune=True)
        assert cross_validation.maybe_tune("Mean baseline", estimator, config) is estimator
        assert cross_validation.maybe_tune("TabPFN", estimator, config) is estimator

    def test_tuning_wraps_in_a_grouped_search(self):
        from sklearn.model_selection import GroupKFold, RandomizedSearchCV

        import cross_validation

        config = self._config(tune=True, tune_iterations=3, tune_inner_folds=2)
        wrapped = cross_validation.maybe_tune("Ridge", object(), config)

        assert isinstance(wrapped, RandomizedSearchCV)
        assert isinstance(wrapped.cv, GroupKFold), (
            "The inner splitter must be GroupKFold — a row-wise inner split would "
            "tune against leaked participants."
        )
        assert wrapped.cv.get_n_splits() == 2
        assert wrapped.n_iter == 3

    def test_search_spaces_target_the_regressor_step(self):
        """Every key must be prefixed ``regressor__`` to reach the estimator
        inside the preprocessing Pipeline; an unprefixed key raises at fit time,
        which would be hours into a tuned run."""
        import cross_validation

        for name, space in cross_validation.PARAM_DISTRIBUTIONS.items():
            bad = [key for key in space if not key.startswith("regressor__")]
            assert not bad, f"{name}: keys must target the pipeline's regressor step: {bad}"

    def test_a_tuned_fit_runs_end_to_end_and_picks_a_parameter(self):
        import cross_validation

        df = make_frame(n_participants=12, per_participant=6)
        X = df[cross_validation.NUM_FEATURES + cross_validation.CAT_FEATURES]
        y = df["trust"]
        groups = df["ProlificID"]

        config = self._config(tune=True, tune_iterations=2, tune_inner_folds=2)
        model = cross_validation.maybe_tune(
            "Ridge", cross_validation.build_model_factories(config)["Ridge"](), config
        )
        model.fit(X, y, groups=groups)

        assert "regressor__alpha" in model.best_params_
        assert len(model.predict(X)) == len(df)


class TestCVOutputNaming:
    """A quick exploratory run must not overwrite the committed headline result.

    Both this and ``MLP/train.py``'s checkpoint naming encode non-default
    configuration in the filename. That is not cosmetic: a ``--folds 3 --repeats 1``
    run writing to ``cv_metrics.json`` replaces a 25-fold result with a 3-fold one,
    and the only trace is a smaller file.
    """

    @staticmethod
    def _stem(**kwargs):
        import cross_validation

        return cross_validation.output_stem(cross_validation.CVConfig(**kwargs))

    def test_canonical_run_keeps_the_bare_name(self):
        assert self._stem(folds=5, repeats=5) == "cv"

    def test_non_default_shape_is_encoded(self):
        assert self._stem(folds=3, repeats=1) == "cv_3x1"

    def test_tuning_is_encoded(self):
        stem = self._stem(folds=5, repeats=5, tune=True, tune_iterations=20, tune_inner_folds=3)
        assert stem == "cv_tuned20x3"

    def test_shape_and_tuning_combine(self):
        stem = self._stem(folds=3, repeats=1, tune=True, tune_iterations=5, tune_inner_folds=2)
        assert stem == "cv_3x1_tuned5x2"

    def test_distinct_configurations_get_distinct_stems(self):
        configurations = [
            dict(folds=5, repeats=5),
            dict(folds=3, repeats=1),
            dict(folds=5, repeats=5, tune=True),
            dict(folds=3, repeats=1, tune=True),
            dict(folds=10, repeats=2),
        ]
        stems = [self._stem(**c) for c in configurations]
        assert len(set(stems)) == len(stems), f"Colliding output names: {stems}"
