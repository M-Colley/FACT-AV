"""Tests for helper functions in main_group_pysr_trust_calibration_more_predictors.py.

PySR requires a working Julia installation. These tests are skipped automatically
when Julia is not available (e.g. fresh CI runners, hash-mismatch download failures).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip the whole module if PySR/Julia cannot be imported.
pytest.importorskip("pysr", reason="PySR / Julia not available in this environment")

from main_group_pysr_trust_calibration_more_predictors import (  # noqa: E402
    build_feature_matrix,
    find_equal_groups,
    split_groups,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(
    n_p1: int = 20,
    n_p2: int = 5,
    p1_trust: list | None = None,
) -> pd.DataFrame:
    """Minimal DataFrame that satisfies the schema expected by the helpers."""
    if p1_trust is None:
        # P1 has 14 rows with trust=3 (meets ≥14 threshold) and 6 with trust=4
        p1_trust = [3.0] * 14 + [4.0] * (n_p1 - 14)

    rng = np.random.default_rng(0)
    n = n_p1 + n_p2
    return pd.DataFrame(
        {
            "ProlificID": ["P1"] * n_p1 + ["P2"] * n_p2,
            "INTRODUCTION": ["ambiguous"] * n,
            "SCENARIO": ["3Spurig"] * n,
            "trust": p1_trust + [float(i % 5 + 1) for i in range(n_p2)],
            "mIoU": rng.uniform(0.3, 0.9, n),
            "Age": [30] * n,
            "Gender": ["A1"] * n,
            "Education": ["A3"] * n,
            "Job": ["A2"] * n,
            "License": ["Y"] * n,
            "DrivingFrequency": ["A1"] * n,
            "Distance": ["A2"] * n,
        }
    )


# ---------------------------------------------------------------------------
# find_equal_groups
# ---------------------------------------------------------------------------

class TestFindEqualGroups:
    def test_detects_group_with_14_identical_trust_values(self):
        df = _make_df()
        groups = find_equal_groups(df)
        assert ("P1", "ambiguous", "3Spurig") in groups

    def test_p2_not_in_groups_when_below_threshold(self):
        df = _make_df()
        groups = find_equal_groups(df)
        assert ("P2", "ambiguous", "3Spurig") not in groups

    def test_returns_empty_set_when_no_threshold_met(self):
        df = _make_df(n_p1=5, p1_trust=[1.0, 2.0, 3.0, 4.0, 5.0])
        groups = find_equal_groups(df)
        assert len(groups) == 0

    def test_returns_a_set(self):
        df = _make_df()
        groups = find_equal_groups(df)
        assert isinstance(groups, set)

    def test_group_key_is_tuple_of_three(self):
        df = _make_df()
        groups = find_equal_groups(df)
        for key in groups:
            assert isinstance(key, tuple)
            assert len(key) == 3


# ---------------------------------------------------------------------------
# split_groups
# ---------------------------------------------------------------------------

class TestSplitGroups:
    def test_returns_two_dataframes(self):
        df = _make_df()
        equal_df, other_df = split_groups(df)
        assert isinstance(equal_df, pd.DataFrame)
        assert isinstance(other_df, pd.DataFrame)

    def test_no_index_overlap_between_splits(self):
        df = _make_df()
        equal_df, other_df = split_groups(df)
        overlap = equal_df.index.intersection(other_df.index)
        assert len(overlap) == 0

    def test_split_sizes_sum_to_original(self):
        df = _make_df()
        equal_df, other_df = split_groups(df)
        assert len(equal_df) + len(other_df) == len(df)

    def test_equal_df_contains_balanced_participant(self):
        df = _make_df()
        equal_df, _ = split_groups(df)
        assert "P1" in equal_df["ProlificID"].values

    def test_empty_equal_df_when_no_threshold_met(self):
        df = _make_df(n_p1=5, p1_trust=[1.0, 2.0, 3.0, 4.0, 5.0])
        equal_df, other_df = split_groups(df)
        assert len(equal_df) == 0
        assert len(other_df) == len(df)

    def test_other_df_retains_all_columns(self):
        df = _make_df()
        _, other_df = split_groups(df)
        assert set(df.columns) == set(other_df.columns)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_numpy_array(self):
        df = _make_df()
        result = build_feature_matrix(df)
        assert isinstance(result, np.ndarray)

    def test_row_count_matches_input(self):
        df = _make_df()
        result = build_feature_matrix(df)
        assert result.shape[0] == len(df)

    def test_has_more_columns_than_raw_inputs(self):
        # 2 numeric (mIoU, Age) + one-hot for 5 categorical cols + 3 ordinals
        df = _make_df()
        result = build_feature_matrix(df)
        assert result.shape[1] > 5

    def test_numeric_features_are_in_first_two_columns(self):
        df = _make_df()
        result = build_feature_matrix(df)
        # First column should be mIoU values
        np.testing.assert_array_almost_equal(result[:, 0], df["mIoU"].to_numpy())

    def test_all_values_are_finite(self):
        df = _make_df()
        result = build_feature_matrix(df)
        assert np.all(np.isfinite(result))

    def test_dtype_is_float(self):
        df = _make_df()
        result = build_feature_matrix(df)
        assert np.issubdtype(result.dtype, np.floating)
