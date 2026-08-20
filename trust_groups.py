"""Canonical "equal-trust group" detection, shared by the PySR pipelines.

``main_group_pysr_trust_calibration.py`` and
``main_group_pysr_trust_calibration_more_predictors.py`` each carried their own
copy of this logic. The two copies had drifted: one returned a ``list`` and the
other a ``set``, so ``key in combinations`` was O(n) in one script and O(1) in
the other, and a de-duplication bug fixed in one would not have reached the
other. They now both import from here.

Caveat worth restating wherever these functions are used: the split conditions on
the *outcome's own variance*. Participants are assigned to ``all_equal_df`` or
``other_rows_df`` on the basis of how much their trust ratings vary, and trust is
what the subsequent search is trying to predict. Any R^2 difference between the
two subsets is therefore partly mechanical and the split must be reported as
exploratory, not as a discovered subgroup effect.
"""

from __future__ import annotations

import pandas as pd

GROUP_KEYS = ["ProlificID", "INTRODUCTION", "SCENARIO"]

# A cell qualifies as "equal trust" if the participant answered almost flatly
# (>= FLAT_THRESHOLD identical ratings) or split roughly evenly between two
# ratings (both >= SPLIT_THRESHOLD, frequencies within MAX_SPLIT_DIFFERENCE).
FLAT_THRESHOLD = 14
SPLIT_THRESHOLD = 7
MAX_SPLIT_DIFFERENCE = 1

GroupKey = tuple[str, str, str]


def find_equal_groups(df: pd.DataFrame) -> set[GroupKey]:
    """``(ProlificID, INTRODUCTION, SCENARIO)`` cells whose trust ratings cluster.

    A cell qualifies if either:

    * its single most common trust value occurs >= 14 times -- the participant
      answered almost flatly; or
    * its two most common trust values each occur >= 7 times **and those two
      frequencies differ by at most 1** -- the participant split roughly evenly
      between two ratings.

    The ``<= 1`` compares *frequencies*, not the trust values themselves, so a
    10/10 split between trust 1 and trust 5 qualifies. The criterion is about how
    concentrated a participant's responses are, not how close the two ratings
    are.

    Computed with a single groupby pass. The original implementation re-scanned
    the whole frame once per row (O(N^2)) and tracked the ">= 7" comparison in a
    dict keyed across *different* cells, so a count from one cell could
    spuriously qualify another.
    """
    trust_counts = df.groupby(GROUP_KEYS)["trust"].value_counts().unstack(fill_value=0)

    combinations: set[GroupKey] = set()
    for key, row in trust_counts.iterrows():
        counts = sorted(row.to_numpy(), reverse=True)
        if counts and counts[0] >= FLAT_THRESHOLD:
            combinations.add(key)
        elif (
            len(counts) >= 2
            and counts[0] >= SPLIT_THRESHOLD
            and counts[1] >= SPLIT_THRESHOLD
            and abs(counts[0] - counts[1]) <= MAX_SPLIT_DIFFERENCE
        ):
            combinations.add(key)

    return combinations


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition ``df`` into ``(all_equal_df, other_rows_df)``.

    The complement is taken by *index*. An earlier version used
    ``df[~df.isin(all_equal_df)].dropna()``, an element-wise value-and-index
    aligned comparison that silently discarded far more rows than intended.
    """
    combinations = find_equal_groups(df)
    if not combinations:
        return df.iloc[0:0].copy(), df.copy()

    equal_frames = [
        df[
            (df["ProlificID"] == participant_id)
            & (df["INTRODUCTION"] == intro)
            & (df["SCENARIO"] == scenario)
        ]
        for participant_id, intro, scenario in combinations
    ]
    # ``sort_index()`` on both halves: ``find_equal_groups`` returns a set, so
    # ``pd.concat`` would otherwise stitch the frames together in an arbitrary,
    # run-dependent order. Row order does not change the least-squares objective,
    # but it does change the stochastic search trajectory and the order points
    # are drawn in, so pinning it is part of making a run reproducible.
    all_equal_df = pd.concat(equal_frames).sort_index()
    other_rows_df = df.drop(index=all_equal_df.index).sort_index()
    return all_equal_df, other_rows_df
